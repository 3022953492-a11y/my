#!/usr/bin/env python3
"""
双流融合层（Joint Fusion Encoder）实现
使用Cross-Attention进行图流和序列流的融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 查询、键、值投影
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: 查询张量 [batch_size, seq_len_q, d_model]
            key: 键张量 [batch_size, seq_len_k, d_model]
            value: 值张量 [batch_size, seq_len_v, d_model]
            mask: 注意力掩码
        """
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k, seq_len_v = key.size(1), value.size(1)
        
        # 线性投影
        q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # 输出投影
        output = self.w_o(attn_output)
        
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: 输入序列 [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class CrossAttentionFusionLayer(nn.Module):
    """交叉注意力融合层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(CrossAttentionFusionLayer, self).__init__()
        
        # 图流到序列流的交叉注意力
        self.graph_to_seq_attention = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 序列流到图流的交叉注意力
        self.seq_to_graph_attention = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, graph_features, sequence_features):
        """
        Args:
            graph_features: 图特征 [batch_size, d_model]
            sequence_features: 序列特征 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = sequence_features.shape
        
        # 扩展图特征以匹配序列维度
        graph_features_expanded = graph_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 图流到序列流的交叉注意力
        seq_enhanced, attn1 = self.graph_to_seq_attention(
            query=sequence_features,
            key=graph_features_expanded,
            value=graph_features_expanded
        )
        seq_enhanced = self.norm1(sequence_features + self.dropout1(seq_enhanced))
        
        # 序列流到图流的交叉注意力
        # 首先对序列特征进行池化得到序列级别的表示
        seq_pooled = torch.mean(sequence_features, dim=1)  # [batch_size, d_model]
        
        graph_enhanced, attn2 = self.seq_to_graph_attention(
            query=graph_features.unsqueeze(1),
            key=seq_pooled.unsqueeze(1),
            value=seq_pooled.unsqueeze(1)
        )
        graph_enhanced = self.norm2(graph_features.unsqueeze(1) + self.dropout2(graph_enhanced))
        graph_enhanced = graph_enhanced.squeeze(1)
        
        # 前馈网络（应用于序列特征）
        ff_output = self.feed_forward(seq_enhanced)
        seq_output = self.norm3(seq_enhanced + self.dropout3(ff_output))
        
        return graph_enhanced, seq_output, attn1, attn2


class GatedFusion(nn.Module):
    """门控融合机制"""
    
    def __init__(self, d_model):
        super(GatedFusion, self).__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(self, graph_features, sequence_features):
        """
        Args:
            graph_features: 图特征 [batch_size, d_model]
            sequence_features: 序列特征 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = sequence_features.shape
        
        # 扩展图特征
        graph_expanded = graph_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 计算门控权重
        concat_features = torch.cat([sequence_features, graph_expanded], dim=-1)
        gate_weights = self.gate(concat_features)
        
        # 门控融合
        fused_features = gate_weights * sequence_features + (1 - gate_weights) * graph_expanded
        
        # 最终投影
        output = self.fusion_proj(torch.cat([fused_features, sequence_features], dim=-1))
        
        return output


class FusionEncoder(nn.Module):
    """融合编码器主模块"""
    
    def __init__(self, d_model=256, num_heads=8, num_layers=2, d_ff=512, 
                 fusion_type='cross_attention', dropout=0.1):
        super(FusionEncoder, self).__init__()
        
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        # 特征对齐投影
        self.graph_proj = nn.Linear(d_model, d_model)
        self.seq_proj = nn.Linear(d_model, d_model)
        
        # 选择融合方式
        if fusion_type == 'cross_attention':
            self.fusion_layers = nn.ModuleList([
                CrossAttentionFusionLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            self.final_fusion = GatedFusion(d_model)
            
        elif fusion_type == 'gated_fusion':
            self.fusion_layer = GatedFusion(d_model)
            
        elif fusion_type == 'concat_mlp':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, graph_features, sequence_features):
        """
        Args:
            graph_features: 图特征 [batch_size, d_model]
            sequence_features: 序列特征 [batch_size, seq_len, d_model]
        Returns:
            fused_features: 融合特征 [batch_size, seq_len, d_model]
        """
        # 特征对齐
        graph_aligned = self.graph_proj(graph_features)
        seq_aligned = self.seq_proj(sequence_features)
        
        # 融合处理
        if self.fusion_type == 'cross_attention':
            # 多层交叉注意力融合
            graph_fused = graph_aligned
            seq_fused = seq_aligned
            
            for layer in self.fusion_layers:
                graph_fused, seq_fused, attn1, attn2 = layer(graph_fused, seq_fused)
            
            # 最终门控融合
            fused_features = self.final_fusion(graph_fused, seq_fused)
            
        elif self.fusion_type == 'gated_fusion':
            fused_features = self.fusion_layer(graph_aligned, seq_aligned)
            
        elif self.fusion_type == 'concat_mlp':
            batch_size, seq_len, d_model = seq_aligned.shape
            
            # 扩展图特征
            graph_expanded = graph_aligned.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 拼接特征
            concat_features = torch.cat([seq_aligned, graph_expanded], dim=-1)
            
            # MLP融合
            fused_features = self.fusion_mlp(concat_features)
        
        # 输出投影
        output = self.output_proj(fused_features)
        
        return output


def test_fusion_encoder():
    """测试融合编码器"""
    # 创建测试数据
    batch_size, seq_len, d_model = 2, 50, 256
    
    graph_features = torch.randn(batch_size, d_model)
    sequence_features = torch.randn(batch_size, seq_len, d_model)
    
    # 测试交叉注意力融合
    print("测试交叉注意力融合...")
    fusion_cross = FusionEncoder(fusion_type='cross_attention')
    fused_cross = fusion_cross(graph_features, sequence_features)
    print(f"交叉注意力融合输出维度: {fused_cross.shape}")
    
    # 测试门控融合
    print("\n测试门控融合...")
    fusion_gated = FusionEncoder(fusion_type='gated_fusion')
    fused_gated = fusion_gated(graph_features, sequence_features)
    print(f"门控融合输出维度: {fused_gated.shape}")
    
    # 测试拼接+MLP融合
    print("\n测试拼接+MLP融合...")
    fusion_concat = FusionEncoder(fusion_type='concat_mlp')
    fused_concat = fusion_concat(graph_features, sequence_features)
    print(f"拼接+MLP融合输出维度: {fused_concat.shape}")
    
    print("融合编码器测试通过!")


if __name__ == "__main__":
    test_fusion_encoder()