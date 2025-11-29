#!/usr/bin/env python3
"""
CTC + CRF 混合解码器实现
结合CTC的帧级别预测和CRF的序列级约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CRF(nn.Module):
    """线性链条件随机场"""
    
    def __init__(self, num_tags, batch_first=True):
        super(CRF, self).__init__()
        
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # 转移矩阵：从标签i到标签j的转移分数
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # 开始和结束状态分数
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        # 化学语法约束（硬约束）
        self.register_buffer('constraint_mask', torch.ones(num_tags, num_tags))
        
        # 设备跟踪
        self._device = None
        
    def to(self, device):
        """重写to方法，确保设备一致性"""
        self._device = device
        return super().to(device)
        
    def _get_device(self, tensor):
        """获取张量的设备，如果未设置则使用张量的设备"""
        if self._device is not None:
            return self._device
        return tensor.device
        
    def _constrain_transitions(self, emissions, tags):
        """应用化学语法约束"""
        # 这里可以添加化学公式的语法约束
        # 例如：数字后面不能直接跟字母，上标后面不能直接跟下标等
        pass
        
    def forward(self, emissions, tags, mask=None):
        """计算CRF的负对数似然"""
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        return self._compute_loss(emissions, tags, mask)
    
    def _compute_loss(self, emissions, tags, mask=None):
        """计算CRF损失"""
        # 前向算法计算配分函数
        partition = self._forward_algorithm(emissions, mask)
        
        # 计算真实路径的分数
        score = self._score_sequence(emissions, tags, mask)
        
        # 负对数似然
        nll = partition - score
        
        return nll.mean()
    
    def _forward_algorithm(self, emissions, mask=None):
        """前向算法计算配分函数"""
        seq_length, batch_size, num_tags = emissions.shape
        device = self._get_device(emissions)
        
        # 初始化前向变量（确保转移矩阵在正确的设备上）
        start_transitions = self.start_transitions.to(device)
        alpha = start_transitions + emissions[0]
        
        for t in range(1, seq_length):
            # 广播计算（确保转移矩阵在正确的设备上）
            emit_scores = emissions[t].unsqueeze(1)  # [batch_size, 1, num_tags]
            trans_scores = self.transitions.unsqueeze(0).to(device)  # [1, num_tags, num_tags]
            
            # 前向递推
            next_alpha = torch.logsumexp(alpha.unsqueeze(2) + trans_scores + emit_scores, dim=1)
            
            # 应用掩码
            if mask is not None:
                alpha = torch.where(mask[t].unsqueeze(1), next_alpha, alpha)
            else:
                alpha = next_alpha
        
        # 加上结束转移（确保结束转移矩阵在正确的设备上）
        end_transitions = self.end_transitions.to(device)
        alpha = alpha + end_transitions.unsqueeze(0)
        
        # 最终配分函数
        partition = torch.logsumexp(alpha, dim=1)
        
        return partition
    
    def _score_sequence(self, emissions, tags, mask=None):
        """计算真实路径的分数"""
        seq_length, batch_size = tags.shape
        device = self._get_device(emissions)
        
        # 开始转移分数（确保转移矩阵在正确的设备上）
        start_transitions = self.start_transitions.to(device)
        score = start_transitions[tags[0]]
        
        # 发射分数和转移分数
        for t in range(seq_length - 1):
            # 当前发射分数
            score += emissions[t].gather(1, tags[t].unsqueeze(1)).squeeze(1)
            
            # 转移分数（确保转移矩阵在正确的设备上）
            transitions = self.transitions.to(device)
            score += transitions[tags[t], tags[t + 1]]
            
            # 应用掩码
            if mask is not None:
                score = score * mask[t]
        
        # 最后一个发射分数
        score += emissions[-1].gather(1, tags[-1].unsqueeze(1)).squeeze(1)
        
        # 结束转移分数（确保结束转移矩阵在正确的设备上）
        end_transitions = self.end_transitions.to(device)
        score += end_transitions[tags[-1]]
        
        return score
    
    def decode(self, emissions, mask=None):
        """维特比解码"""
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        return self._viterbi_decode(emissions, mask)
    
    def _viterbi_decode(self, emissions, mask=None):
        """维特比算法解码"""
        seq_length, batch_size, num_tags = emissions.shape
        device = self._get_device(emissions)
        
        # 初始化（确保在正确的设备上）
        viterbi = torch.zeros(seq_length, batch_size, num_tags, device=device)
        backpointers = torch.zeros(seq_length, batch_size, num_tags, dtype=torch.long, device=device)
        
        # 第一步（确保转移矩阵在正确的设备上）
        start_transitions = self.start_transitions.to(device)
        viterbi[0] = start_transitions + emissions[0]
        
        for t in range(1, seq_length):
            # 广播计算（确保转移矩阵在正确的设备上）
            emit_scores = emissions[t].unsqueeze(1)  # [batch_size, 1, num_tags]
            trans_scores = self.transitions.unsqueeze(0).to(device)  # [1, num_tags, num_tags]
            
            # 维特比递推
            scores = viterbi[t-1].unsqueeze(2) + trans_scores + emit_scores
            
            # 最大分数和回溯指针
            viterbi[t], backpointers[t] = torch.max(scores, dim=1)
            
            # 应用掩码
            if mask is not None:
                viterbi[t] = torch.where(mask[t].unsqueeze(1), viterbi[t], 
                                       torch.full_like(viterbi[t], -1e9, device=device))
        
        # 结束转移（确保结束转移矩阵在正确的设备上）
        end_transitions = self.end_transitions.to(device)
        viterbi_end = viterbi[-1] + end_transitions.unsqueeze(0)
        
        # 回溯解码
        best_paths = []
        best_scores, best_last_tags = torch.max(viterbi_end, dim=1)
        
        for i in range(batch_size):
            path = [best_last_tags[i].item()]
            
            for t in range(seq_length - 1, 0, -1):
                path.append(backpointers[t, i, path[-1]].item())
            
            path.reverse()
            best_paths.append(path)
        
        return best_paths, best_scores


class CTCLossWithCRF(nn.Module):
    """CTC损失与CRF损失的组合"""
    
    def __init__(self, num_classes, blank_idx=0, crf_weight=0.5):
        super(CTCLossWithCRF, self).__init__()
        
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean')
        self.crf = CRF(num_classes)
        self.crf_weight = crf_weight
        
    def forward(self, logits, targets, input_lengths, target_lengths, crf_tags=None):
        """
        Args:
            logits: CTC输出 [T, N, C]
            targets: 目标序列 [N, S]
            input_lengths: 输入序列长度
            target_lengths: 目标序列长度
            crf_tags: CRF标签序列
        """
        # CTC损失
        ctc_loss = self.ctc_loss(logits.log_softmax(2), targets, input_lengths, target_lengths)
        
        # CRF损失（如果提供了CRF标签）
        if crf_tags is not None:
            # 将logits转换为CRF需要的格式
            emissions = logits.transpose(0, 1)  # [N, T, C]
            crf_loss = self.crf(emissions, crf_tags)
            
            # 组合损失
            total_loss = (1 - self.crf_weight) * ctc_loss + self.crf_weight * crf_loss
        else:
            total_loss = ctc_loss
            
        return total_loss


class CTC_CRF_Decoder(nn.Module):
    """CTC + CRF混合解码器"""
    
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=2, 
                 dropout=0.1, blank_idx=0, crf_weight=0.5):
        super(CTC_CRF_Decoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.blank_idx = blank_idx
        
        # CTC投影层
        self.ctc_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # CRF层
        self.crf = CRF(num_classes)
        self.crf_weight = crf_weight
        
        # 设备跟踪
        self._device = None
        
        # 化学语法约束初始化
        self._init_chemical_constraints()
        
    def to(self, device):
        """重写to方法，确保CRF层也正确设置设备"""
        self._device = device
        # 先移动主模型到设备
        result = super().to(device)
        # 然后确保CRF层也正确设置设备
        if hasattr(self, 'crf') and self.crf is not None:
            self.crf.to(device)
        return result
        
    def _init_chemical_constraints(self):
        """初始化化学公式语法约束"""
        # 这里可以设置化学公式的语法规则
        # 例如：数字后面不能直接跟字母，上标后面不能直接跟下标等
        
        # 创建约束掩码（允许所有转移）
        constraint_mask = torch.ones(self.num_classes, self.num_classes)
        
        # 示例约束：禁止某些转移
        # constraint_mask[digit_idx, letter_idx] = 0  # 数字不能直接转移到字母
        
        self.crf.constraint_mask = constraint_mask
        
    def forward(self, features, targets=None, input_lengths=None, 
                target_lengths=None, crf_tags=None, training=True):
        """
        Args:
            features: 融合特征 [batch_size, seq_len, input_dim]
            targets: 目标序列（训练时使用）
            input_lengths: 输入序列长度
            target_lengths: 目标序列长度
            crf_tags: CRF标签序列
            training: 是否为训练模式
        """
        batch_size, seq_len, _ = features.shape
        
        # CTC投影
        ctc_logits = self.ctc_projection(features)  # [batch_size, seq_len, num_classes]
        
        # 转换为CTC需要的格式 [seq_len, batch_size, num_classes]
        ctc_logits = ctc_logits.transpose(0, 1)
        
        if training:
            # 训练模式：计算CTC + CRF损失
            if targets is not None and input_lengths is not None and target_lengths is not None:
                # CTC损失
                ctc_loss = nn.CTCLoss(blank=self.blank_idx, reduction='mean')(
                    ctc_logits.log_softmax(2), targets, input_lengths, target_lengths
                )
                
                # CRF损失（如果提供了CRF标签）
                if crf_tags is not None:
                    # 将logits转换为CRF需要的格式
                    emissions = ctc_logits.transpose(0, 1)  # [batch_size, seq_len, num_classes]
                    crf_loss = self.crf(emissions, crf_tags)
                    
                    # 组合损失
                    total_loss = (1 - self.crf_weight) * ctc_loss + self.crf_weight * crf_loss
                    
                    return {
                        'loss': total_loss,
                        'ctc_loss': ctc_loss,
                        'crf_loss': crf_loss,
                        'logits': ctc_logits
                    }
                else:
                    # 仅使用CTC损失
                    return {
                        'loss': ctc_loss,
                        'ctc_loss': ctc_loss,
                        'crf_loss': torch.tensor(0.0),
                        'logits': ctc_logits
                    }
            else:
                raise ValueError("训练模式下需要提供targets, input_lengths和target_lengths")
        else:
            # 推理模式：使用CRF解码
            emissions = ctc_logits.transpose(0, 1)  # [batch_size, seq_len, num_classes]
            
            # CRF维特比解码
            best_paths, best_scores = self.crf.decode(emissions)
            
            return {
                'predictions': best_paths,
                'scores': best_scores,
                'logits': ctc_logits
            }
    
    def ctc_decode(self, logits, input_lengths):
        """仅使用CTC进行贪婪解码"""
        # 贪婪解码
        _, predicted = torch.max(logits, dim=2)  # [seq_len, batch_size]
        predicted = predicted.transpose(0, 1)  # [batch_size, seq_len]
        
        # 移除空白标签和重复标签
        decoded = []
        for i in range(predicted.size(0)):
            seq = []
            prev = -1
            for j in range(input_lengths[i] if input_lengths is not None else predicted.size(1)):
                char = predicted[i, j].item()
                if char != self.blank_idx and char != prev:
                    seq.append(char)
                prev = char
            decoded.append(seq)
        
        return decoded
    
    def apply_chemical_rules(self, predictions):
        """应用化学公式语法规则后处理"""
        corrected_predictions = []
        
        for pred in predictions:
            corrected = []
            
            # 示例规则：确保数字和字母的正确顺序
            # 这里可以添加更复杂的化学公式语法规则
            
            corrected_predictions.append(corrected)
        
        return corrected_predictions


def test_ctc_crf_decoder():
    """测试CTC+CRF解码器"""
    # 创建测试数据
    batch_size, seq_len, input_dim, num_classes = 2, 50, 256, 37
    
    features = torch.randn(batch_size, seq_len, input_dim)
    
    # 测试解码器
    print("测试CTC+CRF解码器...")
    decoder = CTC_CRF_Decoder(input_dim, num_classes)
    
    # 训练模式测试
    targets = torch.randint(1, num_classes, (batch_size, 20))
    input_lengths = torch.tensor([seq_len] * batch_size)
    target_lengths = torch.tensor([20] * batch_size)
    
    output = decoder(features, targets, input_lengths, target_lengths, training=True)
    print(f"训练输出: {output.keys()}")
    print(f"损失值: {output['loss'].item():.4f}")
    
    # 推理模式测试
    output_inference = decoder(features, training=False)
    print(f"推理输出: {output_inference.keys()}")
    print(f"预测序列数量: {len(output_inference['predictions'])}")
    
    # CTC解码测试
    ctc_predictions = decoder.ctc_decode(output['logits'], input_lengths)
    print(f"CTC解码结果数量: {len(ctc_predictions)}")
    
    print("CTC+CRF解码器测试通过!")


if __name__ == "__main__":
    test_ctc_crf_decoder()