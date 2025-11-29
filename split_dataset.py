#!/usr/bin/env python3
"""
数据集划分工具
将完整数据集划分为训练集和验证集
"""

import os
import json
import random
from typing import List, Dict, Any


def split_dataset(annotations_file: str, train_ratio: float = 0.8):
    """划分数据集为训练集和验证集"""
    
    # 读取完整标注
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"总数据量: {len(annotations)}")
    
    # 随机打乱数据
    random.shuffle(annotations)
    
    # 划分训练集和验证集
    split_index = int(len(annotations) * train_ratio)
    train_annotations = annotations[:split_index]
    val_annotations = annotations[split_index:]
    
    print(f"训练集大小: {len(train_annotations)}")
    print(f"验证集大小: {len(val_annotations)}")
    
    # 保存训练集
    train_file = annotations_file.replace('.json', '_train.json')
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, ensure_ascii=False, indent=2)
    
    # 保存验证集
    val_file = annotations_file.replace('.json', '_val.json')
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, ensure_ascii=False, indent=2)
    
    print(f"训练集已保存到: {train_file}")
    print(f"验证集已保存到: {val_file}")
    
    return train_file, val_file


def update_vocab(annotations_file: str, vocab_file: str):
    """根据完整数据集更新词汇表"""
    
    # 读取完整标注
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 收集所有字符
    all_chars = set()
    for ann in annotations:
        formula = ann['formula']
        all_chars.update(formula)
    
    # 特殊标记
    special_tokens = ['<blank>', '<sos>', '<eos>', '<unk>']
    
    # 构建词汇表
    vocab = {}
    idx = 0
    
    # 添加特殊标记
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    
    # 添加字符
    for char in sorted(all_chars):
        if char not in vocab:
            vocab[char] = idx
            idx += 1
    
    print(f"词汇表大小: {len(vocab)}")
    
    # 保存词汇表
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"词汇表已保存到: {vocab_file}")
    
    return vocab


def main():
    """主函数"""
    
    # 划分完整数据集
    print("=== 划分数据集 ===")
    train_file, val_file = split_dataset('data/annotations_full.json', train_ratio=0.8)
    
    # 更新词汇表
    print("\\n=== 更新词汇表 ===")
    vocab = update_vocab('data/annotations_full.json', 'data/vocab_full.json')
    
    print("\\n=== 数据集划分完成 ===")
    print("生成的文件:")
    print(f"- {train_file}: 训练集 ({len(json.load(open(train_file, 'r')))} 条数据)")
    print(f"- {val_file}: 验证集 ({len(json.load(open(val_file, 'r')))} 条数据)")
    print(f"- data/vocab_full.json: 完整词汇表 ({len(vocab)} 个字符)")


if __name__ == "__main__":
    main()