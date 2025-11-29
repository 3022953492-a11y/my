#!/usr/bin/env python3
"""
评估指标模块
包含准确率、字符错误率、词错误率等计算函数
"""

import numpy as np
import torch


def calculate_accuracy(predictions, targets):
    """
    计算准确率
    
    Args:
        predictions: 预测序列列表
        targets: 目标序列列表
        
    Returns:
        accuracy: 准确率
    """
    if len(predictions) != len(targets):
        raise ValueError("预测序列和目标序列数量不一致")
    
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def calculate_cer(predictions, targets):
    """
    计算字符错误率 (Character Error Rate)
    
    Args:
        predictions: 预测序列列表
        targets: 目标序列列表
        
    Returns:
        cer: 字符错误率
    """
    if len(predictions) != len(targets):
        raise ValueError("预测序列和目标序列数量不一致")
    
    total_chars = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        # 使用动态规划计算编辑距离
        distance = _levenshtein_distance(pred, target)
        total_errors += distance
        total_chars += len(target)
    
    cer = total_errors / total_chars if total_chars > 0 else 0.0
    return cer


def calculate_wer(predictions, targets):
    """
    计算词错误率 (Word Error Rate)
    
    Args:
        predictions: 预测序列列表
        targets: 目标序列列表
        
    Returns:
        wer: 词错误率
    """
    if len(predictions) != len(targets):
        raise ValueError("预测序列和目标序列数量不一致")
    
    total_words = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        # 将序列按空格分割成单词
        pred_words = pred.split()
        target_words = target.split()
        
        # 使用动态规划计算编辑距离
        distance = _levenshtein_distance(pred_words, target_words)
        total_errors += distance
        total_words += len(target_words)
    
    wer = total_errors / total_words if total_words > 0 else 0.0
    return wer


def _levenshtein_distance(seq1, seq2):
    """
    计算两个序列之间的编辑距离（Levenshtein距离）
    
    Args:
        seq1: 序列1
        seq2: 序列2
        
    Returns:
        distance: 编辑距离
    """
    if len(seq1) < len(seq2):
        return _levenshtein_distance(seq2, seq1)
    
    if len(seq2) == 0:
        return len(seq1)
    
    previous_row = list(range(len(seq2) + 1))
    
    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        
        for j, c2 in enumerate(seq2):
            # 计算插入、删除、替换的代价
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row
    
    return previous_row[-1]


def calculate_precision_recall_f1(predictions, targets, average='macro'):
    """
    计算精确率、召回率和F1分数
    
    Args:
        predictions: 预测序列列表
        targets: 目标序列列表
        average: 平均方式 ('macro', 'micro', 'weighted')
        
    Returns:
        precision: 精确率
        recall: 召回率
        f1: F1分数
    """
    # 对于序列任务，需要更复杂的实现
    # 这里简化实现，仅计算整体匹配
    
    if len(predictions) != len(targets):
        raise ValueError("预测序列和目标序列数量不一致")
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return precision, recall, f1


def calculate_sequence_accuracy(predictions, targets):
    """
    计算序列级别的准确率
    
    Args:
        predictions: 预测序列列表
        targets: 目标序列列表
        
    Returns:
        sequence_accuracy: 序列准确率
    """
    if len(predictions) != len(targets):
        raise ValueError("预测序列和目标序列数量不一致")
    
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1
    
    sequence_accuracy = correct / total if total > 0 else 0.0
    return sequence_accuracy


def calculate_beam_search_metrics(beam_results, targets):
    """
    计算束搜索相关的评估指标
    
    Args:
        beam_results: 束搜索结果列表
        targets: 目标序列列表
        
    Returns:
        top1_accuracy: 束搜索第一候选的准确率
        top5_accuracy: 束搜索前五候选的准确率
    """
    if len(beam_results) != len(targets):
        raise ValueError("束搜索结果和目标序列数量不一致")
    
    top1_correct = 0
    top5_correct = 0
    total = len(beam_results)
    
    for beam_result, target in zip(beam_results, targets):
        # 检查第一候选
        if beam_result[0]['sequence'] == target:
            top1_correct += 1
            top5_correct += 1
        else:
            # 检查前五候选
            for candidate in beam_result[:5]:
                if candidate['sequence'] == target:
                    top5_correct += 1
                    break
    
    top1_accuracy = top1_correct / total if total > 0 else 0.0
    top5_accuracy = top5_correct / total if total > 0 else 0.0
    
    return top1_accuracy, top5_accuracy


def compute_metrics(predictions, targets):
    """
    计算综合评估指标
    
    Args:
        predictions: 预测序列列表
        targets: 目标序列列表
        
    Returns:
        metrics_dict: 包含各种评估指标的字典
    """
    metrics = {}
    
    # 计算各种指标
    metrics['accuracy'] = calculate_accuracy(predictions, targets)
    metrics['cer'] = calculate_cer(predictions, targets)
    metrics['wer'] = calculate_wer(predictions, targets)
    metrics['precision'], metrics['recall'], metrics['f1'] = calculate_precision_recall_f1(predictions, targets)
    metrics['sequence_accuracy'] = calculate_sequence_accuracy(predictions, targets)
    
    return metrics


def decode_predictions(model_output, vocab, config=None):
    """
    解码模型输出为可读的化学公式
    
    Args:
        model_output: 模型输出张量
        vocab: 词汇表
        config: 配置对象（可选）
        
    Returns:
        predictions: 解码后的预测序列列表
    """
    predictions = []
    
    # 简单的贪婪解码
    for batch_idx in range(model_output.size(0)):
        pred_sequence = []
        
        # 遍历序列中的每个时间步
        for time_step in range(model_output.size(1)):
            # 获取当前时间步的最大概率字符索引
            char_idx = torch.argmax(model_output[batch_idx, time_step]).item()
            
            # 将索引转换为字符
            if char_idx < len(vocab):
                char = vocab[char_idx]
                # 跳过特殊标记
                if char not in ['<blank>', '<sos>', '<eos>', '<pad>']:
                    pred_sequence.append(char)
        
        # 合并字符成字符串
        predictions.append(''.join(pred_sequence))
    
    return predictions


def test_metrics():
    """测试评估指标函数"""
    predictions = ["H2O", "CO2", "H2SO4"]
    targets = ["H2O", "CO2", "H2SO4"]
    
    accuracy = calculate_accuracy(predictions, targets)
    cer = calculate_cer(predictions, targets)
    wer = calculate_wer(predictions, targets)
    precision, recall, f1 = calculate_precision_recall_f1(predictions, targets)
    sequence_accuracy = calculate_sequence_accuracy(predictions, targets)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"字符错误率: {cer:.4f}")
    print(f"词错误率: {wer:.4f}")
    print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
    print(f"序列准确率: {sequence_accuracy:.4f}")
    
    # 测试错误情况
    predictions_err = ["H2O", "CO3", "H2SO4"]  # 有一个错误
    accuracy_err = calculate_accuracy(predictions_err, targets)
    cer_err = calculate_cer(predictions_err, targets)
    
    print(f"\n错误情况测试:")
    print(f"准确率: {accuracy_err:.4f}")
    print(f"字符错误率: {cer_err:.4f}")


if __name__ == "__main__":
    test_metrics()