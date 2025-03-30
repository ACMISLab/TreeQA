import argparse
import json
import os
import logging
import re
import string
from collections import Counter
from typing import List, Tuple, Dict, Any

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_text(s: str) -> str:
    """小写、去除标点、冠词和多余空格"""
    if not s:
        return ""
    s = str(s).lower()
    # 移除标点符号
    s = s.translate(str.maketrans('', '', string.punctuation))
    # 移除冠词 (a, an, the)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # 移除多余空格
    s = ' '.join(s.split())
    return s

def calculate_f1_precision_recall(prediction: str, ground_truths: List[str]) -> Tuple[float, float, float]:
    """
    计算预测与真实答案列表之间的最大 F1, Precision, Recall 分数 (基于 Token)。

    Args:
        prediction: 预测的答案字符串。
        ground_truths: 包含一个或多个真实答案字符串的列表。

    Returns:
        一个元组 (f1, precision, recall)。
    """
    if not ground_truths:
        return (0.0, 0.0, 0.0) if not prediction else (0.0, 0.0, 0.0) # 或者根据需要处理
    if not prediction:
        return (0.0, 0.0, 0.0) # 预测为空，无法匹配

    normalized_prediction = normalize_text(prediction)
    prediction_tokens = Counter(normalized_prediction.split())

    if not prediction_tokens: # 规范化后预测为空
         return (0.0, 0.0, 0.0)

    max_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for gt in ground_truths:
        normalized_gt = normalize_text(gt)
        gt_tokens = Counter(normalized_gt.split())

        if not gt_tokens:
            continue # 跳过空的真实答案

        common_tokens = prediction_tokens & gt_tokens
        num_common = sum(common_tokens.values())

        # Precision = (共享 Token 数) / (预测 Token 总数)
        precision = num_common / sum(prediction_tokens.values()) if sum(prediction_tokens.values()) > 0 else 0.0
        # Recall = (共享 Token 数) / (真实答案 Token 总数)
        recall = num_common / sum(gt_tokens.values()) if sum(gt_tokens.values()) > 0 else 0.0
        # F1 Score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > max_f1:
            max_f1 = f1
            best_precision = precision
            best_recall = recall

    return max_f1, best_precision, best_recall

def calculate_em(prediction: str, ground_truths: List[str]) -> bool:
    """
    计算 Exact Match (EM) 分数。
    检查规范化后的预测是否与任何一个规范化后的真实答案完全匹配。
    """
    if not ground_truths:
        return not prediction # 如果真实答案为空，只有当预测也为空时才算匹配
    if not prediction:
        return False # 预测为空，但真实答案不为空

    normalized_prediction = normalize_text(prediction)

    for gt in ground_truths:
        normalized_gt = normalize_text(gt)
        if normalized_prediction == normalized_gt:
            return True
    return False

def evaluate_results(input_file_path: str, error_file_path: str = None):
    """
    评估 JSONL 文件中的结果。

    Args:
        input_file_path: 输入的 JSONL 文件路径。
        error_file_path: (可选) 保存错误记录的文件路径。
    """
    if not os.path.exists(input_file_path):
        logging.error(f"输入文件未找到: {input_file_path}")
        return

    total_records = 0
    em_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    # 用于计算平均值的累加器
    metric_accumulators: Dict[str, float] = {
        "logic_init_time": 0.0,
        "self_adaptive_time": 0.0,
        "final_reasoning_time": 0.0,
        "total_processing_time": 0.0,
        "logic_init_tokens": 0.0,
        "self_adaptive_tokens": 0.0,
        "final_reasoning_tokens": 0.0,
        "total_tokens": 0.0,
        "fix_count": 0.0,
    }
    metric_keys = list(metric_accumulators.keys())
    error_records = []

    logging.info(f"开始评估文件: {input_file_path}")

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            try:
                data = json.loads(line)
                total_records += 1

                # --- 提取信息 ---
                record_id = data.get("id", f"line_{i+1}")
                question = data.get("question", "")
                # 确保 original_answer 是列表且元素为字符串
                original_answers_raw = data.get("original_answer", [])
                if original_answers_raw is None:
                    original_answers = []
                elif isinstance(original_answers_raw, list):
                    original_answers = [str(ans) for ans in original_answers_raw if ans is not None]
                else: # 如果不是列表，尝试将其作为单个答案处理
                    original_answers = [str(original_answers_raw)]

                # 优先使用 'final_answer'，如果不存在则尝试解析 'processed_answer' 中的 'answer'
                prediction = data.get("final_answer")
                if prediction is None:
                    processed_answer_str = data.get("processed_answer")
                    if isinstance(processed_answer_str, str):
                        try:
                            processed_answer_json = json.loads(processed_answer_str)
                            prediction = processed_answer_json.get("answer")
                        except json.JSONDecodeError:
                            prediction = None # 如果解析失败，则预测为空
                    elif isinstance(processed_answer_str, dict): # 如果已经是dict
                         prediction = processed_answer_str.get("answer")


                if prediction is None:
                    prediction = "" # 保证 prediction 是字符串
                else:
                    prediction = str(prediction) # 确保是字符串

                # --- 计算分数 ---
                em = calculate_em(prediction, original_answers)
                f1, precision, recall = calculate_f1_precision_recall(prediction, original_answers)

                em_scores.append(em)
                f1_scores.append(f1)
                precision_scores.append(precision)
                recall_scores.append(recall)

                # --- 累加指标 ---
                for key in metric_keys:
                    value = data.get(key, 0) # 如果指标不存在，默认为 0
                    if isinstance(value, (int, float)):
                        metric_accumulators[key] += value
                    else:
                         logging.warning(f"记录 {record_id} 的指标 '{key}' 不是数值类型: {value} (类型: {type(value)})，计为 0")


                # --- 记录错误 ---
                if not em and error_file_path:
                    error_records.append({
                        "id": record_id,
                        "question": question,
                        "original_answer": original_answers,
                        "predicted_answer": prediction,
                        "f1": f1, # 也记录下 F1 分数
                        "fix_count": data.get("fix_count", "N/A")
                    })

            except json.JSONDecodeError as e:
                logging.error(f"解析第 {i+1} 行 JSON 时出错: {e}")
            except Exception as e:
                logging.error(f"处理第 {i+1} 行时发生意外错误: {e}")

    if total_records == 0:
        logging.warning("文件中没有找到有效的记录进行评估。")
        return

    # --- 计算平均值 ---
    avg_em = sum(em_scores) / total_records
    avg_f1 = sum(f1_scores) / total_records
    avg_precision = sum(precision_scores) / total_records
    avg_recall = sum(recall_scores) / total_records

    average_metrics: Dict[str, float] = {}
    for key in metric_keys:
        average_metrics[f"avg_{key}"] = metric_accumulators[key] / total_records

    # --- 打印报告 ---
    print("\n--- 评估结果报告 ---")
    print(f"处理总记录数: {total_records}")
    print("-" * 20)
    print("性能指标:")
    print(f"  平均 Exact Match (EM): {avg_em:.4f} ({avg_em:.2%})")
    print(f"  平均 F1 Score:         {avg_f1:.4f}")
    print(f"  平均 Precision:        {avg_precision:.4f}")
    print(f"  平均 Recall:           {avg_recall:.4f}")
    print("-" * 20)
    print("平均资源消耗:")
    print(f"  平均 Logic Init Time:        {average_metrics['avg_logic_init_time']:.4f} s")
    print(f"  平均 Self-Adaptive Time:     {average_metrics['avg_self_adaptive_time']:.4f} s")
    print(f"  平均 Final Reasoning Time:   {average_metrics['avg_final_reasoning_time']:.4f} s")
    print(f"  平均 Total Processing Time:  {average_metrics['avg_total_processing_time']:.4f} s")
    print("-" * 10)
    print(f"  平均 Logic Init Tokens:      {average_metrics['avg_logic_init_tokens']:.2f}")
    print(f"  平均 Self-Adaptive Tokens:   {average_metrics['avg_self_adaptive_tokens']:.2f}")
    print(f"  平均 Final Reasoning Tokens: {average_metrics['avg_final_reasoning_tokens']:.2f}")
    print(f"  平均 Total Tokens Consumed:  {average_metrics['avg_total_tokens']:.2f}")
    print("-" * 10)
    print(f"  平均 Fix Count:            {average_metrics['avg_fix_count']:.2f}")
    print("-" * 20)

    # --- 保存错误记录 ---
    if error_file_path and error_records:
        try:
            # 推断输出目录并创建
            error_dir = os.path.dirname(error_file_path)
            if error_dir and not os.path.exists(error_dir):
                os.makedirs(error_dir)
                logging.info(f"创建错误文件目录: {error_dir}")

            with open(error_file_path, 'w', encoding='utf-8') as errfile:
                for record in error_records:
                    errfile.write(json.dumps(record, ensure_ascii=False) + '\n')
            logging.info(f"已将 {len(error_records)} 条错误记录保存到: {error_file_path}")
        except Exception as e:
            logging.error(f"保存错误文件时出错: {e}")

    logging.info("评估完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 QA 模型生成的 JSONL 结果文件。")
    parser.add_argument("--input_file", type=str, help="需要评估的 JSONL 结果文件路径。")
    parser.add_argument("--error_file", type=str, default=None,
                        help="(可选) 用于保存评估错误记录的 JSONL 文件路径。如果未提供，则不保存错误记录。")

    args = parser.parse_args()

    evaluate_results(args.input_file, args.error_file)