
import os
import re
import json
from utils.math_reward import last_boxed_only_string, remove_boxed

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))
from math_verify import parse, verify

def judge_math_item(item):

    pred_raw = item.pred if hasattr(item, 'pred') else item.get('pred', "")
    golden_answers = item.golden_answers if hasattr(item, 'golden_answers') else item.get('golden_answers', [])
    gold_raw = golden_answers[0] if golden_answers else ""

    
    gold_parsed = parse(str(gold_raw))
    
    
    
    
    if not gold_parsed:
        gold_parsed = parse(f"\\boxed{{{str(gold_raw)}}}")

    
    pred_parsed = parse(str(pred_raw))

    
    try:
        is_right = verify(gold_parsed, pred_parsed)
    except Exception:
        is_right = False

    return is_right, str(gold_parsed), str(pred_parsed)

def evaluate_results(results, experiment_name, result_log_file):
    correct = 0
    total = len(results)
    
    
    scores_list = [] 
    
    
    os.makedirs(os.path.dirname(result_log_file), exist_ok=True)

    with open(result_log_file, "a", encoding="utf-8") as f:
        header = f"\n{'='*20} {experiment_name} {'='*20}\n"
        print(header.strip()) 
        f.write(header)
        
        for i, item in enumerate(results):
            
            question = item.question if hasattr(item, 'question') else item.get('question', "")
            pred_raw = item.pred if hasattr(item, 'pred') else item.get('pred', "")

            
            is_right, gold_val, pred_val = judge_math_item(item)
            
            
            current_score = 1.0 if is_right else 0.0
            scores_list.append(current_score)
            
            if is_right: 
                correct += 1

            
            log_entry = (
                f"\n[ID]: {i}\n"
                f"[Question]: {str(question)[:50]}...{str(question)[-50:]}\n" 
                f"[Gold Parsed]: {gold_val}\n"
                f"[Pred Parsed]: {pred_val}\n"
                f"[Pred All]: {pred_raw[:50]}...{pred_raw[-50:]}\n"
                f"[Result]: {' Correct' if is_right else ' Wrong'}\n"
                f"{'-'*30}\n"
            )
            
            f.write(log_entry)
            
            
            if i < 3: 
                print(log_entry.strip())

        
        acc = correct / total * 100 if total > 0 else 0
        summary = (
            f"\n ({experiment_name}):\n"
            f"Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%\n"
            f"{'='*50}\n"
        )
        print(summary)
        f.write(summary)
        
    
    return acc, scores_list