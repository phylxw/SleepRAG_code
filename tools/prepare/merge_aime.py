import json
import os
import random
from datasets import load_dataset



AIME_VAL_SETS = [
    "MathArena/aime_2023_I",
    "MathArena/aime_2023_II",
    "MathArena/aime_2024_I",
    "MathArena/aime_2024_II"
]


AIME_TEST_SETS = [
    "MathArena/aime_2025_I",
    "MathArena/aime_2025_II"
]

def normalize_instance(item):
    """problem -> question, solution -> golden_answers"""
    question = item.get("problem") or item.get("question")
    answer = item.get("solution") or item.get("answer")
    if answer: answer = str(answer).strip()
    
    return {
        "id":  None,
        "question": question,
        "golden_answers": [answer] if answer else []
    }

def merge_aime(output_path, cfg,is_val):

    if is_val == False:
        target_datasets = AIME_TEST_SETS
        print(f"Test")
        print(f"2025")
    else:
        target_datasets = AIME_VAL_SETS
        print(f"Validation")
        print(f"2023 + 2024")

    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_data = []
    
    
    for ds_name in target_datasets:
        print(f" Loading: {ds_name} ...")
        try:
            ds = load_dataset(ds_name, split="test") 
        except:
            try:
                ds = load_dataset(ds_name, split="train")
            except Exception as e:
                print(f"skip {ds_name}: {e}")
                continue
                
        for item in ds:
            processed = normalize_instance(item)
            if processed['question'] and processed['golden_answers']:
                all_data.append(processed)

    
    
    if is_val:
        print("mix 2023 and 2024 ...")
        random.seed(42)
        random.shuffle(all_data)
    
    
    start_index = int(cfg.parameters.get("start_index", 0) or 0)
    debug_num = cfg.parameters.get("debug_num")
    
    total_len = len(all_data)
    
    if debug_num:
        limit = int(debug_num)
        end_index = min(start_index + limit, total_len)
    else:
        end_index = total_len

    
    final_data = all_data[start_index : end_index]

    
    for idx, item in enumerate(final_data):
        real_id = start_index + idx
        item['id'] = str(real_id) 

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("OK")