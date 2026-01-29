import os
import json
from datasets import load_dataset

import os
import json
from datasets import load_dataset

def normalize_code_instance(item, dataset_type="mbpp"):

    normalized = {}
    
    if dataset_type == "mbpp":
        
        question = item.get("prompt") or item.get("text")
        answer = item.get("code")
        
        normalized = {
            "id": str(item.get("task_id", "")),
            
            "contents": question, 
            "question": question,
            "golden_answers": [answer] if answer else [],
            **item 
        }
        
    
    
    return normalized

def prepare_mbpp(corpus_file, test_file, cfg, need_split):
 if not os.path.exists(corpus_file):

    try:


        corpus_split = "train+prompt"

        mbpp_corpus_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=corpus_split)

        os.makedirs(os.path.dirname(corpus_file), exist_ok=True)
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for item in mbpp_corpus_ds:
                processed = normalize_code_instance(item, dataset_type="mbpp")
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")

        print(f" ✅ : {corpus_file} ({corpus_split}, {len(mbpp_corpus_ds)} )")
    except Exception as e:
        print(f"fail: {e}")
        return False

    is_val = need_split

    try:

        if is_val:
            print(f"Validation Split")
            candidate_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="validation")
        else:
            print(f"Test Split")
            candidate_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

        
        p_start = int(cfg.parameters.get("start_index", 0) or 0)
        p_debug = cfg.parameters.get("debug_num") 
        
        candidate_len = len(candidate_ds)
        
        
        if p_debug:
            limit = int(p_debug)
            p_end = min(p_start + limit, candidate_len)
        else:
            p_end = candidate_len
            
        
        if p_start >= candidate_len:
            final_ds = []
        else:
            
            final_ds = candidate_ds.select(range(p_start, p_end))

        
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in final_ds:
                processed = normalize_code_instance(item, dataset_type="mbpp")
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")
        
        print(f" ✅ : {test_file} ({len(final_ds)} )")
        
    except Exception as e:
        print(f"fail: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True