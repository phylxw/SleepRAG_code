import os
import json
from datasets import load_dataset

def normalize_code_instance(item, dataset_type="humaneval"):
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
        
    elif dataset_type == "humaneval":
        
        question = item.get("prompt")
        answer = item.get("canonical_solution")
        
        normalized = {
            "id": str(item.get("task_id", "")),
            
            
            "contents": question,
            
            "question": question,
            "golden_answers": [answer] if answer else [],
            **item 
        }
    
    return normalized

def prepare_humaneval(corpus_file, test_file, cfg, need_split):
    is_val = need_split
    if not os.path.exists(corpus_file):
        try:
            
            target_split = cfg.experiment.get("corpus_split", "train+validation+test+prompt")
            mbpp_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=target_split)
            
            os.makedirs(os.path.dirname(corpus_file), exist_ok=True)
            with open(corpus_file, 'w', encoding='utf-8') as f:
                for item in mbpp_ds:
                    processed = normalize_code_instance(item, dataset_type="mbpp")
                    f.write(json.dumps(processed, ensure_ascii=False) + "\n")
            
            print(f" ✅ : {corpus_file} ({len(mbpp_ds)} )")
        except Exception as e:
            print(f"fail : {e}")
            return False

    try:
        he_ds = load_dataset("openai_humaneval", split="test") 
        total_len = len(he_ds)
        mid_point = total_len // 2 
        
        if is_val:
            print(f"Validation")
            print(f"{mid_point} (Index 0-{mid_point-1})")
            
            candidate_ds = he_ds.select(range(0, mid_point))
        else:
            print(f"Test")
            print(f"{total_len - mid_point} (Index {mid_point}-{total_len-1})")
            
            candidate_ds = he_ds.select(range(mid_point, total_len))

        
        
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
                processed = normalize_code_instance(item, dataset_type="humaneval")
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")
        
        print(f" ✅ HumanEval : {test_file} ({len(final_ds)} )")
        
    except Exception as e:
        print(f"fail: {e}")
        
        import traceback
        traceback.print_exc()
        return False

    return True

