import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from omegaconf import DictConfig
from collections import Counter

def _format_sciknow_instance(item):

    question_raw = item.get("question", "").strip()
    choices = item.get("choices", None)
    
    
    
    answer_raw = item.get("answerKey")
    
    
    if answer_raw is None or str(answer_raw).strip() == "":
        answer_raw = item.get("answer", "")

    
    if answer_raw is None or str(answer_raw).strip() == "":
        return "", "", False

    
    if not choices:
        return "", "", False

    
    ans_str = str(answer_raw).strip()
    labels_pool = ["A", "B", "C", "D", "E", "F", "G", "H"]
    final_answer = ans_str 
    options_str = ""

    
    if isinstance(choices, list):
        
        for idx, text in enumerate(choices):
            label = labels_pool[idx] if idx < len(labels_pool) else str(idx)
            options_str += f"\n({label}) {text}"
            
            
            if ans_str == str(text) or ans_str == str(text).strip():
                final_answer = label

        
        
        if isinstance(answer_raw, int) and 0 <= answer_raw < len(choices):
            final_answer = labels_pool[answer_raw]
        
        
        elif ans_str.isdigit():
            idx = int(ans_str)
            if 0 <= idx < len(choices):
                final_answer = labels_pool[idx]

    
    
    elif isinstance(choices, dict) and "text" in choices:
        texts = choices["text"]
        labels = choices.get("label", labels_pool[:len(texts)])
        
        for l, t in zip(labels, texts):
            options_str += f"\n({l}) {t}"
            
            if ans_str == str(t) or ans_str == str(t).strip():
                final_answer = l
                
    
    if len(str(final_answer)) == 1 and str(final_answer).upper() in labels_pool:
        final_answer = str(final_answer).upper()

    
    q_text = question_raw + options_str
    
    return q_text, final_answer, True

def prepare_sciknow(corpus_path: str, test_path: str, cfg: DictConfig , need_split) -> bool:
    
    memory_exists = os.path.exists(corpus_path)
    if memory_exists:
        print(f"exist: {corpus_path}")
    else:
        print(f"[Init] fail...")
    
    is_val = need_split

    try:
        ds = load_dataset("hicai-zju/SciKnowEval", split="test") 
    except Exception as e:
        print(f"❌ SciKnowEval 下载失败: {e}")
        return False
    
    raw_data = list(ds)
    domain_counter = Counter()
    type_counter = Counter()
    valid_candidates = []
    
    for item in tqdm(raw_data, desc="Scanning"):
        
        ans = item.get("answer")
        ans_key = item.get("answerKey")
        
        has_ans = (ans is not None and str(ans).strip() != "")
        has_key = (ans_key is not None and str(ans_key).strip() != "")
        
        if not has_ans and not has_key:
            continue
            
        d = item.get("domain", "Unknown")
        if isinstance(d, list) and d: d = d[0]
        t = item.get("type", "Unknown")
        
        domain_counter[str(d)] += 1
        type_counter[str(t)] += 1
        valid_candidates.append(item)
    
    if len(valid_candidates) == 0:

        return False

    
    
    
    target_domain = cfg.experiment.get("target_domain")

    
    final_data = []
    skipped_domain = 0
    skipped_type = 0
    
    for item in valid_candidates:
        
        d = item.get("domain", "")
        if isinstance(d, list) and d: d = d[0]
        
        if target_domain and d != target_domain:
            skipped_domain += 1
            continue

        
        t = str(item.get("type", "")).lower()
        if "mcq" not in t and "multiple_choice" not in t:
            skipped_type += 1
            continue
            
        final_data.append(item)

    if len(final_data) == 0:

        return False

    
    random.seed(42) 
    
    
    
    total_limit = cfg.experiment.get("total_limit")
    if total_limit:
        limit_val = int(total_limit)
        if limit_val < len(final_data):
            final_data = final_data[:limit_val]

    
    split_idx_1 = int(len(final_data) * 0.8)
    corpus_pool = final_data[:split_idx_1]      
    final_test_pool = final_data[split_idx_1:]  


    
    if is_val:
        split_ratio = cfg.parameters.get("split_ratio", 0.9)
        split_idx_2 = int(len(corpus_pool) * split_ratio)
        
        real_corpus_data = corpus_pool[:split_idx_2]      
        target_test_data = corpus_pool[split_idx_2:]
        
    else:
        real_corpus_data = corpus_pool
        target_test_data = final_test_pool

    
    
    if not memory_exists:
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        with open(corpus_path, "w", encoding="utf-8") as f:
            count = 0
            for i, item in enumerate(tqdm(real_corpus_data, desc="Writing Corpus")):
                q_text, a_text, is_valid = _format_sciknow_instance(item)
                if is_valid:
                    content = f"Question: {q_text}\nAnswer: {a_text}"
                    f.write(json.dumps({"id": str(count), "contents": content}, ensure_ascii=False) + "\n")
                    count += 1
    else:
        print(f"skip")
            
    
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    start_index = int(cfg.parameters.get("start_index", 0) or 0)
    debug_num = cfg.parameters.get("debug_num")
    
    if debug_num:
        limit = int(debug_num)
        end_idx = min(start_index + limit, len(target_test_data))
        test_data_slice = target_test_data[start_index : end_idx]
    else:
        test_data_slice = target_test_data[start_index:]

    with open(test_path, "w", encoding="utf-8") as f:
        count = 0 
        for i, item in enumerate(tqdm(test_data_slice, desc="Writing Test")):
            q_text, a_text, is_valid = _format_sciknow_instance(item)
            if is_valid:
                f.write(json.dumps({
                    "id": str(count), 
                    "question": q_text,
                    "golden_answers": [a_text]
                }, ensure_ascii=False) + "\n")
                count += 1
            
    print("SciKnowEval OK")
    return True