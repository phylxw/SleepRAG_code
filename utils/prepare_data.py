from datasets import load_dataset
from omegaconf import DictConfig
import os
import json
from tqdm import tqdm
import random 
from tools.prepare.merge_hmmt import merge_hmmt
from tools.prepare.merge_aime import merge_aime
from tools.prepare.sci_split import prepare_sciknow
from tools.prepare.humaneval_split import prepare_humaneval
from tools.prepare.mbpp_split import prepare_mbpp

def _get_available_column(dataset, candidates, default):

    cols = []
    if hasattr(dataset, "column_names"):
        cols = dataset.column_names
    elif hasattr(dataset, "features"):
        cols = dataset.features.keys()
    
    
    for cand in candidates:
        if cand in cols:
            return cand
    return default

def prepare_data(cfg: DictConfig, corpus_file: str, test_file: str, need_split):

    is_val = False 
    if cfg.experiment.tag == "sci":
        
        return prepare_sciknow(corpus_file, test_file, cfg, need_split)
    if cfg.experiment.tag == "humaneval":
        
        return prepare_humaneval(corpus_file, test_file, cfg, need_split)
    if cfg.experiment.tag == "mbpp":
        
        return prepare_mbpp(corpus_file, test_file, cfg, need_split)
    if (cfg.experiment.tag != "math_self") and (cfg.experiment.tag != "gsm8k_self"):
        is_val = need_split
        need_split = False
        
    
    
    q_col_cfg = cfg.experiment.field_map.question
    a_col_cfg = cfg.experiment.field_map.answer
    
    
    q_candidates = [q_col_cfg, "problem", "question", "input", "content", "Question"]
    a_candidates = [a_col_cfg, "solution", "answer", "ground_truth", "output", "completion", "Correct Answer"]

    
    
    
    c_name = cfg.experiment.get("corpus_dataset_name") or cfg.experiment.get("dataset_name")
    c_config = cfg.experiment.get("corpus_dataset_config") or cfg.experiment.get("dataset_config")
    c_split = cfg.experiment.get("corpus_split", "train")

    
    
    split_ratio = cfg.parameters.get("split_ratio", 0.9)
    
    
    
    

    if not os.path.exists(corpus_file) or need_split: 
        print(f"\n [Memory]: {c_name} | Split: {c_split}")
        try:
            ds_corpus = load_dataset(c_name, c_config, split=c_split)
        except Exception as e:
            print(f"fail: {e}")
            return False

        target_level = cfg.experiment.get("level_filter", None) 
        
        if cfg.experiment.tag == "hmmtex" or cfg.experiment.tag == "aimeex":
            print(f"[Mode] HMMT :MATH Level 5 ")
            target_level = "Level 5" 

            if "solution" in ds_corpus.column_names:
                a_candidates.insert(0, "solution") 

        
        if target_level:
            level_candidates = ["level", "difficulty", "grade"]
            level_col = _get_available_column(ds_corpus, level_candidates, None)

            if level_col:
                original_len = len(ds_corpus)
                
                ds_corpus = ds_corpus.filter(
                    lambda x: x[level_col] is not None and ("5" in str(x[level_col]))
                )
                print(f"[Filter]({target_level}): {original_len} -> {len(ds_corpus)}")
            else:
                print(f"fail")

        
        
        target_type = cfg.experiment.get("problem_type", "all")
        
        if target_type and target_type.lower() != "all":
            print(f"find: '{target_type}'")

            type_candidates = ["problem_type", "subject", "category", "type"]
            type_col = _get_available_column(ds_corpus, type_candidates, None)
            
            if type_col:
                original_len = len(ds_corpus)
                
                ds_corpus = ds_corpus.filter(
                    lambda x: x[type_col] is not None and target_type.lower() in str(x[type_col]).lower()
                )
                print(f"{original_len} -> {len(ds_corpus)} {type_col})")
            else:
                print(f"fail : {ds_corpus.column_names}")
        

        
        max_limit = cfg.parameters.get("total_num", None) 
        if max_limit is not None and len(ds_corpus) > int(max_limit):
            print(f"{max_limit} ")
            ds_corpus = ds_corpus.select(range(int(max_limit)))

        q_col_mem = _get_available_column(ds_corpus, q_candidates, q_col_cfg)
        a_col_mem = _get_available_column(ds_corpus, a_candidates, a_col_cfg)
        print(f"   👉 自动匹配列名: Q='{q_col_mem}', A='{a_col_mem}'")

        
        if need_split and split_ratio > 0:
            print(f"[Split]: Corpus {len(ds_corpus)} {1 - split_ratio} for(Test File)")
            
            ds_corpus = ds_corpus.shuffle(seed=42)
            
            
            split_idx = int(len(ds_corpus)*split_ratio)
            if split_idx < 0: split_idx = 0
            
            
            ds_memory = ds_corpus.select(range(0, split_idx)) 
            ds_val = ds_corpus.select(range(split_idx, len(ds_corpus))) 
        else:
            print(f"[Full]: all")
            ds_memory = ds_corpus
            ds_val = None

        
        if not os.path.exists(corpus_file):
            with open(corpus_file, "w", encoding="utf-8") as f:
                for i, item in enumerate(tqdm(ds_memory, desc="Writing Corpus")):
                    q_text = item.get(q_col_mem, "")
                    a_text = item.get(a_col_mem, "")
                    if q_text:
                        
                        content = f"Question: {q_text}\nAnswer: {a_text}"
                        f.write(json.dumps({"id": str(i), "contents": content}) + "\n")
        
        
        if need_split and ds_val is not None:
            print(f"[Split]: {test_file}")
            
            
            start_idx = int(cfg.parameters.get("start_index", 0) or 0)
            debug_num = cfg.parameters.get("debug_num")
            
            total_val_len = len(ds_val)
            
            
            if debug_num:
                limit = int(debug_num)
                end_idx = min(start_idx + limit, total_val_len)
            else:
                end_idx = total_val_len
            
            
            if start_idx >= total_val_len:
                selected_val = []
            else:
                
                indices = range(start_idx, end_idx)
                selected_val = ds_val.select(indices)
                print(f"[{start_idx}:{end_idx}] | : {len(selected_val)} ")

            with open(test_file, "w", encoding="utf-8") as f:
                for i, item in enumerate(tqdm(selected_val, desc="Writing Validation Set")):
                    
                    real_id = start_idx + i 
                    
                    q_text = item.get(q_col_mem, "")
                    a_text = item.get(a_col_mem, "")
                    
                    
                    f.write(json.dumps({
                        "id": str(real_id),
                        "question": q_text,
                        "golden_answers": [str(a_text)] 
                    }) + "\n")
            
            print(f"Done")
            return True  

    else:
        print(f"[Memory]: {corpus_file}")

    
    
    
    if cfg.experiment.tag == "hmmtex":
        print(f"HMMT")
        merge_hmmt(test_file, cfg, is_val)
        return True
    
    if cfg.experiment.tag == "aimeex":
        print(f"AIME")
        merge_aime(test_file, cfg, is_val)
        return True
    
    t_name = cfg.experiment.get("test_dataset_name") or c_name
    t_config = cfg.experiment.test_dataset_config if "test_dataset_config" in cfg.experiment else c_config
    t_split = cfg.experiment.get("test_split", "test")

    print(f"\n🔨 [Test] : {t_name} | Split: {t_split}")
    try:
        ds_test = load_dataset(t_name, t_config, split=t_split)
    except Exception as e:
        print(f"fail: {e}")
        return False

    
    is_gpqa = "gpqa" in str(t_name).lower()
    
    
    is_sciknow = "sci" in str(t_name).lower()

    if not is_gpqa and not is_sciknow:
        q_col_test = _get_available_column(ds_test, q_candidates, q_col_cfg)
        a_col_test = _get_available_column(ds_test, a_candidates, a_col_cfg)
        print(f" Q='{q_col_test}', A='{a_col_test}'")
    elif is_sciknow:
        print(f" [Mode] SciKnowEval")
    else:
        print(f" [Mode] GPQA")

    
    with open(test_file, "w", encoding="utf-8") as f:
        start_idx = int(cfg.parameters.get("start_index", 0) or 0)
        debug_num = cfg.parameters.get("debug_num")
        
        total_len = len(ds_test)
        if debug_num:
            limit = int(debug_num)
            end_idx = min(start_idx + limit, total_len)
        else:
            end_idx = total_len
            
        indices = range(start_idx, end_idx)
        selected_data = ds_test.select(indices)
        
        print(f"num: {len(selected_data)}")

        for i, item in enumerate(selected_data):
            real_id = start_idx + i

            
            if is_sciknow:
                
                question_raw = item.get("question", "")
                choices = item.get("choices", []) 
                answer_raw = item.get("answer", "")
                
                options_str = ""
                labels = ['A', 'B', 'C', 'D', 'E', 'F']
                
                if isinstance(choices, list):
                    for idx, choice_text in enumerate(choices):
                        label = labels[idx] if idx < len(labels) else str(idx)
                        options_str += f"\n({label}) {choice_text}"
                else:
                    options_str = f"\n{str(choices)}"

                q_text = question_raw + options_str
                a_text = str(answer_raw)

            elif is_gpqa:
                
                question_raw = item.get("Question", "")
                correct_ans = item.get("Correct Answer", "")
                inc_ans_1 = item.get("Incorrect Answer 1", "")
                inc_ans_2 = item.get("Incorrect Answer 2", "")
                inc_ans_3 = item.get("Incorrect Answer 3", "")
                
                options = [correct_ans, inc_ans_1, inc_ans_2, inc_ans_3]
                random.shuffle(options)
                
                labels = ['A', 'B', 'C', 'D']
                try:
                    correct_idx = options.index(correct_ans)
                    final_ans = labels[correct_idx] 
                except ValueError:
                    final_ans = "Error"

                options_str = ""
                for label, content in zip(labels, options):
                    options_str += f"\n({label}) {content}"
                
                q_text = question_raw + options_str
                a_text = final_ans 

            else:
                
                
                q_text = item.get(q_col_test, "")
                a_text = item.get(a_col_test, "")

            
            f.write(json.dumps({
                "id": str(real_id),
                "question": q_text,
                "golden_answers": [str(a_text)]
            }) + "\n")
            
    return True