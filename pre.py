import os
import json
import time
import torch
from huggingface_hub import snapshot_download

import hydra
from omegaconf import DictConfig, OmegaConf

from flashrag.config import Config
from flashrag.pipeline import SequentialPipeline
from flashrag.utils import get_retriever, get_generator, Dataset
from flashrag.prompt import PromptTemplate

import transformers
transformers.logging.set_verbosity_error()



from utils.prepare_data import prepare_data
from utils.build_index import build_index
from utils.toolfunction import format_base_prompt
from utils.generator.gemini import GeminiGenerator
from utils.generator.sglang import SGLangGenerator
from tools.evaluate import evaluate_results
from tools.retrieverwrapper import BEMRRetrieverWrapper
from tools.memoryscore import _load_memory_corpus,_calculate_scores,_print_stats_and_save,_visualize_results
from tools.evalcode import evaluate_code_results

from cluster import cluster


def analyze_memory_usage(rag_results, cfg: DictConfig, corpus_file: str, vis_image_file: str, 
                         old_stats: dict = None, baseline_score = None ,root_dir: str = None, corpus_tag: str = None):
    freq_file = cfg.paths.freq_file
    print("\n [Analysis] (Bayesian Belief Update)...")

    
    all_memory_ids, id_to_content = _load_memory_corpus(corpus_file)

    
    
    memory_scores, new_stats, correct_count = _calculate_scores(rag_results, all_memory_ids, cfg, old_stats,baseline_score)

    
    if root_dir and corpus_tag:
        stats_file = cfg.paths.stats_file
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(new_stats, f, ensure_ascii=False, indent=2)
            print(f"[BEMR] TextGrad to: {stats_file}")
        except Exception as e:
            print(f"fail: {e}")

    
    
    sorted_memories = _print_stats_and_save(
        memory_scores, 
        id_to_content, 
        len(rag_results), 
        correct_count, 
        freq_file
    )

    
    _visualize_results(cfg, sorted_memories, vis_image_file)




@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    
    print("Visible GPU count:", torch.cuda.device_count())
    root_dir = cfg.paths.root
    
    
    
    
    
    
    
    corpus_name = cfg.experiment.get("corpus_dataset_name") or cfg.experiment.dataset_name
    corpus_tag = corpus_name.split('/')[-1] 
    
    
    test_name = cfg.experiment.get("test_dataset_name") or cfg.experiment.dataset_name
    test_tag = test_name.split('/')[-1]

    print(f"  Corpus Tag: {corpus_tag} | Test Tag: {test_tag}")

    
    
    
    
    
    corpus_file = cfg.paths.corpus_file
    index_dir = cfg.paths.index_dir
    result_dir = cfg.paths.result_dir
    
    
    
    jsonl_dir =  cfg.paths.jsonl
    test_file = os.path.join(jsonl_dir, f"{test_tag}_test_data.jsonl")
    
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    result_log_file = os.path.join(result_dir, f"{test_tag}_on_{corpus_tag}_{cfg.model.source}_{cfg.parameters.mode}_{timestamp}.txt")
    
    
    vis_image_file = os.path.join(result_dir, f"{test_tag}_on_{corpus_tag}_dist_{timestamp}.png")

    if os.path.exists(result_log_file): os.remove(result_log_file)
    print(f"result: {result_log_file}")
    print(f"mode: {cfg.parameters.mode} | {cfg.model.source}")
    print(f"Memory: {corpus_name} |  Test: {test_name}")

    
    need_split = cfg.parameters.get("split_corpus_for_val", False) 
    if not prepare_data(cfg, corpus_file, test_file,need_split): return
    
    
    if cfg.parameters.mode in ['rag', 'all']:
        build_index(corpus_file, index_dir)
    
    
    generator = None
    config = None 
    
    model_source = cfg.model.source
    
    if model_source == "gemini":
        print(f"[Init] Gemini: {cfg.model.gemini_name}...")
        api_key = os.environ.get("GEMINI_API_KEY") 
        generator = GeminiGenerator(cfg.model.gemini_name, api_key)
        
        
        gemini_config_dict = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "device": "cpu",
            "retrieval_method": cfg.experiment.retrieval_method,
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "generator_model": "huggingface", 
            "generator_model_path": "gpt2",   
        }
        config = Config(config_dict=gemini_config_dict)

    elif model_source == "sglang":
        print(f" [Init] SGLang Client...")
        
        
        sglang_base_url = cfg.model.get("sglang_api_url", "http://127.0.0.1:30000/v1")
        
        sglang_model_name = cfg.model.get("sglang_model_name", "Qwen/Qwen3-4B-Instruct-2507")
        
        
        sglang_config_dict = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "retrieval_method": cfg.experiment.retrieval_method,
            
            
            "device": "cpu",
            "gpu_num": 0,
            "generator_model": "openai",               
            "generator_model_path": sglang_model_name, 
            "generation_method": "openai", 
            "batch_size": cfg.model.batch_size,
            "max_input_len": cfg.model.max_input_len,
            "max_new_tokens": cfg.model.max_new_tokens,
        }
        
        config = Config(config_dict=sglang_config_dict)

        
        generator = SGLangGenerator(
            base_url=sglang_base_url,
            model_name=sglang_model_name,
            max_new_tokens=cfg.model.max_new_tokens,
            batch_size=cfg.model.batch_size,
            temperature=0.7, 
        )
        print(f"SGLang Generator ({sglang_model_name}) to {sglang_base_url}")

    elif model_source == "huggingface":
        hf_name = cfg.model.hf_name
        print(f"[Init]  HF model: {hf_name}...")
        try:
            model_path = snapshot_download(repo_id=hf_name)
        except:
            print("fail")
            return

        hf_config_dict = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "device": cfg.model.device,
            "gpu_num": torch.cuda.device_count(),
            "generator_model": "huggingface",
            "generator_model_path": model_path,
            "generation_method": "huggingface",
            "batch_size": cfg.model.batch_size,
            "max_input_len": cfg.model.max_input_len,
            "max_new_tokens": cfg.model.max_new_tokens,
        }
        config = Config(config_dict=hf_config_dict)
        generator = get_generator(config)
        
        
        if hasattr(generator, 'tokenizer'):
            generator.tokenizer.padding_side = 'left' 
            if generator.tokenizer.pad_token is None:
                generator.tokenizer.pad_token = generator.tokenizer.eos_token
                generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id
            generator.tokenizer.model_max_length = cfg.model.max_input_len
        
        if hasattr(generator, 'model'):
            if hasattr(generator.model.config, 'pad_token_id') and generator.model.config.pad_token_id is None:
                generator.model.config.pad_token_id = generator.tokenizer.pad_token_id
        print(f"Tokenizer OK")
    
    else:
        print(f"unsupport MODEL_SOURCE: {model_source}")
        return


    is_code_task = False
    code_dataset_type = "math" 
    if "humaneval" in cfg.experiment.tag.lower():
        is_code_task = True
        code_dataset_type = "humaneval"
    elif "mbpp" in cfg.experiment.tag.lower():
        is_code_task = True
        code_dataset_type = "mbpp"

    
    
    
    
    stats_file = cfg.paths.stats_file
    memory_stats = {}
    if os.path.exists(stats_file):
        print(f"[BEMR] load: {stats_file}")
        with open(stats_file, 'r', encoding='utf-8') as f:
            memory_stats = json.load(f)
    else:
        print(f"[BEMR] init Alpha=1, Beta=1)")
        

    
    with open(test_file, "r") as f:
        test_dataset_raw = [json.loads(line) for line in f]

    acc_baseline = 0
    acc_rag = 0
    baseline_score = []

    
    if cfg.parameters.mode in ['baseline', 'all']:
        print("\n[Task A] Baseline ...")
        
        baseline_inputs = []
        for item in test_dataset_raw:
            sys_msg = cfg.experiment.prompts.sys_msg
            formatted_prompt = format_base_prompt(sys_msg, item['question'],model_source)
            baseline_inputs.append(formatted_prompt)

        baseline_preds = generator.generate(baseline_inputs)
        

        if is_code_task:
            baseline_results = []
            for item, pred in zip(test_dataset_raw, baseline_preds):
                
                
                res_item = item.copy() 
                res_item['pred'] = pred
                baseline_results.append(res_item)
        
        
            acc_baseline,baseline_score  = evaluate_code_results(
                results=baseline_results, 
                experiment_name=f"Baseline (No RAG)",
                result_log_file=result_log_file,
                dataset_type=code_dataset_type
            )
        else:
            baseline_results = []
            for item, pred in zip(test_dataset_raw, baseline_preds):
                baseline_results.append({
                    "question": item['question'],
                    "golden_answers": item['golden_answers'],
                    "pred": pred
                })
            
            acc_baseline,baseline_score = evaluate_results(baseline_results, "Baseline (No RAG)", result_log_file)

    
    if cfg.parameters.mode in ['rag', 'all']:
        print("\n[Task B] FlashRAG (Few-shot Retrieval)...")
        
        
        rag_config_dict = OmegaConf.to_container(cfg, resolve=True) 
        
        rag_update = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "retrieval_method": cfg.experiment.retrieval_method,
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "topk": cfg.parameters.retrieval_topk,
            
            "device": cfg.model.device,
            "generator_model_path": config['generator_model_path'] if 'generator_model_path' in config else "gpt2"
        }
        
        
        rag_config = Config(config_dict=rag_update)
        retriever = get_retriever(rag_config)
        print("[BEMR]...")
        retriever = BEMRRetrieverWrapper(retriever, memory_stats, cfg)

        rag_system_part = cfg.experiment.prompts.rag_system
        
        prompt_tpl = PromptTemplate(rag_config, system_prompt=rag_system_part, user_prompt="")

        pipeline = SequentialPipeline(rag_config, prompt_template=prompt_tpl, retriever=retriever, generator=generator)
        dataset_obj = Dataset(rag_config, test_file)
        
        rag_results = pipeline.run(dataset_obj)
        
        if is_code_task:
            print("[Data Merge]...")
            
            
            rag_eval_data = []
            for raw_item, rag_item in zip(test_dataset_raw, rag_results):
                
                merged_item = raw_item.copy()
                
                merged_item['pred'] = rag_item.pred 
                rag_eval_data.append(merged_item)

            
            acc_rag, scores_list = evaluate_code_results(
                results=rag_eval_data, 
                experiment_name=f"FlashRAG ({corpus_tag}) - Code",
                result_log_file=result_log_file,
                dataset_type=code_dataset_type
            )
            
            
            
            print(f" [Inject]{len(scores_list)}...")
            for i, item in enumerate(rag_results):
                
                item.score = scores_list[i]
        else:
            
            acc_rag,_ = evaluate_results(rag_results, f"FlashRAG ({corpus_tag} Memory)", result_log_file)
        
        
        
        analyze_memory_usage(rag_results, cfg, corpus_file, vis_image_file, old_stats=memory_stats,baseline_score = baseline_score,root_dir=root_dir,corpus_tag=corpus_tag)

    
    if cfg.parameters.mode == 'all':
        summary = (
            f"\n{'='*20} result {'='*20}\n"
            f"dataset: {cfg.experiment.test_dataset_name}\n"
            f"model: {model_source}\n"
            f"Baseline: {acc_baseline:.2f}%\n"
            f"FlashRAG: {acc_rag:.2f}%\n"
            f"gain: {acc_rag - acc_baseline:+.2f}%\n"
            f"{'='*50}\n"
        )
        print(summary)
        with open(result_log_file, "a", encoding="utf-8") as f:
            f.write(summary)

if __name__ == "__main__":
    main()