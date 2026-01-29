import subprocess
import sys
import os
import time
from datetime import datetime
import hydra
from omegaconf import DictConfig
from utils.logger import setup_logging,Logger
import logging
import shutil
from hydra import compose, initialize
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

exp_override = None #["experiment=mbpp"]
sglang_url = None #"http://127.0.0.1:30001/v1" 
expert_url = None #"http://127.0.0.1:30001/v1"
expert_source= None #"qwen"
def run_step(script_name, step_desc, overrides, env=None, hydra_overrides=None):
    print(f"\n{'='*80}")
    print(f" [Step: {step_desc}] start {script_name}...")

    cmd = [sys.executable, script_name]

    if hydra_overrides:
        print(" Hydra Overrides:")
        for o in hydra_overrides:
            cmd.append(o)
            print(f"   - {o}")

    print(f"(Overrides):")
    for key, value in overrides.items():
        final_value = value
        if isinstance(value, str) and (os.path.exists(os.path.dirname(value)) or os.path.isabs(value)):
            final_value = os.path.abspath(value)

        cmd.append(f"++{key}={final_value}")
        print(f"   - {key} = {final_value}")

    print(f"{'-'*80}")

    current_env = os.environ.copy()
    if env:
        current_env.update(env)

    current_env["PYTHONUNBUFFERED"] = "1"

    start_time = time.time()
    try:
        subprocess.run(cmd, env=current_env, check=True)
    except subprocess.CalledProcessError:
        print(f"\n [Error] {script_name} fail。")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"[Success] {script_name} complete (time: {elapsed:.2f}s)")


def get_round_paths(root_dir, pipeline_id, round_idx, tag="sci"):

    base_dir = os.path.join(root_dir, "results", pipeline_id, f"round_{round_idx}")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return {
        "dir": base_dir,

        "corpus": os.path.join(base_dir, f"{tag}_corpus.jsonl"),
        "optimized_memory": os.path.join(base_dir, f"{tag}_optimized_memory_topk.jsonl"),
        "test": os.path.join(base_dir, f"{tag}_test.jsonl"),

        "stats": os.path.join(base_dir, f"{tag}_memory_stats.json"),
        "stats_optimized": os.path.join(base_dir, f"{tag}_memory_optimized_stats.json"),
        "stats_after": os.path.join(base_dir, f"{tag}_memory_after_stats.json"),

        "freq": os.path.join(base_dir, f"{tag}_memory_freq.jsonl"),
        "freq_after": os.path.join(base_dir, f"{tag}_memory_after_freq.jsonl"),

        "cluster_output": os.path.join(base_dir, f"{tag}_clustered_result.jsonl"),
        "cluster_summary": os.path.join(base_dir, f"{tag}_cluster_summary.jsonl"),
        "cluster_vis": os.path.join(base_dir, f"{tag}_visualization.png"),
        "cluster_plot": os.path.join(base_dir, f"{tag}_cluster_distribution.png"),

        "rag_cache": os.path.join(root_dir, "rag_result_cache", pipeline_id, f"round_{round_idx}")
    }

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config", overrides=exp_override)


    EXP_TAG = cfg.experiment.get("tag", "experiment")
    TOTAL_ROUNDS = cfg.parameters.get("total_rounds", 2)

    RESUME_PATH = cfg.parameters.get("resume_path", None) 

    root_dir = cfg.paths.root if "paths" in cfg and "root" in cfg.paths else os.getcwd()
    root_dir = os.path.abspath(root_dir)

    # init
    pipeline_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{EXP_TAG}_Loop"
    setup_logging(root_dir, pipeline_timestamp) 
    print(f"\n[Pipeline Start] | ID: {pipeline_timestamp}")
    
    if RESUME_PATH:
        print(f"[Resume Mode] : {RESUME_PATH}")
    
    print(f"root: {root_dir}")
    
    client_env = os.environ.copy()

    for r in range(TOTAL_ROUNDS):
        print(f"\n\n{'#'*80}")
        print(f" ======(Round {r})=====")
        print(f"{'#'*80}")

        curr_paths = get_round_paths(root_dir, pipeline_timestamp, r, tag=EXP_TAG)
        prev_paths = get_round_paths(root_dir, pipeline_timestamp, r-1, tag=EXP_TAG) if r > 0 else None

        skip_prepro = False

        # ==============================================================================
        # Input Source
        # ==============================================================================
        if r == 0:
            if RESUME_PATH and os.path.exists(RESUME_PATH):
                # === Resume ===
                print(f"[Round 0 - Resume] mode：-> {RESUME_PATH}")

                input_corpus = os.path.join(RESUME_PATH, f"{EXP_TAG}_optimized_memory_topk.jsonl")
                if not os.path.exists(input_corpus):
                    print("no optimized_memory，reading corpus...")
                    input_corpus = os.path.join(RESUME_PATH, f"{EXP_TAG}_corpus.jsonl")

                input_stats = os.path.join(RESUME_PATH, f"{EXP_TAG}_memory_after_stats.json")
                input_freq  = os.path.join(RESUME_PATH, f"{EXP_TAG}_memory_after_freq.jsonl")
                
                skip_prepro = True

                if os.path.exists(input_stats): shutil.copy(input_stats, curr_paths['stats'])
                if os.path.exists(input_freq):  shutil.copy(input_freq, curr_paths['freq'])
                if os.path.exists(input_corpus): shutil.copy(input_corpus, curr_paths['corpus'])

                input_corpus = curr_paths['corpus']
                input_stats  = curr_paths['stats']
                input_freq   = curr_paths['freq']

            else:
                # === Fresh Start ===
                print(f" [Round 0 - Fresh] mode")
                input_corpus = curr_paths['corpus']
                input_stats  = curr_paths['stats']
                input_freq   = curr_paths['freq']
                
        else:
            # === Loop ===
            print(f" [Round {r}] mode")
            input_corpus = prev_paths['optimized_memory']
            input_stats  = prev_paths['stats_after']
            input_freq   = prev_paths['freq_after']

        is_fresh_start = (r == 0 and not RESUME_PATH)

        if not is_fresh_start:
            for f_path, f_name in [(input_corpus, "Corpus"), (input_stats, "Stats"), (input_freq, "Freq")]:
                if not os.path.exists(f_path):
                    print(f"fail：no {f_name} in {f_path}")
                    if r == 0 and RESUME_PATH:
                        sys.exit(1)
        else:
            print("[Fresh Start]")

        # --------------------------------------------------
        # Step 1: Pre-process
        # --------------------------------------------------
        if r == 0:
            pre_overrides = {
                "paths.stats_file": curr_paths['stats'],
                "paths.optimized_memory": curr_paths['corpus'],
                "paths.stats_optimized_file": curr_paths['stats'],
                "paths.freq_file": curr_paths['freq'], 
                "paths.corpus_file": curr_paths['corpus'],
                "paths.test_file": curr_paths['test'],
                "paths.result_dir": curr_paths['dir'], 
            }

            if sglang_url:
                pre_overrides["model.sglang_api_url"] = sglang_url
            if expert_url:
                pre_overrides["expert_model.sglang_api_url"] = expert_url
            if expert_source:
                pre_overrides["expert_model.source"] = expert_source
            if skip_prepro:
                # === Resume  ===
                print("[Resume]")

                eval_overrides = {
                    "paths.optimized_memory": curr_paths['corpus'],
                    "paths.stats_optimized_file": curr_paths['stats'],
                    "paths.stats_after_file": curr_paths['stats_after'],
                    "paths.freq_after_file": curr_paths['freq_after'],
                    "paths.rag_cache_dir": curr_paths['rag_cache'],
                    "parameters.is_first": False,
                    "paths.result_dir": curr_paths['dir'], 
                }

                if sglang_url:
                    eval_overrides["model.sglang_api_url"] = sglang_url
                if expert_url:
                    pre_overrides["expert_model.sglang_api_url"] = expert_url
                if expert_source:
                    pre_overrides["expert_model.source"] = expert_source
            else:
                
                run_step("evallast.py", f"R{r}-0. evalfirst", overrides=pre_overrides, env=client_env,hydra_overrides=exp_override)
                run_step("pre.py", f"R{r}-1. pre", pre_overrides, env=client_env,hydra_overrides=exp_override)

        # --------------------------------------------------
        # Step 2: Clustering
        # --------------------------------------------------
        cluster_overrides = {
            "paths.cluster_output": curr_paths['cluster_output'],
            "paths.cluster_summary": curr_paths['cluster_summary'],
            "paths.cluster_vis": curr_paths['cluster_vis'],
            "paths.cluster_plot": curr_paths['cluster_plot'],
            "paths.corpus_file": input_corpus,
            "paths.stats_file": input_stats,
            "paths.freq_file": input_freq
        }
        if sglang_url:
            cluster_overrides["model.sglang_api_url"] = sglang_url   
        if expert_url:
            pre_overrides["expert_model.sglang_api_url"] = expert_url   
        if expert_source:
            pre_overrides["expert_model.source"] = expert_source                 
        run_step("cluster.py", f"R{r}-2. 聚类", cluster_overrides, env=client_env,hydra_overrides=exp_override)

        # --------------------------------------------------
        # Step 3: Optimizer
        # --------------------------------------------------
        opt_overrides = {
            "paths.corpus_file": input_corpus,
            "paths.stats_file": input_stats,
            "paths.freq_file": input_freq,
            "paths.cluster_output": curr_paths['cluster_output'],
            "paths.cluster_summary": curr_paths['cluster_summary'],
            "paths.optimized_memory": curr_paths['optimized_memory'],
            "paths.stats_optimized_file": curr_paths['stats_optimized'],
        }
        if sglang_url:
            opt_overrides["model.sglang_api_url"] = sglang_url
        if expert_url:
            pre_overrides["expert_model.sglang_api_url"] = expert_url
        if expert_source:
            pre_overrides["expert_model.source"] = expert_source
        run_step("optimizer.py", f"R{r}-3. optimizer", opt_overrides, env=client_env,hydra_overrides=exp_override)
        # --------------------------------------------------
        # Step 4: Eval
        # --------------------------------------------------
        eval_overrides = {
            "paths.optimized_memory": curr_paths['optimized_memory'],
            "paths.stats_optimized_file": curr_paths['stats_optimized'],
            "paths.stats_after_file": curr_paths['stats_after'],
            "paths.freq_after_file": curr_paths['freq_after'],
            "paths.rag_cache_dir": curr_paths['rag_cache'],
            "parameters.is_first": False,
            "paths.result_dir": curr_paths['dir'], 
        }
        if sglang_url:
            eval_overrides["model.sglang_api_url"] = sglang_url
        if expert_url:
            pre_overrides["expert_model.sglang_api_url"] = expert_url
        if expert_source:
            pre_overrides["expert_model.source"] = expert_source
        if not os.path.exists(curr_paths['stats_optimized']):
            shutil.copy(input_stats, curr_paths['stats_optimized'])

        run_step("eval.py", f"R{r}-4. eavl", eval_overrides, env=client_env,hydra_overrides=exp_override)
        run_step("evallast.py", f"R{r}-5. evallast", eval_overrides, env=client_env,hydra_overrides=exp_override)

        print(f"\n {r} round OK!")

    print(f"\ncomplete!")

if __name__ == "__main__":
    main()