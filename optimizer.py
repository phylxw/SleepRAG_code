import os
import json
from typing import Dict, List
import hydra
from omegaconf import DictConfig
import logging


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


from tools.optimize.callllm import init_llm, call_llm_batch
from tools.optimize.callexpert import init_expert_llm, call_expert, call_expert_batch
from tools.optimize.memoryload import load_clustered_memories, load_cluster_summary
from optimize.selector import select_ids_from_stats
from optimize.prune import prune

from optimize.textgrad_opt import textgrad_opt 
from optimize.evolve import evolve_high_score_opt

@hydra.main(version_base=None, config_path="conf", config_name="config")
def optimize_memory(cfg: DictConfig):
    
    
    
    init_llm(cfg)          
    init_expert_llm(cfg)   

    
    cluster_file = cfg.paths.cluster_output
    summary_file = cfg.paths.cluster_summary
    stats_file = cfg.paths.stats_file
    output_file = cfg.paths.optimized_memory
    stats_optimized_file = cfg.paths.stats_optimized_file
    log_file_path = cfg.paths.textgrad_log

    
    if not os.path.exists(stats_file):
        print(f"fail: {stats_file}")
        return
    with open(stats_file, 'r', encoding='utf-8') as f:
        memory_stats = json.load(f)

    
    memories, id_order = load_clustered_memories(cluster_file)
    cluster_to_ids = load_cluster_summary(summary_file)
    
    if not memories: 
        print("empty")
        return

    high_ids, bad_ids, evolve_ids = select_ids_from_stats(memory_stats, cfg)


    to_delete_ids = prune(memories, memory_stats, high_ids,cfg=cfg)
    new_supplement_ids = evolve_high_score_opt(cfg, memories, memory_stats, log_file_path,evolve_ids)

    
    optimized_ids = textgrad_opt(cfg, memories, memory_stats,log_file_path, cluster_to_ids, bad_ids, to_delete_ids)
    current_memory_ids = set(memories.keys())
    old_ids_set = set(id_order)
    
    new_ids = [mid for mid in memories.keys() if mid not in old_ids_set]
    
    if new_ids:
        
        final_save_order = id_order + new_ids
    else:
        final_save_order = id_order

    kept_count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for mid in final_save_order:
            
            if mid not in memories: continue
            
            if mid in to_delete_ids: continue
            
            
            f.write(json.dumps(memories[mid], ensure_ascii=False) + "\n")
            kept_count += 1

    print("\n========== Stats Sync ==========")
    
    
    for del_id in to_delete_ids:
        if del_id in memory_stats:
            del memory_stats[del_id]

    
    all_changed_ids = optimized_ids.union(new_supplement_ids)
    
    for opt_id in all_changed_ids:
        if opt_id in memory_stats:
            memory_stats[opt_id]['alpha'] = 1.0
            memory_stats[opt_id]['beta'] = 1.0
            
            memory_stats[opt_id]['neg_queries'] = []
            memory_stats[opt_id]['pos_queries'] = []

    
    
    cleaned_count = 0
    for mid in memory_stats:
        stats = memory_stats[mid]
        
        stats['pos_queries'] = []
        stats['neg_queries'] = []
        cleaned_count += 1
            
    print(f"delete {len(to_delete_ids)}")
    print(f"rewrite {len(all_changed_ids)}")
    print(f"clean {cleaned_count} ")
    
    try:
        with open(stats_optimized_file, 'w', encoding='utf-8') as f:
            json.dump(memory_stats, f, ensure_ascii=False, indent=2)
        print(f"[BEMR] : {stats_optimized_file}")
    except Exception as e:
        print(f"fail: {e}")

if __name__ == "__main__":
    optimize_memory()