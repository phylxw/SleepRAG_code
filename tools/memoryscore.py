
import os
from omegaconf import DictConfig, OmegaConf
import json
from tqdm import tqdm
from tools.evaluate import judge_math_item
import matplotlib.pyplot as plt
from tools.score.bemr import _calculate_bemr_final_score
import copy


def _load_memory_corpus(corpus_file: str):
    all_memory_ids = set()
    id_to_content = {} 
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                mid = str(item['id'])
                all_memory_ids.add(mid)
                id_to_content[mid] = item.get("contents", "")
    except Exception as e:
        print(f" {corpus_file}，fail: {e}")
    return all_memory_ids, id_to_content

def _calculate_scores(rag_results, all_memory_ids, cfg, old_stats=None, baseline_scores=None):

    INIT_VAL = cfg.parameters.INIT_VAL
    
    if old_stats:
        memory_stats = copy.deepcopy(old_stats)
        
        for mid in all_memory_ids:
            if mid not in memory_stats:
                memory_stats[mid] = {'alpha': INIT_VAL, 'beta': INIT_VAL, 'pos_queries': [], 'neg_queries': []}
    else:
        memory_stats = {mid: {'alpha': INIT_VAL, 'beta': INIT_VAL, 'pos_queries': [], 'neg_queries': []} for mid in all_memory_ids}

    correct_count = 0
    
    
    for i, item in enumerate(tqdm(rag_results, desc="Scoring & Capturing Gradients (BEMR)")):
        
        
        if cfg.experiment.tag in ["humaneval", "mbpp"]:
            is_rag_correct = (item.score == 1.0)
        else:
            try:
                is_rag_correct, _, _ = judge_math_item(item)
            except Exception:
                is_rag_correct = False
        
        if is_rag_correct: correct_count += 1

        
        if baseline_scores and i < len(baseline_scores):
            
            is_base_correct = (baseline_scores[i] == 1.0)
        else:
            
            
            
            
            is_base_correct = False 

        
        q = getattr(item, 'question', '') or getattr(item, 'prompt', '') or ''
        q = q.strip()
        gold_list = getattr(item, 'golden_answers', [])
        a = gold_list[0] if gold_list else "No Answer Provided"
        current_query = f"[Question]: {q}\n   [Target Answer]: {str(a)[:500]}"

        
        retrieved_docs = getattr(item, 'retrieval_result', [])
        
        for doc in retrieved_docs:
            doc_id = str(doc.get('id')) if isinstance(doc, dict) else str(getattr(doc, 'id', None))
            
            if doc_id and doc_id in memory_stats:
                
                
                
                
                
                if is_rag_correct and not is_base_correct:
                    memory_stats[doc_id]['alpha'] += 2.0  
                    if current_query not in memory_stats[doc_id]['pos_queries']:
                        memory_stats[doc_id]['pos_queries'].append(current_query)
                
                
                
                elif not is_rag_correct and is_base_correct:
                    memory_stats[doc_id]['beta'] += 2.0   
                    if current_query not in memory_stats[doc_id]['neg_queries']:
                        memory_stats[doc_id]['neg_queries'].append(current_query)
                
                
                
                
                elif is_rag_correct and is_base_correct:
                    memory_stats[doc_id]['alpha'] += 0.05  
                
                
                
                
                elif not is_rag_correct and not is_base_correct:
                    memory_stats[doc_id]['beta'] += 0.25
                    
                    if current_query not in memory_stats[doc_id]['neg_queries']:
                        memory_stats[doc_id]['neg_queries'].append(current_query)

    
    final_scores_map = {}
    for mid, stats in memory_stats.items():
        total = stats['alpha'] + stats['beta']
        
        score = stats['alpha'] / total if total > 0 else 0.5
        final_scores_map[mid] = score
    
    return final_scores_map, memory_stats, correct_count

def _print_stats_and_save(memory_scores, id_to_content, total_questions, correct_count, freq_file ,is_write = True):

    sorted_memories = sorted(memory_scores.items(), key=lambda x: (-x[1], x[0]))
    
    
    total_mem = len(sorted_memories)
    positive_mem = sum(1 for _, v in sorted_memories if v > 0.51)
    negative_mem = sum(1 for _, v in sorted_memories if v < 0.49)
    zero_mem = sum(1 for _, v in sorted_memories if v < 0.51 and v > 0.49)
    
    print(f" statistics:")
    print(f"   - total: {total_mem}")
    print(f"   - contribute: {positive_mem} ({(positive_mem/total_mem)*100:.1f}%)")
    print(f"   - distribute: {negative_mem} ({(negative_mem/total_mem)*100:.1f}%)")
    print(f"   - zero: {zero_mem}")
    print(correct_count)
    print(total_questions)
    print(f"   - rate: {correct_count/total_questions*100:.2f}%")

    if is_write :
        
        try:
            os.makedirs(os.path.dirname(freq_file), exist_ok=True)
            
            with open(freq_file, "w", encoding="utf-8") as f:
                for rank, (mid, score) in enumerate(sorted_memories, start=1):
                    record = {
                        "rank": rank,
                        "memory_id": mid,
                        "freq": round(score, 3), 
                        "contents": id_to_content.get(mid, "")
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("OK")
        except Exception as e:
            print(f"fail : {e}")
        
    return sorted_memories

def _visualize_results(cfg: DictConfig, sorted_memories, vis_image_file: str):

    if cfg.experiment.visualize_memory:
        print(f"[Visual]: {vis_image_file}")
        try:
            ids = [m[0] for m in sorted_memories]
            scores = [m[1] for m in sorted_memories]
            
            display_limit = 30
            if len(ids) > display_limit * 2:
                plot_ids = ids[:display_limit] + ["..."] + ids[-display_limit:]
                plot_scores = scores[:display_limit] + [0] + scores[-display_limit:]
                
                colors = []
                for s in plot_scores:
                    if s > 0: colors.append('skyblue')
                    elif s < 0: colors.append('salmon')
                    else: colors.append('lightgrey')
            else:
                plot_ids = ids
                plot_scores = scores
                colors = ['skyblue' if s > 0 else 'salmon' if s < 0 else 'lightgrey' for s in plot_scores]

            plt.figure(figsize=(15, 6))
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            bars = plt.bar(plot_ids, plot_scores, color=colors, edgecolor='navy')
            plt.title(f'Memory  Score', fontsize=14)
            plt.ylabel('Score')
            plt.xticks(rotation=90, fontsize=8) 
            
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if plot_ids[i] != "...": 
                    y_pos = height if height >= 0 else height - (max(scores)*0.05)
                    va = 'bottom' if height >= 0 else 'top'
                    plt.text(bar.get_x() + bar.get_width()/2., y_pos, f'{int(height*1000)/1000}',
                             ha='center', va=va, fontsize=8)
            
            plt.tight_layout()
            plt.savefig(vis_image_file, dpi=300)
            print("OK")
        except ImportError:
            print("fail")
    else:
        print("\n [Top 10 High-Utility Memories]")
        for mid, score in sorted_memories[:10]:
            print(f"   ID: {mid:<5} | Score: {score}")