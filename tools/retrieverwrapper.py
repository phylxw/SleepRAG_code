import numpy as np
import math




class BEMRRetrieverWrapper:

    def __init__(self, original_retriever, memory_stats, cfg):
        self.retriever = original_retriever
        self.memory_stats = memory_stats
        self.cfg = cfg
        self.INIT_VAL = cfg.parameters.INIT_VAL
        
        
        
        if hasattr(cfg, 'parameters'):
            self.final_topk = cfg.parameters.get("final_topk", 3)
        else:
            self.final_topk = 3

        self.lambda1 = cfg.parameters.get('bemr_lambda1', 1.0)
        self.lambda2 = cfg.parameters.get('bemr_lambda2', 0.5)
        
        print(f" [Wrapper] BEMR | Top-{self.final_topk}")

    def _calculate_ucb_score(self, doc_id, sim_score):
        stats = self.memory_stats.get(str(doc_id), {'alpha': self.INIT_VAL , 'beta': self.INIT_VAL})
        alpha = stats['alpha']
        beta = stats['beta']
        total = alpha + beta
        
        mean_utility = alpha / total
        exploration = math.sqrt(math.log(max(total, 1)) / total)
        
        
        
        ucb_part = (self.lambda1 * mean_utility) + (self.lambda2 * exploration)
        
        
        
        bm25_part = 0.001 * sim_score
        
        final_score = ucb_part + bm25_part
        return final_score, ucb_part  

    
    def search(self, query_list, num=None, return_score=False):
        
        
        
        
        INITIAL_POOL_SIZE = 20 
        search_k = max(num if num else 0, INITIAL_POOL_SIZE)
        
        
        raw_output = self.retriever.batch_search(query_list, num=search_k, return_score=True)
        
        if isinstance(raw_output, tuple):
            batch_hits, batch_scores = raw_output
        else:
            batch_hits = raw_output
            batch_scores = [[0.0] * len(h) for h in batch_hits]

        reranked_results = []
        reranked_scores = []

        
        for q_idx, (hit_list, score_list) in enumerate(zip(batch_hits, batch_scores)):
            
            
            debug_info = [] 
            
            
            if not score_list:
                reranked_results.append([])
                reranked_scores.append([])
                continue
                
            min_s, max_s = min(score_list), max(score_list)
            denominator = max_s - min_s if (max_s - min_s) > 1e-6 else 1.0
            
            scored_hits = []
            
            
            for i, hit in enumerate(hit_list):
                doc_id = hit.get('id')
                raw_bm25 = score_list[i] 
                
                
                norm_bm25 = (raw_bm25 - min_s) / denominator
                
                
                final_score, pure_ucb = self._calculate_ucb_score(doc_id, norm_bm25)
                
                
                hit['score'] = final_score
                scored_hits.append(hit)
                
                
                stats = self.memory_stats.get(str(doc_id), {'alpha': self.INIT_VAL, 'beta': self.INIT_VAL})
                
                
                debug_info.append({
                    "id": doc_id,
                    "bm25_raw": raw_bm25,
                    "bm25_norm": norm_bm25,
                    "pure_ucb": pure_ucb,    
                    "final_score": final_score, 
                    "stats": f"{stats['alpha']:.1f}/{stats['beta']:.1f}"
                })
            
            scored_hits.sort(key=lambda x: x['score'], reverse=True)
            
            
            cutoff = self.final_topk
            if num and num < self.final_topk:
                cutoff = num
            truncated_hits = scored_hits[:cutoff]
            truncated_scores = [h['score'] for h in truncated_hits]
            
            reranked_results.append(truncated_hits)
            reranked_scores.append(truncated_scores)

            debug_info.sort(key=lambda x: x['final_score'], reverse=True)


        if return_score:
            return reranked_results, reranked_scores
        else:
            return reranked_results
        
    def batch_search(self, query_list, num=None, return_score=False):

        print(f" [Wrapper] , BEMR...")
        return self.search(query_list, num, return_score)

    def __getattr__(self, name):
        
        print(f"[Wrapper Bypass] : {name}")
        return getattr(self.retriever, name)