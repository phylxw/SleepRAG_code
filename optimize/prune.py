from __future__ import annotations
import math
from typing import Dict, Any, Optional, Set, List

try:
    from omegaconf import DictConfig
except Exception:
    DictConfig = Any

def _beta_mean(alpha: float, beta: float) -> float:
    s = alpha + beta
    return alpha / s if s > 0 else 0.5

def _beta_var(alpha: float, beta: float) -> float:
    s = alpha + beta
    if s <= 1:
        return 0.25
    return (alpha * beta) / (s * s * (s + 1.0))

def _lcb(alpha: float, beta: float, z: float) -> float:
    mu = _beta_mean(alpha, beta)
    var = _beta_var(alpha, beta)
    return mu - z * math.sqrt(max(var, 1e-12))

def _ucb(alpha: float, beta: float, z: float) -> float:
    mu = _beta_mean(alpha, beta)
    var = _beta_var(alpha, beta)
    return mu + z * math.sqrt(max(var, 1e-12))

def _safe_get_cfg(cfg: Any, path: str, default: Any) -> Any:
    if cfg is None:
        return default
    cur = cfg
    for key in path.split("."):
        try:
            if isinstance(cur, dict):
                cur = cur.get(key, None)
            else:
                cur = getattr(cur, key) if hasattr(cur, key) else cur.get(key)
        except Exception:
            return default
        if cur is None:
            return default
    return cur

def prune(
    memories: Dict[str, Dict[str, Any]], 
    memory_stats: Dict[str, Dict[str, Any]], 
    high_ids: List[str],  
    cfg: Optional[Any] = None
) -> Set[str]:
    """
    Tri-Stream Safe Pruning:
    Only prune memories that are strictly worse than the cluster leader, 
    AND ONLY if both are High-Score Candidates.
    
    This preserves:
    - Bad memories (for TextGrad Repair)
    - High-Friction memories (for Evolution)
    """
    enabled = bool(_safe_get_cfg(cfg, "optimizer.prune.enabled", True))
    verbose = bool(_safe_get_cfg(cfg, "optimizer.prune.verbose", True))
    
    if not enabled:
        if verbose:
            print("[Pruning] Disabled by config.")
        return set()

    
    high_ids_set = set(high_ids)

    
    z = float(_safe_get_cfg(cfg, "optimizer.prune.z", 1.64))
    min_cluster_size = int(_safe_get_cfg(cfg, "optimizer.prune.min_cluster_size", 2))
    min_obs_best = int(_safe_get_cfg(cfg, "optimizer.prune.min_obs_best", 5))
    max_prune_per_cluster = int(_safe_get_cfg(cfg, "optimizer.prune.max_prune_per_cluster", 2))

    if verbose:
        print("\n==========Retention Stream Pruning==========")
        print(f"   Strategy: Only prune redundant 'High IDs' that are statistically inferior to the leader.")

    
    cluster_groups: Dict[int, list[str]] = {}
    for mid, rec in memories.items():
        cid = rec.get("cluster_id", None)
        if cid is None: continue
        try:
            cid_int = int(cid)
            if cid_int < 0: continue 
            cluster_groups.setdefault(cid_int, []).append(str(mid))
        except: continue

    to_delete: Set[str] = set()
    pruned_count = 0
    
    
    for cid, members in cluster_groups.items():
        if len(members) < min_cluster_size:
            continue

        
        stats_list = []
        for mid in members:
            st = memory_stats.get(mid, {"alpha": 1.0, "beta": 1.0})
            alpha = float(st.get("alpha", 1.0))
            beta = float(st.get("beta", 1.0))
            n_obs = max(0.0, alpha + beta - 2.0)
            
            stats_list.append({
                "id": str(mid),
                "mean": _beta_mean(alpha, beta),
                "lcb": _lcb(alpha, beta, z),
                "ucb": _ucb(alpha, beta, z),
                "n_obs": n_obs,
                "is_high": str(mid) in high_ids_set  
            })

        
        stats_list.sort(key=lambda x: (-x["mean"], -x["n_obs"]))
        best = stats_list[0]

        
        
        
        if not best["is_high"]:
            continue
        
        
        
        if best["n_obs"] < min_obs_best:
            continue

        pruned_here = 0
        
        
        for mem in stats_list[1:]:
            if pruned_here >= max_prune_per_cluster:
                break
            
            
            
            
            if not mem["is_high"]:
                continue

            
            
            
            if mem["ucb"] < best["lcb"]:
                to_delete.add(mem["id"])
                pruned_here += 1
                pruned_count += 1
                
                if verbose:
                    print(f"   Cluster {cid}: Pruned redundant High-ID {mem['id'][:8]} (UCB={mem['ucb']:.2f}) < Leader {best['id'][:8]} (LCB={best['lcb']:.2f})")

    if verbose:
        print(f" [Pruning Summary] Plan to delete {pruned_count} redundant high-score memories.")

    return to_delete