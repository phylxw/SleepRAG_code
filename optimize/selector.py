from typing import Dict, List, Tuple, Set
from omegaconf import DictConfig, OmegaConf

def select_ids_from_stats(memory_stats: Dict[str, dict], cfg: DictConfig) -> Tuple[List[str], List[str], List[str]]:
    """
    Select IDs for the Tri-Stream Optimization Framework (ICML).
    
    Logic Flow:
    1. Pre-calculate metrics (WinRate, Friction, Obs).
    2. Stream 1 (Evolve): Pick High Friction items (High Alpha & High Beta).
    3. Stream 2 (High): Pick High WinRate items (High Alpha & Low Beta) - EXCLUDING Evolve IDs.
    4. Stream 3 (Bad): Pick Low WinRate items (High Beta).
    """
    INIT_VAL = cfg.parameters.INIT_VAL
    scores: List[dict] = []

    # ---- Config ----
    top_k_high = int(cfg.optimizer.get("top_k_high", 50))
    bottom_k_low = int(cfg.optimizer.get("bottom_k_low", 80))
    top_k_evolve = int(cfg.optimizer.get("top_k_evolve", 50))

    # Thresholds
    legacy_freq_th = float(cfg.optimizer.get("low_freq_threshold", 1))
    min_obs = float(cfg.optimizer.get("min_obs_threshold", legacy_freq_th))
    evolve_win_rate_th = float(cfg.optimizer.get("evolve_win_rate_threshold", 0.5))

    # =========================================================
    # 0. Pre-calculation
    # =========================================================
    for mid, stats in memory_stats.items():
        alpha = float(stats.get("alpha", INIT_VAL))
        beta = float(stats.get("beta", INIT_VAL))
        total = alpha + beta

        win_rate = alpha / total if total > 1e-6 else 0.5

        n_obs = max(0.0, total - (INIT_VAL * 2))
        
        #  Friction
        friction = (alpha * beta) / total if total > 1e-6 else 0.0

        scores.append({
            "mid": str(mid),
            "win_rate": win_rate,
            "n_obs": n_obs,
            "alpha": alpha,
            "beta": beta,
            "total": total,
            "friction": friction,
            "neg_len": len(stats.get("neg_queries", []))
        })

    evolve_candidates = []
    for s in scores:
        if s["win_rate"] < evolve_win_rate_th:
            continue
        if s["n_obs"] < min_obs:
            continue
        if s["beta"] <= (INIT_VAL + 0.5): 
            continue
            
        evolve_candidates.append(s)

    evolve_candidates.sort(key=lambda x: (-x["friction"], -x["neg_len"]))

    evolve_final = evolve_candidates[:top_k_evolve]
    evolve_ids = [x["mid"] for x in evolve_final]
    evolve_ids_set = set(evolve_ids)
    high_candidates = []
    for s in scores:
        if s["win_rate"] < 0.5:
            continue
        if s["n_obs"] < min_obs:
            continue
        if s["mid"] in evolve_ids_set:
            continue
            
        high_candidates.append(s)
    high_candidates.sort(key=lambda x: (-x["win_rate"], -x["alpha"]))
    high_final = high_candidates[:top_k_high]
    high_ids = [x["mid"] for x in high_final]

    # =========================================================
    # Stream 3: Low-Score Restoration
    # =========================================================
    bad_candidates = []
    for s in scores:
        if s["win_rate"] >= 0.5:
            continue
        if s["n_obs"] < min_obs:
            continue
            
        bad_candidates.append(s)
    bad_candidates.sort(key=lambda x: (x["win_rate"], -x["total"]))

    bad_final = bad_candidates[:bottom_k_low]
    bad_ids = [x["mid"] for x in bad_final]
    print(f"\n [Tri-Stream Selection Report]")
    
    print(f" Evolution Stream (Top {len(evolve_ids)}) | Criteria: High Friction")
    if evolve_final:
        print(f"    Sample: ID={evolve_final[0]['mid']} | Win={evolve_final[0]['win_rate']:.2f} | Beta={evolve_final[0]['beta']:.1f} | Fric={evolve_final[0]['friction']:.2f}")
    else:
        print("    [Empty] No candidates met criteria.")

    print(f"  Retention Stream (Top {len(high_ids)}) | Criteria: High WinRate")
    if high_final:
        print(f"    Sample: ID={high_final[0]['mid']} | Win={high_final[0]['win_rate']:.2f} | Alpha={high_final[0]['alpha']:.1f}")

    print(f"  Restoration Stream (Top {len(bad_ids)}) | Criteria: Low WinRate")
    if bad_final:
        print(f"    Sample: ID={bad_final[0]['mid']} | Win={bad_final[0]['win_rate']:.2f} | Total={bad_final[0]['total']:.1f}")

    return high_ids, bad_ids, evolve_ids