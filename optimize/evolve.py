import os
import re
import uuid
from typing import Set, List, Dict, Optional,Tuple
from omegaconf import DictConfig

# Tooling
from tools.optimize.callllm import call_llm_batch
from tools.optimize.callexpert import call_expert_batch
from utils.opt.toolfunction import _extract_memory_blocks,_basic_guard

import re
from typing import List, Tuple


# ------------------------------------------------------------------------------
# Acceptance test (for high-score evolve candidates)
# ------------------------------------------------------------------------------

_EVOLVE_VERDICT_RE = re.compile(r"Verdict:\s*(PASS|FAIL)", re.IGNORECASE)
_EVOLVE_FEEDBACK_RE = re.compile(r"Feedback:\s*(.*)", re.IGNORECASE | re.DOTALL)

def _evolve_parse_acceptance(output: str):
    if not output:
        return {"verdict": "FAIL", "feedback": "No judge output."}
    m = _EVOLVE_VERDICT_RE.search(output)
    verdict = (m.group(1).upper() if m else "FAIL")
    m2 = _EVOLVE_FEEDBACK_RE.search(output)
    feedback = (m2.group(1).strip() if m2 else "").strip()
    if not feedback:
        feedback = "OK" if verdict == "PASS" else "Missing feedback."
    return {"verdict": verdict, "feedback": feedback}

def _evolve_acceptance_batch(cfg, items):
    prompts = []
    for it in items:
        prompts.append(cfg.optimizer.prompts.expert_judge.format(
            failed=(it.get("failed_queries","") or "").strip(),
            old=(it.get("old_memory","") or "").strip(),
            new=(it.get("new_memory","") or "").strip(),
        ))
    if not prompts:
        return []
    outs = call_expert_batch(prompts, cfg)
    return [_evolve_parse_acceptance(o) for o in outs]

def _evolve_acceptance_enabled(cfg) -> bool:
    opt = getattr(cfg, "optimizer", None)
    if opt is None:
        return True
    acc = getattr(opt, "acceptance", None)
    if acc is None:
        return bool(getattr(opt, "acceptance_enabled", True))
    return bool(getattr(acc, "enabled", True))

# ------------------------------------------------------------------------------
# Rollback / retry logic (shared for SUPPLEMENT and SPLIT)
# ------------------------------------------------------------------------------

def _get_max_retries(cfg) -> int:
    try:
        return int(getattr(cfg.parameters, "max_retries", 2) or 2)
    except Exception:
        return int(getattr(cfg.parameters, "max_retries", 2) or 2)

def _judge_one_candidate(cfg, *, failed_queries: str, old_memory: str, new_memory: str) -> Optional[str]:
    """Return None if PASS; else return feedback string."""
    if not _evolve_acceptance_enabled(cfg):
        return None
    if not (failed_queries or "").strip():
        return None
    res = _evolve_acceptance_batch(cfg, [{
        "failed_queries": failed_queries,
        "old_memory": old_memory,
        "new_memory": new_memory,
    }])[0]
    if (res.get("verdict") or "").upper() == "PASS":
        return None
    return res.get("feedback", "Acceptance FAIL.")

# ------------------------------------------------------------------------------
# Main high-score evolution
# ------------------------------------------------------------------------------

def evolve_high_score_opt(cfg: DictConfig, memories: Dict, memory_stats: Dict,log_file_path, high_ids: List[str]) -> Set[str]:
    """High-score memory evolution (SUPPLEMENT / SPLIT) with robust parsing & rollback.
    Semantics: both SUPPLEMENT and SPLIT add exactly ONE new memory; champion memory is never modified.
    """
    print("\n========== Ace Evolution ==========")

    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    except Exception:
        pass
    def tee_print(msg):
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")
        except Exception:
            pass

    # --- 2) Target selection: only champions with failed queries ---
    target_ids: List[str] = []
    for mid in list(high_ids):
        if mid not in memories:
            continue
        stats = memory_stats.get(mid, {}) or {}
        neg_queries = stats.get("neg_queries", []) or []
        if len(neg_queries) > 0:
            target_ids.append(mid)

    # Optional debug limit
    debug_limit = int(getattr(cfg.optimizer, "debug_high_score_limit", 0) or 0)
    if debug_limit > 0:
        target_ids = target_ids[:debug_limit]

    if target_ids:
        print(f" ID list: {target_ids}")
    if not target_ids:
        return set()

    batch_size = int(cfg.optimizer.llm_batch_size)
    new_created_ids_total: Set[str] = set()

    for i in range(0, len(target_ids), batch_size):
        chunk_ids = target_ids[i : i + batch_size]
        print(f" [Expert-Batch] {i} - {i+len(chunk_ids)}...")

        expert_prompts: List[str] = []
        chunk_metadata: List[dict] = []

        for mid in chunk_ids:
            rec = memories[mid]
            base_text = rec.get("contents", "")
            stats = memory_stats.get(mid, {}) or {}
            neg_queries = stats.get("neg_queries", []) or []

            top_k_neg = int(getattr(cfg.optimizer, "high_grad_topk_neg", 5) or 5)
            neg_text = "\n".join([f"- {q}" for q in neg_queries[:top_k_neg]])

            try:
                prompt = cfg.optimizer.prompts.high_grad_expert.format(content=base_text, neg_queries=neg_text)
            except Exception as e:
                print(f"❌ Prompt fail (MID: {mid}): {e}")
                continue

            expert_prompts.append(prompt)
            chunk_metadata.append({
                "mid": mid,
                "base_text": base_text,
                "neg_text": neg_text,
                "expert_prompt_content": prompt,
            })

        if not expert_prompts:
            continue

        expert_outputs = call_expert_batch(expert_prompts, cfg)

        student_prompts: List[str] = []
        student_tasks: List[dict] = []

        for meta, expert_resp in zip(chunk_metadata, expert_outputs):
            mid = meta["mid"]

            log_info = {
                "mid": mid,
                "type": "evolve_high_score",
                "expert_prompt": meta.get("expert_prompt_content", ""),
                "expert_output": expert_resp,
                "action": "UNKNOWN",
                "gradient": "N/A",
                "split_num": 1,  # kept for backward compatibility in logs
                "student_prompt": "N/A",
            }

            if not expert_resp:
                tee_print(f" [Error] MID: {mid} - Expert output is empty.")
                continue

            action_match = re.search(r"\\box\{(IGNORE|SUPPLEMENT|SPLIT)\}", expert_resp)
            action = action_match.group(1).strip() if action_match else "IGNORE"

            gradient_match = re.search(r"\\gradient\{(.*?)\}", expert_resp, re.DOTALL)
            advice = gradient_match.group(1).strip() if gradient_match else "No specific advice provided."
            gradient = advice

            # IMPORTANT: SPLIT is forced to ONE new memory (no 1-to-many)
            split_num = 1

            log_info["action"] = action
            log_info["gradient"] = advice
            log_info["split_num"] = split_num

            if action == "IGNORE":
                with open(log_file_path, "a", encoding="utf-8") as log_f:
                    mid_display = str(mid)[:8]
                    grad = str(advice)
                    grad_prev = f"{grad[:20]}...{grad[-20:]}" if len(grad) > 40 else grad
                    
                    log_lines = [
                        f" [{mid_display}] | EVOLVE |  IGNORED",
                        f"   Strategy: High-Score-Evolve",
                        f"   Action  : IGNORE",
                        f"   Reason  : {grad_prev}",
                    ]
                    log_f.write("\n".join(log_lines))
                    log_f.flush()
                continue

            if action == "SUPPLEMENT":
                tpl = cfg.optimizer.prompts.appgrad_high_supplement
                s_prompt = tpl.format(original_content=meta["base_text"], advice=advice)
                log_info["student_prompt"] = s_prompt
                student_prompts.append(s_prompt)
                student_tasks.append({
                    "parent_mid": mid,
                    "action": "SUPPLEMENT",
                    "log": log_info,
                    "old_content": meta.get("base_text",""),
                    "neg_text": meta.get("neg_text",""),
                })

            elif action == "SPLIT":
                tpl = cfg.optimizer.prompts.appgrad_high_split
                # allow templates with/without {num}
                try:
                    s_prompt = tpl.format(neg_text=meta["neg_text"], advice=advice, num=1)
                except Exception:
                    s_prompt = tpl.format(neg_text=meta["neg_text"], advice=advice)
                log_info["student_prompt"] = s_prompt
                student_prompts.append(s_prompt)
                student_tasks.append({
                    "parent_mid": mid,
                    "action": "SPLIT",
                    "log": log_info,
                    "old_content": meta.get("base_text",""),
                    "neg_text": meta.get("neg_text",""),
                })

        if not student_prompts:
            continue

        for task in student_tasks:
            action_type = task.get("action", "UNKNOWN").upper()
            mid_display = task['parent_mid'][:8] 

            tee_print(f"   -> [{mid_display}] |  {action_type} |  High-Score-Evolve")
        tee_print("   ------------------------------------------------")


        student_outputs = call_expert_batch(student_prompts, cfg)


        max_retries = _get_max_retries(cfg)


        candidates = []
        for task, raw_out in zip(student_tasks, student_outputs):
            candidates.append({
                "task": task,
                "history": [{"out": raw_out, "judge": None}],
                "status": "PENDING", # PENDING, PASS, FAIL
                "final_output": None,
                "fail_reason": ""
            })

        max_retries = _get_max_retries(cfg)
        min_len = int(getattr(cfg.optimizer, "min_memory_len", 20) or 20)
        max_len = int(getattr(cfg.optimizer, "max_memory_len", 2000) or 2000)
        
        for round_idx in range(max_retries + 1):

            to_judge_indices = []
            judge_payloads = []
            
            for i, cand in enumerate(candidates):
                if cand["status"] != "PENDING":
                    continue
                    
                last_attempt = cand["history"][-1]
                raw_txt = last_attempt["out"]

                blocks = _extract_memory_blocks(raw_txt)

                valid_block = None
                for b in blocks:
                    if _basic_guard(b, min_len=min_len, max_len=max_len):
                        valid_block = b
                        break
                
                if not valid_block:
                    cand["fail_reason"] = "Rejected by basic guard (length/format/banned)."
                    last_attempt["judge"] = {"verdict": "FAIL", "feedback": cand["fail_reason"]}
                else:
                    last_attempt["parsed_block"] = valid_block

                    meta = cand["task"]
                    neg_text = meta.get("neg_text", "")
                    
                    if _evolve_acceptance_enabled(cfg) and neg_text:
                        to_judge_indices.append(i)
                        judge_payloads.append({
                            "failed_queries": neg_text,
                            "old_memory": meta.get("old_content", ""),
                            "new_memory": valid_block
                        })
                    else:

                        cand["status"] = "PASS"
                        cand["final_output"] = valid_block
                        last_attempt["judge"] = {"verdict": "PASS", "feedback": "OK (Skipped)"}


            if judge_payloads:
                tee_print(f" [Batch Judge] Round {round_idx}:  {len(judge_payloads)} ...")
                judge_results = _evolve_acceptance_batch(cfg, judge_payloads)
                
                for idx, res in zip(to_judge_indices, judge_results):
                    cand = candidates[idx]
                    last_attempt = cand["history"][-1]
                    last_attempt["judge"] = res
                    
                    if res["verdict"] == "PASS":
                        cand["status"] = "PASS"
                        cand["final_output"] = last_attempt["parsed_block"]
                    else:
                        cand["fail_reason"] = res["feedback"]

            if round_idx == max_retries:
                break

            to_rewrite_indices = []
            retry_prompts = []

            for i, cand in enumerate(candidates):
                if cand["status"] == "PENDING":
                    to_rewrite_indices.append(i)
                    last_attempt = cand["history"][-1]
                    prev_out = last_attempt.get("out", "")
                    fb = cand.get("fail_reason", "Improve logic.")
                    base_prompt = cand["task"]["log"].get("student_prompt", "")
                    neg_q = cand["task"].get("neg_text", "")

                    retry_prompt = cfg.optimizer.prompts.retry_prompt.format(
                            failed=neg_q,
                            ori=base_prompt,
                            bad=(prev_out or ""),  
                            feedback=(fb or "")
                        )
                    
                    retry_prompts.append(retry_prompt)

            if not retry_prompts:
                break
            tee_print(f"\n  [Batch Retry] Round {round_idx+1}: {len(retry_prompts)} rollback...")

            for idx in to_rewrite_indices[:2]:
                    mid = candidates[idx]["task"]["parent_mid"][:8]
                    fb = candidates[idx].get("fail_reason", "")
                    tee_print(f"      ->  ID: {mid} | reason: {fb[:50]}...")
            retry_outputs = call_expert_batch(retry_prompts, cfg)

            for idx, new_out in zip(to_rewrite_indices, retry_outputs):
                mid = candidates[idx]["task"]["parent_mid"][:8]
                clean = new_out.strip().replace('\n', ' ')
                prev = clean if len(clean) < 40 else f"{clean[:20]}...{clean[-20:]}"
                tee_print(f" ->[Retry] ID: {mid} | {prev}")
                
                candidates[idx]["history"].append({"out": new_out, "judge": None})

        for cand in candidates:
            task = cand["task"]
            log_info = task["log"]
            parent_mid = task["parent_mid"]
            action_type = task["action"]
            accepted = (cand["status"] == "PASS" and cand["final_output"] is not None)
            final_txt = cand["final_output"] if accepted else cand["history"][-1]["out"]
            final_txt = (final_txt or "").strip().replace('\n', ' ')
            content_prev = f"{final_txt[:30]}...{final_txt[-30:]}" if len(final_txt) > 60 else final_txt
            
            last_judge = cand["history"][-1].get("judge") or {}
            judge_verdict = last_judge.get("verdict", "FAIL")
            judge_fb = last_judge.get("feedback", cand.get("fail_reason", "Unknown"))
            judge_prev = f"{judge_fb[:50]}..." if len(judge_fb) > 50 else judge_fb

            with open(log_file_path, "a", encoding="utf-8") as log_f:
                status_str = " ACCEPTED" if accepted else " REJECTED"
                grad = str(log_info.get("gradient", ""))
                grad_prev = f"{grad[:20]}...{grad[-20:]}" if len(grad) > 40 else grad
                
                log_lines = [
                    f"  [{parent_mid[:8]}] | {action_type} | {status_str}",
                    f"   Strategy: High-Score-Evolve (Batched)",
                    f"   Action  : {action_type}",
                    f"   Gradient: {grad_prev}",
                    f"   Result  : {content_prev}",
                    f"   Judge   : {judge_verdict} ({judge_prev})",
                    "-" * 60 + "\n"
                ]
                log_f.write("\n".join(log_lines))
                log_f.flush()

            if accepted:
                new_id = str(uuid.uuid4())
                suffix = "supplement" if action_type == "SUPPLEMENT" else "split"
                _save_new_memory(memories, memory_stats, new_id, cand["final_output"], parent_mid, f"high_score_{suffix}")
                new_created_ids_total.add(new_id)
                tee_print(f"  [NEW] {parent_mid[:8]} -> {new_id[:8]} ({action_type})")

    print(f" [Evolve]  {len(new_created_ids_total)} ")
    return new_created_ids_total

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _save_new_memory(memories, memory_stats, new_id, content, parent_id, opt_type):
    memories[new_id] = {
        "id": new_id,
        "contents": content,
        "cluster_id": -1,
        "opt_type": opt_type,
        "parent_id": parent_id,
    }
    memory_stats[new_id] = {
        "alpha": 1.0,
        "beta": 1.0,
        "neg_queries": [],
        "pos_queries": [],
    }

def _write_log(log_file_path: str, info: dict, result_content: str):
    try:
        with open(log_file_path, "a", encoding="utf-8") as f:
            log_entry = (
                f"\n{'='*60}\n"
                f" Parent Memory ID: {info.get('mid','')}\n"
                f"-- Expert Prompt (Input) ---\n{info.get('expert_prompt','')}\n\n"
                f"--️ Expert Output (Raw) ---\n{info.get('expert_output','')}\n\n"
                f"---  Parsed Decision ---\n"
                f"   Action   : {info.get('action','')}\n"
                f"   Advice   : {info.get('gradient','')}\n"
                f"   Split Num: {info.get('split_num',1)}\n\n"
                f"---  Student Prompt ---\n{info.get('student_prompt','')}\n\n"
                f"--- Final Result (New Memories) ---\n{result_content}\n"
                f"{'='*60}\n"
            )
            f.write(log_entry)
            f.flush()
    except Exception as e:
        print(f"fail: {e}")