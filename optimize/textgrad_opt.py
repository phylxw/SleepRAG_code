import os
import re
import uuid
import logging
from typing import List, Dict, Set, Any, Optional
from dataclasses import dataclass, field


from tools.optimize.callllm import call_llm_batch
from tools.optimize.callexpert import call_expert_batch
from utils.opt.toolfunction import _extract_single_memory, _basic_guard


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ICML_WarRoom")

@dataclass
class OptimizationTask:

    mid: str
    original_content: str
    stats: Dict
    
    diagnosis_prompt: str = ""
    expert_action: str = "WAITING" 
    expert_advice: str = ""
    
    student_prompt: str = ""
    generated_content: str = ""
    
    judge_verdict: str = "PENDING"
    judge_feedback: str = ""
    retry_count: int = 0
    
    final_accepted_content: Optional[str] = None
    
    is_new_node: bool = False 
    parent_id: Optional[str] = None

class TextGradOptimizer:
    def __init__(self, cfg, memories, memory_stats, log_path):
        self.cfg = cfg
        self.memories = memories
        self.memory_stats = memory_stats
        self.log_path = log_path
        self.batch_size = cfg.optimizer.llm_batch_size
        self.max_retries = cfg.parameters.get("max_retries", 2)
        
        
        
        self.action_re = re.compile(r'(?:\\box\{|Action:\s*)(REFINE|EXPAND|REPLACE|CREATE)', re.IGNORECASE)
        
        
        
        self.advice_re = re.compile(r'(?:\\advice\{|Advice:\s*)(.*?)(?:\}|(?=$))', re.DOTALL | re.IGNORECASE)
        self.verdict_re = re.compile(r"Verdict:\s*(PASS|FAIL)", re.IGNORECASE)

        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*20} New Optimization Session {'='*20}\n")

    def log(self, msg):

        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    def run(self, target_ids: List[str], to_delete_ids: Set[str]) -> Set[str]:

        
        valid_ids = [mid for mid in target_ids if mid in self.memories and mid not in to_delete_ids]
        self.log(f" {len(valid_ids)} ")
        
        optimized_ids = set()

        
        for i in range(0, len(valid_ids), self.batch_size):
            chunk_ids = valid_ids[i : i + self.batch_size]
            self.log(f"\n Processing Batch {i//self.batch_size + 1} ({len(chunk_ids)} items)")
            
            
            tasks = self._init_tasks(chunk_ids)
            
            
            
            self._batch_diagnose(tasks)
            
            
            
            new_tasks = self._batch_execute_and_expand(tasks)
            
            
            if new_tasks:
                self.log(f" [EXPAND Triggered] Added {len(new_tasks)} new split-nodes to current batch.")
                tasks.extend(new_tasks)

            
            
            self._batch_evaluate_loop(tasks)
            
            
            
            batch_success_ids = self._commit_changes(tasks)
            optimized_ids.update(batch_success_ids)
            
        return optimized_ids

    def _init_tasks(self, mids) -> List[OptimizationTask]:
        tasks = []
        for mid in mids:
            rec = self.memories[mid]
            tasks.append(OptimizationTask(
                mid=mid,
                original_content=rec.get("contents", ""),
                stats=self.memory_stats.get(mid, {})
            ))
        return tasks

    
    
    
    def _batch_diagnose(self, tasks: List[OptimizationTask]):
        prompts = []
        for task in tasks:
            neg_queries = task.stats.get('neg_queries', [])
            
            if not neg_queries:
                
                prompt = self.cfg.optimizer.prompts.low_grad_polish.format(
                    content=task.original_content
                )
            else:
                
                top_k_negs = "\n".join([f"- {q}" for q in neg_queries[:3]])
                prompt = self.cfg.optimizer.prompts.low_grad_expert.format(
                    content=task.original_content,
                    neg_queries=top_k_negs
                )
            
            task.diagnosis_prompt = prompt
            prompts.append(prompt)

        self.log(f" [Expert] Diagnosing {len(prompts)} memories...")
        outputs = call_expert_batch(prompts, self.cfg)

        for task, out in zip(tasks, outputs):
            if not out: continue
            
            m_act = self.action_re.search(out)
            task.expert_action = m_act.group(1) if m_act else "REFINE" 
            
            
            m_adv = self.advice_re.search(out)
            gradient = m_adv.group(1).strip() if m_adv else out.strip()
            task.expert_advice = gradient
            
            
            preview = gradient[:60] + "..." if len(gradient) > 60 else gradient
            self.log(f"  -> ID:{task.mid[:6]} | Action: {task.expert_action}")
            self.log(f"     Gradient: {preview}")

    
    
    
    def _batch_execute_and_expand(self, tasks: List[OptimizationTask]) -> List[OptimizationTask]:
        prompts = []
        active_tasks = [] 
        new_spawned_tasks = [] 

        for task in tasks:
            if task.expert_action == "WAITING": continue
            
            neg_text = "\n".join(task.stats.get('neg_queries', [])[:3])
            gradient = task.expert_advice

            
            if task.expert_action == "EXPAND":
                
                
                p_old = self.cfg.optimizer.prompts.appgrad_low_refine.format(
                    content=task.original_content, 
                    gradient=f"Keep the general definition, but distinguish from new concept. Advice: {gradient}"
                )
                task.student_prompt = p_old
                prompts.append(p_old)
                active_tasks.append(task)

                
                
                new_mid = str(uuid.uuid4())
                
                
                new_task = OptimizationTask(
                    mid=new_mid,
                    original_content="", 
                    stats={"neg_queries": task.stats.get('neg_queries', [])}, 
                    expert_action="CREATE", 
                    is_new_node=True,
                    parent_id=task.mid
                )
                
                
                p_new = self.cfg.optimizer.prompts.appgrad_low_replace.format(
                    neg_queries=neg_text, 
                    gradient=f"Create a NEW memory specific to these queries. Advice: {gradient}"
                )
                new_task.student_prompt = p_new
                
                
                prompts.append(p_new)
                active_tasks.append(new_task) 
                new_spawned_tasks.append(new_task)

            elif task.expert_action == "REPLACE":
                p = self.cfg.optimizer.prompts.appgrad_low_replace.format(neg_queries=neg_text, gradient=gradient)
                task.student_prompt = p
                prompts.append(p)
                active_tasks.append(task)

            else: 
                p = self.cfg.optimizer.prompts.appgrad_low_refine.format(content=task.original_content, gradient=gradient)
                task.student_prompt = p
                prompts.append(p)
                active_tasks.append(task)

        if not prompts: return []

        self.log(f"️ [Student] Drafting updates for {len(prompts)} tasks (incl. expansions)...")
        
        
        outputs = call_expert_batch(prompts, self.cfg) 
        
        for t, out in zip(active_tasks, outputs):
            
            clean_content = _extract_single_memory(out)
            t.generated_content = clean_content if clean_content else out
            
            
            if t.is_new_node:
                self.log(f"     [NEW NODE] Generated content for {t.mid[:6]} (Parent: {t.parent_id[:6]})")

        return new_spawned_tasks

    
    
    
    def _batch_evaluate_loop(self, tasks: List[OptimizationTask]):
        
        for retry_idx in range(self.max_retries + 1):
            
            pending_tasks = [t for t in tasks if t.judge_verdict != "PASS" and t.generated_content]
            if not pending_tasks:
                break
                
            self.log(f"️ [Judge] Round {retry_idx}: Evaluating {len(pending_tasks)} candidates...")
            
            
            judge_prompts = []
            for t in pending_tasks:
                neg_q = "\n".join(t.stats.get('neg_queries', [])[:3])
                
                p = self.cfg.optimizer.prompts.expert_judge.format(failed = neg_q, old = t.original_content, new = t.generated_content)
                judge_prompts.append(p)
            
            
            judge_outs = call_expert_batch(judge_prompts, self.cfg)
            
            
            retry_prompts = []
            retry_tasks = []
            
            for t, out in zip(pending_tasks, judge_outs):
                verdict_match = self.verdict_re.search(out)
                verdict = verdict_match.group(1).upper() if verdict_match else "FAIL"
                
                feedback = out.split("Feedback:")[-1].strip() if "Feedback:" in out else out[-100:]
                
                t.judge_verdict = verdict
                t.judge_feedback = feedback
                
                if verdict == "PASS":
                    t.final_accepted_content = t.generated_content
                    self.log(f"  [PASS] ID:{t.mid[:6]}")
                else:
                    self.log(f"  [FAIL] ID:{t.mid[:6]} | Feedback: {feedback[:50]}...")
                    if retry_idx < self.max_retries:
                        
                        new_prompt = self.cfg.optimizer.prompts.retry_prompt.format(ori = t.student_prompt, failed = neg_q, bad = t.generated_content,feedback = feedback)
                        retry_prompts.append(new_prompt)
                        retry_tasks.append(t)

            
            if retry_prompts:
                self.log(f" [Retry] Regenerating {len(retry_prompts)} items...")
                retry_outs = call_expert_batch(retry_prompts, self.cfg)
                for t, out in zip(retry_tasks, retry_outs):
                    t.generated_content = _extract_single_memory(out) or out
                    t.retry_count += 1
            else:
                break

    
    
    
    def _commit_changes(self, tasks: List[OptimizationTask]) -> Set[str]:
        success_ids = set()
        for t in tasks:
            if t.final_accepted_content:
                
                if t.is_new_node:
                    
                    self.memories[t.mid] = {
                        "id": t.mid,
                        "contents": t.final_accepted_content,
                        "cluster_id": -1, 
                        "opt_type": "textgrad_expand",
                        "parent_id": t.parent_id
                    }
                    
                    self.memory_stats[t.mid] = {
                        "alpha": 0.5, 
                        "beta": 0.5, 
                        "neg_queries": [], 
                        "pos_queries": []
                    }
                    self.log(f" [EXPAND] Created New Node: {t.mid[:8]}")
                else:
                    
                    self.memories[t.mid]["contents"] = t.final_accepted_content
                    self.memories[t.mid]["cluster_id"] = -1 
                    self.memories[t.mid]["opt_type"] = "textgrad_v2"
                    
                    if t.mid in self.memory_stats:
                        self.memory_stats[t.mid]['neg_queries'] = []
                    self.log(f" [UPDATE] Updated Node: {t.mid[:8]}")
                
                success_ids.add(t.mid)
        return success_ids




def textgrad_opt(cfg, memories, memory_stats, log_file_path, cluster_to_ids, bad_ids, to_delete_ids):
    optimizer = TextGradOptimizer(cfg, memories, memory_stats, log_file_path)
    target_ids_list = list(bad_ids)
    return optimizer.run(target_ids_list, to_delete_ids)