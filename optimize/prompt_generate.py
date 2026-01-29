from typing import Dict, List
import hydra
from omegaconf import DictConfig

def generate_gradient_prompt(content: str, neg_queries: List[str], cfg: DictConfig) -> str:
    neg_text = "\n".join([f"- {q}" for q in neg_queries[:5]])
    raw_prompt = cfg.optimizer.prompts.gradient_generate
    prompt = raw_prompt.format(content=content, neg_text=neg_text)
    return prompt

def apply_gradient_prompt(content: str, gradient: str, good_examples: str, cfg: DictConfig) -> str:
    momentum_part = ""
    if good_examples:
        momentum_part = f"\n[Reference (Momentum)]\nHigh-quality neighbors:\n{good_examples}\n"

    template = cfg.optimizer.prompts.apply_gradient
    return template.format(content=content, gradient=gradient, momentum_part=momentum_part)

def summarize_experience_prompt(target_text: str, good_neighbors: List[str], cfg: DictConfig) -> str:
    good_examples_text = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(good_neighbors))
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=target_text, good_examples=good_examples_text)
    return prompt

def expand_low_freq_memory_prompt(text: str, good_examples: str, cfg: DictConfig) -> str:
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=text, good_examples=good_examples)
    return prompt