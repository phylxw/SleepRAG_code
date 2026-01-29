import os
import concurrent.futures
from typing import List
from omegaconf import DictConfig
from utils.toolfunction import clean_special_chars
import logging
from tqdm import tqdm


GLOBAL_EXPERT_CLIENT = None

def init_expert_llm(cfg: DictConfig):
    global GLOBAL_EXPERT_CLIENT
    expert_cfg = cfg.expert_model
    source = expert_cfg.source
    if source == "gemini":
        try:
            import google.generativeai as genai
            api_key = os.environ.get("EXPERT_API_KEY") or os.environ.get("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            GLOBAL_EXPERT_CLIENT = genai.GenerativeModel(expert_cfg.name)
            print(f" [Expert-Init] Gemini ({expert_cfg.name}) OK")
        except ImportError:
            print(" [Expert-Init] fail")

    elif source in ["openai", "sglang", "qwen"]:
        try:
            from openai import OpenAI
            
            base_url = os.environ.get("EXPERT_BASE_URL", "https://api.openai.com/v1")
            api_key = os.environ.get("EXPERT_API_KEY")
            
            
            if source == "sglang":
                base_url = expert_cfg.get("sglang_api_url", "http://127.0.0.1:30000/v1")
                api_key = "EMPTY"
            
            
            elif source == "qwen":
                base_url = your base_url
                api_key = your api_key
              if not api_key:
                    api_key = xxxxxxxx
            GLOBAL_EXPERT_CLIENT = OpenAI(base_url=base_url, api_key=api_key)
            print(f" [Expert-Init] {source.upper()} Client ({expert_cfg.name}) 就绪 | URL: {base_url}")
        except ImportError:
            print(" [Expert-Init] fail")
    else:
        print(f" [Expert-Init] unkonwn: {source}")


def call_expert(prompt: str, cfg: DictConfig) -> str:
    global GLOBAL_EXPERT_CLIENT
    if GLOBAL_EXPERT_CLIENT is None: return None

    source = cfg.expert_model.source
    model_name = cfg.expert_model.name
    
    try:
        if source == "gemini":
            resp = GLOBAL_EXPERT_CLIENT.generate_content(prompt)
            return clean_special_chars(resp.text.strip())
        
        
        
        elif source in ["openai", "sglang", "qwen"]:
            resp = GLOBAL_EXPERT_CLIENT.chat.completions.create(
                model=model_name, 
                messages=[
                    {"role": "system", "content": "You are a helpful and critical AI expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                stream=False  
            )
            return clean_special_chars(resp.choices[0].message.content.strip())

    except Exception as e:
        print(f"❌ [Expert Error]: {e}")
        return None



logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def call_expert_batch(prompts: List[str], cfg: DictConfig) -> List[str]:
    if not prompts: return []
    
    source = cfg.expert_model.source
    
    
    if source in ["sglang", "openai", "qwen"]:
        
        
        
        max_workers = 16 
        
        results = [None] * len(prompts)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(call_expert, p, cfg): i 
                for i, p in enumerate(prompts)
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(prompts), desc=f"🧠 {source.upper()} Batch"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"\n❌ [Expert Batch Error] Task {idx} failed: {e}")
                    results[idx] = ""
                    
        return results

    
    results = []
    for p in tqdm(prompts, desc="Gemini Expert"):
        results.append(call_expert(p, cfg))
    return results