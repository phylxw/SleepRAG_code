from typing import List
from openai import OpenAI
import os
import concurrent.futures
import logging
from tqdm import tqdm  


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class SGLangGenerator:

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_new_tokens: int = 1024,
        batch_size: int = 32, 
        temperature: float = 0.0,
    ):
        self.client = OpenAI(
            api_key=os.getenv("SGLANG_API_KEY", "EMPTY"),
            base_url=base_url.rstrip("/"),
        )
        self.model = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_input_len = 4096

    def generate(self, prompts: List[str]) -> List[str]:

        if not prompts:
            return []

        
        def _send_request(prompt):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                
                print(f"SGLang Request Error: {e}")
                return ""

        results = [None] * len(prompts)
        
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_idx = {
                executor.submit(_send_request, p): i 
                for i, p in enumerate(prompts)
            }
            
            
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(prompts), desc="SGLang Inference"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"Task {idx} generated an exception: {exc}")
                    results[idx] = ""

        return results