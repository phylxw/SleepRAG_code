from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig
from typing import List
import os
import torch
from utils.toolfunction import clean_special_chars
import logging
from tqdm import tqdm 
import concurrent.futures

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def init_llm(cfg: DictConfig):

    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_SGLANG_CLIENT
    
    model_source = cfg.model.optimize

    if model_source == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            print(f"[Init] Gemini API ({cfg.model.gemini_name}) OK")
        else:
            print("[Init] fail")
            
    elif model_source == "huggingface":
        hf_name = cfg.model.hf_name
        try:
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                hf_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            ).eval()
            
            
            GLOBAL_TOKENIZER.padding_side = 'left'
            if GLOBAL_TOKENIZER.pad_token is None:
                GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token
                GLOBAL_TOKENIZER.pad_token_id = GLOBAL_TOKENIZER.eos_token_id
            
            print(f"[Init] complete")
        except Exception as e:
            print(f"[Init] fail: {e}")


    elif model_source == "sglang":
        try:
            from openai import OpenAI
            
            api_url = cfg.model.get("sglang_api_url", "http://127.0.0.1:30000/v1")
            api_key = "EMPTY" 
            
            GLOBAL_SGLANG_CLIENT = OpenAI(base_url=api_url, api_key=api_key)
            print(f"[Init] SGLang Client  {api_url}")
        except ImportError:
            print("[Init] fail")

def call_llm(prompt: str, cfg: DictConfig, max_new_tokens: int = None, verbose: bool = True) -> str:
    model_source = cfg.model.optimize
    if max_new_tokens is None:
        max_new_tokens = cfg.model.max_new_tokens

    
    if model_source == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            return "Skipped (No GEMINI_API_KEY)"
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(cfg.model.gemini_name)
            if verbose:
                print("[Gemini]...", end="", flush=True)
            resp = model.generate_content(prompt)
            if verbose:
                print("OK")
            return clean_special_chars(resp.text.strip())
        except Exception as e:
            if verbose: print(f"\n❌ [Gemini Error]: {e}")
            return ""

    
    elif model_source == "huggingface":
        if GLOBAL_MODEL is None:
            if verbose: print("[Local] fail")
            return ""

        try:
            if verbose:
                print(" 🚀 [Local] ...", end="", flush=True)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = GLOBAL_TOKENIZER.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = GLOBAL_TOKENIZER(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=cfg.model.max_input_len,
            ).to(GLOBAL_MODEL.device)

            with torch.no_grad():
                generated_ids = GLOBAL_MODEL.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = GLOBAL_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if verbose:
                print("OK")
            return clean_special_chars(response.strip())
        except Exception as e:
            if verbose: print(f"\n❌ [Local Error]: {e}")
            return ""

    
    elif model_source == "sglang":
        if GLOBAL_SGLANG_CLIENT is None:
            return "Skipped (Client Not Initialized)"
        
        model_name = cfg.model.get("sglang_model_name", "Qwen/Qwen3-4B-Instruct-2507")
        try:
            if verbose:
                print(" 🚀 [SGLang] ...", end="", flush=True)
            
            resp = GLOBAL_SGLANG_CLIENT.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_new_tokens
            )
            content = resp.choices[0].message.content
            
            if verbose:
                print("OK")
            return clean_special_chars(content.strip())
        except Exception as e:
            if verbose: print(f"\n[SGLang Error]: {e}")
            return ""

    return ""


def call_llm_batch(prompts: List[str], cfg: DictConfig, max_new_tokens: int = None) -> List[str]:
    if not prompts:
        return []
    
    model_source = cfg.model.optimize
    if max_new_tokens is None:
        max_new_tokens = cfg.model.max_new_tokens

    
    if model_source == "sglang":
        max_workers = cfg.model.get("batch_size", 32)
        
        results = [None] * len(prompts)
        
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(call_llm, p, cfg, max_new_tokens, verbose=False): i 
                for i, p in enumerate(prompts)
            }
            
            
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(prompts), desc="🚀 SGLang Batch"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    
                    print(f"\n[Batch Error] Task {idx} failed: {e}")
                    results[idx] = ""
        
        return results

    
    if model_source == "gemini":
        results = []
        
        for p in tqdm(prompts, desc="Gemini Batch"):
            results.append(call_llm(p, cfg, max_new_tokens=max_new_tokens))
        return results

    
    if model_source == "huggingface":
        if GLOBAL_MODEL is None:
            return [""] * len(prompts)

        try:
            print(f" 🚀 [Local-Batch]{len(prompts)} ...", end="", flush=True)
            
            messages_list = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": p}
                ]
                for p in prompts
            ]
            text_list = [
                GLOBAL_TOKENIZER.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for msgs in messages_list
            ]

            model_inputs = GLOBAL_TOKENIZER(
                text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.model.max_input_len,
            ).to(GLOBAL_MODEL.device)

            with torch.no_grad():
                generated_ids = GLOBAL_MODEL.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

            results = []
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
                new_token_ids = output_ids[len(input_ids):]
                text = GLOBAL_TOKENIZER.decode(new_token_ids, skip_special_tokens=True)
                results.append(clean_special_chars(text.strip()))
            print("OK")
            return results

        except Exception as e:
            print(f"\n [Local-Batch Error]: {e}")
            return [""] * len(prompts)

    return [""] * len(prompts)