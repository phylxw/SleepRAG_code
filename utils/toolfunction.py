import torch

def clean_special_chars(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace('\u2028', ' ').replace('\u2029', ' ')


def has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False
    
def format_base_prompt(system_text, user_text,model_source):
    
    if model_source == "gemini":
        return f"{system_text}\n\n{user_text}" if system_text else user_text

    prompt = ""
    if system_text:
        prompt += f"<|im_start|>system\n{system_text}<|im_end|>\n"
    
    prompt += f"<|im_start|>user\n{user_text}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\nLet's think step by step.\n" 
    
    return prompt