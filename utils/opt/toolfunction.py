import re
from typing import List, Optional, Dict, Any, Tuple
from utils.opt.memorywrap import parse_memory







_MEMORY_BLOCK_RE = re.compile(
    r"(?:\\{1,2})memory\$?\s*(.*?)\s*(?:\\{1,2})endmemory\$?\s*",
    re.DOTALL | re.IGNORECASE
)


_MARKER_STRIP_RE = re.compile(
    r"(?:\\{1,2})memory\$?|(?:\\{1,2})endmemory\$?",
    re.IGNORECASE
)





def clean_memory_text(s: str) -> str:
    if not s:
        return ""
    
    s = re.sub(r"\s+", " ", s)
    
    s = _MARKER_STRIP_RE.sub("", s)
    return s.strip()

def extract_memory_blocks(raw_output: str) -> List[str]:

    if not raw_output:
        return []

    
    blocks = []
    for m in _MEMORY_BLOCK_RE.finditer(raw_output):
        cleaned = clean_memory_text(m.group(1))
        if cleaned:
            blocks.append(cleaned)
    
    if blocks:
        return blocks

    
    
    external_parse = parse_memory(raw_output)
    if external_parse:
        cleaned = clean_memory_text(external_parse)
        if cleaned:
            return [cleaned]

    
    fallback = clean_memory_text(raw_output)
    return [fallback] if fallback else []

def _extract_single_memory(raw_output: str) -> str:
    blocks = extract_memory_blocks(raw_output)
    return blocks[0] if blocks else ""





def _clean_block(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)         
    s = _MARKER_STRIP_RE.sub("", s)    
    return s.strip()

def _find_memory_spans(raw_output: str) -> List[Tuple[int, int, str]]:
    """Return list of (start_idx, end_idx_exclusive, inner_text) for each memory block."""
    if not raw_output:
        return []
    spans: List[Tuple[int, int, str]] = []
    for m in _MEMORY_BLOCK_RE.finditer(raw_output): 
        inner = _clean_block(m.group(1))
        spans.append((m.start(), m.end(), inner))
    return spans

def _extract_memory_blocks(raw_output: str) -> List[str]:
    spans = _find_memory_spans(raw_output or "")
    blocks = [inner for (_, _, inner) in spans if inner]
    if blocks:
        return blocks

    
    return [_clean_block(raw_output or "")] if (raw_output or "").strip() else []





def _basic_guard(text: str, *, min_len: int = 20, max_len: int = 2000) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < min_len or len(t) > max_len:
        return False
    
    banned = [
        "As an AI",
        "As a language model",
        "I can't",
        "I cannot",
        "I am unable"
    ]
    if any(b in t for b in banned):
        return False
    return True

