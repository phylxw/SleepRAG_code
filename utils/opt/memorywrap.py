import re


_MEM_BLOCK_RE = re.compile(
    r"\\memory\$?\s*(.*?)\s*\\endmemory\$?\s*",
    re.DOTALL | re.IGNORECASE
)

def clean_text(s: str) -> str:
    if not s:
        return ""
    
    s = s.replace("\\n", " ")
    
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def parse_memory(raw: str) -> str:
    if not raw:
        return ""
    m = _MEM_BLOCK_RE.search(raw)
    if m:
        return clean_text(m.group(1))
    
    return clean_text(raw)


    