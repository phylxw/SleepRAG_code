from typing import Dict
import numpy as np
from utils.toolfunction import has_cuda
from sentence_transformers import SentenceTransformer

def build_embeddings_for_memories(memories: Dict[str, dict], model_name: str) -> Dict[str, np.ndarray]:
    device = "cuda" if has_cuda() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    ids = list(memories.keys())
    texts = []
    for mid in ids:
        rec = memories[mid]
        text = rec.get("contents", "")
        texts.append(text)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    id_to_emb = {mid: embeddings[i] for i, mid in enumerate(ids)}
    print(f"{len(id_to_emb)}")
    return id_to_emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
