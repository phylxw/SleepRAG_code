import os
import json
import time
from typing import Dict, List, Tuple, Set

def load_clustered_memories(path: str) -> Tuple[Dict[str, dict], List[str]]:
    memories: Dict[str, dict] = {}
    order: List[str] = []
    if not os.path.exists(path):
        return {}, []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            mid = str(obj["id"])
            memories[mid] = obj
            order.append(mid)
    return memories, order


def load_cluster_summary(path: str) -> Dict[int, List[str]]:
    cluster_to_ids: Dict[int, List[str]] = {}
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            cid = int(obj["cluster_id"])
            ids = [str(x) for x in obj.get("memory_ids", [])]
            cluster_to_ids[cid] = ids
    return cluster_to_ids


def load_memory_freq(path: str) -> Dict[str, int]:
    freq_map: Dict[str, int] = {}
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            mid = str(obj.get("memory_id", obj.get("id", "")))
            if not mid: continue
            freq = int(obj.get("freq", 0))
            freq_map[mid] = freq
    print(f"{len(freq_map)}")
    return freq_map