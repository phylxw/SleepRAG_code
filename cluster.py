import os
import json
import re
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns  
from sentence_transformers import SentenceTransformer

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from transformers import AutoModelForCausalLM, AutoTokenizer


try:
    import umap  
except Exception:
    umap = None

try:
    import hdbscan  
except Exception:
    hdbscan = None


import hydra
from omegaconf import DictConfig


GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_SGLANG_CLIENT = None



def clean_special_chars(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace('\u2028', ' ').replace('\u2029', ' ')

def normalize_text(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"\d+(\.\d+)?", " <num> ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def import_torch_and_check_gpu():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def _cfg_get(cfg: DictConfig, key: str, default: Any) -> Any:
    try:
        return cfg.get(key, default)  
    except Exception:
        return default



def init_llm(cfg: DictConfig):
    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_SGLANG_CLIENT

    model_source = cfg.model.source

    if model_source == "gemini":
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            print(f"[Init] Gemini API ({cfg.model.gemini_name})")
        else:
            print("init fail")

    elif model_source == "huggingface":
        hf_name = cfg.model.hf_name
        print(f"loading : {hf_name} ...")
        try:
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                hf_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            ).eval()
            print("✅ [Init] ！")
        except Exception as e:
            print(f"❌ [Init] : {e}")

    elif model_source == "sglang":
        try:
            from openai import OpenAI
            api_url = cfg.model.get("sglang_api_url", "http://127.0.0.1:30000/v1")
            api_key = "EMPTY"
            GLOBAL_SGLANG_CLIENT = OpenAI(base_url=api_url, api_key=api_key)
            print(f"✅ [Init] SGLang Client {api_url}")
        except ImportError:
            print("❌ [Init]")

def call_llm(prompt: str, cfg: DictConfig) -> str:
    model_source = cfg.model.source

    if model_source == "gemini":
        import google.generativeai as genai
        if not os.environ.get("GEMINI_API_KEY"):
            return "Skipped (No Key)"
        model = genai.GenerativeModel(cfg.model.gemini_name)
        try:
            print("thinking", end="", flush=True)
            resp = model.generate_content(prompt)
            print("complete")
            return clean_special_chars(resp.text.strip())
        except Exception as e:
            print(f"\n❌ [Gemini Error]: {e}")
            time.sleep(1)
            return "Unknown Topic"

    elif model_source == "huggingface":
        if GLOBAL_MODEL is None:
            return "Skipped (Model Not Loaded)"
        try:
            print(" [Local] reasoning", end="", flush=True)
            messages = [{"role": "user", "content": prompt}]
            text = GLOBAL_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = GLOBAL_TOKENIZER([text], return_tensors="pt").to(GLOBAL_MODEL.device)
            with torch.no_grad():
                generated_ids = GLOBAL_MODEL.generate(model_inputs.input_ids, max_new_tokens=50, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = GLOBAL_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("complete!")
            return clean_special_chars(response.strip())
        except Exception as e:
            print(f"\n❌ [Local Error]: {e}")
            return "Unknown Topic"

    elif model_source == "sglang":
        if GLOBAL_SGLANG_CLIENT is None:
            return "Skipped (Client Not Initialized)"
        model_name = cfg.model.get("sglang_model_name", "Qwen/Qwen3-4B-Instruct-2507")
        try:
            print("  [SGLang] reasoning...", end="", flush=True)
            resp = GLOBAL_SGLANG_CLIENT.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            content = resp.choices[0].message.content
            print("complete!")
            return clean_special_chars(content.strip())
        except Exception as e:
            print(f"\n❌ [SGLang Error]: {e}")
            return "Unknown Topic"

    return "Unknown Config"



def load_questions(jsonl_path: str):
    print(f"{jsonl_path}...")
    if not os.path.exists(jsonl_path):
        print(f"❌: {jsonl_path}")
        return [], [], []

    ids, questions, raw_contents = [], [], []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = obj.get("contents", "")
            raw_contents.append(content)

            
            if "Question:" in content:
                q_part = content.split("Answer:")[0].replace("Question:", "").strip()
                if not q_part and content:
                    q_part = content
            else:
                q_part = content

            mid = obj.get("memory_id", obj.get("id"))
            ids.append(str(mid))
            questions.append(clean_special_chars(q_part))

    return ids, questions, raw_contents



def build_embeddings(questions: List[str], model_name: str, device_cfg: str = "cuda") -> np.ndarray:
    device = device_cfg if (device_cfg == "cuda" and torch.cuda.is_available()) else "cpu"
    model = SentenceTransformer(model_name, device=device)
    q_norm = [normalize_text(q) for q in questions]

    emb = model.encode(
        q_norm,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.asarray(emb)

def preprocess_embeddings_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    if embeddings.shape[0] < n_components:
        return embeddings

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)

    explained_variance = float(np.sum(pca.explained_variance_ratio_))
    return reduced


def _auto_kmeans(embeddings: np.ndarray, cfg: DictConfig) -> np.ndarray:
    n = embeddings.shape[0]
    k_min = int(_cfg_get(cfg.cluster, "kmeans_k_min", 2))
    k_max = int(_cfg_get(cfg.cluster, "kmeans_k_max", min(50, max(3, int(np.sqrt(n)) + 5))))
    k_max = min(k_max, n - 1)

    if k_max < k_min:
        model = KMeans(n_clusters=2, random_state=42, n_init='auto')
        return model.fit_predict(embeddings)

    sample_size = int(_cfg_get(cfg.cluster, "silhouette_sample", 2000))
    sample_size = min(sample_size, n)

    best_k, best_score, best_labels = None, -1.0, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(embeddings)

        
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(
                embeddings, labels,
                metric="euclidean",
                sample_size=sample_size,
                random_state=42
            )
        except Exception:
            continue

        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    if best_labels is None:
        n_clusters = int(cfg.cluster.kmeans_n_clusters)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        return model.fit_predict(embeddings)
    return best_labels


def cluster_questions_auto(embeddings: np.ndarray, cfg: DictConfig) -> np.ndarray:
    method = cfg.cluster.method

    if method == "kmeans":
        n_clusters = int(cfg.cluster.kmeans_n_clusters)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = model.fit_predict(embeddings)

    elif method == "kmeans_auto":
        labels = _auto_kmeans(embeddings, cfg)

    elif method == "agglomerative":
        threshold = float(cfg.cluster.distance_threshold)
        linkage = _cfg_get(cfg.cluster, "agglom_linkage", "average")  
        metric = _cfg_get(cfg.cluster, "agglom_metric", "cosine")     

        
        if linkage == "ward":
            metric = "euclidean"
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric=metric,
            linkage=linkage
        )
        labels = model.fit_predict(embeddings)

    elif method == "hdbscan":
        if hdbscan is None:
            cfg.cluster.method = "agglomerative"
            return cluster_questions_auto(embeddings, cfg)

        
        min_cluster_size = int(_cfg_get(cfg.cluster, "hdbscan_min_cluster_size", 8))
        min_samples = _cfg_get(cfg.cluster, "hdbscan_min_samples", None)
        metric = _cfg_get(cfg.cluster, "hdbscan_metric", "euclidean")  
        cluster_selection_method = _cfg_get(cfg.cluster, "hdbscan_selection_method", "eom")
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method
        )
        labels = model.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int(np.sum(labels == -1))

    else:
        raise ValueError(f"unknown: {method}")

    return labels



def plot_cluster_stats(labels: np.ndarray, save_path: str, method_name: str):
    unique_labels, counts = np.unique(labels, return_counts=True)

    
    valid_mask = (counts > 1) & (unique_labels != -1)
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]

    if len(valid_counts) == 0:
        return

    sorted_indices = np.argsort(valid_counts)[::-1]
    sorted_plot_labels = valid_labels[sorted_indices]
    sorted_plot_counts = valid_counts[sorted_indices]

    plt.figure(figsize=(12, 6))
    x_ticks = [str(lbl) for lbl in sorted_plot_labels]

    if len(x_ticks) > 50:
        x_ticks = x_ticks[:50]
        sorted_plot_counts = sorted_plot_counts[:50]

    plt.bar(x_ticks, sorted_plot_counts, color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Cluster ID')
    plt.ylabel('Count')
    plt.title(f'Top Cluster Size Distribution ({method_name})')
    plt.xticks(rotation=90 if len(x_ticks) > 20 else 0, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)


def plot_dimensionality_reduction(embeddings: np.ndarray, labels: np.ndarray, cfg: DictConfig, save_path: str):
    method = cfg.cluster.vis_method

    n = embeddings.shape[0]
    if n < 5:
        return

    X = embeddings
    vis_pca_dims = int(_cfg_get(cfg.cluster, "vis_pca_dims", 50))
    if X.shape[1] > vis_pca_dims and vis_pca_dims > 2:
        X = PCA(n_components=vis_pca_dims, random_state=42).fit_transform(X)

    reducer = None
    if method == "tsne":
        perp = min(int(cfg.cluster.tsne_perplexity), n - 1)
        metric = _cfg_get(cfg.cluster, "tsne_metric", "cosine")
        reducer = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
            init='pca',
            learning_rate='auto',
            metric=metric
        )
        reduced_emb = reducer.fit_transform(X)

    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        reduced_emb = reducer.fit_transform(X)

    elif method == "umap":
        if umap is None:
            cfg.cluster.vis_method = "tsne"
            return plot_dimensionality_reduction(embeddings, labels, cfg, save_path)

        n_neighbors = int(_cfg_get(cfg.cluster, "umap_n_neighbors", 15))
        min_dist = float(_cfg_get(cfg.cluster, "umap_min_dist", 0.05))
        metric = _cfg_get(cfg.cluster, "umap_metric", "cosine")
        supervised = bool(_cfg_get(cfg.cluster, "umap_supervised", True))
        target_weight = float(_cfg_get(cfg.cluster, "umap_target_weight", 0.5))

        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(n_neighbors, n - 1),
            min_dist=min_dist,
            metric=metric,
            target_metric="categorical",
            target_weight=target_weight if supervised else 0.0
        )

        
        if supervised and labels is not None:
            reduced_emb = reducer.fit_transform(X, y=labels)
        else:
            reduced_emb = reducer.fit_transform(X)

    else:
        return

    plt.figure(figsize=(12, 10))

    
    if np.any(labels == -1):
        noise_mask = (labels == -1)
        non_noise = ~noise_mask
        plt.scatter(reduced_emb[noise_mask, 0], reduced_emb[noise_mask, 1], c="lightgray", s=8, alpha=0.5, linewidths=0)
        sc = plt.scatter(reduced_emb[non_noise, 0], reduced_emb[non_noise, 1], c=labels[non_noise], cmap='tab20', s=10, alpha=0.75, linewidths=0)
        plt.colorbar(sc, label='Cluster ID (noise=-1 excluded)')
    else:
        sc = plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels, cmap='tab20', s=10, alpha=0.75, linewidths=0)
        plt.colorbar(sc, label='Cluster ID')

    plt.title(f'{method.upper()} Visualization')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    
    return reduced_emb


def plot_top_clusters_kde(reduced_emb: np.ndarray, labels: np.ndarray, save_path: str, top_k: int = 3):
    
    if reduced_emb is None or len(reduced_emb) == 0:
        return

    unique_labels, counts = np.unique(labels, return_counts=True)
    
    
    valid_mask = (unique_labels != -1) & (counts >= 5)
    if not np.any(valid_mask):
        return
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]
    
    
    sorted_indices = np.argsort(valid_counts)[::-1]
    top_labels = valid_labels[sorted_indices][:top_k]
    
    print(f"   target clust ID: {top_labels}")
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c='lightgray', s=5, alpha=0.3, label='Other')

    colors = sns.color_palette("tab10", len(top_labels)) 
    
    for i, cid in enumerate(top_labels):
        
        mask = (labels == cid)
        subset = reduced_emb[mask]
        
        
        label_text = f'Cluster {cid} (n={len(subset)})'
        
        try:

            sns.kdeplot(
                x=subset[:, 0], 
                y=subset[:, 1], 
                fill=True, 
                alpha=0.2,    
                color=colors[i], 
                warn_singular=False
            )

            plt.scatter(
                subset[:, 0], 
                subset[:, 1], 
                s=10, 
                color=colors[i], 
                alpha=0.8, 
                label=label_text  
            )
            
        except Exception as e:
            print(f" Cluster {cid} fail: {e}")

    plt.title(f'KDE Density Plot for Top {len(top_labels)} Clusters')
    
    
    plt.legend(loc='best')
    plt.tight_layout()
    
    
    kde_save_path = save_path.replace(".png", "_kde.png")
    plt.savefig(kde_save_path, dpi=300)

def tfidf_keywords_per_cluster(questions, cluster_labels, max_features=5000, top_k=10):
    q_norm = [normalize_text(q) for q in questions]
    try:
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=max_features, stop_words="english")
        X = vectorizer.fit_transform(q_norm)
        vocab = np.array(vectorizer.get_feature_names_out())

        cluster_keywords = {}
        for cid in np.unique(cluster_labels):
            if cid == -1:
                continue
            idx = np.where(cluster_labels == cid)[0]
            if len(idx) < 2:
                continue
            tfidf_mean = np.asarray(X[idx].mean(axis=0)).ravel()
            top_idx = tfidf_mean.argsort()[::-1][:top_k]
            cluster_keywords[cid] = vocab[top_idx].tolist()
        return cluster_keywords
    except ValueError:
        return {}

def llm_label_cluster(cid, questions, cluster_labels, cluster_keywords, cfg: DictConfig, max_examples=5):
    idx = np.where(cluster_labels == cid)[0]
    examples_idx = np.random.choice(idx, min(len(idx), max_examples), replace=False)
    examples = [questions[i] for i in examples_idx]
    kw = ", ".join(cluster_keywords.get(cid, []))

    prompt = f"""You are a Math Education Expert.
I have grouped similar math problems together.
Keywords: [{kw}]
Examples:
{chr(10).join(f"- {q}" for q in examples)}

Task: Provide a **very short category name** (3-6 words) for this problem type.
Output ONLY the category name. Do not explain.
"""
    callback = call_llm(prompt, cfg)
    return callback.replace('"', "").strip()



@hydra.main(version_base=None, config_path="conf", config_name="config")
def cluster(cfg: DictConfig):

    
    init_llm(cfg)

    
    input_file = cfg.paths.freq_file
    output_file = cfg.paths.cluster_output
    summary_file = cfg.paths.cluster_summary
    plot_file = cfg.paths.cluster_plot
    vis_plot_file = cfg.paths.cluster_vis

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)

    ids, questions, raw_contents = load_questions(input_file)
    if not ids:
        return

    
    embeddings = build_embeddings(questions, cfg.model.embedding_name, cfg.model.device)

    
    if cfg.cluster.enable_pca_preprocess:
        embeddings = preprocess_embeddings_pca(embeddings, n_components=int(cfg.cluster.pca_preprocess_dims))

    
    labels = cluster_questions_auto(embeddings, cfg)

    
    plot_cluster_stats(labels, save_path=plot_file, method_name=cfg.cluster.method)
    
    
    reduced_emb = plot_dimensionality_reduction(embeddings, labels, cfg, save_path=vis_plot_file)

    
    
    if reduced_emb is not None:
        plot_top_clusters_kde(reduced_emb, labels, save_path=vis_plot_file, top_k=3)

    
    keywords_map = tfidf_keywords_per_cluster(questions, labels)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
    cluster_labels_text = {}
    for cid, count in sorted_clusters[:10]:
        
        if cid == -1:
            continue
        label_text = llm_label_cluster(cid, questions, labels, keywords_map, cfg)
        cluster_labels_text[int(cid)] = label_text
        print(f"   🏷️ Cluster {cid} ({count} queries): {label_text}")
        if cfg.model.source == "gemini":
            time.sleep(1)

    with open(output_file, "w", encoding="utf-8") as f:
        for qid, q, raw, cid in zip(ids, questions, raw_contents, labels):
            obj = {
                "id": qid,
                "contents": raw,
                "cluster_id": int(cid),
                "cluster_label": cluster_labels_text.get(int(cid), f"Cluster {cid}"),
                "cluster_keywords": keywords_map.get(int(cid), [])
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    cluster_aggregation = {}
    for qid, cid in zip(ids, labels):
        cid_int = int(cid)
        if cid_int not in cluster_aggregation:
            cluster_aggregation[cid_int] = {
                "cluster_id": cid_int,
                "cluster_label": cluster_labels_text.get(cid_int, f"Cluster {cid_int}"),
                "count": 0,
                "memory_ids": []
            }
        cluster_aggregation[cid_int]["memory_ids"].append(qid)
        cluster_aggregation[cid_int]["count"] += 1

    with open(summary_file, "w", encoding="utf-8") as f:
        for cid in sorted(cluster_aggregation.keys(), key=lambda k: cluster_aggregation[k]['count'], reverse=True):
            f.write(json.dumps(cluster_aggregation[cid], ensure_ascii=False) + "\n")

    print("OK")

if __name__ == "__main__":
    cluster()
