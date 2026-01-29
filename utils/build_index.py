from datasets import load_dataset
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
import os
import json
import tqdm
import bm25s

def build_index(corpus_file: str, index_dir: str):
    print(f"🔨 [Index] {corpus_file} ...")
    corpus_texts = []
    
    
    with open(corpus_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                
                
                content = item.get('contents') or item.get('question') or item.get('prompt') or item.get('text')
                
                if content:
                    corpus_texts.append(content)
                else:
                    
                    print(f"[Line {i}] warn。Keys: {list(item.keys())}")
            except json.JSONDecodeError:
                continue

    if not corpus_texts:
        raise ValueError(f"fail：{corpus_file} 。")
    
    
    corpus_tokens = bm25s.tokenize(corpus_texts)
    retriever_builder = bm25s.BM25()
    retriever_builder.index(corpus_tokens)
    retriever_builder.save(index_dir)
    
    
    with open(os.path.join(index_dir, "stopwords.tokenizer.json"), "w") as f:
        json.dump([], f)
    with open(os.path.join(index_dir, "vocab.tokenizer.json"), "w") as f:
        vocab = corpus_tokens.vocab
        
        json.dump({"word_to_id": vocab, "stem_to_sid": vocab, "word_to_stem": {k: k for k in vocab}}, f)
    print("OK")