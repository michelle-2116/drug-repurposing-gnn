import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

FAISS_PATH = "data/pubmed/pubmed.faiss"
ABSTRACTS_PATH = "data/pubmed/abstracts.csv"

print("Loading RAG components...")
index = faiss.read_index(FAISS_PATH)
df = pd.read_csv(ABSTRACTS_PATH)
encoder = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
print(f"Index: {index.ntotal} vectors | Abstracts: {len(df)}")

def retrieve(query: str, top_k: int = 5):
    embedding = encoder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    scores, indices = index.search(embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(df):
            row = df.iloc[idx]
            results.append({
                'pmid': row['pmid'],
                'text': row['text'],
                'score': float(score)
            })
    return results

def print_results(query: str):
    print(f"\nQuery: {query}")
    print("-" * 60)
    results = retrieve(query)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] PMID: {r['pmid']} | Score: {r['score']:.4f}")
        print(r['text'][:300] + "...")

