#PubMed RAG Index Builder
#Run on Google Colab with T4 GPU
#Runtime -> Change runtime type -> T4 GPU -> Save
#!pip install sentence-transformers faiss-cpu
#Upload hetionet-v1.0-nodes.tsv when prompted
from google.colab import files
uploaded = files.upload()

import os
import requests
import pandas as pd
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

OUT_DIR = "/content/pubmed"
os.makedirs(OUT_DIR, exist_ok=True)

nodes_df = pd.read_csv('hetionet-v1.0-nodes.tsv', sep='\t')
diseases = nodes_df[nodes_df['kind'] == 'Disease']['name'].tolist()
print(f"Found {len(diseases)} diseases")

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def search_and_fetch(query, max_results=50):
    try:
        r = requests.get(BASE_URL + "esearch.fcgi", params={
            'db': 'pubmed', 'term': query,
            'retmax': max_results, 'retmode': 'json'
        })
        pmids = r.json().get('esearchresult', {}).get('idlist', [])
        if not pmids:
            return []
        r2 = requests.get(BASE_URL + "efetch.fcgi", params={
            'db': 'pubmed', 'id': ','.join(pmids),
            'rettype': 'abstract', 'retmode': 'text'
        })
        chunks = r2.text.strip().split('\n\n\n')
        results = []
        for i, chunk in enumerate(chunks):
            if len(chunk) > 100:
                pmid = pmids[i] if i < len(pmids) else ''
                results.append({'pmid': pmid, 'text': chunk.strip(), 'query': query})
        return results
    except Exception as e:
        print(f"  Error: {e}")
        return []

abstracts = []
seen_pmids = set()

for disease in diseases:
    query = f"{disease} drug treatment therapy mechanism"
    results = search_and_fetch(query, max_results=50)
    new = [r for r in results if r['pmid'] not in seen_pmids]
    seen_pmids.update(r['pmid'] for r in new)
    abstracts.extend(new)
    print(f"  {disease}: +{len(new)} | Total: {len(abstracts)}")
    time.sleep(0.4)

for query in [
    "drug repurposing knowledge graph",
    "drug repositioning machine learning",
    "drug repurposing gene expression",
    "computational drug repurposing",
    "drug target interaction prediction",
]:
    results = search_and_fetch(query, max_results=100)
    new = [r for r in results if r['pmid'] not in seen_pmids]
    seen_pmids.update(r['pmid'] for r in new)
    abstracts.extend(new)
    print(f"  +{len(new)} | Total: {len(abstracts)}")
    time.sleep(0.4)

print(f"\nTotal abstracts: {len(abstracts)}")
df = pd.DataFrame(abstracts)
df.to_csv(f"{OUT_DIR}/abstracts.csv", index=False)

print("\nLoading BioBERT...")
encoder = SentenceTransformer(
    'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
)
embeddings = encoder.encode(
    df['text'].tolist(),
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    device='cuda'
)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(embeddings)
index.add(embeddings)
print(f"FAISS index: {index.ntotal} vectors")
faiss.write_index(index, f"{OUT_DIR}/pubmed.faiss")
print("Saved pubmed.faiss")

import shutil
shutil.make_archive("/content/pubmed_rag", "zip", "/content/pubmed")
files.download("/content/pubmed_rag.zip")