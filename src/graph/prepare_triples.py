import pandas as pd
import os

EDGES_FILE = "data/hetionet/hetionet-v1.0-edges.sif.gz"
NODES_FILE = "data/hetionet/hetionet-v1.0-nodes.tsv"
OUT_DIR = "data/hetionet"

edges_df = pd.read_csv(
    EDGES_FILE, sep='\t', compression='gzip',
    names=['source', 'relation', 'target']
)

edges_df[['source', 'relation', 'target']].to_csv(
    f"{OUT_DIR}/hetionet_triples.tsv", sep='\t', index=False, header=False
)

treats = edges_df[edges_df['relation'] == 'CtD'].copy()
treats.to_csv(f"{OUT_DIR}/drug_disease_treats.tsv", sep='\t', index=False)

nodes_df = pd.read_csv(NODES_FILE, sep='\t')
n_drugs = len(nodes_df[nodes_df['kind'] == 'Compound'])
n_diseases = len(nodes_df[nodes_df['kind'] == 'Disease'])

print(f"Drugs:    {n_drugs}")
print(f"Diseases: {n_diseases}")
print(f"Known drug-disease pairs: {len(treats)}")
print(f"Possible pairs: {n_drugs * n_diseases}")
print(f"Connected: {len(treats) / (n_drugs * n_diseases) * 100:.2f}%")