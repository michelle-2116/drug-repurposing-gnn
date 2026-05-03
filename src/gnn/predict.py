import torch
import pandas as pd

MODEL_PATH = "models/trained_model.pkl"
EDGES_FILE = "data/hetionet/hetionet-v1.0-edges.sif.gz"
NODES_FILE = "data/hetionet/hetionet-v1.0-nodes.tsv"
TRIPLES_DIR = "models/training_triples"

print("Loading model...")
model = torch.load(MODEL_PATH, map_location='cpu')
model.eval()
print("Model loaded.")

nodes_df = pd.read_csv(NODES_FILE, sep='\t')
edges_df = pd.read_csv(EDGES_FILE, sep='\t', compression='gzip',
                       names=['source', 'relation', 'target'])

print("Loading entity mappings...")
entity_df = pd.read_csv(f"{TRIPLES_DIR}/entity_to_id.tsv.gz", sep='\t',
                        compression='gzip', header=0)
relation_df = pd.read_csv(f"{TRIPLES_DIR}/relation_to_id.tsv.gz", sep='\t',
                          compression='gzip', header=0)

entity_to_id = dict(zip(entity_df['label'], entity_df['id']))
id_to_entity = dict(zip(entity_df['id'], entity_df['label']))
relation_to_id = dict(zip(relation_df['label'], relation_df['id']))

diseases_df = nodes_df[nodes_df['kind'] == 'Disease'][['id', 'name']]
compounds_df = nodes_df[nodes_df['kind'] == 'Compound'][['id', 'name']]

print(f"Entities: {len(entity_to_id)} | Relations: {len(relation_to_id)}")

def predict_drugs(disease_name: str, top_k: int = 10):
    matches = diseases_df[diseases_df['name'].str.contains(disease_name, case=False, na=False)]
    if matches.empty:
        print(f"\nNo disease found matching '{disease_name}'")
        print("Try one of these:")
        print(diseases_df['name'].head(20).to_string())
        return

    disease_doid = matches.iloc[0]['id']
    disease_label = matches.iloc[0]['name']
    print(f"\nQuerying: {disease_label} ({disease_doid})")

    if disease_doid not in entity_to_id:
        print(f"Disease not in model entity index.")
        return

    disease_idx = entity_to_id[disease_doid]
    relation_id = relation_to_id['CtD']

    known_pairs = edges_df[edges_df['relation'] == 'CtD']
    known_drugs = set(known_pairs[known_pairs['target'] == disease_doid]['source'].tolist())

    compound_ids = [entity_to_id[c] for c in compounds_df['id'] if c in entity_to_id]

    scores = []
    with torch.no_grad():
        for cid in compound_ids:
            hrt = torch.tensor([[cid, relation_id, disease_idx]])
            score = model.score_hrt(hrt).item()
            scores.append((cid, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<5} {'Drug ID':<35} {'Score':<10} {'Status'}")
    print("-" * 65)
    for rank, (cid, score) in enumerate(scores[:top_k], 1):
        drug_doid = id_to_entity[cid]
        drug_name = compounds_df[compounds_df['id'] == drug_doid]['name'].values
        drug_label = drug_name[0] if len(drug_name) > 0 else drug_doid
        status = "KNOWN" if drug_doid in known_drugs else "NOVEL"
        print(f"{rank:<5} {drug_label:<35} {score:<10.4f} {status}")

