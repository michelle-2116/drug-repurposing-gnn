import pandas as pd
import networkx as nx

EDGES_FILE = "data/hetionet/hetionet-v1.0-edges.sif.gz"
NODES_FILE = "data/hetionet/hetionet-v1.0-nodes.tsv"

edges_df = pd.read_csv(EDGES_FILE, sep='\t', compression='gzip',
                       names=['source', 'relation', 'target'])
nodes_df = pd.read_csv(NODES_FILE, sep='\t')

print("Building graph...")
G = nx.MultiGraph() #bidirectional
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'], relation=row['relation'])
print("Graph ready.")

id_to_name = dict(zip(nodes_df['id'], nodes_df['name']))

RELATION_LABELS = {
    'CbG':  'binds gene',
    'CcSE': 'causes side effect',
    'CdG':  'downregulates gene',
    'CpD':  'palliates disease',
    'CrC':  'resembles compound',
    'CtD':  'treats disease',
    'CuG':  'upregulates gene',
    'DaG':  'disease associates gene',
    'DdG':  'disease downregulates gene',
    'DlA':  'disease localizes anatomy',
    'DpS':  'disease presents symptom',
    'DrD':  'disease resembles disease',
    'DuG':  'disease upregulates gene',
    'AdG':  'anatomy downregulates gene',
    'AeG':  'anatomy expresses gene',
    'AuG':  'anatomy upregulates gene',
    'GcG':  'gene covaries gene',
    'GiG':  'gene interacts gene',
    'GpBP': 'gene participates biological process',
    'GpCC': 'gene participates cellular component',
    'GpMF': 'gene participates molecular function',
    'GpPW': 'gene participates pathway',
    'Gr>G': 'gene regulates gene',
    'PCiC': 'pharmacologic class includes compound',
}

def get_path(drug_id: str, disease_id: str):
    if drug_id not in G or disease_id not in G:
        return None
    try:
        path = nx.shortest_path(G, source=drug_id, target=disease_id)
        steps = []
        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            src_name = id_to_name.get(src, src)
            tgt_name = id_to_name.get(tgt, tgt)
            edge_data = G.get_edge_data(src, tgt)
            if edge_data:
                rel_code = list(edge_data.values())[0]['relation']
                rel_label = RELATION_LABELS.get(rel_code, rel_code)
            else:
                rel_label = 'connects to'
            steps.append(src_name + ' --[' + rel_label + ']--> ' + tgt_name)
        return steps
    except nx.NetworkXNoPath:
        return None
    except nx.NodeNotFound:
        return None

def print_path(drug_id: str, disease_id: str):
    drug_name = id_to_name.get(drug_id, drug_id)
    disease_name = id_to_name.get(disease_id, disease_id)
    print('\n' + drug_name + ' -> ' + disease_name)
    print('-' * 60)
    path = get_path(drug_id, disease_id)
    if not path:
        print("No path found.")
        return
    for step in path:
        print('  ' + step)

