import os
import pandas as pd
import networkx as nx

EDGES_FILE = "data/hetionet/hetionet-v1.0-edges.sif.gz"
NODES_FILE = "data/hetionet/hetionet-v1.0-nodes.tsv"

for f in [EDGES_FILE, NODES_FILE]:
    if not os.path.exists(f):
        print(f"Missing: {f}")
        exit()

nodes_df = pd.read_csv(NODES_FILE, sep='\t')
print(f"Total nodes: {len(nodes_df)}")
print(nodes_df['kind'].value_counts().to_string())

edges_df = pd.read_csv(EDGES_FILE, sep='\t', compression='gzip',
                       names=['source', 'relation', 'target'])
print(f"\nTotal edges: {len(edges_df)}")
print(edges_df['relation'].value_counts().to_string())

G = nx.from_pandas_edgelist(
    edges_df,
    source='source',
    target='target',
    edge_attr='relation',
    create_using=nx.MultiDiGraph()
)
print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")