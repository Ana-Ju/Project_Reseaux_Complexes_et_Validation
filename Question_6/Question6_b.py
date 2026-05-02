NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

import networkx as nx
import os
from networkx.algorithms import community
from sklearn.metrics import adjusted_rand_score

#  using ARI to compare communities with attributes
def get_matching_score(G, clusters, attribute_name):
    # handle case where major_index is not present
    labels = nx.get_node_attributes(G, attribute_name)
    if not labels and attribute_name == 'major_index':
        labels = nx.get_node_attributes(G, 'major')

    # assign each node to a cluster id
    predicted_groups = {}
    for cluster_id, nodes in enumerate(clusters):
        for node in nodes:
            predicted_groups[node] = cluster_id

    true_labels = []
    detected_labels = []
    
    for node in G.nodes():
        # skip missing values (0 = unknown)
        val = labels.get(node, 0)
        if val != 0: 
            true_labels.append(val)
            detected_labels.append(predicted_groups[node])
            
    if len(true_labels) < 2:
        return 0.0
        
    return adjusted_rand_score(true_labels, detected_labels)


if __name__ == "__main__":
    data_folder = "fb100/data/"
    
    ## take small graphs so it runs faster
    all_files = [f for f in os.listdir(data_folder) if f.endswith(".gml")]
    selected_files = sorted(all_files, key=lambda x: os.path.getsize(data_folder + x))[:2]
    
    target_attrs = ["major_index", "dorm", "year", "gender"]
    
    for filename in selected_files:
        print(f"\n--- Processing: {filename} ---")
        
        G = nx.read_gml(data_folder + filename)
        G = nx.convert_node_labels_to_integers(G)
        
        # keep largest connected component
        main_component = sorted(nx.connected_components(G), key=len, reverse=True)[0]
        G_core = G.subgraph(main_component).copy()
        
        print(f"Nodes in core: {len(G_core)}")

        # try different algorithms
        # Greedy Modularity (standard approach)
        greedy_groups = list(community.greedy_modularity_communities(G_core))
        
        # Label Propagation (fast, based on neighbors)
        lpa_groups = list(community.asyn_lpa_communities(G_core))
        
        # Louvain (very common for social networks)
        louvain_groups = list(community.louvain_communities(G_core, seed=42))
        
        results = {
            "Greedy Modularity": greedy_groups,
            "Label Propagation": lpa_groups,
            "Louvain Method": louvain_groups
        }
        
        # compare with attributes
        for method_name, clusters in results.items():
            print(f"\nAlgorithm: {method_name} ({len(clusters)} groups)")
            
            for attr in target_attrs:
                score = get_matching_score(G_core, clusters, attr)
                print(f"  {attr:<12} -> ARI: {score:.4f}")
