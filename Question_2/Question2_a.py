NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

import networkx as nx

def get_lcc(filepath):
    G = nx.read_gml(filepath)
    
    # extracts only the main component
    lcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc_nodes).copy()

caltech = get_lcc('/home/ana-julia/Documents/TSP/ReseauxComplexes/fb100/data/Caltech36.gml')
mit = get_lcc('/home/ana-julia/Documents/TSP/ReseauxComplexes/fb100/data/MIT8.gml')
hopkins = get_lcc('/home/ana-julia/Documents/TSP/ReseauxComplexes/fb100/data/Johns Hopkins55.gml')

print("Nodes LCC Caltech:", caltech.number_of_nodes())
print("Nodes LCC MIT:", mit.number_of_nodes())
print("Nodes LCC Hopkins:", hopkins.number_of_nodes())

