NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

import networkx as nx
import matplotlib.pyplot as plt

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

# question 1: For these three networks plot the degree distribution for each of the three networks that you downloaded.
degrees_caltech = [d for n, d in caltech.degree()]
degrees_mit = [d for n, d in mit.degree()]
degrees_hopkins = [d for n, d in hopkins.degree()]

# Plots

# Caltech
plt.figure() 
plt.hist(degrees_caltech, bins=30, color='blue', edgecolor='black')
plt.title("Degree Distribution - Caltech")
plt.xlabel("Degree (Number of friends)")
plt.ylabel("Frequency")
plt.show() 

# MIT
plt.figure()
plt.hist(degrees_mit, bins=50, color='red', edgecolor='black')
plt.title("Degree Distribution - MIT")
plt.xlabel("Degree (Number of friends)")
plt.ylabel("Frequency")
plt.show()

# Johns Hopkins
plt.figure()
plt.hist(degrees_hopkins, bins=50, color='green', edgecolor='black')
plt.title("Degree Distribution - Johns Hopkins")
plt.xlabel("Degree (Number of friends)")
plt.ylabel("Frequency")
plt.show()

# question 2: Compute the global clustering coefficient and mean local clustering coefficient for each of the 3 networks.
# In addition, compute the edge density of each network.

# Caltech 
global_cc_caltech = nx.transitivity(caltech)
mean_local_cc_caltech = nx.average_clustering(caltech)
density_caltech = nx.density(caltech) # edge density

print("\n\nCaltech ")
print("Global Clustering: ", global_cc_caltech)
print("Mean Local Clustering: ", mean_local_cc_caltech)
print("Density:", density_caltech)

# Johns Hopkins 
global_cc_hopkins = nx.transitivity(hopkins)
mean_local_cc_hopkins = nx.average_clustering(hopkins)
density_hopkins = nx.density(hopkins) # edge density

print("\nJohns Hopkins ")
print("Global Clustering: ", global_cc_hopkins)
print("Mean Local Clustering: ", mean_local_cc_hopkins)
print("Density:", density_hopkins)

#  MIT 
global_cc_mit = nx.transitivity(mit)
mean_local_cc_mit = nx.average_clustering(mit)
density_mit = nx.density(mit) # edge density

print("\nMIT ")
print("Global Clustering: ", global_cc_mit)
print("Mean Local Clustering:", mean_local_cc_mit)
print("Density:", density_mit)
