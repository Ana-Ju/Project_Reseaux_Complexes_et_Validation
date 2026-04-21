NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def get_lcc(filepath):
    G = nx.read_gml(filepath)
    
    # extracts only the main component
    lcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc_nodes).copy()

# question 1: Assortativity Analysis with the Facebook100 Dataset of the FB100 networks, 
# investigate the assortativity patterns for five vertex attributes: 
# (i) student/faculty status, (ii) major, (iii) vertex degree, (iiii) dorm, (iiiii) gender

data_dir = '/home/ana-julia/Documents/TSP/ReseauxComplexes/fb100/data/'

# lists to store results
network_sizes = []

assort_student = []
assort_major = []
assort_dorm = []
assort_gender = []
assort_degree = []

# get all the .gml files
gml_files = [f for f in os.listdir(data_dir) if f.endswith('.gml')]

print("Starting assortativity calculations ...")


for filename in gml_files:
    filepath = os.path.join(data_dir, filename)
    
    # load graph + reduce to main component
    G = get_lcc(filepath)
    network_sizes.append(G.number_of_nodes())

    # some graphs don't have all attributes, so this avoids crashes
    # student/faculty
    try:
        assort_student.append(nx.attribute_assortativity_coefficient(G, 'student_fac'))
    except:
        assort_student.append(np.nan)  # np.nan (Not a Number)
        
    # major
    try:
        assort_major.append(nx.attribute_assortativity_coefficient(G, 'major_index'))
    except:
        assort_major.append(np.nan)
        
    # dorm
    try:
        assort_dorm.append(nx.attribute_assortativity_coefficient(G, 'dorm'))
    except:
        assort_dorm.append(np.nan)
        
    # gender
    try:
        assort_gender.append(nx.attribute_assortativity_coefficient(G, 'gender'))
    except:
        assort_gender.append(np.nan)

    # computing degree assortativity
    try:
        assort_degree.append(nx.degree_assortativity_coefficient(G))
    except:
        assort_degree.append(np.nan)


print("Finished. Generating plots...")

sizes_array = np.array(network_sizes)

# grouped like this to avoid repeating the same plotting code
plot_data = [
    ('Student/Faculty', assort_student, 'blue'),
    ('Major', assort_major, 'red'),
    ('Dorm', assort_dorm, 'green'),
    ('Gender', assort_gender, 'purple'),
    ('Degree', assort_degree, 'orange')
]

for title, assort_list, color in plot_data:
    
    assort_array = np.array(assort_list, dtype=float)
    
    #  remove NaNs (some networks don't have all attributes)
    valid_mask = ~np.isnan(assort_array)
    
    valid_sizes = sizes_array[valid_mask]
    valid_assorts = assort_array[valid_mask]
            
    # scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(valid_sizes, valid_assorts, alpha=0.6, color=color, edgecolors='none')
    plt.xscale('log')
    plt.axhline(0, color='black', linestyle='--') 
    plt.title("Assortativity vs Network Size -- " + title)
    plt.xlabel("Network Size (n)")
    plt.ylabel("Assortativity Coefficient")
    plt.show()
    
    #  histogram
    plt.figure(figsize=(8, 5))
    plt.hist(valid_assorts, bins=20, color=color, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--') 
    plt.title("Assortativity Distribution -- " + title)
    plt.xlabel("Assortativity Coefficient")
    plt.ylabel("Frequency")
    plt.show()