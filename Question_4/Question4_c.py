NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

import random
import itertools

# Question 4 (c) (2 points) Evaluating a link predictor:
def evaluate_link_prediction(graph, predictor, f=0.1, k_values=[50,100,200,300,400]):
    # 1. copy graph and get edges
    G = graph.copy()
    edges = list(G.edges())
    
    # 2. remove a fraction f of edges
    num_remove = int(f * len(edges))
    removed_edges = random.sample(edges, num_remove)
    G.remove_edges_from(removed_edges)
    
    # save removed edges in a set 
    # Sorting them with min/max avoids the need to check both (A,B) and (B,A) later
    removed_set = set()
    for u, v in removed_edges:
        removed_set.add((min(u, v), max(u, v)))


    # 3. For each node pair in the graph |V |×|V |, for each node pair compute the link predictor metrics of interest p, these are the predicted ”friendship” Epredict.
