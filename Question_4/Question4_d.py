NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

import networkx as nx
import os
import random
import itertools
import math
from abc import ABC, abstractmethod

# Question 4 (b) - Link prediction metrics
class LinkPrediction(ABC):
    def __init__(self, graph):
        self.graph = graph
        self.N = len(graph)

    def neighbors(self, v):
        neighbors_list = self.graph.neighbors(v)
        return list(neighbors_list)

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Fit must be implemented")

class CommonNeighbors(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self, node_pairs):
        res = []
        for pair in node_pairs:
            u, v = pair[0], pair[1]
            n_u = set(self.neighbors(u))
            n_v = set(self.neighbors(v))
            
            # intersection size
            score = len(n_u.intersection(n_v))
            res.append(((u, v), score))
        return res

class Jaccard(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self, node_pairs):
        res = []
        for u, v in node_pairs:
            n_u = set(self.neighbors(u))
            n_v = set(self.neighbors(v))
            
            inter = n_u.intersection(n_v)
            union = n_u.union(n_v)
            
            # handle div by zero
            if len(union) == 0:
                score = 0.0
            else:
                score = len(inter) / len(union)
                
            res.append(((u, v), score))
        return res

class AdamicAdar(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self, node_pairs):
        res = []
        for u, v in node_pairs:
            n_u = set(self.neighbors(u))
            n_v = set(self.neighbors(v))
            common = n_u.intersection(n_v)
            
            score = 0.0
            for z in common:
                deg = len(self.neighbors(z))
                # log(1) is 0, ignore deg <= 1 to avoid error
                if deg > 1:
                    score += 1.0 / math.log(deg)
                    
            res.append(((u, v), score))
        return res


# Question 4 (c) - Evaluation function
def evaluate_link_prediction(graph, predictor, f=0.1, k_values=[50,100,200,300,400]):
    G = graph.copy()
    edges = list(G.edges())
    
    # remove a fraction f of edges
    num_remove = int(f * len(edges))
    removed = random.sample(edges, num_remove)
    G.remove_edges_from(removed)
    
    # Sorting them with min/max avoids the need to check both (A,B) and (B,A) later
    removed_set = set()
    for u, v in removed:
        removed_set.add((min(u, v), max(u, v)))
        
    # generate all possible node pairs (not already connected)
    node_pairs = []
    for u, v in itertools.combinations(G.nodes(), 2):
        if not G.has_edge(u, v):
            node_pairs.append((u, v))
            
    # compute scores using our predictor class
    model = predictor(G)
    scores = model.fit(node_pairs)
    
    # sort by decreasing score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    results = {}
    #compute precision and recall for each k
    for k in k_values:
        top_k = scores[:k]
        
        tp = 0
        for pair, score in top_k:
            u, v = pair
            if (min(u, v), max(u, v)) in removed_set:
                tp += 1
                
        precision = tp / k
        recall = tp / len(removed_set)
        
        results[k] = {
            "top@k": precision,
            "precision": precision,
            "recall": recall
        }
        
        print(f"k={k:3d} | TP: {tp:3d} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    return results


# Question 4 (d) - Testing on a large set of graphs (> 10)

if __name__ == "__main__":
    path = 'fb100/data/'
    all_files = os.listdir(path)
    
    # filter only the .gml files
    gml_files = []
    for f in all_files:
        if f.endswith('.gml'):
            gml_files.append(f)
            
    # sort them by size to avoid the giant networks 
    gml_files.sort(key=lambda x: os.path.getsize(path + x))
    
    # grab 12 smallest graphs to satisfy the >10 rule
    test_files = gml_files[:12]
    
    metrics = [CommonNeighbors, Jaccard, AdamicAdar]
    
    for f in test_files:
        print("\n=== graph:", f, "===")
        
        # load graph using simple string concatenation
        G = nx.read_gml(path + f)
        
        # get largest connected component
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        lcc = G.subgraph(comps[0]).copy()
        
        for m in metrics:
            print("running", m.__name__)
            evaluate_link_prediction(lcc, m, 0.1)
