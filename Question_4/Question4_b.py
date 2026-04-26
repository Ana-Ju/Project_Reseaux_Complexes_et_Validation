NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

from abc import ABC, abstractmethod
import math
# import networkx as nx
# import numpy as np

# Question 4 (b) implement the following link prediction metrics: common neighbors, jaccard, Adamic/Adar. 
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

# Common Neighbors
class CommonNeighbors(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self, node_pairs):
        scores = []
        for u, v in node_pairs:
            n_u = set(self.neighbors(u))
            n_v = set(self.neighbors(v))
            
            # intersection size
            score = len(n_u.intersection(n_v))
            scores.append(((u, v), score))
            
        return scores

# Jaccard
class Jaccard(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self, node_pairs):
        scores = []
        for pair in node_pairs:
            u, v = pair[0], pair[1]
            n_u = set(self.neighbors(u))
            n_v = set(self.neighbors(v))
            
            intersection = n_u.intersection(n_v)
            union = n_u.union(n_v)
            
            # handle div by zero
            if len(union) == 0:
                score = 0.0
            else:
                score = len(intersection) / len(union)
                
            scores.append(((u, v), score))
            
        return scores

#Adamic/Adar
class AdamicAdar(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self, node_pairs):
        scores = []
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
                    
            scores.append(((u, v), score))
            
        return scores
