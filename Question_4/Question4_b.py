NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
import math

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
        scores = []
        for u, v in node_pairs:
            n_u = set(self.neighbors(u))
            n_v = set(self.neighbors(v))
            
            # intersection size
            score = len(n_u.intersection(n_v))
            scores.append(((u, v), score))
            
        return scores
