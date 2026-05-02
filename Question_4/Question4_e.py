NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

import torch
import numpy as np # keep for safety
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx, negative_sampling
import random
import time # to check how long it takes
import os

# --- my model ---
class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden):
        super(GNNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.conv2(h, edge_index)
        return h

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        
        # print("shapes:", src.shape, dst.shape) # debug
        
        score = (src*dst).sum(dim=1)
        return score

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


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
    
    for f in test_files:
        print("\n=== graph:", path + f, "===")
        
        # 1. load graph
        G = nx.read_gml(path + f)
        
        # get LCC
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(comps[0]).copy()
        
        G = nx.convert_node_labels_to_integers(G)
        data = from_networkx(G)
        
        # check if it has features, if not use degree as proxy
        if data.x is None:
            degs = [val for (node, val) in G.degree()]
            data.x = torch.tensor(degs, dtype=torch.float).view(-1, 1)

        edges = list(G.edges())
        random.shuffle(edges)
        
        test_size = int(len(edges) * 0.1)
        test_edges = edges[:test_size]
        train_edges = edges[test_size:]
        
        # make undirected for training
        train_idx = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
        train_idx = torch.cat([train_idx, train_idx[[1, 0]]], dim=1)
        
        # init model for each new graph
        model = GNNLinkPredictor(data.x.size(1), 64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # 2. Training
        # start_time = time.time()
        model.train()
        for epoch in range(150):
            optimizer.zero_grad()
            
            # negative samples
            neg_idx = negative_sampling(train_idx, num_nodes=data.num_nodes, num_neg_samples=train_idx.size(1))
            
            edge_label_idx = torch.cat([train_idx, neg_idx], dim=1)
            labels = torch.cat([torch.ones(train_idx.size(1)), torch.zeros(neg_idx.size(1))])
            
            out = model(data.x, train_idx, edge_label_idx)
            loss = loss_fn(out, labels)
            
            loss.backward()
            optimizer.step()

        # 3. Evaluation
        print("running GNN")
        model.eval()
        with torch.no_grad():
            
            # using a fast vectorized non-edge generation so the computer don't freeze
            N = data.num_nodes
            
            adj = torch.ones((N, N), dtype=torch.bool)
            adj[train_idx[0], train_idx[1]] = False
            adj = torch.triu(adj, diagonal=1)
            
            row, col = torch.nonzero(adj, as_tuple=True)
            candidates = torch.stack([row, col], dim=0)
            
            scores = model(data.x, train_idx, candidates)
            
            # sort predictions
            _, sorted_idx = torch.sort(scores, descending=True)
            
            test_set = set(tuple(sorted(e)) for e in test_edges)
            total_hidden = len(test_set)
            
            ks = [50, 100, 200, 300, 400]
            for k in ks:
                if k > len(sorted_idx): 
                    break
                    
                top_k = sorted_idx[:k]
                top_edges = candidates[:, top_k].t().tolist()
                
                tp = 0
                for e in top_edges:
                    if tuple(sorted(e)) in test_set:
                        tp += 1
                        
                prec = tp / k
                rec = tp / total_hidden if total_hidden>0 else 0
                
                # print format to match previous output
                print(f"k={k:>3} | TP: {tp:>3} | Precision: {prec:.4f} | Recall: {rec:.4f}")