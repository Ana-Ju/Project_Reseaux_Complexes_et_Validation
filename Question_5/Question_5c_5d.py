NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from sklearn.metrics import mean_absolute_error, f1_score 

# GCN from 5(b)
class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()
    
        self.gc1 = nn.Linear(num_features, hidden_dim, bias=False)
        self.gc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.dropout = dropout

    def forward(self, x, adj_norm):
   
        h = self.gc1(x)
        h = torch.sparse.mm(adj_norm, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        out = self.gc2(h)
        out = torch.sparse.mm(adj_norm, out)
        
        return out


if __name__ == "__main__":
    # loading the Brown network 
    graph_path = 'fb100/data/Brown11.gml'
    print(f"Loading {graph_path}...")
    
    G = nx.read_gml(graph_path)
    G = nx.convert_node_labels_to_integers(G)
    num_nodes = G.number_of_nodes()

    # precompute normalized adjacency matrix
    A = nx.adjacency_matrix(G).todense()
    A = torch.tensor(A, dtype=torch.float32)
    A_tilde = A + torch.eye(num_nodes)
    
    # degree matrix math
    D_tilde = torch.sum(A_tilde, dim=1)
    D_inv = torch.pow(D_tilde, -0.5)
    D_inv[torch.isinf(D_inv)] = 0.
    D_mat = torch.diag(D_inv)

    # final sparse adj matrix
    adj_norm = torch.mm(torch.mm(D_mat, A_tilde), D_mat).to_sparse()

    # identity matrix since we are trying to predict attributes
    X = torch.eye(num_nodes)

    # experiment setup
    attrs = ['major_index', 'dorm', 'gender'] # removed year to strictly match prompt (c)
    fractions = [0.1, 0.2, 0.3]               # removed 0.4 to strictly match prompt (c)
    
    # Dictionaries to hold the results for parts (c) and (d)
    results_acc = {}
    results_mae = {}
    results_f1 = {}
    
    for attr in attrs:
        # get raw labels from the graph
        raw_labels = list(nx.get_node_attributes(G, attr).values())
        
        # fallback if the dataset uses 'major' instead of 'major_index'
        if not raw_labels and attr == 'major_index':
            raw_labels = list(nx.get_node_attributes(G, 'major').values())
            
        if not raw_labels:
            continue

        # map labels to continuous integers starting from 0 for PyTorch
        raw_labels = np.array(raw_labels).astype(int)
        unique_vals = np.unique(raw_labels)
        mapping = {val: i for i, val in enumerate(unique_vals)}
        labels = torch.tensor([mapping[v] for v in raw_labels], dtype=torch.long)
        
        n_classes = len(unique_vals)
        
        acc_row = []
        mae_row = []
        f1_row = []
        
        for frac in fractions:
            # shuffle nodes to randomly select the missing ones
            nodes = list(range(num_nodes))
            random.shuffle(nodes)
            
            num_remove = int(frac * num_nodes)
            test_idx = nodes[:num_remove]
            train_idx = nodes[num_remove:]

            # init model and optimizer
            model = GCN(num_features=num_nodes, hidden_dim=16, num_classes=n_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            loss_fn = nn.CrossEntropyLoss()

            # training loop
            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(X, adj_norm)
                
                # loss only on the nodes we kept (train_idx)
                loss = loss_fn(out[train_idx], labels[train_idx])
                loss.backward()
                optimizer.step()

            # evaluation
            model.eval()
            with torch.no_grad():
                out = model(X, adj_norm)
                
                # predictions on the nodes we removed (test_idx)
                preds = out[test_idx].argmax(dim=1)
                
                # Convert to numpy for sklearn metrics
                y_true = labels[test_idx].numpy()
                y_pred = preds.numpy()
                
                # Part (C) Metric
                acc = (y_pred == y_true).sum() / len(test_idx)
                
                # Part (D) Metrics
                mae = mean_absolute_error(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted') # weighted is best for imbalanced fb100 attributes
                
                acc_row.append(acc)
                mae_row.append(mae)
                f1_row.append(f1)
                
        # Store rows for printing later
        name = 'Major' if attr == 'major_index' else attr.capitalize()
        results_acc[name] = acc_row
        results_mae[name] = mae_row
        results_f1[name] = f1_row

    # --- Print Formatting ---
    def print_table(title, data_dict):
        print(f"\n{title} (Brown)")
        print(f"{'':<10} {'0.1':<6} {'0.2':<6} {'0.3':<6}")
        for name, row in data_dict.items():
            print(f"  {name:<8} {row[0]:.3f}  {row[1]:.3f}  {row[2]:.3f}")

    print_table("Table: Accuracy (Eq. 1)", results_acc)
    print_table("Table: Mean Absolute Error", results_mae)
    print_table("Table: F1-Score", results_f1)