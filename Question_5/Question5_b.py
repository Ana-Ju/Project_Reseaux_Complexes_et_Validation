NOM = "MENDES GOUVEIA DA SILVA"
PRENOM = "Ana Julia"

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        # linear layers to handle the feature weights (X * W)
        self.gc1 = nn.Linear(num_features, hidden_dim, bias=False)
        self.gc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        # adj_norm is the precomputed renormalized matrix: D^(-1/2) * A_tilde * D^(-1/2)
        # first layer: transform features then aggregate from neighbors
        h = self.gc1(x)
        h = torch.sparse.mm(adj_norm, h)
        h = F.relu(h)
        
        # simple dropout to avoid overfitting
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # apply second transformation and aggregate again
        out = self.gc2(h)
        out = torch.sparse.mm(adj_norm, out)
        
        # no softmax here because loss will handle it
        return out