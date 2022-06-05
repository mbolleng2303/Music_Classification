import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer as GraphSageLayer
from layers.mlp_readout_layer import MLPReadout


class GraphNet2(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """

    def __init__(self, net_params):
        super().__init__()
        in_dim_node = 3011#net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        self.layer_type = 'edge'
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.readout = net_params['readout']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim) # node feat is an integer
        #self.embedding_e = nn.Linear(1, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([GraphSageLayer(int(hidden_dim/2**l), int(hidden_dim/2**(l+1)), F.relu,
                                                    dropout, aggregator_type, batch_norm, residual) for l in
                                     range(n_layers - 1)])
        self.layers.append(GraphSageLayer(int(hidden_dim/2**(n_layers-1)), out_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes, net_params)

    def forward(self, g, h, e):
        # input embedding
        #torch.set_default_dtype(torch.float64)
        #g = dgl.sampling.sample_neighbors(g, list(range(0, g.ndata['feat'].size()[0])), 30)
        #g = dgl.khop_graph(g, 1)

        h = g.ndata['feat']
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)
        #e = g.edata['feat']
        #e = self.embedding_e(np.reshape(e, (-1, 1)).float())
        i = 0
        for conv in self.layers:
            h = conv(g, h)
            i+=1
        h_out = self.MLP_layer(h)
        return h_out

    def loss(self, pred, label):
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.07366666666666667, 0.05466666666666667, 0.07633333333333334, 0.016, 0.042333333333333334, 0.033666666666666664, 0.036333333333333336, 0.016, 0.08933333333333333, 0.026333333333333334, 0.07466666666666667, 0.041, 0.04733333333333333, 0.354, 0.018333333333333333]))
        loss = criterion(pred.float(), label.float())
        return loss

