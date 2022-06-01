import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.mlp_readout_layer import MLPReadout
from layers.ConvLayer1D import ConvLayer1D

class Conv1DNet(torch.nn.Module):
    """
    Convolutional network with multiple convLayer layers
    """
    def __init__(self, net_params):
        super().__init__()
        self.in_dim = net_params['in_dim']
        #  self.out_dim = net_params['out_dim']
        self.hidden_dim = net_params['hidden_dim']
        self.n_classes = net_params['n_classes']
        self.size = net_params['size']
        self.pool_ratio = net_params['pool_ratio']
        self.pool = net_params['pool']
        self.kernel_size = net_params['kernel_size']
        self.in_feat_dropout = net_params['in_feat_dropout']
        self.dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.first_pool = 1
        self.decrease = net_params['decrease_conv']  # False
        self.activation = torch.relu
        # self.embedding_h = nn.Linear(self.in_dim, self.hidden_dim) # node feat is an integer
        self.feat_extraction = ConvLayer1D(self.in_dim, self.hidden_dim, self.activation, self.dropout, self.batch_norm,net_params, residual=self.residual,pooling=bool(self.first_pool))
        if self.decrease:
            self.mlp_input_size = int(self.size[0] * self.size[1] * self.hidden_dim / 2**self.first_pool)  # need to compute by hand /2 if first pooling = True feat_extraction
            self.layers = nn.ModuleList([ConvLayer1D(self.hidden_dim*2**i, self.hidden_dim*2**(i+1), self.activation, self.dropout, self.batch_norm, net_params, residual=self.residual) for i in range(self.n_layers)])
        else:
            self.mlp_input_size = int(self.size[0] * self.size[
                1] * self.hidden_dim / 2**(self.n_layers + self.first_pool))  # need to compute by hand /2 if first pooling = True feat_extraction
            self.layers = nn.ModuleList([ConvLayer1D(self.hidden_dim, self.hidden_dim,
                                                     self.activation, self.dropout, self.batch_norm, net_params,
                                                     residual=self.residual) for i in range(self.n_layers)])
        self.MLP_layer = MLPReadout(self.mlp_input_size, self.n_classes, net_params)

    def forward(self, h):
        batch = len(h[:, 0, 0, 0])
        h = h.clone().detach().float()
        h = torch.tensor(np.reshape(h.detach().numpy(), (batch, 1, -1)))
        h = self.feat_extraction(h)
        h = torch.dropout(h, p=self.in_feat_dropout, train=True)
        for conv in self.layers:
            h = conv(h)
        h = torch.tensor(np.reshape(h.detach().numpy(), (batch, -1)))
        assert h.shape[1] == self.mlp_input_size, 'error in computing outpout total dim shape input = {} output shape = {}'.format(h.shape, self.mlp_input_size)
        h_out = self.MLP_layer(h)
        return h_out

    def loss(self, pred, label):
        """
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        """
        # weighted cross-entropy for unbalanced classes
        V = label.size(0)
        label=label.long()
        # label_count = torch.bincount(np.reshape(label[:,1],(-1)))
        # label_count = label_count[label_count.nonzero()].squeeze()
        '''cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()'''
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(pred.float(), label.float())

        return loss


