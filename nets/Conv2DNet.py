import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.mlp_readout_layer import MLPReadout
from layers.ConvLayer2D import ConvLayer2D

class Conv2DNet(torch.nn.Module):
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
        self.activation = torch.relu
        self.first_pool = 0
        self.inc_channel = net_params['inc_channel']
        #int(self.size[0]*self.size[1]*self.hidden_dim/#  need to compute by hand
        # self.embedding_h = nn.Linear(self.in_dim, self.hidden_dim) # node feat is an integer
        self.feat_extraction = ConvLayer2D(self.in_dim, self.hidden_dim, self.activation, self.dropout, self.batch_norm,net_params, residual=self.residual, pooling=bool(self.first_pool))
        if self.inc_channel:
            self.mlp_input_size = 1800 #int(self.size[0] * self.size[1] * self.hidden_dim / (2**(self.n_layers + self.first_pool)))
            self.layers = nn.ModuleList([ConvLayer2D(self.hidden_dim*2**i, self.hidden_dim*2**(i+1), self.activation, self.dropout, self.batch_norm, net_params, residual=self.residual) for i in range(self.n_layers)])
        else:
            self.mlp_input_size = 3600 #int(self.size[0]*self.size[1]*self.hidden_dim/(((2**self.first_pool)*(self.pool_ratio**2)**self.n_layers)))
            self.layers = nn.ModuleList([ConvLayer2D(self.hidden_dim, self.hidden_dim,
                                                     self.activation, self.dropout, self.batch_norm, net_params,
                                                     residual=self.residual) for _ in range(self.n_layers)])
        self.MLP_layer = MLPReadout(self.mlp_input_size, self.n_classes, net_params)

    def forward(self, h):
        batch = len(h[:, 0, 0, 0])
        h = h.clone().detach().float()
        h = self.feat_extraction(h)
        h = torch.dropout(h, p=self.in_feat_dropout, train=True)
        for conv in self.layers:
            h = conv(h)
        h = torch.tensor(np.reshape(h.detach().numpy(), (batch, -1)))
        assert h.shape[1] == self.mlp_input_size, 'error in computing outpout total dim shape input = {} ' \
                                                  'output shape = {} you have to change self.mlp_input_size =.... to match the input mlp'.format(h.shape, self.mlp_input_size)
        h_out = self.MLP_layer(h)
        return h_out

    def loss(self, pred, label):
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(
            [0.07366666666666667, 0.05466666666666667, 0.07633333333333334, 0.016, 0.042333333333333334,
             0.033666666666666664, 0.036333333333333336, 0.016, 0.08933333333333333, 0.026333333333333334,
             0.07466666666666667, 0.041, 0.04733333333333333, 0.354, 0.018333333333333333]))
        loss = criterion(pred.float(), label.float())
        return loss



