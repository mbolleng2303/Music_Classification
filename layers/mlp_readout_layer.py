import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    MLP Layer used after conv vector representation
"""
class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, net_params):  # L=nb_hidden_layers
        self.L = net_params['L_mlp']
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(self.L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** self.L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.output_dim = output_dim
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        y = F.softmax(y, dim=1)
        return y
