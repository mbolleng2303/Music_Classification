import numpy as np
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore', '.*User provided device_type of \'cuda\', but CUDA is not available. Disabling.*', )


class ConvLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 batch_norm, net_params, residual=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)
        self.kernel_size = net_params['kernel_size']
        self.pool_ratio = net_params['pool_ratio']
        self.pool_type = net_params['pool']
        if in_feats != out_feats:
            self.residual = False
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm2d(int(out_feats))
        self.conv = torch.nn.Conv2d(in_channels=in_feats,out_channels=out_feats, kernel_size=self.kernel_size, stride=1, padding=1)
        if self.pool_type == 'max':
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=self.pool_ratio)
        else :
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=self.pool_ratio)
        if activation is not None :
            self.activation = activation
        else:
            self.activation = torch.nn.ReLU()

    def forward(self, h):
        h = self.dropout(h)
        h_in = h  # for residual connection
        h = self.dropout(h)
        h = self.conv(h)
        h = self.pool(h)
        h = self.activation(h)
        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.residual:
            h = h_in + h  # residual connection

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, kernel_size={}, residual={}, pool ={})'.format(self.__class__.__name__,
                                                                                        self.in_channels,
                                                                                        self.out_channels,
                                                                                        self.kernel_size,
                                                                                        self.residual,
                                                                                        self.pool,)