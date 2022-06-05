import torch.nn as nn
'User provided device_type of \'cuda\', but CUDA is not available. Disabling'
"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""


class GraphSageLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type
        self.batch_norm = batch_norm
        self.residual = residual
        if in_feats != out_feats:
            self.residual = False
        self.dropout = nn.Dropout(p=dropout)
        from dgl.nn.pytorch import SAGEConv, GatedGraphConv, GraphConv, TAGConv, RelGraphConv,GMMConv, GINConv
        #self.sageconv = GraphConv(in_feats, out_feats, activation=activation)
        self.sageconv = SAGEConv(in_feats, out_feats, aggregator_type, activation=activation, feat_drop=dropout)
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)
    def forward(self, g, h ,e = None):
        h_in = h  # for residual connection
        #h = self.dropout(h)
        #g = dgl.sampling.select_topk(g, 30, 'feat')
        if e is not None:

            h = self.sageconv(g, h, e)
        else:
            h = self.sageconv(g, h)

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.residual:
            h = h_in + h  # residual connection

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
                                                                                        self.in_channels,
                                                                                        self.out_channels,
                                                                                        self.aggregator_type,
                                                                                        self.residual)


