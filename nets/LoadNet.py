
from nets.Conv2DNet import Conv2DNet
from nets.Conv1DNet import Conv1DNet
from nets.GraphNet import GraphNet
def convnet(net_params):

    return


def load_model(MODEL_NAME, net_params):

    if MODEL_NAME =='Conv2DNet':
        return Conv2DNet(net_params)
    if MODEL_NAME == 'Conv1DNet':
        return Conv1DNet(net_params)
    if MODEL_NAME == 'GraphNet':
        return GraphNet(net_params)
    else :
        raise NotImplementedError



