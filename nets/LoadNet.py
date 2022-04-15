
from nets.ConvNet import ConvNet


def convnet(net_params):

    return


def load_model(MODEL_NAME, net_params):

    models = {
            'ConvNet': convnet(net_params)
        }

    return ConvNet(net_params)
