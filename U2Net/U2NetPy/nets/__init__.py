import os

from .u2net import U2NETP, U2NET
from .u2net_groupconv import U2NET as U2NETGroupConv
from .u2net_dsconv import U2NET as U2NETDSConv


def get_net(net_name, return_pretrain_model_path=True, only_return_d0=False):

    if net_name == 'u2net':
        net = U2NET(3, 1, only_return_d0)
    elif net_name == 'u2netp':
        net = U2NETP(3,1, only_return_d0)
    elif net_name == 'u2net_groupconv':
        net = U2NETGroupConv(3, 1, only_return_d0)
    elif net_name == 'u2net_dsconv':
        net = U2NETDSConv(3, 1, only_return_d0)
    
    if return_pretrain_model_path:
        model_dir = './models/pretrain_models' # relative to: ../U2NetPy
        if net_name == 'u2net_groupconv':
            load_model_path = os.path.join(model_dir, "u2net_groups2.pth")
        else:
            load_model_path = os.path.join(model_dir, net_name+".pth")
        return net, load_model_path
    return net
