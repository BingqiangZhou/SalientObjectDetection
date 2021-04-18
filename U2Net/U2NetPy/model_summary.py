import torch
from torchsummary import summary  # torchsummary-1.5.1
from thop import profile #  thop-0.0.31
from ptflops import get_model_complexity_info # ptflops-0.6.4

from u2net import U2NETP
from u2net_groupconv import U2NET as U2NETGroupConv
from u2net_dsconv import U2NET as U2NETDSConv

to_device = 'cpu'
input_size = (3, 400, 400)
verbose = False

models = {
    "U2NETP": U2NETP(3, 1),
    "U2NETGroupConv": U2NETGroupConv(3, 1),
    "U2NETDSConv": U2NETDSConv(3, 1)
}

for key in models.keys():
    model = models[key].to(to_device)
    
    # thop
    input_tensor = torch.randn(1, *input_size, device=to_device)
    flops, params = profile(model, (input_tensor, ), verbose=verbose)
    print(f"{key} flops: {flops}, params: {params}")
    
    # ptflops
    macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                           print_per_layer_stat=False, verbose=verbose)
    print(key)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    #  torchsummary
    # summary(model, input_size=input_size,  device=to_device) # (channels, h, w)

# U2NETP
# Input size (MB): 1.83 (1, 3, 400, 400)
# Forward/backward pass size (MB): 2270.41
# Params size (MB): 4.32
# Estimated Total Size (MB): 2276.56
# Computational complexity (GMac): 31.16 
# Number of parameters (M): 1.13

# U2NETGroupConv
# Input size (MB): 1.83 (1, 3, 400, 400)
# Forward/backward pass size (MB): 3136.27
# Params size (MB): 84.19
# Estimated Total Size (MB): 3222.29
# Computational complexity (GMac):  52.36
# Number of parameters (M): 22.07

# U2NETDSConv
# Input size (MB): 1.83 (1, 3, 400, 400)
# Forward/backward pass size (MB): 4042.85
# Params size (MB): 19.61
# Estimated Total Size (MB): 4064.29
# Computational complexity (GMac): 11.9
# Number of parameters (M): 5.14

