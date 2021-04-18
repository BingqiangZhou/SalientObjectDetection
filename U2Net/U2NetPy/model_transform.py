import os
import torch
import torchvision

from load_net import *

# reference docs: https://pytorch.org/docs/stable/onnx.html#example-end-to-end-alexnet-from-pytorch-to-onnx


def model2onnx(model, onnx_path, input_size, input_names=[ "input_image" ], output_names=[ "output_pred" ]):

    dummy_input = torch.randn(*input_size, device='cpu')
    # model = torchvision.models.alexnet(pretrained=True).cuda()
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=11, input_names=input_names, output_names=output_names)

def load_model_from_onnx(onnx_path):
    # onnx
    #########################################################################################
    ## conda install -c conda-forge onnx

    # import onnx
    # # Load the ONNX model
    # model = onnx.load("alexnet.onnx")

    # # Check that the IR is well formed
    # onnx.checker.check_model(model)

    # # Print a human readable representation of the graph
    # onnx.helper.printable_graph(model.graph)
    #########################################################################################

    # caffe2
    #########################################################################################
    # #...continuing from above
    # import caffe2.python.onnx.backend as backend
    # import numpy as np

    # rep = backend.prepare(model, device="CUDA:0") # or "CPU"
    # # For the Caffe2 backend:
    # #     rep.predict_net is the Caffe2 protobuf for the network
    # #     rep.workspace is the Caffe2 workspace for the network
    # #       (see the class caffe2.python.onnx.backend.Workspace)
    # outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
    # # To run networks with more than one input, pass a tuple
    # # rather than a single numpy ndarray.
    # print(outputs[0])
    #########################################################################################
    pass

# 注意：为了让部署时，只输出d0（side fuse的结果），导出模型前，先修改网络模型中的输出，只输出d0

# model_name = 'u2netp'
# model = U2NETP(3,1)
is_parallel_model = True
model_path = "./saved_models/u2net/u2net_bce_itr_68000_train_0.505270_tar_0.054967.pth"
# model_path = "./saved_models/u2net/u2net_bce_itr_58000_train_0.552608_tar_0.061580.pth"
state_dict = torch.load(model_path, map_location="cpu")
if is_parallel_model:
    new_state_dict = {}
    for key in state_dict.keys():
        new_state_dict.update({key[7:]: state_dict[key]}) # remove "module." in key
    state_dict.clear()
    state_dict = new_state_dict
input_size = (1, 3, 400, 400)
model = U2NET(3,1)
model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
# print(model_name, model, state_dict.keys())
model.load_state_dict(state_dict)
# model.cuda()
model2onnx(model, f"./saved_models/{model_name}.onnx", input_size)
