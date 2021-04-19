import os
import sys
import torch
import torchvision

pyfiles_dir = os.path.dirname(os.path.realpath(__file__)).replace('tools', '')
# print(pyfiles_dir)
sys.path.append(pyfiles_dir)

from nets import get_net

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

def model_transform(net_name, model_path, input_size, is_parallel_model=False):
    
    model = get_net(net_name, False, only_return_d0=True)
    state_dict = torch.load(model_path, map_location="cpu")
    if is_parallel_model:
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict.update({key[7:]: state_dict[key]}) # remove "module." in key
        state_dict.clear()
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model_dir, model_name = os.path.split(model_path)
    model_name, ext = os.path.splitext(model_name)
    
    onnx_model_path = f"{model_dir}/{input_size[2]}_{input_size[3]}_{model_name}.onnx"
    model2onnx(model, onnx_model_path, input_size)
    print("model save to: ", onnx_model_path)


# model_path = "./saved_models/u2net/u2net_bce_itr_118000_train_0.383985_tar_0.038499.pth"
model_path = "./models/backup/u2netdsconv_bce_itr_54000_train_1.149171_tar_0.148460.pth"
# net_name = 'u2netp'
net_name = 'u2net_dsconv'
# net_name = 'u2net_groupconv'
is_parallel_model = True
# input_size = (1, 3, 400, 400)
input_size = (1, 3, 288, 288)
model_transform(net_name, model_path, input_size, is_parallel_model)
