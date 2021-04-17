import os
import torch
import torchvision

from sod import U2NETP, U2NET

# reference docs: https://pytorch.org/docs/stable/onnx.html#example-end-to-end-alexnet-from-pytorch-to-onnx


def model2onnx(model, onnx_path, input_size, input_names=[ "input_image" ], output_names=[ "output_pred" ]):

    dummy_input = torch.randn(*input_size, device='cuda')
    # model = torchvision.models.alexnet(pretrained=True).cuda()
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, input_names=input_names, output_names=output_names)

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


# model = torchvision.models.alexnet(pretrained=True).cuda()
# torch.save(model.state_dict(), "alexnet.pth")
# model_name = 'u2netp'
# model = U2NETP(3,1)
model_name = 'u2net'
model = U2NET(3,1)
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '.pth')
model.load_state_dict(torch.load(model_dir))
model.cuda()
model2onnx(model, f"./saved_models/{model_name}.onnx", (1, 3, 400, 400))
