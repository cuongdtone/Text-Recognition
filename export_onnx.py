# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 8/2/2022


from src.models.crnn import CRNN
from config import *

import torch.onnx
import onnx
import onnxruntime


path = '../WildOCR/src/crnn_r18.onnx'
net = CRNN(args)
device = torch.device('cuda' if torch.cuda.is_available() and args['cuda'] else 'cpu')
net.load_state_dict(torch.load(args['weight'], map_location=device)['state_dict'])
net.to(device)
net.eval()

x = torch.randn(1, 3, 64, 1000, requires_grad=True)
torch_out = net(x)

# Export the model
torch.onnx.export(net,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size', 3: 'imgH'},    # variable length axes
                                'output': {0: 'batch_size'}})

onnx_model = onnx.load(path)
onnx.checker.check_model(onnx_model)
net = onnxruntime.InferenceSession(path)
print(net)
