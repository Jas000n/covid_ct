import io
import numpy as np
import torch.onnx
from model import my_resnet

torch_model = my_resnet()
model_path = "model_weights.pth"
batch_size = 1

torch_model.load_state_dict(torch.load(model_path))
torch_model.eval()

x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = torch_model(x)

torch.onnx.export(torch_model, x, "model.onnx", export_params=True, opset_version=11, do_constant_folding=True, input_names = ['input'], output_names = ['output'], dynamic_axes = {'input' : {0: 'batch_size'}, 'output': {0: 'batch_size'}})