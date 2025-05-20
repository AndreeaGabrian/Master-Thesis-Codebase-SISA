# import torch
# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("CUDA version:", torch.version.cuda)
#     print("Device:", torch.cuda.get_device_name(0))

from torchvision.models import resnet18
from torchviz import make_dot
import torch

model = resnet18(pretrained=True)
x = torch.randn(1, 3, 224, 224)
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("resnet18_architecture2", format="png")
