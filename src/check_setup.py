import torch
import torchvision
import numpy as np

print("Setup check")
print("-----------")
print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("NumPy version:", np.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    device = "GPU"
else:
    print("No GPU found. Using CPU.")

x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print("Test tensor:", x)
print("Device used:", x.device)

print("Everything works.")