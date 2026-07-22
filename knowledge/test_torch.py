import torch

cuda_available = torch.cuda.is_available()
print(f"当前是否可用:{cuda_available}")

if cuda_available:
    print(f"设备名：{torch.cuda.get_device_name()}")
else:
    print("当前使用 CPU 模式，未检测到可用的 CUDA 设备")