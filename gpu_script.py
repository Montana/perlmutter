import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.rand(10000, 10000).cuda()
    y = torch.matmul(x, x)
    print("Matrix multiplication complete.")
