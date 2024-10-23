import torch
from torch.amp import GradScaler  

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize GradScaler for mixed precision training
scaler = GradScaler()

print(f'Device:{device}')