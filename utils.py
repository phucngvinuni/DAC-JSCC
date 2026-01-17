import torch
import os
import math

def get_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images (or batches).
    img1, img2: Tensor [N, C, H, W] or [C, H, W]
    max_val: Maximum value of the image range (default 1.0)
    """
    mse = torch.mean((img1.float() - img2.float()) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))

def save_model(model, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, filename))
    # print(f"Model saved to {os.path.join(save_path, filename)}")

def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += param.numel()
    return total_param

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
