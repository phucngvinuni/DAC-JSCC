import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. LINEARIZED CHANNEL (KÊNH SẠCH)
# ==========================================
class LinearizedChannel(nn.Module):
    def __init__(self, map_path, device):
        super().__init__()
        self.device = device
        
        # Load Map
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"❌ Thiếu file {map_path}. Hãy chạy 'create_linear_map.py' trước!")
            
        with open(map_path, 'r') as f:
            data = json.load(f)
            
        # Chuyển sang Tensor để tính toán gradient (Khả vi)
        self.good_dac = torch.tensor(data['dac_indices'], dtype=torch.float32).to(device)
        self.good_sensor = torch.tensor(data['sensor_values'], dtype=torch.float32).to(device)
        
        self.num_levels = len(self.good_dac)
        self.sensor_min = self.good_sensor.min()
        self.sensor_max = self.good_sensor.max()
        
        # Giả lập nhiễu: 3% của dải đo (Bạn có thể chỉnh số này)
        self.noise_std = (self.sensor_max - self.sensor_min) * 0.03
        
        print(f"🔧 Channel Init: {self.num_levels} levels | Range Sensor: [{self.sensor_min:.0f}, {self.sensor_max:.0f}]")

    def forward(self, x):
        # x: [Batch, Latent] trong khoảng [0, 1]
        
        # 1. Map x sang Index liên tục [0, num_levels - 1]
        idx_cont = x * (self.num_levels - 1)
        
        # 2. Nội suy Soft (Differentiable Interpolation)
        # Giúp Gradient truyền ngược qua bảng tra cứu
        idx_floor = torch.floor(idx_cont).long().clamp(0, self.num_levels - 1)
        idx_ceil = (idx_floor + 1).clamp(0, self.num_levels - 1)
        alpha = idx_cont - idx_floor.float()
        
        val_floor = self.good_sensor[idx_floor]
        val_ceil = self.good_sensor[idx_ceil]
        
        # Giá trị sensor lý tưởng
        sensor_val = (1 - alpha) * val_floor + alpha * val_ceil
        
        # 3. Thêm nhiễu
        noise = torch.randn_like(x) * self.noise_std
        sensor_noisy = sensor_val + noise
        
        # 4. Normalize về [0, 1] cho Decoder
        out = (sensor_noisy - self.sensor_min) / (self.sensor_max - self.sensor_min)
        return out

# ==========================================
# 2. MODEL (V2 - CÓ BATCHNORM)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
            nn.Sigmoid() 
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 32 * 7 * 7)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        x = self.linear(x)
        return self.net(x)

# ==========================================
# 3. TRAIN LOOP
# ==========================================
def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training MNIST Linearized | Latent: {args.latent_dim}")
    
    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), 
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), 
                             batch_size=args.batch_size, shuffle=False)

    # Init
    channel = LinearizedChannel(args.map_path, device).to(device)
    encoder = Encoder(latent_dim=args.latent_dim).to(device)
    decoder = Decoder(latent_dim=args.latent_dim).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        encoder.train(); decoder.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", leave=False)
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            
            latent = encoder(imgs)
            
            # Regularization nhẹ để giữ Latent không bị dính vào 0 hoặc 1 quá chặt
            # Nhưng với kênh sạch, ta giảm hệ số này xuống thấp (0.01)
            reg_loss = torch.mean((latent - 0.5)**2) * 0.01
            
            noisy = channel(latent)
            recon = decoder(noisy)
            
            mse = criterion(recon, imgs)
            loss = mse + reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += mse.item()
            pbar.set_postfix({'MSE': f"{mse.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        
        # Validate & Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), os.path.join(args.out_dir, "linear_mnist_enc.pth"))
            torch.save(decoder.state_dict(), os.path.join(args.out_dir, "linear_mnist_dec.pth"))
            print(f"✅ Epoch {epoch+1}: Saved Best Loss {best_loss:.4f}")

    # --- KẾT THÚC: VẼ BIỂU ĐỒ PHÂN BỐ DAC ---
    print("\n📊 Đang vẽ biểu đồ phân bố...")
    visualize_distribution(args, test_loader, channel, device)
    visualize_reconstruction(args, test_loader, channel, device)

def visualize_distribution(args, loader, channel, device):
    encoder = Encoder(latent_dim=args.latent_dim).to(device)
    encoder.load_state_dict(torch.load(os.path.join(args.out_dir, "linear_mnist_enc.pth")))
    encoder.eval()
    
    # Load DAC map để map từ latent -> dac thật
    with open(args.map_path, 'r') as f:
        map_data = json.load(f)
    valid_dacs = np.array(map_data['dac_indices'])
    
    all_dacs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            latent = encoder(imgs).cpu().numpy().flatten()
            
            # Map Latent [0,1] -> Index -> Real DAC
            indices = np.round(latent * (len(valid_dacs) - 1)).astype(int)
            indices = np.clip(indices, 0, len(valid_dacs) - 1)
            real_dacs = valid_dacs[indices]
            
            all_dacs.append(real_dacs)
            
    all_dacs = np.concatenate(all_dacs)
    
    plt.figure(figsize=(10, 5))
    plt.hist(all_dacs, bins=len(valid_dacs), color='green', edgecolor='black', alpha=0.7)
    plt.title(f"Phân bố DAC trên Kênh Tuyến Tính (Latent={args.latent_dim})")
    plt.xlabel("DAC Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(args.out_dir, "linear_distribution.png"))
    print(f"📈 Biểu đồ đã lưu tại: {args.out_dir}/linear_distribution.png")

def visualize_reconstruction(args, loader, channel, device):
    encoder = Encoder(latent_dim=args.latent_dim).to(device)
    decoder = Decoder(latent_dim=args.latent_dim).to(device)
    encoder.load_state_dict(torch.load(os.path.join(args.out_dir, "linear_mnist_enc.pth")))
    decoder.load_state_dict(torch.load(os.path.join(args.out_dir, "linear_mnist_dec.pth")))
    encoder.eval(); decoder.eval()
    
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        recon = decoder(channel(encoder(imgs)))
    
    imgs = imgs.cpu().numpy()
    recon = recon.cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.subplot(2, 5, i+1); plt.imshow(imgs[i][0], cmap='gray'); plt.axis('off')
        plt.subplot(2, 5, i+6); plt.imshow(recon[i][0], cmap='gray'); plt.axis('off')
    plt.savefig(os.path.join(args.out_dir, "linear_result.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--map_path', type=str, default='good_dac_map.json')
    parser.add_argument('--out_dir', type=str, default='./mnist_checkpoints')
    
    args = parser.parse_args()
    
    if os.path.exists(args.map_path):
        train(args)
    else:
        print("⚠️ Chưa có file map. Hãy chạy 'create_linear_map.py' trước!")