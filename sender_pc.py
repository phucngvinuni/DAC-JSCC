import time
import serial
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import sys

# --- CẤU HÌNH ---
TX_PORT = 'COM27'   # Sửa cổng COM
BAUD_RATE = 115200
SYMBOL_DURATION = 0.3

# --- MODEL DEFINITION (Copy y hệt file train) ---
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

def main(args):
    device = torch.device('cpu')
    
    # 1. Load Map JSON
    if not os.path.exists(args.map_path):
        print("❌ Thiếu file map JSON"); return
    with open(args.map_path, 'r') as f:
        map_data = json.load(f)
    valid_dacs = np.array(map_data['dac_indices']) # Danh sách DAC xịn
    
    # 2. Load Model
    encoder = Encoder(latent_dim=args.latent_dim).to(device)
    try:
        encoder.load_state_dict(torch.load("linear_mnist_enc.pth", map_location=device))
        encoder.eval()
    except Exception as e:
        print(f"❌ Lỗi load model: {e}"); return

    # 3. Lấy ảnh MNIST ngẫu nhiên
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    idx = np.random.randint(0, len(test_dataset))
    img, label = test_dataset[idx]
    img_tensor = img.unsqueeze(0).to(device)
    
    print(f"🖼️  Đang gửi ảnh số: {label}")

    # 4. Encode & Map
    with torch.no_grad():
        latent = encoder(img_tensor).numpy()[0] # [0, 1]
    
    # Map Latent -> Index -> DAC
    idx = np.round(latent * (len(valid_dacs) - 1)).astype(int)
    idx = np.clip(idx, 0, len(valid_dacs) - 1)
    dac_to_send = valid_dacs[idx]
    
    print(f"📦 Gói tin ({len(dac_to_send)} symbols): {list(dac_to_send)}")

    # 5. Gửi
    try:
        ser = serial.Serial(TX_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"🔌 Đã kết nối {TX_PORT}")
        
        input("\n👉 Bấm ENTER để phát...")
        print("🚀 ĐANG GỬI...")
        
        for i, val in enumerate(dac_to_send):
            ser.write(f"{val}\n".encode())
            _ = ser.readline() # Handshake
            sys.stdout.write(f"\r  📡 Symbol {i+1}/{len(dac_to_send)}: {val}   ")
            sys.stdout.flush()
            time.sleep(SYMBOL_DURATION)
            
        ser.write(b"0\n"); ser.close()
        print("\n✅ Xong!")
        
        # Hiện ảnh gốc để so sánh
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Ảnh gốc (Số {label})")
        plt.show()
        
    except Exception as e:
        print(f"❌ Lỗi Serial: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_path', type=str, default='good_dac_map.json')
    parser.add_argument('--latent_dim', type=int, default=16)
    args = parser.parse_args()
    main(args)