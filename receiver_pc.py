import time
import serial
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import sys
import os

# --- CẤU HÌNH ---
RX_PORT = 'COM20'   # Sửa cổng COM
BAUD_RATE = 115200
SYMBOL_DURATION = 2.0 
TRIGGER_THRESHOLD = 1000 

# --- MODEL DEFINITION (Copy y hệt file train) ---
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

def main(args):
    device = torch.device('cpu')

    # 1. Load Min/Max từ JSON Map
    if not os.path.exists(args.map_path):
        print("❌ Thiếu file map JSON"); return
    
    with open(args.map_path, 'r') as f:
        map_data = json.load(f)
    
    # Lấy min/max của các điểm "xịn"
    valid_sensors = np.array(map_data['sensor_values'])
    s_min = valid_sensors.min()
    s_max = valid_sensors.max()
    print(f"🔧 Sensor Range (Clean): [{s_min:.1f}, {s_max:.1f}]")

    # 2. Load Model
    decoder = Decoder(latent_dim=args.latent_dim).to(device)
    try:
        decoder.load_state_dict(torch.load("linear_mnist_dec.pth", map_location=device))
        decoder.eval()
    except Exception as e:
        print(f"❌ Lỗi load model: {e}"); return

    # 3. Kết nối & Auto Trigger
    try:
        ser = serial.Serial(RX_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"🔌 Receiver tại {RX_PORT}. Chờ tín hiệu > {TRIGGER_THRESHOLD}...")
        
        received_vals = []
        while True:
            ser.write(b'r')
            line = ser.readline().decode().strip()
            if line.isdigit():
                val = int(line)
                sys.stdout.write(f"\r   Sensor: {val}   ")
                sys.stdout.flush()
                
                if val > TRIGGER_THRESHOLD:
                    print(f"\n⚡ BẮT ĐẦU THU...")
                    received_vals.append(val)
                    time.sleep(SYMBOL_DURATION)
                    for i in range(args.latent_dim - 1):
                        ser.write(b'r')
                        line = ser.readline().decode().strip()
                        v = int(line) if line.isdigit() else 0
                        received_vals.append(v)
                        print(f"  📥 {i+2}/{args.latent_dim}: {v}")
                        time.sleep(SYMBOL_DURATION)
                    break
            time.sleep(0.1)
            
    except Exception as e:
        print(e); return

    print(f"\n✅ Dữ liệu thô: {received_vals}")

    # 4. Giải mã
    latent = (np.array(received_vals) - s_min) / (s_max - s_min)
    latent = np.clip(latent, 0, 1)
    
    tensor = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        recon_img = decoder(tensor).numpy()[0] # [1, 28, 28]

    # 5. Hiển thị
    plt.figure(figsize=(5, 5))
    plt.imshow(recon_img.squeeze(), cmap='gray')
    plt.title("Kết quả MNIST (Linearized Channel)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_path', type=str, default='good_dac_map.json')
    parser.add_argument('--latent_dim', type=int, default=16)
    args = parser.parse_args()
    main(args)