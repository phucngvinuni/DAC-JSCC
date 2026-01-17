import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

class RealChannel(nn.Module):
    def __init__(self, csv_path='final_merged_dataset.csv', device='cuda', dac_range=(140, 170)):
        """
        Mô phỏng kênh truyền thực tế dựa trên dữ liệu thu thập được.
        Sử dụng nội suy tuyến tính để đảm bảo tính khả vi (Differentiable).
        """
        super().__init__()
        self.device = device
        
        # 1. Load và Xử lý Dữ liệu
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"LỖI: Không tìm thấy file '{csv_path}'. Hãy thu thập dữ liệu trước!")
            
        print(f"--- Đang khởi tạo Real Channel từ {csv_path} ---")
        df = pd.read_csv(csv_path)
        
        # Lọc dữ liệu trong vùng tuyến tính bạn mong muốn (VD: 140-170)
        # Điều này quan trọng để loại bỏ vùng chết (0-139) gây nhiễu việc học
        min_dac, max_dac = dac_range
        df = df[(df['dac_input'] >= min_dac) & (df['dac_input'] <= max_dac)]
        
        if len(df) == 0:
            raise ValueError(f"Dữ liệu trống sau khi lọc vùng {dac_range}. Kiểm tra lại CSV!")

        # 2. Tính toán Thống kê (Mean & Std) cho từng mức DAC
        # Group theo mức DAC input để tìm đặc tính phân phối của Output tại mức đó
        stats = df.groupby('dac_input')['sensor_output'].agg(['mean', 'std'])
        
        # Reindex để đảm bảo có đủ các mức từ min đến max (tránh lủng lỗ)
        full_idx = np.arange(min_dac, max_dac + 1)
        stats = stats.reindex(full_idx)
        
        # Nội suy (Fill) các giá trị NaN nếu thiếu mẫu ở một vài mức
        stats = stats.interpolate(method='linear').bfill().ffill()
        
        # Lưu các thông số cấu hình
        self.dac_min = min_dac
        self.dac_max = max_dac
        self.num_levels = self.dac_max - self.dac_min + 1
        
        # Normalization Bounds (Lấy min/max thực tế của Sensor để chuẩn hóa về 0-1)
        # Bạn có thể hard-code số này nếu muốn cố định (VD: 1141, 4921)
        self.sensor_min = df['sensor_output'].min()
        self.sensor_max = df['sensor_output'].max()
        
        print(f"DAC Range: [{self.dac_min}, {self.dac_max}] ({self.num_levels} levels)")
        print(f"Sensor Output Range: [{self.sensor_min:.0f}, {self.sensor_max:.0f}]")
        
        # 3. Chuyển thống kê sang Tensor (đưa vào Buffer để không bị update bởi Optimizer)
        # register_buffer giúp lưu các tensor này vào state_dict khi save model
        self.register_buffer('means', torch.tensor(stats['mean'].values, dtype=torch.float32))
        self.register_buffer('stds', torch.tensor(stats['std'].values, dtype=torch.float32))

    def forward(self, x):
        """
        Mô phỏng kênh truyền: Input [0,1] -> Kênh thực (Nội suy) -> Output [0,1]
        x: Latent vector từ Encoder (Batch, ...), giá trị trong khoảng [0, 1]
        """
        # 1. Ánh xạ x [0, 1] sang không gian chỉ số liên tục (Continuous Index)
        # Ví dụ: x=0.5 -> idx_cont = 15.0 (nằm giữa dải)
        # Phải giữ nguyên số thực (float), KHÔNG được làm tròn (int) ở đây!
        idx_cont = x * (self.num_levels - 1)
        
        # 2. Nội suy tuyến tính (Linear Interpolation) - Cốt lõi của sự khả vi
        # Tìm 2 điểm lân cận: Floor (dưới) và Ceil (trên)
        idx_floor = torch.floor(idx_cont).long()
        idx_ceil = idx_floor + 1
        
        # Kẹp giá trị index để không bị tràn mảng (Out of bounds)
        idx_floor = torch.clamp(idx_floor, 0, self.num_levels - 1)
        idx_ceil = torch.clamp(idx_ceil, 0, self.num_levels - 1)
        
        # Tính trọng số pha trộn (alpha)
        # alpha = phần lẻ. Ví dụ idx_cont=15.3 -> alpha=0.3
        alpha = idx_cont - idx_floor.float()
        
        # Lấy Mean và Std tại 2 điểm lân cận
        mean_floor = self.means[idx_floor]
        mean_ceil = self.means[idx_ceil]
        std_floor = self.stds[idx_floor]
        std_ceil = self.stds[idx_ceil]
        
        # Pha trộn Mean và Std (Công thức nội suy)
        # Kết quả: mu và sigma sẽ thay đổi mượt mà theo x
        mu = (1 - alpha) * mean_floor + alpha * mean_ceil
        sigma = (1 - alpha) * std_floor + alpha * std_ceil
        
        # 3. Reparameterization Trick (Thêm nhiễu)
        # y = mu + sigma * epsilon
        # epsilon là nhiễu chuẩn N(0,1), độc lập với x
        epsilon = torch.randn_like(x)
        y_real = mu + sigma * epsilon
        
        # 4. Chuẩn hóa ngược lại về [0, 1] để đưa vào Decoder
        # Decoder chỉ hiểu tín hiệu trong khoảng 0-1 (do Sigmoid/ReLU)
        y_norm = (y_real - self.sensor_min) / (self.sensor_max - self.sensor_min)
        
        return y_norm