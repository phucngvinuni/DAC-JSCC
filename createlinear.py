import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Cấu hình
CSV_PATH = 'final_merged_dataset.csv'
OUT_JSON = 'good_dac_map.json'

def create_monotonic_map():
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.lower() for c in df.columns]
    
    # Tính trung bình sensor cho mỗi mức DAC (để loại bỏ nhiễu ngẫu nhiên)
    stats = df.groupby('dac_input')['sensor_output'].mean()
    dac_values = stats.index.values
    sensor_values = stats.values

    # 2. Thuật toán Greedy: Tìm chuỗi tăng dần dài nhất
    # Chúng ta sẽ duyệt từ 0 đến 255, chỉ giữ lại điểm nào cao hơn điểm đã chọn trước đó
    
    good_dac = []
    good_sensor = []
    
    current_max_sensor = -1
    noise_margin = 10 # Sensor phải tăng ít nhất 10 đơn vị mới tính là tăng (tránh nhiễu)
    
    for dac, sens in zip(dac_values, sensor_values):
        # Bỏ qua vùng tối hẳn (ví dụ sensor < 100)
        if sens < 100: continue
        
        # Nếu giá trị sensor tăng so với đỉnh cũ -> Chọn
        if sens > current_max_sensor + noise_margin:
            good_dac.append(int(dac))
            good_sensor.append(float(sens))
            current_max_sensor = sens

    # 3. Lưu kết quả
    print(f"✅ Tìm thấy {len(good_dac)} mức DAC tốt (Monotonic) trên tổng số {len(dac_values)}")
    print(f"   Min DAC: {good_dac[0]} | Max DAC: {good_dac[-1]}")
    
    # Lưu vào JSON để dùng khi train
    data = {
        'dac_indices': good_dac,
        'sensor_values': good_sensor
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(data, f)
    print(f"💾 Đã lưu bảng map vào: {OUT_JSON}")

    # 4. Vẽ so sánh
    plt.figure(figsize=(12, 6))
    plt.plot(dac_values, sensor_values, 'r-', alpha=0.3, label='Gốc (Răng cưa)')
    plt.scatter(good_dac, good_sensor, c='b', s=10, label='Đã lọc (Tuyến tính)')
    plt.title("Tuyến tính hóa Kênh truyền")
    plt.xlabel("DAC Input")
    plt.ylabel("Sensor Output")
    plt.legend()
    plt.savefig("lin.png")

if __name__ == "__main__":
    create_monotonic_map()