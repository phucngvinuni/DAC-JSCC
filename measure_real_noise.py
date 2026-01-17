"""
Script đo nhiễu thực tế từ DAC-ADC hardware
"""
import serial
import time
import numpy as np

def measure_noise_level(port='/dev/ttyUSB0', baudrate=115200):
    """Đo noise bằng cách gửi cùng 1 giá trị nhiều ln"""
    
    ser = serial.Serial(port, baudrate, timeout=2)
    time.sleep(2)
    
    print("🔬 Đo nhiễu DAC-ADC...")
    
    # Test với 10 giá tr khác nhau
    test_values = [0, 512, 1024, 2048, 3072, 3584, 4095]
    noise_percentages = []
    
    for dac_val in test_values:
        print(f"\n📡 Gửi DAC = {dac_val}")
        readings = []
        
        # Gửi 50 lần cùng 1 giá trị
        for _ in range(50):
            ser.write(f"{dac_val}\n".encode())
            time.sleep(0.05)
            
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                try:
                    sensor_val = float(line.split()[-1])
                    readings.append(sensor_val)
                except:
                    pass
        
        if len(readings) > 10:
            readings = np.array(readings)
            mean = readings.mean()
            std = readings.std()
            noise_pct = (std / mean) * 100 if mean > 0 else 0
            
            print(f"   Mean: {mean:.2f}, Std: {std:.4f}")
            print(f"   Noise: {noise_pct:.2f}%")
            noise_percentages.append(noise_pct)
    
    ser.close()
    
    avg_noise = np.mean(noise_percentages)
    print(f"\n" + "="*60)
    print(f"📊 KẾT QUẢ:")
    print(f"   Nhiễu trung bnh: {avg_noise:.2f}%")
    print(f"\n💡 CẬP NHẬT CODE:")
    print(f"   Thay 0.03 → {avg_noise/100:.4f} trong train_small_dataset.py")
    print(f"   Dòng code:")
    print(f"   self.noise_std = (self.sensor_max - self.sensor_min) * {avg_noise/100:.4f}")
    print("="*60)

if __name__ == "__main__":
    measure_noise_level()
