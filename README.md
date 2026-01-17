
# DAC-JSCC: Analog Visible Light Communication with Deep Learning

**Course:** ELEC4010 - Introduction to Microelectronics (VinUniversity)  
**Project:** 8-bit R-2R DAC with Op-Amp and Transistor LED Driver  

## 📖 Overview
This project implements a **hardware-in-the-loop Deep Joint Source-Channel Coding (Deep JSCC)** system. Unlike traditional digital communication that transmits raw binary data (0s and 1s), this system compresses images into "semantic features" (latent vectors) and transmits them as **analog light intensities** using a custom-built 8-bit R-2R DAC.

The system uses a Deep Neural Network (Autoencoder) to learn robust feature representations that survive the noise and non-linearity of the physical optical channel.

## 🎥 Demo
Check out the system in action, transmitting handwritten digits from the transmitter PC to the receiver PC via light:

<video src="demo.mp4" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>

*(If the video does not load, please open `demo.mp4` locally)*

---

## 🛠 Hardware Setup

### Components
| Component | Function |
|-----------|----------|
| **Arduino Uno/R4** | Microcontroller for DAC control and ADC sampling |
| **R-2R Ladder** | 1kΩ and 2kΩ Precision Resistors (1%) |
| **LM358** | Op-Amp configured as Voltage Follower (Buffer) |
| **2N2222** | NPN Transistor (Emitter-Follower LED Driver) |
| **Blue LED** | High-brightness optical transmitter |
| **TEMT6000** | Ambient Light Sensor (Receiver) |

### Circuit Topology
1.  **DAC:** 8-bit GPIO $\to$ R-2R Network $\to$ Analog Voltage.
2.  **Buffer:** DAC Output $\to$ LM358 (+ Input).
3.  **Driver:** LM358 Output $\to$ 2N2222 Base $\to$ LED.

---

## 📂 Project Structure

### Hardware Interface & Deployment
*   `sender_pc.py`: **(Transmitter)** Encodes MNIST images into latent vectors, maps them to optimized DAC values, and sends them via Serial.
*   `receiver_pc.py`: **(Receiver)** Listens for analog signals, decodes the latent vectors, and reconstructs the image.
*   `measure_real_noise.py`: Diagnostic tool to measure the Signal-to-Noise Ratio (SNR) of your specific hardware setup.

### Training & Core Logic
*   `createlinear.py`: Analyzes channel data (`final_merged_dataset.csv`) to find the most linear/monotonic DAC values. Outputs `good_dac_map.json`.
*   `trainlinear.py`: Trains the Neural Network (Encoder/Decoder). It uses the generated map to simulate the physical channel during training (Differentiable Channel Approximation).
*   `channel.py` / `real_channel.py`: PyTorch layers simulating AWGN, Rayleigh, and Real hardware channels.
*   `utils.py`: Helper functions for model saving and PSNR calculation.

### Data
*   `final_merged_dataset.csv`: **(Required)** This file contains the calibration data (DAC Input vs. Sensor Output) collected from your hardware.
*   `good_dac_map.json`: Generated map linking Neural Network outputs to specific, clean DAC levels.

---

## 🚀 Usage Guide

### 1. Prerequisites
Install the required libraries:
```bash
pip install torch torchvision numpy pandas matplotlib pyserial tqdm scikit-learn
```

### 2. Channel Linearization
Before training, we must identify which DAC values produce a linear, monotonic response to help the AI learn better. This filters out the non-linear regions of the LED/Transistor.

```bash
python createlinear.py
```
*   **Input:** `final_merged_dataset.csv`
*   **Output:** `good_dac_map.json`

### 3. Training the Model
Train the Autoencoder. The system simulates the hardware channel using the map created in the previous step.

```bash
python trainlinear.py --epochs 15 --latent_dim 16
```
*   **Output:** Saves `linear_mnist_enc.pth` and `linear_mnist_dec.pth`.

### 4. Running the Real-World Experiment

**Step A: Configure Serial Ports**
Open `sender_pc.py` and `receiver_pc.py` and update the COM ports to match your computers:
```python
TX_PORT = 'COM27' # Update in sender_pc.py
RX_PORT = 'COM20' # Update in receiver_pc.py
```

**Step B: Start the Receiver**
Run this on the receiving computer (or terminal):
```bash
python receiver_pc.py --latent_dim 16
```
*The receiver will enter a listening loop, waiting for a trigger signal.*

**Step C: Start the Transmitter**
Run this on the transmitting computer:
```bash
python sender_pc.py --latent_dim 16
```
*   The script will load a random MNIST digit, encode it, and ask you to press ENTER to transmit.
*   Observe the LED blinking and the image appearing on the Receiver screen!

---

## 📊 Performance
*   **Compression:** Transmits a 784-pixel image using only 16 analog symbols (~98% compression).
*   **Linearization:** The `createlinear.py` script ensures that despite the hardware's non-linear IV curve, the Neural Network sees a "clean" monotonic channel.

## 👥 Authors
*   **Vo Viet Duc**
*   **Nguyen Hong Phuc**
```
