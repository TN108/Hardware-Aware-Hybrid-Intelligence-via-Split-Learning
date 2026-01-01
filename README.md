# Hardware-Aware Hybrid Intelligence via Split Learning


## 📖 Overview

A practical implementation of **split learning** for resource-limited edge devices, specifically targeting bandwidth constraints. This project demonstrates a distributed ML system that splits a CNN between an Arduino Nano 33 BLE Sense (client) and a Raspberry Pi (server).

The Arduino runs initial convolutional layers locally, generating compressed feature maps transmitted over Bluetooth Low Energy (BLE). The Pi processes these features through remaining network layers for final classification.

### Why Split Learning?

Edge AI deployment faces critical trade-offs:

| Approach | Problem |
|----------|---------|
| **Local Inference** | Resource-constrained devices can't run full models |
| **Cloud Offloading** | High bandwidth, power consumption, privacy risks, latency |

**Split Learning** intelligently partitions neural networks across devices, combining the benefits of both approaches.

## ✨ Features

- 🎯 **83.40% accuracy** on binary classification (dog vs. not-dog)
- 📉 **92.9% bandwidth reduction** vs. raw image transmission
- 🔒 **Privacy-preserving**: Raw visual data never leaves the device
- 📦 **4KB feature tensors** transmitted instead of full images
- ⚡ **Real-time inference** with BLE bidirectional communication
- 🔧 **TinyML ready**: Runs on devices with <1MB SRAM

## 🏗️ Architecture

```
┌─────────────────────┐         BLE          ┌─────────────────────┐
│  Arduino Nano       │  ─────────────────>  │  Raspberry Pi       │
│  33 BLE Sense       │                       │  Server             │
│                     │                       │                     │
│  📷 Image Capture   │   4KB Feature         │  🧠 ServerNet       │
│  🔧 ClientNet       │   Tensor (8×4×4)      │  • Remaining Conv   │
│  • Initial Conv     │   128 int8 values     │  • FC Layers        │
│  • Pooling          │                       │  • Classifier       │
│  • Feature Extract  │  <─────────────────   │  ✅ Binary Result   │
│                     │   Result (0 or 1)     │                     │
└─────────────────────┘                       └─────────────────────┘
```

### Network Partition

| Component | Device | Layers | Parameters | FLOPs |
|-----------|--------|--------|------------|-------|
| **ClientNet** | Arduino | Initial Conv + Pooling | ~5,000 | ~50K |
| **ServerNet** | Raspberry Pi | FC + Classifier | ~50,000 | ~500K |
| **Offload Ratio** | - | - | 90% | 90% |

## 🛠️ Hardware Requirements

### Arduino Nano 33 BLE Sense

| Specification | Value |
|--------------|-------|
| Microcontroller | nRF52840 |
| Processor | ARM Cortex-M4F @ 64 MHz |
| SRAM | 256 KB |
| Flash Memory | 1 MB |
| Wireless | Bluetooth 5.0 / BLE |
| Camera | OV7675 (QVGA) via breakout |

### Raspberry Pi (3B+ or 4B)

| Specification | Pi 3B+ | Pi 4B |
|--------------|--------|-------|
| Processor | Cortex-A53 | Cortex-A72 |
| Cores | 4 | 4 |
| Clock Speed | 1.4 GHz | 1.5-1.8 GHz |
| RAM | 1 GB | 2/4/8 GB |
| Bluetooth | 4.2 BLE | 5.0 BLE |

## 📦 Installation

### Prerequisites

- Arduino IDE (1.8.x or later)
- Python 3.7+
- Raspberry Pi OS (Bullseye or later)

### 1. Clone Repository

```bash
git clone https://github.com/TN108/Hardware-Aware-Hybrid-Intelligence-via-Split-Learning.git
cd Hardware-Aware-Hybrid-Intelligence-via-Split-Learning
```

### 2. Install Arduino Dependencies

Open Arduino IDE and install:

```
Tools > Manage Libraries > Search and Install:
- ArduinoBLE
- Arduino_TensorFlowLite
- Arduino_OV767X
```

### 3. Install Raspberry Pi Dependencies

```bash
cd raspberry_pi
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
bleak>=0.14.0
```

## 🚀 Quick Start

### Step 1: Train the Model (PC)

```bash
# Train the full split learning model
python training/train_split_and_cache.py

# Train Pi-side binary classifier
python training/train_pi_remaining_conv_binary.py
```

**Output:**
- `models/client_model.tflite` - Arduino weights
- `models/server_conv_dogbin.pth` - Raspberry Pi model
- `models/meta.json` - Quantization parameters

### Step 2: Deploy to Arduino

```bash
# 1. Open arduino_client/arduino_client.ino in Arduino IDE
# 2. Connect Arduino Nano 33 BLE Sense via USB
# 3. Select: Tools > Board > Arduino Nano 33 BLE
# 4. Select: Tools > Port > [Your COM Port]
# 5. Click Upload
```

### Step 3: Run Raspberry Pi Server

```bash
cd raspberry_pi
python recv_features_convbin_and_reply.py
```

**Expected Output:**
```
[INFO] Starting BLE server...
[INFO] Scanning for Arduino...
[INFO] Connected to Nano33SplitClient
[INFO] Receiving features...
[INFO] Feature tensor received: 4096 bytes
[INFO] Running inference...
[RESULT] Classification: DOG DETECTED
[INFO] Result sent to Arduino
```

## 📊 Results

### Performance Metrics

| Metric | Value | Comparison |
|--------|-------|------------|
| **Test Accuracy** | 83.40% | Binary classification |
| **Feature Tensor Size** | 4,096 bytes | 4 KB |
| **Raw Image Size (32×32×3)** | 3,072 bytes | Baseline |
| **High-res Image (160×120×3)** | 57,600 bytes | 56 KB |
| **Bandwidth Reduction** | 92.9% | vs. high-res |
| **Total Latency** | 420-950 ms | End-to-end |

### Latency Breakdown

| Operation | Time (ms) |
|-----------|-----------|
| Image Capture | 50-100 |
| Feature Extraction | 100-200 |
| BLE Transmission | 200-500 |
| Pi Inference | 20-50 |
| BLE Response | 50-100 |

### Training Progress

- **Training Epochs**: 25
- **Initial Loss**: 1.9461
- **Final Loss**: 0.9780
- **Initial Validation Accuracy**: 29.21%
- **Final Test Accuracy**: 83.40%
- **Dataset**: CIFAR-10 (binary subset: dog vs. not-dog)



## 🔬 Technical Details

### Data Pipeline

```
Raw Capture (160×120×3 RGB565)
    ↓
Downsample (32×32×3 RGB)
    ↓
Grayscale Conversion (32×32×1)
    ↓
ClientNet Feature Extraction (8×4×4 int8)
    ↓
BLE Transmission (4096 bytes)
    ↓
ServerNet Inference
    ↓
Binary Classification (0 or 1)
```

### BLE Communication Protocol

| Parameter | Value |
|-----------|-------|
| Service UUID | `12340000-0000-1000-8000-00805f9b34ef0` |
| Feature Characteristic | `12340000-0000-1000-8000-00805f9b34ef1` |
| Response Characteristic | `12340000-0000-1000-8000-00805f9b34ef2` |
| Device Name | `Nano33SplitClient` |
| MTU Size | 20-512 bytes |
| Total Payload | 4096 bytes (chunked) |

### Communication Sequence

```
Arduino                    Raspberry Pi
   |                            |
   |--- Advertise ------------->|
   |<-- Connect Request --------|
   |--- Connection Ack -------->|
   |                            |
   |--- Feature Chunk 1 ------->|
   |--- Feature Chunk 2 ------->|
   |--- ...                     |
   |--- Feature Chunk N ------->|
   |                            |
   |                    [Inference]
   |                            |
   |<-- Binary Result (0/1) ----|
   |                            |
```

## 🚧 Challenges & Solutions

### 1. Hardware Conflict
**Problem**: Arduino can't run camera and BLE simultaneously  
**Solution**: Sequential workflow - capture → compute → disable camera → transmit

### 2. BLE Instability
**Problem**: Frequent connection drops during transmission  
**Solution**: Implemented retry logic, connection parameter tuning, checksum verification

### 3. Memory Constraints
**Problem**: 256KB SRAM limits computation complexity  
**Solution**: Simplified feature extraction, careful buffer management, streaming processing

### 4. Model Incompatibility
**Problem**: Initial Pi model expected raw images, not feature tensors  
**Solution**: Trained custom ServerNet accepting 8×4×4 feature tensors as input

## 🔮 Future Work

- [ ] **Exact Model Deployment**: Port trained weights to Arduino (currently approximated)
- [ ] **INT8 Quantization**: Reduce feature size from 4KB to 1KB (75% reduction)
- [ ] **Multi-Class Classification**: Extend from binary to full 10-class CIFAR-10
- [ ] **Latency Optimization**: Implement asynchronous processing pipelines
- [ ] **Energy Profiling**: Comprehensive power consumption analysis
- [ ] **Scalability**: Support multiple Arduino clients per Pi server
- [ ] **Federated Learning**: Collaborative model improvement across devices

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- 🔧 Optimizing BLE communication protocols
- 🧠 Implementing exact TFLite model deployment
- 📱 Adding support for other edge devices (ESP32, STM32)
- ⚡ Real-time performance improvements
- 📊 Energy consumption benchmarking

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

**Team 13**

| Member | ID |
|--------|------------|
| Hisham | 24280041 |
| Usman Shahid | 24030029 |
| Talha Nasir | 24280040 |
| Khadija Hakim | 24280056 |


## 📖 References

1. Gupta, O., & Raskar, R. (2018). Distributed learning of deep neural network over multiple agents. *Journal of Network and Computer Applications*.

2. Thapa, C., et al. (2022). SplitFed: When Federated Learning Meets Split Learning. *AAAI Conference on Artificial Intelligence*.

3. Banbury, C. R., et al. (2021). Micronets: Neural network architectures for deploying tinyml applications on commodity microcontrollers. *Proceedings of Machine Learning and Systems*.

4. TensorFlow Lite for Microcontrollers. https://www.tensorflow.org/lite/microcontrollers

## 🙏 Acknowledgments

This project wouldn't be possible without the open-source communities behind:

- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [PyTorch](https://pytorch.org/)
- [Arduino](https://www.arduino.cc/)
- [bleak BLE library](https://github.com/hbldh/bleak)

<div align="center">




[![GitHub stars](https://img.shields.io/github/stars/yourusername/Hardware-Aware-Hybrid-Intelligence-via-Split-Learning?style=social)](https://github.com/yourusername/Hardware-Aware-Hybrid-Intelligence-via-Split-Learning)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/Hardware-Aware-Hybrid-Intelligence-via-Split-Learning?style=social)](https://github.com/yourusername/Hardware-Aware-Hybrid-Intelligence-via-Split-Learning/fork)

</div>
