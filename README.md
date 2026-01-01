📋 Overview
This project demonstrates a working split learning system that partitions a CNN between an Arduino Nano 33 BLE Sense (client) and a Raspberry Pi (server). The Arduino runs initial convolutional layers locally, generating compressed feature maps transmitted over Bluetooth Low Energy (BLE). The Raspberry Pi processes these features through remaining network layers for final classification.
Key Achievements

✅ 83.40% accuracy on binary classification
✅ 92.9% bandwidth reduction compared to transmitting raw images
✅ 4KB feature tensors instead of full images
✅ Privacy-preserving: Raw visual data never leaves the device
✅ End-to-end BLE communication with bidirectional feedback

🎯 Motivation
Edge AI deployment faces a critical trade-off:

Local Inference: Resource-constrained devices can't run full models
Cloud Offloading: High bandwidth, power consumption, privacy risks, and latency

Split Learning solves this by intelligently partitioning the neural network across devices.
🏗️ System Architecture
┌─────────────────┐         BLE          ┌──────────────────┐
│  Arduino Nano   │  ──────────────────>  │  Raspberry Pi    │
│  33 BLE Sense   │                       │     Server       │
│                 │                       │                  │
│  • Image        │   4KB Feature         │  • Remaining     │
│    Capture      │   Tensors (8×4×4)     │    Conv Layers   │
│  • Initial      │   (128 int8 values)   │  • FC Layers     │
│    Conv Layers  │                       │  • Classifier    │
│  • Feature      │  <────────────────    │  • Inference     │
│    Extraction   │   Binary Result       │                  │
└─────────────────┘    (0 or 1)          └──────────────────┘
Network Partition
ComponentDeviceLayersClientNetArduino Nano 33 BLE SenseInitial Conv + PoolingServerNetRaspberry PiFC Layers + Classifier
🔧 Hardware Requirements
Arduino Nano 33 BLE Sense

Processor: ARM Cortex-M4 @ 64 MHz
SRAM: 256 KB
Flash: 1 MB
Camera: OV7675 (QVGA)
Wireless: Bluetooth 5.0 (BLE)

Raspberry Pi (3B+ or 4B)

Processor: ARM Cortex-A (Quad-core)
RAM: 1-8 GB
Storage: 16+ GB SD card
BLE: Integrated or USB dongle

📦 Software Dependencies
Arduino
cpp- TensorFlow Lite Micro
- ArduinoBLE library
- Arduino_OV767X (camera support)
Raspberry Pi
bashpip install torch torchvision numpy bleak
🚀 Getting Started
1. Clone the Repository
bashgit clone https://github.com/yourusername/split-learning-edge.git
cd split-learning-edge
2. Training Phase (PC)
bashpython train_split_and_cache.py
python train_pi_remaining_conv_binary.py
This generates:

client_model.tflite - Arduino weights
server_conv_dogbin.pth - Raspberry Pi model
meta.json - Quantization parameters

3. Arduino Setup
bash# Open Arduino IDE
# Load arduino_client/arduino_client.ino
# Install required libraries (ArduinoBLE, TensorFlow Lite Micro)
# Upload to Arduino Nano 33 BLE Sense
4. Raspberry Pi Setup
bashcd raspberry_pi
python recv_features_convbin_and_reply.py
```

## 📊 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 83.40% |
| Feature Tensor Size | 4,096 bytes (4KB) |
| Raw Image Size | 57,600 bytes (160×120×3) |
| Bandwidth Reduction | 92.9% |
| Inference Latency | ~420-950 ms |

### Training Progress

- Initial Loss: 1.9461
- Final Loss: 0.9780
- Training Epochs: 25
- Dataset: CIFAR-10 (binary: dog vs. not-dog)

## 🗂️ File Structure
```
split-learning-edge/
│
├── arduino_client/
│   └── arduino_client.ino          # Client-side TFLM inference & BLE
│
├── raspberry_pi/
│   ├── server_model.py              # ServerNet architecture
│   ├── ble_receiver.py              # BLE communication handler
│   ├── recv_features_convbin_and_reply.py  # Main server script
│   └── server_conv_dogbin.pth       # Trained model weights
│
├── training/
│   ├── train_split_and_cache.py     # End-to-end training
│   └── train_pi_remaining_conv_binary.py  # Pi-side classifier
│
└── README.md
🔬 Technical Details
Data Pipeline

Capture: 160×120 RGB565 frame
Preprocessing: Resize to 32×32, convert to grayscale
Feature Extraction: ClientNet produces 8×4×4 tensor (128 int8 values)
Transmission: 4KB over BLE in chunks
Inference: ServerNet processes features
Response: Binary result (0/1) sent back

BLE Communication Protocol

Service UUID: 12340000-0000-1000-8000-00805f9b34ef0
Feature Characteristic: ...ef1
Response Characteristic: ...ef2
Device Name: Nano33SplitClient
MTU: 20-512 bytes per packet

🚧 Challenges & Solutions
Challenge 1: Hardware Conflict
Problem: Arduino can't run camera and BLE simultaneously
Solution: Sequential workflow - capture → compute → transmit
Challenge 2: BLE Instability
Problem: Frequent connection drops
Solution: Retry mechanisms, connection tuning, checksum verification
Challenge 3: Memory Constraints
Problem: 256KB SRAM limits computation
Solution: Simplified feature extraction, careful memory management
📈 Future Improvements

 Deploy exact trained weights to Arduino (currently using approximation)
 Implement int8 quantization for 75% further bandwidth reduction
 Extend to multi-class classification (full CIFAR-10)
 Optimize latency with asynchronous processing
 Add energy consumption profiling
 Support multiple Arduino clients per server

🤝 Contributing
We welcome contributions! Areas for improvement:

Optimizing BLE communication
Implementing exact model deployment on Arduino
Adding support for other edge devices
Improving real-time performance

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👥 Team
Team 13

Hisham (24280041)
Usman Shahid (24030029)
Talha Nasir (24280040)
Khadija Hakim (24280056)

📚 References

Gupta, O., & Raskar, R. (2018). Distributed learning of deep neural network over multiple agents.
Thapa, C., et al. (2022). SplitFed: When Federated Learning Meets Split Learning.
Banbury, C. R., et al. (2021). Micronets: Neural network architectures for deploying tinyml applications.

🙏 Acknowledgments
Special thanks to the open-source communities behind:

TensorFlow Lite Micro
PyTorch
Arduino
bleak BLE library

