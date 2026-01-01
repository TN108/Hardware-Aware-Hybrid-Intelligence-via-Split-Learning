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
