# 🛒 Surveillance Shoplifting Detection using Temporal Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

> **End-to-end video classification framework for detecting theft and shoplifting acts in surveillance video feeds, featuring uniform equidistant frame sampling, synchronized spatio-temporal data augmentation, and temporal architectures (3D CNN, CNN+BiGRU with Temporal Attention, and Video Transformers) built from scratch.**

---

## 📌 Project Overview

Distinguishing between **Normal Shopping Behavior** (browsing, inspecting merchandise, walking) and **Shoplifting / Theft Acts** (reaching, grabbing, concealing items into clothing or bags, rapid flight) presents fundamental computer vision challenges:
1. **Long-Range Temporal Context**: Theft is an action spanning multiple seconds; static single-frame analysis cannot distinguish reaching for an item from stealing it.
2. **Variable Clip Durations**: Surveillance video clips range widely in duration ($N$ frames). Deep temporal models require fixed-size temporal tensors $(T, C, H, W)$.
3. **Motion Preservation under Augmentation**: Frame-by-frame data augmentation creates jarring spatial teleportations and reverses motion vectors, breaking temporal coherence.
4. **Model Architectures from Scratch**: Designing custom spatio-temporal networks (3D CNNs, CNN-RNNs, Video Transformers) without relying on external pretrained vision backbones.

---

## 🔄 End-to-End Pipeline

```
Surveillance Video Clip (.mp4, variable N frames)
  │
  ├─► Uniform Frame Sampling Strategy (Equidistant Skip Interval)
  │     └── Extracts exactly T frames spanning full duration [0, N-1]
  │
  ├─► Synchronized Spatial Clip Augmenter (Consistent across all T frames)
  │     ├── Random Resized Crop (identical bounding box across time)
  │     ├── Random Horizontal Flip (identical flip orientation across time)
  │     ├── Color Jitter (identical brightness, contrast, saturation shifts)
  │     └── Channel Normalization (ImageNet mean & std)
  │
  ├─► PyTorch Video Dataset & DataLoader
  │     └── Output Batch Tensor: (B, T, C, H, W)
  │
  ├─► Temporal Deep Learning Architectures (Built from Scratch)
  │     ├── Option 1: Spatiotemporal 3D CNN (Conv3d + 3D ResBlocks + 3D AvgPool)
  │     ├── Option 2: Custom 2D CNN + BiGRU + Temporal Attention Pooling
  │     └── Option 3: Video Vision Transformer (Self-Attention + Positional Encoding)
  │
  └─► Classification Head & Multi-Metric Evaluation
        ├── BCEWithLogitsLoss + AdamW + CosineAnnealingLR
        └── Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC
```

---

## 🏗️ Model Architectures Built from Scratch

| Architecture | Core Temporal Mechanism | Trainable Parameters | Tensor Flow |
|:---|:---|:---:|:---|
| **Spatiotemporal 3D CNN** | Spatio-temporal `Conv3d` + Residual Connections | **~3.5M** | $(B, T, C, H, W) \to (B, C, T, H, W) \to (B,)$ |
| **CNN + BiGRU + Attention** | 2D CNN + 2-Layer Bidirectional GRU + Temporal Salience Attention | **~670K** | $(B, T, C, H, W) \to (B \times T, D) \to (B, T, 2H) \to (B,)$ |
| **Video Vision Transformer** | Patch Embedding + Temporal Positional Encoding + Multi-Head Self-Attention | **~245K** | $(B, T, C, H, W) \to (B, T+1, D) \to (B,)$ |

---

## ⚙️ Key Implementation Highlights

### 1. Uniform Frame Sampling Strategy
For any video with $N$ total frames and target sample count $T$:
- **When $N \ge T$**:
  $$\text{step} = \frac{N}{T}, \quad \text{idx}_i = \min\left(\left\lfloor (i + 0.5) \cdot \text{step} \right\rfloor, N - 1\right)$$
- **When $N < T$**:
  $$\text{idx}_i = \min\left(\text{round}\left(i \cdot \frac{N - 1}{T - 1}\right), N - 1\right)$$
Guarantees consistent temporal span and avoids out-of-bounds errors regardless of clip length.

### 2. Synchronized Spatial Clip Augmentation
Evaluates all transformation parameters **once per video clip**:
- Same random horizontal flip applied to all $T$ frames.
- Same crop coordinates $(i, j, h, w)$ applied to all $T$ frames.
- Same brightness, contrast, and saturation multipliers applied to all $T$ frames.

### 3. Temporal Attention Pooling
In the CNN-RNN architecture, attention weights $\alpha_t = \text{Softmax}(W_a h_t)$ are learned dynamically across the $T$ time steps to isolate the moment of theft/concealment.

---

## 📊 Evaluation & Verification Checklist

- [x] **Dataset Exploration (EDA)**: Class balance, duration distribution, total frame counts, and spatial resolution analyzed and visualized.
- [x] **Frame Sampling Verification**: Unit-tested across $N \in [3, 8, 16, 24, 50, 95, 180, 500]$ frames with $T=16$.
- [x] **Augmentation Consistency**: Verified via code assertion and visual filmstrip plotting.
- [x] **Dataset Tensor Shape**: Verified batch tensor format $(B, T, C, H, W) = (4, 16, 3, 112, 112)$.
- [x] **Models from Scratch**: 3D CNN, CNN-RNN with Attention, and Video Transformer implemented without external pretrained weights.
- [x] **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC plotted.

---

## 🚀 Getting Started

Open and run the notebook in Jupyter or Google Colab:
```bash
jupyter notebook shoplifting_detection.ipynb
```
File location:
- [shoplifting_detection.ipynb](file:///d:/Downloads/Vision_intern/week_5_submit/shoplifting_detection.ipynb)
- Root copy: [shoplifting_detection.ipynb](file:///d:/Downloads/Vision_intern/shoplifting_detection.ipynb)
