# Deepfake Detection Using ResNet-18 and Temporal Convolutional Networks (TCN)

This repository contains the code and methodology for a deepfake detection system developed as part of an MCA final-year project. The model is designed to distinguish between real and fake videos by analyzing both spatial and temporal features using a hybrid deep learning approach.

## 🔍 Project Overview

Deepfake videos — AI-generated manipulations of faces and voices — are becoming increasingly realistic and harder to detect. This project addresses that challenge by proposing a robust, real-time video classification system based on:

- **ResNet-18** for extracting spatial features from individual video frames.
- **Temporal Convolutional Networks (TCN)** for learning temporal inconsistencies in facial movements across frames.

The system aims to help in identifying manipulated videos with high accuracy (93.25%) and reliability.

## 🧠 Model Architecture

The architecture combines:

- **ResNet-18**: Captures high-level spatial features like textures, edges, and anomalies in face regions.
- **TCN**: Processes sequences of frames to detect unnatural temporal patterns (e.g., inconsistent blinking, jittery lips).

This hybrid model ensures both single-frame and motion-based anomalies are detected.

## 🗂️ Project Structure

├── may_extraction.py              # Frame extraction from videos  
├── face-only_video.py             # Face detection and face-only video generation  
├── video_dataset.py               # Custom PyTorch Dataset class  
├── training_resnettcn.py          # Model training script  
├── test_resnetcn.py               # Evaluation script with metrics  
├── inference_resnettcn.py         # Inference pipeline for new videos  
├── frames/                        # Directory containing extracted frames  
├── dataset/                       # Videos (real and fake)  
└── requirements.txt               # Dependencies



## 🧪 Dataset

We used a combination of:
- **FaceForensics++**
- **Celeb-DF**

With a custom preprocessing pipeline:
1. Frame extraction
2. Face detection & cropping
3. Temporal sequence generation

## 📊 Results

### 📊 Results

The final model achieves the following performance:

| Metric     | Previous Model | Current Model |
|------------|----------------|----------------|
| Accuracy   | 89.20%         | **93.25%**     |
| Precision  | 88.40%         | **94.00%**     |
| Recall     | 90.10%         | **93.00%**     |
| F1-Score   | 89.20%         | **93.00%**     |
| AUC        | 0.94           | **1.00**       |

**Table: Comparison of performance metrics between the previous and current model**

---

### 🔼 Improvements Over Previous Version

This version of the model shows a **4.05% increase in accuracy** and consistent improvements across all metrics. These gains are primarily due to:

- 🎯 **Focused facial region detection**: Improved preprocessing pipeline using **YOLOv8-Face** to extract face-only frames with high precision.
- 🧠 **ResNet-18 + TCN Architecture**: Combines spatial features (ResNet-18) with temporal dynamics (TCN), allowing better detection of subtle inconsistencies over time.
- 📊 **Balanced and cleaned dataset**: Includes a higher quality, better structured dataset with **balanced class distribution** and rigorous frame-level filtering.

These improvements not only enhanced accuracy but also improved **generalization on unseen data**, making the model more reliable in real-world scenarios.


## 📌 Key Features

- Real-time deepfake detection on raw video input
- Confidence score for each prediction
- Fully automated preprocessing-to-inference pipeline
- Lightweight ResNet-18 + TCN architecture

## 🧑‍💻 Author

**Mayuresh Girish Dhavalikar**  
Email: dhavalikarmayuresh@gmail.com

---

> 🛡️ Disclaimer: This project is for academic and research purposes only. It is not intended for commercial deployment or unauthorized surveillance.
