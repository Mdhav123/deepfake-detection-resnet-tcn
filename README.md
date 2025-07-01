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

- **Accuracy**: 93.25%
- **Metrics**: Confusion Matrix, ROC Curve, AUC
- Robust to low-resolution, compressed, and occluded videos.

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
