# Face Recognition System

This project implements a **robust face recognition system** using a novel architecture called **DB-ACVT (Dual-Branch Attention Convolution Vision Transformer)** and a custom loss function known as **Hardness Adaptive Margin (HAM) Loss**. It focuses on achieving high accuracy even under challenging real-world conditions such as occlusion, pose variation, and lighting inconsistency.

## 🚀 Overview

This system is designed to:
- Accurately identify individuals from facial images.
- Maintain robustness under various occlusion and noise conditions.
- Use adaptive attention mechanisms to focus on identity-relevant features.

---

## 📂 Dataset

- **CelebAMask-HQ** and **CelebA-HQ-Face-Identity** datasets.
- Contains facial images of **17 individuals**, with at least **15 images per identity**.
- Publicly available via GitHub: [CelebA-HQ-Face-Identity](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch)

---

## 🧼 Preprocessing Pipeline

1. **Face Detection**: Using MediaPipe for landmark localization.
2. **Face Alignment**: Affine transformation aligns facial landmarks (e.g., eyes, nose, mouth) to a frontal template.

---

## 🧪 Data Augmentation

To simulate real-world scenarios:
- **SemanticStyleGAN**: Generates variations in lighting, texture, and occlusions.
- **Synthetic Occlusion**: Black rectangles and partial eye/lip masking.
- **Standard Techniques**: Horizontal flipping, Gaussian blur/noise, brightness/contrast changes.

> Aggressive or identity-destroying augmentations are avoided to preserve recognition integrity.

---

## 🧠 Model Architecture: DB-ACVT

A hybrid model combining CNN and Transformer-based mechanisms:

### 1. **Backbone: EfficientNetB0**
- Lightweight and effective feature extraction.
- Uses MBConv blocks and SE modules for scaling efficiency.

### 2. **Attention Modules**
- **Channel Attention**: Global average/max pooling + MLP.
- **Spatial Attention**: 7×7 convolution over pooled features.
- **Residual Fusion**: Improves robustness against occlusion.

### 3. **Global Context: CvT (Convolutional Vision Transformer)**
- Captures long-range dependencies via multi-head attention over convolutional embeddings.

---

## 🔀 Dual-Branch Fusion Mechanism

- **Local Feature Fusion**: Dynamically blends channel & spatial attention.
- **Global & Local Fusion**: Combines local attention features with global Transformer features for final predictions.

> This enables the system to dynamically prioritize relevant facial regions.

---

##  Loss Function: Hardness Adaptive Margin (HAM) Loss

Improves learning from difficult (hard) samples:
- Dynamically adjusts margin based on the angle and hardness of a sample.
- Encourages **intra-class compactness** and **inter-class separation**.

> Outperforms traditional fixed-margin losses like ArcFace and CosFace on hard samples.

---

## 🧾 Embedding & Face Matching

- After training, facial embeddings are extracted and stored in a **face gallery database**.
- **Cosine similarity** is used to compare query embeddings with gallery entries for recognition.

---

## 📚 References

- [CelebA-HQ-Face-Identity GitHub](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch)  
- [SemanticStyleGAN GitHub](https://github.com/seasonSH/SemanticStyleGAN/tree/main)  
- [Face Data Augmentation Techniques](https://hackernoon.com/face-data-augmentation-part-2-image-synthesis)  
- [Dual-Branch Transformer for Hyperspectral Classification](https://www.researchgate.net/publication/380339581)  


