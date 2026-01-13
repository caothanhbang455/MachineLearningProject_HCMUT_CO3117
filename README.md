# Music Genre Classification

## Overview

This project studies **automatic music genre classification** using the **GTZAN dataset**, combining:

- Signal-level exploratory data analysis (EDA)
- Hand-crafted audio feature extraction
- Traditional machine learning models
- Deep learning models on both feature vectors and time–frequency representations

The goal is to **systematically compare representation choices and modeling strategies**, rather than to maximize benchmark accuracy.

---

## Project Structure

```text
ML_Project/
│
├── MusicGenre_Classification.ipynb   # Main notebook (EDA + experiments)
├── requirements.txt                  # Python dependencies
├── train_cnn.py                      # CNN training pipeline (image-based)
├── train_mlp.py                      # MLP training pipeline (feature-based)
│
├── Data/
│   ├── genres_original/              # Original GTZAN wav files
│   ├── features_data.csv             # Extracted feature table (optional)
│   └── images_data/                  # Generated spectrogram / waveform images
│
├── config/
│   └── models.yaml                   # ML model definitions and hyperparameters
│
├── models/
│   ├── cnn.py                        # CNN architectures
│   └── mlp.py                        # MLP architecture
│
├── utils/
│   ├── utils.py                      # Feature extraction and visualization
│   ├── augument.py                   # Audio augmentation utilities
│   └── model_evaluation_utils.py     # Model comparison and evaluation
│
├── checkpoints/                      # Saved model weights
└── runs/                             # TensorBoard logs

```
## Dataset

- **Dataset**: GTZAN Music Genre Dataset  
- **Classes**: 10 genres  
- **Format**: WAV audio files  
- **Duration**: 30 seconds per track  

Each audio file is segmented into **3-second non-overlapping clips** to:

- Increase sample size  
- Enable segment-level and song-level evaluation  
- Reduce overfitting  
---

## Exploratory Data Analysis

The notebook performs detailed EDA at both **signal level** and **dataset level**, including:

### Time-domain analysis
- Raw waveform  
- Amplitude statistics  
- RMS energy  
- Zero Crossing Rate (ZCR)  
- Silence ratio  

### Frequency-domain analysis
- Log spectrogram  
- Mel spectrogram  
- Chromagram  

### Dataset-level visualization
- Class distribution  
- Feature boxplots by genre  
- Correlation heatmap  
- t-SNE visualization of feature space  

Observations regarding **feature redundancy and multicollinearity** are explicitly discussed.

---

## Feature Extraction

For each **3-second segment**, the following feature groups are extracted:

- **Time-domain**: RMS, ZCR, silence ratio  
- **Spectral**: centroid, bandwidth, rolloff, flatness  
- Spectral contrast (multiple frequency bands)  
- Harmonic / percussive energy  
- Tempo-related features  

Feature extraction is implemented in `utils/utils.py`.

---

## Data Splitting Strategy

To prevent **data leakage**, all splits are performed using **group-aware strategies**:

- `StratifiedGroupKFold`  
- Grouped by original audio file name  

This ensures that segments from the same song never appear in both training and testing sets.

---

## Models

### Traditional Machine Learning (feature-based)

Models evaluated include:

- Logistic Regression  
- Support Vector Machine  
- k-Nearest Neighbors  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- AdaBoost  
- XGBoost  
- LightGBM  

Model comparison is performed using **group-aware cross-validation**, with results visualized for interpretability.

---

### Deep Learning (feature-based)

- Multi-Layer Perceptron (MLP)  
- Input: standardized feature vectors  
- Loss: cross-entropy  
- Evaluation: segment-level and song-level accuracy  

Implemented in `train_mlp.py`.

---

### Deep Learning (image-based)

CNN trained on:

- Mel spectrogram images  
- Waveform images  

Adaptive pooling used to handle variable-length inputs.  

Training implemented in `train_cnn.py`.

This branch investigates whether **learned representations outperform handcrafted features**.

---

## Audio Augmentation

Implemented augmentations include:

- Consecutive random cropping  
- Additive Gaussian noise  

Augmentation effects are visualized to ensure realism and signal integrity.

---

## Evaluation

Evaluation is conducted at two levels:

### Segment-level
- Standard classification metrics  

### Song-level
- Majority voting across segments belonging to the same song  

This better reflects real-world usage scenarios.

---

## Reproducibility

### Environment Setup

```bash
pip install -r requirements.txt
