# AI Sepsis Detection

A comprehensive machine learning project for early detection of sepsis in ICU patients using clinical time-series data. This project implements advanced preprocessing, multiple data balancing strategies, and extensive model prototyping with deep learning architectures.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [File Structure](#file-structure)
- [Requirements](#requirements)

## 🎯 Overview

This project addresses the critical challenge of early sepsis detection in ICU settings by leveraging deep learning on multivariate time-series clinical data. The system implements:

- **Rigorous preprocessing** with IQR-based normalization and time-limited forward-fill
- **Exclusion criteria** matching baseline research papers
- **Multiple balancing strategies** (SMOTE, undersampling, class weighting)
- **Comprehensive model prototyping** across 6 architectures with hyperparameter optimization
- **Robust evaluation** with ROC-AUC, PR-AUC, sensitivity, specificity, and F1-score

## ✨ Features

### Data Preprocessing
- **Hourly resampling** of patient time-series using ICULOS (hours since ICU admission)
- **IQR-based outlier detection** and normalization computed from training data only (prevents data leakage)
- **Time-limited forward-fill** with clinically appropriate hold intervals (4h for vitals, 24h for labs)
- **Derived clinical features**: ShockIndex, BUN/Cr ratio, MEWS score, pSOFA score
- **Clinical sanity bounds** to prevent unrealistic values
- **Missingness tracking** using binary mask columns
- **Patient-level exclusion**:
  - Patients with < 8 hours of ICU data
  - Sepsis onset < 4 hours after ICU admission

### Data Split
- **15% Test set** (held-out)
- **68% Training set**
- **17% Validation set**
- Stratified patient-level split (no patient leakage between sets)

### Class Imbalance Handling

#### 1. **SMOTE with LSTM Autoencoder**
- Trains weighted LSTM autoencoder with masking layer to handle `-1` missing values
- Encodes sequences to 64-dimensional embeddings
- Applies SMOTE in embedding space
- Decodes synthetic samples back to original feature space
- Clips extreme values using percentile bounds
- Tracks patient IDs for synthetic samples

#### 2. **Undersampling**
- Random undersampling of majority class to balance dataset
- Maintains all minority class samples
- Preserves patient-level tracking

#### 3. **Class Weighting**
- Applies dynamic class weights (61:1 ratio for original imbalanced data)
- Used during training without modifying dataset

## 📊 Dataset

### Input Features (21 total)

**Vital Signs (7)**:
- `HR`, `SBP`, `DBP`, `Resp`, `O2Sat`, `Temp`, `MAP`

**Lab Values (8)**:
- `WBC`, `Platelets`, `Hgb`, `Creatinine`, `BUN`, `Potassium`, `Glucose`, `Lactate`

**Demographics (2)**:
- `Age`, `Gender`

**Derived Features (4)**:
- `ShockIndex` (HR/SBP)
- `BUN_Cr` (BUN/Creatinine ratio)
- `MEWS` (Modified Early Warning Score)
- `pSOFA` (Partial Sequential Organ Failure Assessment)

**Target**:
- `SepsisLabel` (0 = no sepsis, 1 = sepsis)

### Preprocessing Output
- **Training matrices**: `(N, 4, 21)` - N windows of 4 hours × 21 features
- **Labels**: `(N,)` - Binary sepsis labels
- **Patient IDs**: `(N,)` - Patient tracking for each window

## 🏗️ Pipeline Architecture

```
1. Data Loading & Exclusion Criteria
   ↓
2. Hourly Resampling (ICULOS-based)
   ↓
3. Patient-Level Split (15/68/17)
   ↓
4. IQR Limit Computation (training only)
   ↓
5. Time-Limited Forward Fill
   ↓
6. Derived Feature Engineering
   ↓
7. Normalization & Outlier Clipping
   ↓
8. Missing Value Imputation (-1)
   ↓
9. Sliding Window (4-hour)
   ↓
10. Class Balancing (SMOTE/Undersampling/Weighting)
    ↓
11. Model Training & Evaluation
```

## 🚀 Installation

### Requirements

```bash
pip install numpy pandas scikit-learn tensorflow keras imblearn matplotlib seaborn
pip install fastdtw
```

### Google Colab Setup

The code was originally developed in Google Colab with Google Drive integration:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 💻 Usage

### 1. Preprocessing

Update the paths in the configuration section:

```python
input_path = '/path/to/merged_sepsis_dataset.csv'
output_dir = '/path/to/output'
```

The preprocessing section will generate:
- `sepsis_preprocessed_train.csv`
- `sepsis_preprocessed_val.csv`
- `sepsis_preprocessed_test.csv`
- `iqr_minmax_limits.json`
- `feature_missingness_heatmap.png`

### 2. Build Sliding Window Matrices

```python
OUTPUT_DIR = '/path/to/dataset'
WINDOW_HOURS = 4
FEATURES = ['HR', 'SBP', 'DBP', ...]  # 21 features
```

Outputs:
- `baseline_train_matrices.npz`
- `baseline_val_matrices.npz`
- `baseline_test_matrices.npz`

### 3. Apply SMOTE (Optional)

```python
OUTPUT_DIR = '/path/to/dataset'
EMBEDDING_DIM = 64
BATCH_SIZE = 128
EPOCHS = 30
```

Outputs:
- `smoted_train_matrices_weighted_v2_clipped.npz`
- `lstm_autoencoder_weighted_masked.h5`
- `smote_quality_metrics_weighted_v2_clipped.json`

### 4. Apply Undersampling (Optional)

```python
OUTPUT_DIR = '/path/to/dataset'
RANDOM_STATE = 42
```

Outputs:
- `undersampled_train_matrices.npz`

### 5. Comprehensive Model Prototyping

```python
DATASETS = {
    'Original': {'path': '...', 'use_class_weight': True},
    'SMOTEv2': {'path': '...', 'use_class_weight': False},
    'Undersampled': {'path': '...', 'use_class_weight': False}
}

MODELS = ['LSTM', 'RNN', 'GRU', 'LSTM_Attention', 'CNN', 'Transformer']
OPTIMIZERS = ['adam', 'rmsprop']
ACTIVATIONS = ['relu', 'tanh', 'sigmoid', 'gelu']
```

Runs **144 experiments** (3 datasets × 6 models × 2 optimizers × 4 activations).

Outputs:
- `comprehensive_summary_complete.csv` - All experiment results
- `best_combinations.csv` - Best optimizer/activation per model
- Individual model files and metrics JSONs

## 🧠 Model Architectures

### 1. **LSTM**
- Masking layer for missing values
- 128 LSTM units
- Dropout (0.3)
- Dense layers with dropout

### 2. **RNN**
- Simple RNN with 128 units
- Similar architecture to LSTM

### 3. **GRU**
- GRU with 128 units
- Balanced between LSTM and RNN complexity

### 4. **LSTM with Attention**
- LSTM with return sequences
- Attention mechanism over time steps
- Weighted aggregation of temporal features

### 5. **CNN**
- 1D convolutions (kernel_size=2)
- 64 and 128 filters
- Global max pooling
- Dense classification head

### 6. **Transformer**
- Multi-head attention (4 heads)
- Positional embeddings
- Layer normalization
- Feed-forward network

## 📈 Results

The system generates comprehensive evaluation metrics:

- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Precision-Recall AUC (important for imbalanced data)
- **Sensitivity**: True positive rate (critical for medical applications)
- **Specificity**: True negative rate
- **Precision**: Positive predictive value
- **NPV**: Negative predictive value
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP, TN, FP, FN

Results are saved in:
- Individual experiment JSON files
- Comprehensive CSV summary
- Top-5 models per dataset
- Best combinations report

## 📁 File Structure

```
ai_sepsis_detection.py
├─ Preprocessing Section
│  ├─ Load data
│  ├─ Apply exclusion criteria
│  ├─ Hourly resampling
│  ├─ Patient-level split
│  ├─ IQR normalization
│  ├─ Feature engineering
│  └─ Save preprocessed CSVs
│
├─ Build Input Matrices Section
│  ├─ Load preprocessed CSVs
│  ├─ Create 4-hour sliding windows
│  └─ Save train/val/test matrices
│
├─ SMOTE Section
│  ├─ Train LSTM autoencoder (with masking)
│  ├─ Encode sequences to embeddings
│  ├─ Apply SMOTE
│  ├─ Decode back to sequences
│  └─ Save balanced matrices
│
├─ Undersampling Section
│  ├─ Load original matrices
│  ├─ Random undersample majority class
│  └─ Save balanced matrices
│
├─ Prototyping Section
│  ├─ Define 6 model architectures
│  ├─ Loop over datasets/models/optimizers/activations
│  ├─ Train with early stopping
│  ├─ Evaluate on validation set
│  └─ Save metrics and models
│
└─ Summary Generation Section
   ├─ Collect all metrics JSONs
   ├─ Generate comprehensive summary
   └─ Extract best combinations
```

## 🔧 Configuration

### Feature Hold Intervals
- **Vital signs**: 4 hours
- **Lab values**: 24 hours
- **Demographics**: Static (no forward-fill)

### Clinical Sanity Bounds
```python
clinical_sanity = {
    'HR': (30, 250),
    'SBP': (40, 250),
    'Temp': (30.0, 45.0),
    'O2Sat': (50, 100),
    # ...
}
```

### Training Hyperparameters
- **Epochs**: 50 (with early stopping)
- **Batch Size**: 256
- **Learning Rate**: 0.001
- **Dropout**: 0.3
- **LSTM/GRU/RNN Units**: 128
- **Early Stopping Patience**: 5 epochs

## 📝 Key Implementation Details

### Missing Value Handling
1. **Original missing mask** stored before forward-fill
2. **Time-limited forward-fill** applied per feature
3. **Normalization** applied to non-missing values
4. **Imputation to -1** for ALL originally missing values
5. **Binary mask columns** created (1=present, 0=missing)
6. **Masking layer** in neural networks to ignore -1 values

### Data Leakage Prevention
- IQR limits computed **only from training set**
- Applied consistently to val/test sets
- Patient-level split (no patient in multiple sets)
- Synthetic sample tracking in SMOTE

### Memory Efficiency
- **Memory-mapped loading** (`mmap_mode='r'`) for large files
- **Batch processing** for encoding/decoding
- **Mixed precision training** (float16) when available
- **Garbage collection** after each dataset/model
- **tf.data.Dataset** pipeline with prefetching

## 🎓 Citation

This implementation follows preprocessing and exclusion criteria from:
> *"Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019"*

## ⚠️ Important Notes

1. **Google Colab Specific**: The code uses Google Drive mounting. For local execution, update all file paths.

2. **GPU Recommended**: Training 144 experiments is computationally intensive. Use GPU acceleration for practical runtime.

3. **Checkpoint System**: The prototyping section includes checkpointing to resume interrupted experiments.

4. **Validation Strategy**: Uses held-out validation set for early stopping and model selection, with final test set reserved for final evaluation.

## 📧 Contact

For questions or issues with this implementation, please refer to the original Google Colab notebook or contact the project maintainer.

---

**Note**: This code represents a research implementation originally developed in Google Colab. Adapt file paths and configurations as needed for your environment.
