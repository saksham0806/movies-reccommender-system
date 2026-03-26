# Module 1: Data Preparation Pipeline

## Overview
The first module involves preparing the environment and the datasets necessary to train and benchmark the GUARD (Guided User Attention and Risk Denoising) recommendation system. The system relies on handling both noise and sparse data for robust industrial-scale applications.

## 1. Environment Setup
- **Python**: Ensure Python 3.8+ is installed.
- **Dependencies**: Install PyTorch/TensorFlow, pandas, numpy, scikit-learn, and other data processing libraries.
```bash
pip install torch pandas numpy scikit-learn scipy
```

## 2. Dataset Acquisition
The system evaluation uses the **MovieLens** dataset to evaluate both predictive accuracy (MSE), user engagement (CTR), and risk mitigation via its rich genre/tag metadata.

### Instructions
1. Download the **MovieLens 1M** dataset (or ML-20M) from [GroupLens](https://grouplens.org/datasets/movielens/). This includes user ratings and robust movie metadata (Genres).
2. Extract the dataset (e.g., `ratings.dat`, `movies.dat`, `users.dat`) into a centralized `data/raw/` directory.

## 3. Data Preprocessing & Noise Injection
To effectively test the denoising layer, we must simulate "noisy" datasets (misclicks, exploratory clicks) and sparse interactions.

### Steps
1. **Load Data**: Implement scripts to load user-item ratings from `ratings.dat` using `pandas`. Convert explicit ratings into implicit feedback (e.g., ratings >= 4 is a positive interaction, ratings < 4 is a negative interaction).
2. **Sparsity Simulation**: Sample a fraction of interactions to simulate the exact data sparsity found in industrial deployments.
3. **Noise Injection**: Randomly perturb a percentage (e.g., 5-15%) of user interaction records to act as "false positive" noise or accidental clicks.
4. **Train/Validation/Test Split**: Standardize a chronological or random split function (e.g., 80% train, 10% validation, 10% test).

## 4. Output validation
Ensure that preprocessed datasets (`train.csv`, `val.csv`, `test.csv`) are saved in a structured format in `data/processed/` that can be loaded natively by the main `DataLoader` scripts in subsequent modules.
