# Module 2: Denoising Self-Attention Transformer

## Overview
This module implements the primary recommendation model. It leverages a Denoising Self-Attention Transformer (Wang et al., 2020; Chen et al., 2022) to filter out mistaken inputs or unnecessary noise from implicit feedback, maximizing recommendation accuracy.

## 1. Architecture Implementation
We will build a deep-learning embedding layer followed by an Attention mechanism.
- **Input Embeddings**: Convert user sequences and item descriptors into dense vector representations.
- **Adaptive Binary Masking**: Implement binary masks applied to the self-attention heads explicitly designed to prune or drop out identified false-positive features or anomaly interactions.
- **Feed-Forward Network (FFN)**: Pass the masked attention output into an FFN for higher-level feature fusion.

## 2. Model Training Procedure
The model will learn to recommend items while minimizing error metrics.

### Optimization & Loss
1. Set up the Cross Entropy Loss (`L_ce`) function:
   - Calculate predicting ratings vs ideal ratings (ground truth).
   - The model must minimize this metric alongside mitigating noise.
2. Initialize Adam or AdamW optimizers.

### Training Loop
1. Load `data/processed/train.csv` in batches using `DataLoader`.
2. Forward pass: Feed sequential implicit feedback to the transformer.
3. Compute adaptive binary masks dynamically across data points.
4. Backpropagate the Cross Entropy Loss and update layer weights.
5. Save the best model checkpoints to `models/denoising_transformer.pt` based on Validation Cross Entropy Loss.

## 3. Model Inference
Write an inference function (`predict(user_id)`) that outputs a list of recommended items `Y_i` corresponding to user queries through the denoised neural pathways.
