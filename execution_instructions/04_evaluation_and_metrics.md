# Module 4: Evaluation and Metrics

## Overview
In this module, we benchmark the complete GUARD system against current industry standard implementations to prove algorithmic superiority in handling noisy data, computation latency, and user risk mitigation. 

## 1. Baseline Implementation
Prepare two classic baseline models to benchmark our proposed architecture against:
- **Collaborative Filtering (CF)**: Implementation utilizing matrix factorization based on past user experiences.
- **Content-Based Filtering (CBF)**: Implementation utilizing content-driven keywords, genres, and attributes.

## 2. Defining Evaluation Metrics
You must implement computation hooks for the following metrics:
- **Predictive Error (MSE)**: Mean Squared Error between the predicted interaction values and the test dataset ground truth. Ideal for GUARD: $\approx 0.542$.
- **Accuracy (F1-Score)**: Harmonic mean of Precision and Recall. Ideal for GUARD: $\approx 0.824$.
- **User Engagement (CTR)**: Click-through rate indicating how many recommendations lead to an interaction. Ideal for GUARD: $\approx 0.076$. Note that this might be marginally lower than CF due to filtering misclicks.
- **Violation Rate (Safety %)**: Percentage of risky/unsafe contents actively recommended. Ideal for GUARD: $\approx 0.08\%$ or lower.
- **Compute Latency**: Amount of milliseconds required to do inference. Ideal for GUARD: $\approx 26.7$ ms.

## 3. Evaluation Execution
1. Load `data/processed/test.csv`.
2. Generate top $K$ recommendations using CBF, CF, and GUARD (Denoising + CRC).
3. Compute the evaluation metrics across all three architectures over sparse simulated data conditions.
4. Output a serialized metrics report (`benchmarks.json` or `.csv`) displaying comparative performance.
