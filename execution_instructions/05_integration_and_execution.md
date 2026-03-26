# Module 5: End-to-End Integration and Execution

## Overview
This final instruction file dictates how all previous modules are logically linked to run the full GUARD architecture start to finish, reflecting the complete lifecycle from data ingestion to final comparative benchmarking.

## 1. Project Directory Structure
Ensure your codebase looks somewhat like this:
```text
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── denoising.py
│   ├── crc_layer.py
│   └── baselines.py
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── config.yaml
└── main.py
```

## 2. Integrated Execution Flow
Running the complete pipeline will involve the following chronological CLI commands:

### Step A: Data Setup
Trigger Module 1.
```bash
python scripts/preprocess.py --inject-noise True --sparsity-level 0.1
```

### Step B: Pre-training Baselines
Trigger Module 4 CF/CBF baselines to save standard weights.
```bash
python scripts/train.py --model CF
python scripts/train.py --model CBF
```

### Step C: Training GUARD base Model
Trigger Module 2.
```bash
python scripts/train.py --model GUARD_Transformer --epochs 50 --batch-size 256
```

### Step D: Attaching CRC and Evaluating
Trigger Module 3 and 4 together. The evaluation script seamlessly attaches the CRC guardrail to the GUARD_Transformer inference code dynamically.
```bash
python scripts/evaluate.py --apply-crc True --risk-threshold 0.05
```

## 3. Post-Execution Analysis
1. Upon running `evaluate.py`, analyze the generated `.csv` or `.json` to verify that GUARD has successfully attained lower MSE and Violation Rates compared to CF and CBF, especially in sparse data conditions.
2. Confirm the robustness by varying noise injection and observing if F1-Score reliably stays above 0.8.
