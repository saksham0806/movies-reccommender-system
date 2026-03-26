"""
GUARD Recommendation System — End-to-End Pipeline
===================================================
Orchestrates the full lifecycle:
  1. Data preprocessing (with noise injection + sparsity)
  2. Baseline training (CF, CBF)
  3. GUARD Transformer training
  4. Evaluation with CRC guardrail

Usage:
    python main.py
    python main.py --skip-preprocess --epochs 10
"""

import argparse
import subprocess
import sys
import os
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(cmd: list, desc: str):
    """Run a subprocess and stream stdout/stderr."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {desc}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="GUARD End-to-End Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip data preprocessing (use existing processed data)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override GUARD training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--noise-rate", type=float, default=None)
    parser.add_argument("--sparsity-level", type=float, default=None)
    parser.add_argument("--risk-threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    python = sys.executable

    # ── Step A: Data Preprocessing ───────────────────────────────────────
    if not args.skip_preprocess:
        preprocess_cmd = [
            python, "scripts/preprocess.py",
            "--inject-noise", "True",
            "--config", args.config,
        ]
        if args.noise_rate is not None:
            preprocess_cmd += ["--noise-rate", str(args.noise_rate)]
        if args.sparsity_level is not None:
            preprocess_cmd += ["--sparsity-level", str(args.sparsity_level)]
        run(preprocess_cmd, "Step A: Data Preprocessing")
    else:
        print("[INFO] Skipping preprocessing — using existing processed data.")

    # ── Step B: Train Baselines ──────────────────────────────────────────
    run([python, "scripts/train.py", "--model", "CF", "--config", args.config],
        "Step B: Training Collaborative Filtering")

    run([python, "scripts/train.py", "--model", "CBF", "--config", args.config],
        "Step B: Training Content-Based Filtering")

    # ── Step C: Train GUARD Transformer ──────────────────────────────────
    guard_cmd = [python, "scripts/train.py", "--model", "GUARD_Transformer",
                 "--config", args.config]
    if args.epochs is not None:
        guard_cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        guard_cmd += ["--batch-size", str(args.batch_size)]
    run(guard_cmd, "Step C: Training GUARD Denoising Transformer")

    # ── Step D: Evaluate with CRC ────────────────────────────────────────
    eval_cmd = [python, "scripts/evaluate.py",
                "--apply-crc", "True",
                "--config", args.config]
    if args.risk_threshold is not None:
        eval_cmd += ["--risk-threshold", str(args.risk_threshold)]
    run(eval_cmd, "Step D: Evaluation with CRC Guardrail")

    print("\n" + "=" * 60)
    print("  GUARD Pipeline Complete!")
    print("=" * 60)
    print(f"  Benchmarks saved to {cfg['evaluation']['output_file']}.csv / .json")


if __name__ == "__main__":
    main()
