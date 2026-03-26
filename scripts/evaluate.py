"""
Evaluation Script
==================
Benchmarks CF, CBF, and GUARD (with optional CRC) on the test set.
Computes MSE, F1-Score, CTR, Violation Rate, and Compute Latency.

Usage:
    python scripts/evaluate.py --apply-crc True --risk-threshold 0.05
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.denoising import DenoisingTransformer
from models.baselines import CollaborativeFiltering, ContentBasedFiltering
from models.crc_layer import ConformalRiskControl


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def compute_mse(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((labels - scores) ** 2))


def compute_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_ctr(recommended_items: list, positive_items: set) -> float:
    """Fraction of recommendations that match a ground-truth positive."""
    if not recommended_items:
        return 0.0
    hits = sum(1 for iid in recommended_items if iid in positive_items)
    return hits / len(recommended_items)


def compute_violation_rate(recommended_items: list, risky_items: set) -> float:
    """Fraction of recommendations that belong to the risky set."""
    if not recommended_items:
        return 0.0
    violations = sum(1 for iid in recommended_items if iid in risky_items)
    return violations / len(recommended_items)


# ---------------------------------------------------------------------------
# Build risky-item set from movies metadata
# ---------------------------------------------------------------------------

def build_risky_item_set(movies_df: pd.DataFrame, risky_genres: list) -> set:
    risky = set()
    rg = set(risky_genres)
    for _, row in movies_df.iterrows():
        if pd.notna(row["genres"]):
            genres = set(row["genres"].split("|"))
            if genres & rg:
                risky.add(int(row["item_id"]))
    return risky


# ---------------------------------------------------------------------------
# Evaluate a single model
# ---------------------------------------------------------------------------

def evaluate_model(model_name, predict_fn, test_df, candidate_items,
                   top_k, risky_items, crc=None, tau=None):
    """
    Args:
        predict_fn:  callable(user_id, candidate_items, top_k) -> [(item_id, score), ...]
        crc:         optional ConformalRiskControl instance
    Returns:
        dict of metric values
    """
    users = test_df["user_id"].unique()

    all_scores = []
    all_labels = []
    all_ctr = []
    all_violation = []
    latencies = []

    # Ground-truth positives per user
    pos_by_user = test_df[test_df["label"] == 1].groupby("user_id")["item_id"].apply(set).to_dict()

    # For MSE/F1 we score every (user, item) in test set
    for uid in users:
        user_test = test_df[test_df["user_id"] == uid]
        labels = user_test["label"].values
        items = user_test["item_id"].values.tolist()

        t0 = time.perf_counter()
        recs = predict_fn(uid, candidate_items, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Apply CRC if provided
        if crc is not None:
            recs = crc.filter_recommendations(recs, tau)

        latencies.append(latency_ms)

        # Recommended item IDs
        rec_ids = [r[0] for r in recs]
        rec_scores_map = {r[0]: r[1] for r in recs}

        # CTR
        positives = pos_by_user.get(uid, set())
        all_ctr.append(compute_ctr(rec_ids, positives))

        # Violation rate
        all_violation.append(compute_violation_rate(rec_ids, risky_items))

        # Per-interaction scores for MSE / F1
        for iid, lab in zip(items, labels):
            score = rec_scores_map.get(iid, 0.0)
            all_scores.append(score)
            all_labels.append(lab)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    preds_binary = (all_scores >= 0.5).astype(int)

    metrics = {
        "model": model_name,
        "MSE": round(compute_mse(all_labels, all_scores), 4),
        "F1": round(compute_f1(all_labels, preds_binary), 4),
        "CTR": round(float(np.mean(all_ctr)), 4),
        "Violation_Rate_%": round(float(np.mean(all_violation)) * 100, 4),
        "Latency_ms": round(float(np.mean(latencies)), 2),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GUARD Evaluation & Benchmarking")
    parser.add_argument("--apply-crc", type=str, default="False")
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    proc_dir = cfg["data"]["processed_dir"]
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    top_k = cfg["evaluation"]["top_k"]
    apply_crc = args.apply_crc.lower() in ("true", "1", "yes")
    tau = args.risk_threshold if args.risk_threshold is not None else cfg["crc"]["risk_threshold"]

    # ── Load data ────────────────────────────────────────────────────────
    test_df = pd.read_csv(os.path.join(proc_dir, "test.csv"))
    movies_df = pd.read_csv(os.path.join(proc_dir, "movies.csv"))
    candidate_items = sorted(test_df["item_id"].unique().tolist())

    risky_items = build_risky_item_set(movies_df, cfg["crc"]["risky_genres"])
    print(f"[INFO] Test set: {len(test_df)} interactions, "
          f"{len(candidate_items)} candidate items, {len(risky_items)} risky items.")

    # ── Optionally build CRC layer ───────────────────────────────────────
    crc = None
    if apply_crc:
        print("[INFO] Building Conformal Risk Control layer …")
        crc = ConformalRiskControl(
            movies_df=movies_df,
            risky_genres=cfg["crc"]["risky_genres"],
            safe_genres=cfg["crc"]["safe_genres"],
            risk_threshold=tau,
            kde_bandwidth=cfg["crc"]["kde_bandwidth"],
        )

    # ── Evaluate Collaborative Filtering ─────────────────────────────────
    results = []

    cf_path = os.path.join(ckpt_dir, "cf_model.pkl")
    if os.path.exists(cf_path):
        print("\n[EVAL] Collaborative Filtering")
        with open(cf_path, "rb") as f:
            cf_model = pickle.load(f)
        m = evaluate_model("CF", cf_model.predict, test_df, candidate_items,
                           top_k, risky_items)
        results.append(m)
        print(f"       {m}")
    else:
        print("[WARN] CF model not found, skipping.")

    # ── Evaluate Content-Based Filtering ─────────────────────────────────
    cbf_path = os.path.join(ckpt_dir, "cbf_model.pkl")
    if os.path.exists(cbf_path):
        print("\n[EVAL] Content-Based Filtering")
        with open(cbf_path, "rb") as f:
            cbf_model = pickle.load(f)
        m = evaluate_model("CBF", cbf_model.predict, test_df, candidate_items,
                           top_k, risky_items)
        results.append(m)
        print(f"       {m}")
    else:
        print("[WARN] CBF model not found, skipping.")

    # ── Evaluate GUARD Transformer ───────────────────────────────────────
    guard_path = os.path.join(ckpt_dir, "denoising_transformer.pt")
    if os.path.exists(guard_path):
        print("\n[EVAL] GUARD Denoising Transformer" +
              (" + CRC" if apply_crc else ""))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(guard_path, map_location=device, weights_only=False)

        guard_model = DenoisingTransformer(
            num_users=ckpt["num_users"],
            num_items=ckpt["num_items"],
            embed_dim=ckpt["config"]["embedding_dim"],
            num_heads=ckpt["config"]["num_attention_heads"],
            ffn_dim=ckpt["config"]["ffn_dim"],
            dropout=ckpt["config"]["dropout"],
            mask_threshold=ckpt["config"]["mask_threshold"],
        ).to(device)
        guard_model.load_state_dict(ckpt["model_state_dict"])
        guard_model.eval()

        user_map = ckpt["user_map"]
        item_map = ckpt["item_map"]
        inv_item_map = {v: k for k, v in item_map.items()}

        def guard_predict(user_id, cand_items, k):
            uid_mapped = user_map.get(user_id)
            if uid_mapped is None:
                return [(cand_items[i], 0.0) for i in range(min(k, len(cand_items)))]
            mapped_items = []
            orig_items = []
            for iid in cand_items:
                m = item_map.get(iid)
                if m is not None:
                    mapped_items.append(m)
                    orig_items.append(iid)
            if not mapped_items:
                return [(cand_items[i], 0.0) for i in range(min(k, len(cand_items)))]

            with torch.no_grad():
                u_t = torch.tensor([uid_mapped] * len(mapped_items), device=device)
                i_t = torch.tensor(mapped_items, device=device)
                scores = guard_model(u_t, i_t).cpu().numpy()

            scored = sorted(zip(orig_items, scores.tolist()),
                            key=lambda x: x[1], reverse=True)
            return scored[:k]

        m = evaluate_model(
            "GUARD" + ("+CRC" if apply_crc else ""),
            guard_predict, test_df, candidate_items,
            top_k, risky_items, crc=crc if apply_crc else None, tau=tau,
        )
        results.append(m)
        print(f"       {m}")
    else:
        print("[WARN] GUARD model not found, skipping.")

    # ── Save benchmarks ─────────────────────────────────────────────────
    if results:
        out_base = cfg["evaluation"]["output_file"]
        results_df = pd.DataFrame(results)
        results_df.to_csv(out_base + ".csv", index=False)
        with open(out_base + ".json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[DONE] Benchmarks saved to {out_base}.csv and {out_base}.json")
        print("\n" + results_df.to_string(index=False))
    else:
        print("[WARN] No models evaluated.")


if __name__ == "__main__":
    main()
