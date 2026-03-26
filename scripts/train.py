"""
Training Script
================
Unified training entry-point for CF, CBF, and GUARD Transformer models.

Usage:
    python scripts/train.py --model CF
    python scripts/train.py --model CBF
    python scripts/train.py --model GUARD_Transformer --epochs 50 --batch-size 256
"""

import argparse
import os
import sys
import pickle
import time

import numpy as np
import pandas as pd
import yaml

# ── PyTorch (only needed for GUARD) ──────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Project imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.denoising import DenoisingTransformer
from models.baselines import CollaborativeFiltering, ContentBasedFiltering


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class InteractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------

def train_cf(cfg):
    print("\n" + "=" * 60)
    print("Training Collaborative Filtering (Matrix Factorisation)")
    print("=" * 60)
    train_df = pd.read_csv(os.path.join(cfg["data"]["processed_dir"], "train.csv"))

    model = CollaborativeFiltering(
        num_factors=cfg["model"]["embedding_dim"],
        lr=cfg["training"]["learning_rate"],
        reg=cfg["training"]["weight_decay"],
        epochs=min(cfg["training"]["epochs"], 20),  # CF converges faster
    )
    model.fit(train_df)

    out_path = os.path.join(cfg["training"]["checkpoint_dir"], "cf_model.pkl")
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[DONE] CF model saved to {out_path}")


def train_cbf(cfg):
    print("\n" + "=" * 60)
    print("Training Content-Based Filtering")
    print("=" * 60)
    train_df = pd.read_csv(os.path.join(cfg["data"]["processed_dir"], "train.csv"))

    model = ContentBasedFiltering()
    model.fit(train_df)

    out_path = os.path.join(cfg["training"]["checkpoint_dir"], "cbf_model.pkl")
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[DONE] CBF model saved to {out_path}")


# ---------------------------------------------------------------------------
# GUARD Transformer training
# ---------------------------------------------------------------------------

def train_guard(cfg, epochs_override=None, batch_size_override=None, lr_override=None):
    print("\n" + "=" * 60)
    print("Training GUARD Denoising Transformer")
    print("=" * 60)

    proc_dir = cfg["data"]["processed_dir"]
    train_df = pd.read_csv(os.path.join(proc_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(proc_dir, "val.csv"))

    # Build contiguous ID mappings
    all_users = sorted(set(train_df["user_id"]) | set(val_df["user_id"]))
    all_items = sorted(set(train_df["item_id"]) | set(val_df["item_id"]))
    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {it: i for i, it in enumerate(all_items)}

    train_df["user_id_mapped"] = train_df["user_id"].map(user_map)
    train_df["item_id_mapped"] = train_df["item_id"].map(item_map)
    val_df["user_id_mapped"] = val_df["user_id"].map(user_map)
    val_df["item_id_mapped"] = val_df["item_id"].map(item_map)

    # Drop rows with unmapped IDs (if any)
    train_df = train_df.dropna(subset=["user_id_mapped", "item_id_mapped"])
    val_df = val_df.dropna(subset=["user_id_mapped", "item_id_mapped"])
    train_df["user_id_mapped"] = train_df["user_id_mapped"].astype(int)
    train_df["item_id_mapped"] = train_df["item_id_mapped"].astype(int)
    val_df["user_id_mapped"] = val_df["user_id_mapped"].astype(int)
    val_df["item_id_mapped"] = val_df["item_id_mapped"].astype(int)

    num_users = len(all_users)
    num_items = len(all_items)

    epochs = epochs_override or cfg["training"]["epochs"]
    batch_size = batch_size_override or cfg["training"]["batch_size"]
    lr = lr_override or cfg["training"]["learning_rate"]
    patience = cfg["training"]["patience"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model = DenoisingTransformer(
        num_users=num_users,
        num_items=num_items,
        embed_dim=cfg["model"]["embedding_dim"],
        num_heads=cfg["model"]["num_attention_heads"],
        ffn_dim=cfg["model"]["ffn_dim"],
        dropout=cfg["model"]["dropout"],
        mask_threshold=cfg["model"]["mask_threshold"],
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=cfg["training"]["weight_decay"])

    # Datasets
    train_mapped = train_df.rename(columns={"user_id_mapped": "user_id",
                                             "item_id_mapped": "item_id"})[["user_id", "item_id", "label"]]
    val_mapped = val_df.rename(columns={"user_id_mapped": "user_id",
                                         "item_id_mapped": "item_id"})[["user_id", "item_id", "label"]]
    train_loader = DataLoader(InteractionDataset(train_mapped),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(InteractionDataset(val_mapped),
                            batch_size=batch_size)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "denoising_transformer.pt")

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for users, items, labels in train_loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            preds = model(users, items)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_loader.dataset)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for users, items, labels in val_loader:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                preds = model(users, items)
                loss = criterion(preds, labels)
                val_loss += loss.item() * len(labels)
        val_loss /= len(val_loader.dataset)

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "user_map": user_map,
                "item_map": item_map,
                "num_users": num_users,
                "num_items": num_items,
                "config": cfg["model"],
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}.")
                break

    print(f"[DONE] Best val_loss={best_val_loss:.4f} — saved to {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GUARD Model Training")
    parser.add_argument("--model", type=str, required=True,
                        choices=["CF", "CBF", "GUARD_Transformer"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.model == "CF":
        train_cf(cfg)
    elif args.model == "CBF":
        train_cbf(cfg)
    elif args.model == "GUARD_Transformer":
        train_guard(cfg, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
