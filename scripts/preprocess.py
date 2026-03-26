"""
Module 1: Data Preparation Pipeline
====================================
Downloads MovieLens 1M, converts to implicit feedback, injects noise,
simulates sparsity, and produces train/val/test splits.

Usage:
    python scripts/preprocess.py --inject-noise True --sparsity-level 0.1
"""

import argparse
import os
import sys
import zipfile
import urllib.request
import shutil

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_movielens(url: str, raw_dir: str):
    """Download and extract MovieLens 1M if not already present."""
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "ml-1m.zip")
    extract_dir = os.path.join(raw_dir, "ml-1m")

    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"[INFO] MovieLens 1M already exists at {extract_dir}, skipping download.")
        return extract_dir

    print(f"[INFO] Downloading MovieLens 1M from {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"[INFO] Extracting to {raw_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    os.remove(zip_path)
    print("[INFO] Download complete.")
    return extract_dir


def load_ratings(data_dir: str) -> pd.DataFrame:
    """Load ratings.dat (UserID::MovieID::Rating::Timestamp)."""
    path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    return df


def load_movies(data_dir: str) -> pd.DataFrame:
    """Load movies.dat (MovieID::Title::Genres)."""
    path = os.path.join(data_dir, "movies.dat")
    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["item_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    return df


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------

def convert_to_implicit(ratings: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """Convert explicit ratings to implicit binary feedback."""
    ratings = ratings.copy()
    ratings["label"] = (ratings["rating"] >= threshold).astype(int)
    return ratings


def simulate_sparsity(df: pd.DataFrame, keep_frac: float, seed: int) -> pd.DataFrame:
    """Drop a fraction of interactions to simulate data sparsity."""
    if keep_frac >= 1.0:
        return df
    return df.sample(frac=keep_frac, random_state=seed).reset_index(drop=True)


def inject_noise(df: pd.DataFrame, noise_rate: float, seed: int) -> pd.DataFrame:
    """Randomly flip a percentage of labels to simulate false-positive noise."""
    if noise_rate <= 0.0:
        return df
    rng = np.random.RandomState(seed)
    n_flip = int(len(df) * noise_rate)
    flip_idx = rng.choice(df.index, size=n_flip, replace=False)
    df = df.copy()
    df.loc[flip_idx, "label"] = 1 - df.loc[flip_idx, "label"]
    print(f"[INFO] Injected noise: flipped {n_flip} labels ({noise_rate*100:.1f}%).")
    return df


def attach_genres(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Merge genre information onto the ratings dataframe."""
    merged = ratings.merge(movies[["item_id", "genres"]], on="item_id", how="left")
    merged["genres"] = merged["genres"].fillna("Unknown")
    return merged


def split_data(df: pd.DataFrame, train_r: float, val_r: float, seed: int):
    """Random train / val / test split."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    t1 = int(n * train_r)
    t2 = int(n * (train_r + val_r))
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GUARD Data Preprocessing")
    parser.add_argument("--inject-noise", type=str, default="True",
                        help="Whether to inject noise (True/False)")
    parser.add_argument("--noise-rate", type=float, default=None,
                        help="Override noise rate from config")
    parser.add_argument("--sparsity-level", type=float, default=None,
                        help="Override sparsity level from config")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prep = cfg["preprocessing"]
    seed = prep["random_seed"]

    noise_rate = args.noise_rate if args.noise_rate is not None else prep["noise_rate"]
    sparsity = args.sparsity_level if args.sparsity_level is not None else prep["sparsity_level"]
    do_noise = args.inject_noise.lower() in ("true", "1", "yes")

    # Step 1 – Download
    data_dir = download_movielens(cfg["data"]["movielens_url"], cfg["data"]["raw_dir"])

    # Step 2 – Load
    print("[INFO] Loading ratings …")
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    print(f"[INFO] Loaded {len(ratings)} ratings, {len(movies)} movies.")

    # Step 3 – Convert to implicit
    ratings = convert_to_implicit(ratings, prep["positive_threshold"])

    # Step 4 – Attach genres
    ratings = attach_genres(ratings, movies)

    # Step 5 – Sparsity simulation
    keep_frac = 1.0 - sparsity
    ratings = simulate_sparsity(ratings, keep_frac, seed)
    print(f"[INFO] After sparsity simulation: {len(ratings)} interactions.")

    # Step 6 – Noise injection
    if do_noise:
        ratings = inject_noise(ratings, noise_rate, seed)

    # Step 7 – Split
    train, val, test = split_data(
        ratings, prep["train_ratio"], prep["val_ratio"], seed
    )
    print(f"[INFO] Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    # Step 8 – Save
    out_dir = cfg["data"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cols = ["user_id", "item_id", "label", "genres"]
    train[cols].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val[cols].to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test[cols].to_csv(os.path.join(out_dir, "test.csv"), index=False)

    # Also save movies metadata for CRC layer
    movies.to_csv(os.path.join(out_dir, "movies.csv"), index=False)

    print(f"[INFO] Saved processed data to {out_dir}/")
    print("[DONE] Preprocessing complete.")


if __name__ == "__main__":
    main()
