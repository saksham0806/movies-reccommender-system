"""
Module 4 (partial): Baseline Models
=====================================
Collaborative Filtering (Matrix Factorisation) and Content-Based Filtering
used as benchmarks against the GUARD Denoising Transformer.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------------
# Collaborative Filtering — SVD-style Matrix Factorisation
# ---------------------------------------------------------------------------

class CollaborativeFiltering:
    """
    Simple matrix-factorisation collaborative filter using SGD.
    Learns latent factors for users and items to predict interactions.
    """

    def __init__(self, num_factors: int = 64, lr: float = 0.005,
                 reg: float = 0.02, epochs: int = 20):
        self.num_factors = num_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0

    def fit(self, train_df: pd.DataFrame):
        """
        Train on a DataFrame with columns [user_id, item_id, label].
        """
        users = train_df["user_id"].values
        items = train_df["item_id"].values
        labels = train_df["label"].values.astype(np.float32)

        self.user_map = {u: i for i, u in enumerate(sorted(set(users)))}
        self.item_map = {it: i for i, it in enumerate(sorted(set(items)))}
        self.inv_item_map = {v: k for k, v in self.item_map.items()}
        n_users = len(self.user_map)
        n_items = len(self.item_map)

        rng = np.random.RandomState(42)
        self.user_factors = rng.normal(0, 0.1, (n_users, self.num_factors))
        self.item_factors = rng.normal(0, 0.1, (n_items, self.num_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = labels.mean()

        for epoch in range(self.epochs):
            perm = rng.permutation(len(labels))
            total_loss = 0.0
            for idx in perm:
                u = self.user_map.get(users[idx])
                it = self.item_map.get(items[idx])
                if u is None or it is None:
                    continue
                pred = (self.global_mean + self.user_bias[u] + self.item_bias[it]
                        + self.user_factors[u] @ self.item_factors[it])
                pred = 1.0 / (1.0 + np.exp(-pred))  # sigmoid
                err = labels[idx] - pred

                self.user_factors[u] += self.lr * (err * self.item_factors[it]
                                                    - self.reg * self.user_factors[u])
                self.item_factors[it] += self.lr * (err * self.user_factors[u]
                                                     - self.reg * self.item_factors[it])
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[it] += self.lr * (err - self.reg * self.item_bias[it])
                total_loss += err ** 2

            mse = total_loss / len(labels)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [CF] Epoch {epoch+1}/{self.epochs}  MSE={mse:.4f}")

    def predict_score(self, user_id: int, item_id: int) -> float:
        u = self.user_map.get(user_id)
        it = self.item_map.get(item_id)
        if u is None or it is None:
            return self.global_mean
        raw = (self.global_mean + self.user_bias[u] + self.item_bias[it]
               + self.user_factors[u] @ self.item_factors[it])
        return 1.0 / (1.0 + np.exp(-raw))

    def predict(self, user_id: int, candidate_items: list, top_k: int = 10) -> list:
        """Return top-K (item_id, score) for a user."""
        scores = [(iid, self.predict_score(user_id, iid)) for iid in candidate_items]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Content-Based Filtering — Genre cosine similarity
# ---------------------------------------------------------------------------

class ContentBasedFiltering:
    """
    Content-based recommender using genre vectors.
    Builds a user profile from liked items, then ranks candidates by
    cosine similarity to the user profile.
    """

    def __init__(self):
        self.genre_list = []
        self.genre_to_idx = {}
        self.item_vectors = {}
        self.user_profiles = {}

    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame = None):
        """
        Build item genre vectors and user profiles.

        Args:
            train_df:  DataFrame with [user_id, item_id, label, genres]
            movies_df: Optional movies DataFrame (unused if genres in train_df)
        """
        # Build genre vocabulary from training data
        all_genres = set()
        for g in train_df["genres"].dropna():
            all_genres.update(g.split("|"))
        self.genre_list = sorted(all_genres)
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_list)}
        n = len(self.genre_list)

        # Item vectors
        for _, row in train_df.drop_duplicates("item_id").iterrows():
            vec = np.zeros(n)
            if pd.notna(row["genres"]):
                for g in row["genres"].split("|"):
                    if g in self.genre_to_idx:
                        vec[self.genre_to_idx[g]] = 1.0
            self.item_vectors[int(row["item_id"])] = vec

        # User profiles — mean of liked-item vectors
        liked = train_df[train_df["label"] == 1]
        for uid, grp in liked.groupby("user_id"):
            vecs = []
            for iid in grp["item_id"]:
                if int(iid) in self.item_vectors:
                    vecs.append(self.item_vectors[int(iid)])
            if vecs:
                self.user_profiles[int(uid)] = np.mean(vecs, axis=0)
            else:
                self.user_profiles[int(uid)] = np.zeros(n)

        print(f"  [CBF] Built {len(self.user_profiles)} user profiles, "
              f"{len(self.item_vectors)} item vectors.")

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        d = np.linalg.norm(a) * np.linalg.norm(b)
        if d < 1e-9:
            return 0.0
        return float(np.dot(a, b) / d)

    def predict_score(self, user_id: int, item_id: int) -> float:
        profile = self.user_profiles.get(user_id)
        vec = self.item_vectors.get(item_id)
        if profile is None or vec is None:
            return 0.0
        return self._cosine(profile, vec)

    def predict(self, user_id: int, candidate_items: list, top_k: int = 10) -> list:
        scores = [(iid, self.predict_score(user_id, iid)) for iid in candidate_items]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
