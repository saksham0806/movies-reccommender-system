"""
Module 3: Conformal Risk Control (CRC) Layer
==============================================
Runtime guardrail that filters unsafe/risky recommendations using
Kernel Density Estimation over genre feature vectors.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


class ConformalRiskControl:
    """
    Attaches to the GUARD Transformer inference stage.
    Uses KDE to estimate the probability that a recommended item falls in the
    risky genre space, and swaps unsafe items with safe alternatives.
    """

    def __init__(self, movies_df: pd.DataFrame,
                 risky_genres: list, safe_genres: list,
                 risk_threshold: float = 0.05,
                 kde_bandwidth: float = 0.5):
        """
        Args:
            movies_df:       DataFrame with columns [item_id, title, genres]
            risky_genres:    List of genre strings considered risky
            safe_genres:     List of genre strings considered safe
            risk_threshold:  tau — items with R_i > tau are swapped
            kde_bandwidth:   Bandwidth for KDE
        """
        self.risk_threshold = risk_threshold
        self.risky_genres = set(risky_genres)
        self.safe_genres = set(safe_genres)

        # Build genre vocabulary
        all_genres = set()
        for g in movies_df["genres"].dropna():
            all_genres.update(g.split("|"))
        self.genre_list = sorted(all_genres)
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_list)}
        self.num_genres = len(self.genre_list)

        # Build item → genre vector mapping
        self.item_genre_vec = {}
        for _, row in movies_df.iterrows():
            vec = self._genre_vector(row["genres"])
            self.item_genre_vec[int(row["item_id"])] = vec

        # Build safe item pool
        self.safe_pool = self._build_safe_pool(movies_df)

        # Fit KDE on risky items
        self.kde = self._fit_kde(movies_df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _genre_vector(self, genres_str: str) -> np.ndarray:
        vec = np.zeros(self.num_genres, dtype=np.float64)
        if pd.isna(genres_str):
            return vec
        for g in genres_str.split("|"):
            if g in self.genre_to_idx:
                vec[self.genre_to_idx[g]] = 1.0
        return vec

    def _build_safe_pool(self, movies_df: pd.DataFrame) -> list:
        """Items whose genres include at least one safe genre and zero risky genres."""
        safe_items = []
        for _, row in movies_df.iterrows():
            genres = set(row["genres"].split("|")) if pd.notna(row["genres"]) else set()
            has_safe = bool(genres & self.safe_genres)
            has_risky = bool(genres & self.risky_genres)
            if has_safe and not has_risky:
                safe_items.append(int(row["item_id"]))
        return safe_items

    def _fit_kde(self, movies_df: pd.DataFrame):
        """Fit a KDE on genre vectors of items that contain risky genres."""
        risky_vecs = []
        for _, row in movies_df.iterrows():
            genres = set(row["genres"].split("|")) if pd.notna(row["genres"]) else set()
            if genres & self.risky_genres:
                risky_vecs.append(self._genre_vector(row["genres"]))
        if len(risky_vecs) < 2:
            return None  # Not enough data for KDE
        data = np.array(risky_vecs).T  # shape (n_features, n_samples)
        # Add small jitter to avoid singular matrices
        data += np.random.RandomState(42).normal(0, 1e-6, data.shape)
        try:
            return gaussian_kde(data, bw_method="silverman")
        except np.linalg.LinAlgError:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_risk(self, item_id: int) -> float:
        """
        Compute p_h(Y_i) — the likelihood that item_id falls in the risky
        genre vector space.  Returns residual risk R_i in [0, 1].
        """
        if self.kde is None:
            return 0.0

        vec = self.item_genre_vec.get(item_id)
        if vec is None:
            return 0.0

        # KDE probability density (may be > 1 for high-dimensional densities)
        density = float(self.kde.evaluate(vec.reshape(-1, 1))[0])
        # Normalise via sigmoid-like mapping to [0, 1]
        risk = 1.0 / (1.0 + np.exp(-density))
        return risk

    def filter_recommendations(self, rec_list: list,
                                tau: float = None) -> list:
        """
        Filter a ranked recommendation list.

        Args:
            rec_list:  List of (item_id, score) tuples (descending by score)
            tau:       Override risk threshold (default: self.risk_threshold)

        Returns:
            Filtered list of (item_id, score) tuples with unsafe items swapped.
        """
        if tau is None:
            tau = self.risk_threshold

        filtered = []
        used_safe = set()
        safe_idx = 0

        for item_id, score in rec_list:
            risk = self.compute_risk(item_id)
            if risk > tau:
                # Swap with a safe alternative
                replacement = self._get_safe_replacement(used_safe, safe_idx)
                if replacement is not None:
                    filtered.append((replacement, score))
                    used_safe.add(replacement)
                    safe_idx += 1
                else:
                    # Fallback: keep item but flag it (exhausted safe pool)
                    filtered.append((item_id, score))
            else:
                filtered.append((item_id, score))

        return filtered

    def _get_safe_replacement(self, used: set, start_idx: int):
        """Pick the next unused safe item."""
        for i in range(start_idx, len(self.safe_pool)):
            item = self.safe_pool[i]
            if item not in used:
                return item
        return None
