import pandas as pd
import numpy as np
from collections import defaultdict

from .base_comparison import BaseComparison


class AHPWeights(BaseComparison):
    """
    Analytic Hierarchy Process (AHP) weights for pairwise image comparisons.

    Inherits shared data-loading, filtering, and normalisation logic
    from BaseComparison and only implements what is unique to AHP.

    Three internal backends are supported via the *method* parameter:
        "dict"      — sparse dict-based (default, memory-efficient)
        "matrix"    — dense NumPy matrix
        "dataframe" — pandas-based (falls back to dict if not implemented)
    """

    # ------------------------------------------------------------------
    # Override prepare_matches to build index structures required by AHP
    # ------------------------------------------------------------------

    def prepare_matches(self, metric="safety"):
        super().prepare_matches(metric=metric)
        self.image_ids = np.sort(self.images_df["image_id"].unique()).tolist()
        self.image_to_idx = {img: idx for idx, img in enumerate(self.image_ids)}

    # ------------------------------------------------------------------
    # BaseComparison interface
    # ------------------------------------------------------------------

    def calculate(self, metric="safety", method="dict"):
        """
        Compute AHP priority vector for *metric*.

        Parameters
        ----------
        metric : str
        method : str   "dict" | "matrix" | "dataframe"
        """
        self.prepare_matches(metric=metric)
        df_ = self.get_matches().copy()
        print(f"Analyzing {df_.shape[0]} '{metric}' comparisons")

        self._build_votes(df_, method=method)
        self._build_ahp(method=method)
        self._compute_priority_vector(method=method)

    def normalize(self, normalize=True, min_range=0, max_range=10,
                  epsilon=0.0, **kwargs):
        """Build self.scores_df with raw AHPweight and normalised AHPScore."""
        df_ = pd.DataFrame({
            "image_id": self.image_ids,
            "AHPweight": self.priority_vector.tolist()
        })
        df_ = self.normalize_scores(
            df_, raw_col="AHPweight", score_col="AHPScore",
            normalize=normalize, min_range=min_range, max_range=max_range,
            epsilon=epsilon
        )
        self.scores_df = pd.merge(df_, self.images_df, on="image_id", how="left")

    def get_scores(self) -> pd.DataFrame:
        """Return the scored DataFrame (columns: image_id, AHPweight, AHPScore, geo…)."""
        return self.scores_df

    # ------------------------------------------------------------------
    # Internal routing helpers
    # ------------------------------------------------------------------

    def _build_votes(self, df_, method):
        if method == "matrix":
            self._build_votes_matrix(df_)
        else:
            self._build_votes_dict(df_)

    def _build_ahp(self, method):
        if method == "matrix":
            self._build_ahp_matrix()
        else:
            self._build_ahp_dict()

    def _compute_priority_vector(self, method):
        if method == "matrix":
            self._priority_vector_matrix()
            self.priority_vector = self.priority_vector_matrix
        else:
            self._priority_vector_dict()
            self.priority_vector = self.priority_vector_dict

    # ------------------------------------------------------------------
    # Dict backend
    # ------------------------------------------------------------------

    def _build_votes_dict(self, df_):
        self.votes_dict = defaultdict(lambda: [0, 0])
        for _, row in df_.iterrows():
            left   = self.image_to_idx[row["left_id"]]
            right  = self.image_to_idx[row["right_id"]]
            result = row["winner"].strip().lower()

            if result == "left":
                self.votes_dict[(left, right)][0] += 1
            elif result == "right":
                self.votes_dict[(right, left)][0] += 1
            elif result == "equal":
                self.votes_dict[(left, right)][1] += 1
                self.votes_dict[(right, left)][1] += 1

    def _build_ahp_dict(self):
        ahp_dict = {}
        for (i, j), num in self.votes_dict.items():
            if i == j:
                continue
            den = self.votes_dict.get((j, i), [0, 0])
            numerator   = num[0] + 0.5 * num[1]
            denominator = den[0] + 0.5 * num[1]
            ahp_dict[(i, j)] = numerator / denominator if denominator != 0 else 1e6
        self.ahp_dict = ahp_dict

    def _priority_vector_dict(self):
        # Column sums
        col_sums = defaultdict(float)
        for (i, j), value in self.ahp_dict.items():
            col_sums[j] += value

        # Normalise
        normalized = {
            (i, j): value / col_sums[j]
            for (i, j), value in self.ahp_dict.items()
            if col_sums[j] != 0
        }

        # Row-wise mean → priority vector
        n = len(self.image_ids)
        row_sums   = np.zeros(n)
        row_counts = np.zeros(n)
        for (i, _), value in normalized.items():
            row_sums[i]   += value
            row_counts[i] += 1

        self.priority_vector_dict = np.array([
            row_sums[i] / row_counts[i] if row_counts[i] != 0 else 1.0 / n
            for i in range(n)
        ])

    # ------------------------------------------------------------------
    # Matrix backend
    # ------------------------------------------------------------------

    def _build_votes_matrix(self, df_):
        n = len(self.image_ids)
        self.votes_matrix = np.zeros((n, n, 2), dtype=int)
        for _, row in df_.iterrows():
            left   = self.image_to_idx[row["left_id"]]
            right  = self.image_to_idx[row["right_id"]]
            result = row["winner"].strip().lower()

            if result == "left":
                self.votes_matrix[left, right, 0] += 1
            elif result == "right":
                self.votes_matrix[right, left, 0] += 1
            elif result == "equal":
                self.votes_matrix[left,  right, 1] += 1
                self.votes_matrix[right, left,  1] += 1

    def _build_ahp_matrix(self):
        n = len(self.image_ids)
        ahp_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                num = self.votes_matrix[i, j]
                den = self.votes_matrix[j, i]
                numerator   = num[0] + 0.5 * num[1]
                denominator = den[0] + 0.5 * num[1]
                ahp_matrix[i, j] = numerator / denominator if denominator != 0 else 1e6
        self.ahp_matrix = ahp_matrix

    def _priority_vector_matrix(self):
        col_sums  = self.ahp_matrix.sum(axis=0)
        normalized = self.ahp_matrix / col_sums
        self.priority_vector_matrix = normalized.mean(axis=1)

    # ------------------------------------------------------------------
    # Optional: log-scaled helper (static utility)
    # ------------------------------------------------------------------

    @staticmethod
    def log_scaled(priority_vector, scale_min=0, scale_max=10):
        log_p  = np.log(priority_vector + 1e-8)
        lo, hi = log_p.min(), log_p.max()
        scaled = (log_p - lo) / (hi - lo)
        return scaled * (scale_max - scale_min) + scale_min
