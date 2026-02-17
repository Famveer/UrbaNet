import pandas as pd
import numpy as np
from collections import defaultdict

from .base_comparison import BaseComparison


class EloRatings(BaseComparison):
    """
    Elo Rating System for pairwise image comparisons.

    Inherits shared data-loading, filtering, and normalisation logic
    from BaseComparison and only implements what is unique to Elo.
    """

    # ------------------------------------------------------------------
    # Core Elo logic
    # ------------------------------------------------------------------

    def expected_score(self, rating_1, rating_2):
        """Standard Elo expected score formula."""
        return 1 / (1 + 10 ** ((rating_2 - rating_1) / 400))

    def adaptive_K(self, rating1, rating2, match_count1, match_count2):
        """Adjust K-factor based on rating difference and match history."""
        rating_diff = abs(rating1 - rating2)
        K = max(self.min_K, min(self.max_K, self.initial_K - rating_diff / 100))
        if match_count1 < 10 or match_count2 < 10:
            K = max(K, self.initial_K)
        return K

    # ------------------------------------------------------------------
    # BaseComparison interface
    # ------------------------------------------------------------------

    def calculate(self, metric="safety", initial_rating=0, K=100,
                  max_K=40, min_K=10, adaptative_K=False):
        """
        Compute Elo ratings for *metric*.

        Parameters
        ----------
        metric : str
        initial_rating : float   Starting rating for every image.
        K : int                  Base K-factor.
        max_K, min_K : int       Bounds for adaptive K.
        adaptative_K : bool      Whether to use adaptive K.
        """
        self.prepare_matches(metric=metric)
        df_ = self.get_matches().copy()
        print(f"Analyzing {df_.shape[0]} '{metric}' comparisons")

        self.initial_K = K
        self.max_K = max_K
        self.min_K = min_K

        self.current_ratings: dict = {}
        self.match_count: dict = {}

        for _, row in df_.iterrows():
            img1, img2, result = row["left_id"], row["right_id"], row["winner"]

            r1 = self.current_ratings.get(img1, initial_rating)
            r2 = self.current_ratings.get(img2, initial_rating)
            mc1 = self.match_count.get(img1, 0)
            mc2 = self.match_count.get(img2, 0)

            exp1 = self.expected_score(r1, r2)
            exp2 = 1 - exp1

            if result == "equal":
                s1, s2 = 0.5, 0.5
            elif result == "left":
                s1, s2 = 1, 0
            else:
                s1, s2 = 0, 1

            if adaptative_K:
                K = self.adaptive_K(r1, r2, mc1, mc2)

            self.current_ratings[img1] = r1 + K * (s1 - exp1)
            self.current_ratings[img2] = r2 + K * (s2 - exp2)
            self.match_count[img1] = mc1 + 1
            self.match_count[img2] = mc2 + 1

    def normalize(self, normalize=True, min_range=0, max_range=10, **kwargs):
        """Build self.scores_df with raw EloRating and normalised EloScore."""
        df_ = pd.DataFrame(
            list(self.current_ratings.items()),
            columns=["image_id", "EloRating"]
        )
        df_ = self.normalize_scores(
            df_, raw_col="EloRating", score_col="EloScore",
            normalize=normalize, min_range=min_range, max_range=max_range
        )
        self.scores_df = pd.merge(df_, self.images_df, on="image_id", how="left")

    def get_scores(self) -> pd.DataFrame:
        """Return the scored DataFrame (columns: image_id, EloRating, EloScore, geo…)."""
        return self.scores_df
