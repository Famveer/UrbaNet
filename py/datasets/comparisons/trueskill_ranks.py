import pandas as pd
import numpy as np
from collections import defaultdict
from .base_comparison import BaseComparison

try:
    import trueskill
    TRUESKILL_AVAILABLE = True
except ImportError:
    TRUESKILL_AVAILABLE = False


class TrueSkillRanks(BaseComparison):
    """
    TrueSkill Bayesian skill rating system for pairwise image comparisons.

    TrueSkill models each image's skill as a Gaussian distribution with:
        - μ (mu): skill mean
        - σ (sigma): skill uncertainty

    The uncertainty decreases as more matches are played, allowing the
    system to converge faster than Elo while being more robust to upsets.

    Inherits shared data-loading, filtering, and normalisation logic
    from BaseComparison and only implements what is unique to TrueSkill.

    Requires: pip install trueskill
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not TRUESKILL_AVAILABLE:
            raise ImportError(
                "TrueSkill requires the 'trueskill' package. "
                "Install it with: pip install trueskill"
            )

    # ------------------------------------------------------------------
    # BaseComparison interface
    # ------------------------------------------------------------------

    def calculate(
        self,
        metric="safety",
        mu=25.0,
        sigma=None,
        beta=None,
        tau=None,
        draw_probability=0.10,
        backend="scipy",
        sort_by_time=True,
        timestamp_col="timestamp",
    ):
        """
        Compute TrueSkill ratings for *metric*.

        Parameters
        ----------
        metric : str
            Category to score (default: "safety")
        mu : float
            Initial mean skill value (default: 25.0)
        sigma : float
            Initial skill uncertainty. Default: mu/3
        beta : float
            Skill class width (distance guaranteeing ~76% win probability).
            Default: mu/6
        tau : float
            Additive dynamics factor (increases σ slightly each match to
            model skill drift over time). Default: mu/300
        draw_probability : float
            Probability of a draw (0.0 to 1.0). Default: 0.10
        backend : str
            Math backend: "mpmath" (precise) or "scipy" (fast). Default: "scipy"
        sort_by_time : bool
            Sort comparisons by timestamp (recommended: True)
        timestamp_col : str
            Name of timestamp column (default: "timestamp")
        
        Note
        ----
        TrueSkill is sequential - the ORDER of matches affects results!
        Always use sort_by_time=True if you have timestamps.
        """
        # Set defaults based on mu
        if sigma is None:
            sigma = mu / 3
        if beta is None:
            beta = mu / 6
        if tau is None:
            tau = mu / 300

        # Initialize TrueSkill environment
        self.env = trueskill.TrueSkill(
            mu=mu,
            sigma=sigma,
            beta=beta,
            tau=tau,
            draw_probability=draw_probability,
            backend=backend,
        )

        self.prepare_matches(metric=metric, sort_by_time=sort_by_time,
                           timestamp_col=timestamp_col)
        df_ = self.get_matches().copy()
        print(f"Analyzing {df_.shape[0]} '{metric}' comparisons")

        # Initialize all images with default rating
        self.ratings: dict = {}  # image_id -> trueskill.Rating object
        self.match_count: dict = defaultdict(int)

        for _, row in df_.iterrows():
            img1, img2, result = row["left_id"], row["right_id"], row["winner"]

            # Get or create ratings
            if img1 not in self.ratings:
                self.ratings[img1] = self.env.create_rating()
            if img2 not in self.ratings:
                self.ratings[img2] = self.env.create_rating()

            r1 = self.ratings[img1]
            r2 = self.ratings[img2]

            # Update ratings based on outcome
            if result == "equal":
                # Draw
                new_r1, new_r2 = self.env.rate_1vs1(r1, r2, drawn=True)
            elif result == "left":
                # Left wins
                new_r1, new_r2 = self.env.rate_1vs1(r1, r2, drawn=False)
            else:
                # Right wins
                new_r2, new_r1 = self.env.rate_1vs1(r2, r1, drawn=False)

            self.ratings[img1] = new_r1
            self.ratings[img2] = new_r2
            self.match_count[img1] += 1
            self.match_count[img2] += 1

    def normalize(self, normalize=True, min_range=0, max_range=10,
                  use_conservative=True, **kwargs):
        """
        Build self.scores_df with TrueSkill metrics.

        Parameters
        ----------
        normalize : bool
            Whether to normalize to [min_range, max_range]
        min_range, max_range : float
            Normalization range
        use_conservative : bool
            If True, use conservative skill estimate (μ - 3σ) for normalization.
            If False, use just μ. Conservative estimate accounts for uncertainty.
        """
        data = []
        for img_id, rating in self.ratings.items():
            conservative = rating.mu - 3 * rating.sigma
            data.append({
                "image_id": img_id,
                "TrueSkillMu": rating.mu,
                "TrueSkillSigma": rating.sigma,
                "TrueSkillConservative": conservative,
                "matches_played": self.match_count[img_id],
            })

        df_ = pd.DataFrame(data)

        # Choose which metric to normalize
        raw_col = "TrueSkillConservative" if use_conservative else "TrueSkillMu"

        df_ = self.normalize_scores(
            df_,
            raw_col=raw_col,
            score_col="TrueSkillScore",
            normalize=normalize,
            min_range=min_range,
            max_range=max_range,
        )

        self.scores_df = pd.merge(df_, self.images_df, on="image_id", how="left")

    def get_scores(self) -> pd.DataFrame:
        """
        Return the scored DataFrame.

        Columns:
            - image_id
            - TrueSkillMu: skill mean
            - TrueSkillSigma: skill uncertainty (lower = more certain)
            - TrueSkillConservative: μ - 3σ (99.7% confidence lower bound)
            - TrueSkillScore: normalized score [0, 10]
            - matches_played: number of comparisons
            - lat, long, city, country, continent
        """
        return self.scores_df

    # ------------------------------------------------------------------
    # Additional utility methods
    # ------------------------------------------------------------------

    def win_probability(self, img1, img2):
        """
        Calculate probability that img1 beats img2.

        Parameters
        ----------
        img1, img2 : str
            Image IDs

        Returns
        -------
        float : Probability img1 wins (0.0 to 1.0)
        """
        if img1 not in self.ratings or img2 not in self.ratings:
            raise ValueError(f"Unknown image ID: {img1} or {img2}")

        delta_mu = self.ratings[img1].mu - self.ratings[img2].mu
        sum_sigma = (self.ratings[img1].sigma ** 2 + self.ratings[img2].sigma ** 2) ** 0.5
        denom = (2 * self.env.beta ** 2 + sum_sigma ** 2) ** 0.5

        return self.env.cdf(delta_mu / denom)

    def match_quality(self, img1, img2):
        """
        Calculate match quality (0.0 to 1.0, higher = more balanced/interesting).

        Parameters
        ----------
        img1, img2 : str
            Image IDs

        Returns
        -------
        float : Match quality
        """
        if img1 not in self.ratings or img2 not in self.ratings:
            raise ValueError(f"Unknown image ID: {img1} or {img2}")

        return self.env.quality_1vs1(self.ratings[img1], self.ratings[img2])

    def get_leaderboard(self, sort_by="conservative", top_n=None):
        """
        Get ranked leaderboard.

        Parameters
        ----------
        sort_by : str
            "conservative" (μ - 3σ), "mu" (mean skill), or "score" (normalized)
        top_n : int
            Return only top N (None = all)

        Returns
        -------
        pd.DataFrame
        """
        if not hasattr(self, "scores_df"):
            raise RuntimeError("Must call normalize() before getting leaderboard")

        col_map = {
            "conservative": "TrueSkillConservative",
            "mu": "TrueSkillMu",
            "score": "TrueSkillScore",
        }

        if sort_by not in col_map:
            raise ValueError(f"sort_by must be one of {list(col_map.keys())}")

        df = self.scores_df.sort_values(col_map[sort_by], ascending=False)

        if top_n:
            df = df.head(top_n)

        return df.reset_index(drop=True)
