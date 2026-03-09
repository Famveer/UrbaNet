import pandas as pd

from .elo_ratings import EloRatings
from .ahp_weights import AHPWeights
from .q_scores    import QScores
from .trueskill_ranks import TrueSkillRanks

METHOD_MAP = {
    "EloRatings":           EloRatings,
    "ELORatings":           EloRatings,
    "eloratings":           EloRatings,
    "elo_ratings":          EloRatings,
    
    "ahpweights":           AHPWeights,
    "AHPWeights":           AHPWeights,
    "ahp_weights":          AHPWeights,
    
    "qscores":              QScores,
    "Qscores":              QScores,
    "strength_of_schedule": QScores,
    
    "TrueSkill":            TrueSkillRanks,
    "trueskill":            TrueSkillRanks,
    "TS_ranks":             TrueSkillRanks,
    "ts_ranks":             TrueSkillRanks,
}


class Comparisons:
    """
    Unified entry point for pairwise comparison scoring.

    Instantiates the appropriate backend (EloRatings, AHPWeights, or QScores)
    based on *method_name* and exposes a single consistent API regardless
    of which algorithm is running underneath.

    Parameters
    ----------
    df : pd.DataFrame
        Comparisons table with columns:
        left_id, right_id, winner, category,
        left_lat, left_long, left_city, left_country, left_continent
        (and their right_ equivalents).
    method_name : str
        "elo_ratings" | "ahp_weights" | "qscores" | "schedule_of_strength"
    place_level : str
        Geographic scope: "all" | "city" | "country" | "continent".
    n_jobs : int
        Number of parallel workers passed to the backend.
    parallel : bool
        Whether to enable parallel processing in the backend.

    Examples
    --------
    # One-shot
    results = Comparisons(df, method_name="elo_ratings").fit(metric="safety", K=32)

    # Choose your method and run:

    # OPTION 1: Elo
    comp = Comparisons(df, method_name="elo_ratings", place_level="all")
    comp.prepare_matches(metric="safety")
    comp.calculate(K=32, sort_by_time=True)
    comp.normalize(min_range=0, max_range=10)
    results = comp.get_scores()

    # OPTION 2: AHP
    comp = Comparisons(df, method_name="ahp_weights", place_level="all")
    comp.prepare_matches(metric="safety")
    comp.calculate(method="dict")
    comp.normalize(min_range=0, max_range=10)
    results = comp.get_scores()

    # OPTION 3: Q-Scores
    comp = Comparisons(df, method_name="qscores", place_level="all")
    comp.prepare_matches(metric="safety")
    comp.calculate()
    comp.normalize(min_range=0, max_range=10)
    results = comp.get_scores()

    # OPTION 4: TrueSkill
    comp = Comparisons(df, method_name="trueskill", place_level="all")
    comp.prepare_matches(metric="safety")
    comp.calculate(mu=25, sort_by_time=True)
    comp.normalize(min_range=0, max_range=10)
    results = comp.get_scores()
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        method_name: str = "elo_ratings",
        place_level: str = None,
        n_jobs: int = 4,
        parallel: bool = True,
    ):
        method_name = method_name.lower().strip()
        if method_name not in METHOD_MAP:
            raise ValueError(
                f"Unknown method '{method_name}'. "
                f"Choose from: {list(METHOD_MAP.keys())}"
            )
        else:
            print(f"Using {method_name}.")

        self.method_name = method_name
        self._backend = METHOD_MAP[method_name](
            df=df,
            place_level=place_level,
            n_jobs=n_jobs,
            parallel=parallel,
        )

    # ------------------------------------------------------------------
    # Delegate the full shared interface to the backend
    # ------------------------------------------------------------------

    def prepare_matches(self, metric="safety", sort_by_time=False, timestamp_col="timestamp", comparison_threshold=0):
        return self._backend.prepare_matches(metric=metric, sort_by_time=sort_by_time, 
                           timestamp_col=timestamp_col, comparison_threshold=comparison_threshold)

    def get_metrics(self):
        """Return the unique category labels in the dataset."""
        return self._backend.get_metrics()

    def get_matches(self):
        """Return the raw matches DataFrame for the last computed metric."""
        return self._backend.get_matches()
    
    def get_summarize_matches(self):
        return self._backend.get_summarize_matches()
    
    def get_evaluations(self):
        return self._backend.evaluations_df
    
    def get_samples(self):
        return self._backend.samples_df

    def calculate(self, **kwargs):
        """
        Run the scoring algorithm for *metric*.

        Extra keyword arguments are forwarded to the backend:

        elo_ratings   → initial_rating, K, max_K, min_K, adaptative_K
        ahp_weights   → method ("dict" | "matrix")
        qscores       → (none)
        """
        self._backend.calculate(**kwargs)
        return self

    def normalize(self, normalize: bool = True,
                  min_range: float = 0, max_range: float = 10, **kwargs):
        """
        Normalise raw scores to [min_range, max_range].

        Extra keyword arguments (e.g. epsilon for ahp_weights) are forwarded.
        """
        self._backend.normalize(
            normalize=normalize,
            min_range=min_range,
            max_range=max_range,
            **kwargs
        )
        return self

    def get_scores(self) -> pd.DataFrame:
        """
        Return the final scored DataFrame.

        Score column names by method:
            elo_ratings   →  EloRating,  EloScore
            ahp_weights   →  AHPweight,  AHPScore
            qscores       →  Qscore,     QscoreNorm
        """
        return self._backend.get_scores()

    def fit(
        self,
        metric: str = "safety",
        normalize: bool = True,
        min_range: float = 0,
        max_range: float = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Shortcut: calculate → normalize → get_scores in one call.

        Returns
        -------
        pd.DataFrame  — final scored (and optionally normalised) DataFrame.
        """
        return self._backend.fit(
            metric=metric,
            normalize=normalize,
            min_range=min_range,
            max_range=max_range,
            **kwargs,
        )

    def __repr__(self):
        return (
            f"Comparisons(method='{self.method_name}', "
            f"backend={self._backend!r})"
        )
