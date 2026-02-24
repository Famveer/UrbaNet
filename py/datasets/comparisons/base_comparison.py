import pandas as pd
import numpy as np

PLACE_LEVEL_FILTERS = {
    "city":      lambda df: df[df["left_city"]      == df["right_city"]].copy(),
    "country":   lambda df: df[df["left_country"]   == df["right_country"]].copy(),
    "continent": lambda df: df[df["left_continent"] == df["right_continent"]].copy(),
}

GEO_COLUMNS = ["lat", "long", "city", "country", "continent"]


class BaseComparison:
    """
    Shared foundation for all pairwise comparison scoring methods.

    Subclasses must implement:
        calculate(metric, **kwargs)  → run the scoring algorithm
        normalize(...)               → normalise raw scores
        get_scores()                 → return the final scored DataFrame
    """

    def __init__(self, df=None, place_level=None, n_jobs=4, parallel=True):
        self.place_level = place_level.lower().strip()
        self.N_JOBS = n_jobs
        self.parallel = parallel

        if df is not None:
            filter_fn = PLACE_LEVEL_FILTERS.get(self.place_level)
            self.comparisons_df = filter_fn(df) if filter_fn else df.copy()

    # ------------------------------------------------------------------
    # Common accessors
    # ------------------------------------------------------------------

    def get_metrics(self):
        """Return the unique category labels present in the dataset."""
        return self.comparisons_df["category"].unique()

    def get_matches(self):
        """Return the raw matches DataFrame for the last computed metric."""
        return self.matches_df

    # ------------------------------------------------------------------
    # Shared data-prep helpers
    # ------------------------------------------------------------------

    def _player_rename_map(self, player):
        return {
            f"{player}_id":        "image_id",
            f"{player}_lat":       "lat",
            f"{player}_long":      "long",
            f"{player}_city":      "city",
            f"{player}_country":   "country",
            f"{player}_continent": "continent",
        }

    def filter_player(self, cat_df, player="left"):
        """
        Extract and rename one side (left / right) of the comparisons table.
        Returns a DataFrame with columns: image_id, lat, long, city, country, continent.
        """
        cols = [f"{player}_id", "winner", "category",
                f"{player}_lat", f"{player}_long",
                f"{player}_city", f"{player}_country", f"{player}_continent"]

        df_ = cat_df[cols].copy()
        df_.rename(columns=self._player_rename_map(player), inplace=True)
        df_.drop(columns=["winner", "category"], inplace=True)
        df_.sort_values(by=["image_id"], inplace=True)
        return df_

    def prepare_matches(self, metric="safety", sort_by_time=False, timestamp_col="timestamp"):
        """
        Filter comparisons to *metric*, store raw matches, and build the
        de-duplicated images lookup table (self.images_df).

        Parameters
        ----------
        metric : str
            Category to filter by
        sort_by_time : bool
            If True, sort matches by timestamp before processing.
            Important for sequential methods (Elo, TrueSkill).
        timestamp_col : str
            Name of the timestamp column (default: "timestamp")

        Subclasses that need extra artefacts (e.g. image_to_idx) should call
        super().prepare_matches(metric) first, then extend.
        """
        df_ = self.comparisons_df[self.comparisons_df["category"] == metric].copy()
        
        # Sort by timestamp if requested and column exists
        if sort_by_time and timestamp_col in df_.columns:
            df_ = df_.sort_values(by=timestamp_col).reset_index(drop=True)
            print(f" Sorted {len(df_)} comparisons by timestamp")
        elif sort_by_time and timestamp_col not in df_.columns:
            print(f" Warning: sort_by_time=True but '{timestamp_col}' column not found")
        
        self.matches_df = df_

        left_df  = self.filter_player(df_, player="left")
        right_df = self.filter_player(df_, player="right")

        images_df = pd.concat([left_df, right_df], ignore_index=True)
        images_df.drop_duplicates(inplace=True)
        images_df.sort_values(by=["image_id"], inplace=True)
        self.images_df = images_df

    # ------------------------------------------------------------------
    # Generic min-max normaliser (reusable by all subclasses)
    # ------------------------------------------------------------------

    def normalize_scores(self, df, raw_col, score_col,
                         normalize=True, min_range=0, max_range=10, epsilon=0.0):
        """
        Apply min-max normalisation on *raw_col* and write results to *score_col*.
        Returns the modified DataFrame.
        """
        if normalize:
            lo = df[raw_col].min()
            hi = df[raw_col].max()
            lo_out = min_range + epsilon
            hi_out = max_range - epsilon
            df[score_col] = lo_out + ((df[raw_col] - lo) / (hi - lo)) * (hi_out - lo_out)
        return df

    # ------------------------------------------------------------------
    # Interface that subclasses must implement
    # ------------------------------------------------------------------

    def calculate(self, metric="safety", **kwargs):
        raise NotImplementedError

    def normalize(self, normalize=True, min_range=0, max_range=10, **kwargs):
        raise NotImplementedError

    def get_scores(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience one-shot method
    # ------------------------------------------------------------------

    def fit(self, metric="safety", normalize=True,
            min_range=0, max_range=10, **kwargs):
        """
        Run calculate → normalize → get_scores in a single call.

        Returns
        -------
        pd.DataFrame  — final scored (and optionally normalised) DataFrame.
        """
        self.calculate(metric=metric, **kwargs)
        self.normalize(normalize=normalize, min_range=min_range, max_range=max_range)
        return self.get_scores()

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"place_level='{self.place_level}')")
