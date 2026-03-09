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
    
    def _map_outcome(self, choice, player="left"):
        if choice == "equal":
            return "draw"
        elif choice == player:
            return "win"
        else:
            return "lose"

    def get_metrics(self):
        """Return the unique category labels present in the dataset."""
        return self.comparisons_df["category"].unique()

    def get_matches(self):
        """Return the raw matches DataFrame for the last computed metric."""
        return self.matches_df
    
    def get_summarize_matches(self):
        return self.summarize_matches_df
    
    def get_evaluations(self):
        return self.evaluations_df
    
    def get_samples(self):
        return self.samples_df

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

    def filter_player(self, cat_df, player="left", against="right"):
        cols = [f"{player}_id", "winner", "category",
                f"{player}_lat", f"{player}_long",
                f"{player}_city", f"{player}_country", f"{player}_continent",
                f"{against}_id", "timestamp"]

        df_ = cat_df[cols].copy()
        df_.rename(columns={
            **self._player_rename_map(player),
            f"{against}_id": "against"
        }, inplace=True)

        df_["match"] = df_["winner"].apply(lambda x: self._map_outcome(x, player=player))
        df_.drop(columns=["winner"], inplace=True)
        df_.sort_values(by=["image_id"], inplace=True)
        return df_
    
    
    def remove_ids(self, df, ids_to_remove,
                       id_col="image_id",
                       list_cols=("draw_against", "lose_against", "win_against")):
        df = df.loc[~df[id_col].isin(ids_to_remove)].copy()

        def clean_one_column(df, col):
            tmp = df[[id_col, col]].copy()
            tmp[col] = tmp[col].apply(
                lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
            )
            tmp = tmp.explode(col)
            tmp = tmp[tmp[col].isna() | (~tmp[col].isin(ids_to_remove))]
            tmp = (
                tmp.groupby(id_col, sort=False)[col]
                .agg(lambda s: [x for x in s.dropna()])
                .reset_index()
            )
            return tmp

        for col in list_cols:
            cleaned = clean_one_column(df, col)
            df = df.drop(columns=[col]).merge(cleaned, on=id_col, how="left")

        for col in list_cols:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

        df["wins"]   = df["win_against"].apply(len)
        df["draws"]  = df["draw_against"].apply(len)
        df["loses"]  = df["lose_against"].apply(len)
        df["total_games"] = df["wins"] + df["draws"] + df["loses"]

        return df

    def summarize_matches(self, comparison_threshold=0):
        pivot = pd.pivot_table(
                    self.evaluations_df.copy(),
                    index=["image_id", "lat", "long", "city", "country", "continent", "category"],
                    columns=["match"],
                    values=["against"],
                    aggfunc={"against": list}
                ).reset_index()

        # Flatten multi-level columns  →  "win_against", "draw_against", "lose_against"
        pivot.columns = [
            "_".join([col[1], col[0]]) if col[1] and col[0] else "".join(col)
            for col in pivot.columns
        ]

        # Ensure all three outcome columns exist, even if absent from the data
        for col in ["win_against", "draw_against", "lose_against"]:
            if col not in pivot.columns:
                pivot[col] = [[] for _ in range(len(pivot))]
            else:
                pivot[col] = pivot[col].apply(lambda x: x if isinstance(x, list) else [])

        pivot["wins"]   = pivot["win_against"].apply(len)
        pivot["draws"]  = pivot["draw_against"].apply(len)
        pivot["loses"]  = pivot["lose_against"].apply(len)
        pivot["total_games"] = pivot["wins"] + pivot["draws"] + pivot["loses"]
        
        if comparison_threshold>0:
        
            ids_to_remove = pivot[pivot["total_games"]<comparison_threshold]["image_id"].tolist()
            pivot = self.remove_ids(pivot, ids_to_remove)
        
        pivot["win_rate"] = np.where(pivot["total_games"] > 0, pivot["wins"] / pivot["total_games"], 0.0)
        pivot["lose_rate"] = np.where(pivot["total_games"] > 0, pivot["loses"] / pivot["total_games"], 0.0)
        pivot["draw_rate"] = np.where(pivot["total_games"] > 0, pivot["draws"] / pivot["total_games"], 0.0)

        return pivot

    def prepare_matches(self, metric="safety", sort_by_time=False, timestamp_col="timestamp", comparison_threshold=0):
        """
        Filter comparisons to *metric*, store raw matches, and build the
        de-duplicated images lookup table.

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

        left_df  = self.filter_player(df_, player="left",  against="right")
        right_df = self.filter_player(df_, player="right", against="left")

        evaluations_df = pd.concat([left_df, right_df], ignore_index=True)
        evaluations_df.drop_duplicates(inplace=True)
        evaluations_df.sort_values(by=["image_id", "against", "timestamp"], inplace=True)
        self.evaluations_df = evaluations_df
        self.summarize_matches_df = self.summarize_matches(comparison_threshold=comparison_threshold)
        
        samples_df = evaluations_df.drop(columns=["category", "timestamp", "match", "against"]).copy()
        samples_df.drop_duplicates(inplace=True)
        self.samples_df = samples_df

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

    def calculate(self, **kwargs):
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
        self.calculate(**kwargs)
        self.normalize(normalize=normalize, min_range=min_range, max_range=max_range)
        return self.get_scores()

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"place_level='{self.place_level}')")
