import pandas as pd
import numpy as np

from .base_comparison import BaseComparison


class QScores(BaseComparison):
    """
    Schedule-of-Strength Q-Score system for pairwise image comparisons.

    Inherits shared data-loading, filtering, and normalisation logic
    from BaseComparison and only implements what is unique to Q-Scores.

    Formula:  Q = 10/3 * (win_rate + W - L + 1)
        where W = mean win_rate of opponents beaten
              L = mean lose_rate of opponents lost to
    """

    # ------------------------------------------------------------------
    # Override filter_player — QScores needs the opponent id and win/draw/lose label
    # ------------------------------------------------------------------

    def filter_player(self, cat_df, player="left", against="right"):
        cols = [f"{player}_id", "winner", "category",
                f"{player}_lat", f"{player}_long",
                f"{player}_city", f"{player}_country", f"{player}_continent",
                f"{against}_id"]

        df_ = cat_df[cols].copy()
        df_.rename(columns={
            **self._player_rename_map(player),
            f"{against}_id": "against"
        }, inplace=True)

        df_["match"] = df_["winner"].apply(lambda x: self._map_outcome(x, player=player))
        df_.drop(columns=["winner"], inplace=True)
        df_.sort_values(by=["image_id", "category"], inplace=True)
        return df_

    # ------------------------------------------------------------------
    # Override prepare_matches — builds a pivot table instead of a flat list
    # ------------------------------------------------------------------

    def prepare_matches(self, metric="safety"):
        df_ = self.comparisons_df[self.comparisons_df["category"] == metric].copy()

        left_df  = self.filter_player(df_, player="left",  against="right")
        right_df = self.filter_player(df_, player="right", against="left")

        matches = pd.concat([left_df, right_df], ignore_index=True)
        matches.sort_values(by=["image_id", "category"], inplace=True)

        pivot = pd.pivot_table(
            matches,
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

        self.matches_df = pivot

    def get_matches(self):
        return self.matches_df

    # ------------------------------------------------------------------
    # BaseComparison interface
    # ------------------------------------------------------------------

    def calculate(self, metric="safety", **kwargs):
        """Compute Q-scores for *metric*."""
        self.prepare_matches(metric=metric)
        df_ = self.get_matches().copy()
        print(f"Analyzing {df_.shape[0]} '{metric}' images")

        df_["wins"]   = df_["win_against"].apply(len)
        df_["draws"]  = df_["draw_against"].apply(len)
        df_["loses"]  = df_["lose_against"].apply(len)
        df_["total_games"] = df_["wins"] + df_["draws"] + df_["loses"]

        df_["win_rate"]  = df_["wins"]  / df_["total_games"]
        df_["lose_rate"] = df_["loses"] / df_["total_games"]
        df_["draw_rate"] = df_["draws"] / df_["total_games"]

        # Strength-of-schedule weights
        df_["win_weight"]  = df_["win_against"].apply(
            lambda x: df_[df_["image_id"].isin(x)]["win_rate"].sum())
        df_["draw_weight"] = df_["draw_against"].apply(
            lambda x: df_[df_["image_id"].isin(x)]["draw_rate"].sum())
        df_["lose_weight"] = df_["lose_against"].apply(
            lambda x: df_[df_["image_id"].isin(x)]["lose_rate"].sum())

        df_["W"] = df_.apply(lambda r: r["win_weight"]  / r["wins"]  if r["wins"]  > 0 else 0, axis=1)
        df_["D"] = df_.apply(lambda r: r["draw_weight"] / r["draws"] if r["draws"] > 0 else 0, axis=1)
        df_["L"] = df_.apply(lambda r: r["lose_weight"] / r["loses"] if r["loses"] > 0 else 0, axis=1)

        df_["Qscore"] = 10 / 3 * (df_["win_rate"] + df_["W"] - df_["L"] + 1)

        self.scores_df = df_

    def normalize(self, normalize=True, min_range=0, max_range=10, **kwargs):
        """Add a normalised QscoreNorm column to self.scores_df."""
        self.scores_df = self.normalize_scores(
            self.scores_df,
            raw_col="Qscore", score_col="QscoreNorm",
            normalize=normalize, min_range=min_range, max_range=max_range
        )

    def get_scores(self) -> pd.DataFrame:
        """Return the scored DataFrame (columns: image_id, Qscore, QscoreNorm, geo…)."""
        return self.scores_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_outcome(self, choice, player="left"):
        if choice == "equal":
            return "draw"
        elif choice == player:
            return "win"
        else:
            return "lose"
