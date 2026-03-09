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
    # BaseComparison interface
    # ------------------------------------------------------------------

    def get_weight(self, df, list_col, mapper):
        tmp = df[["image_id", list_col]].explode(list_col)
        tmp["value"] = tmp[list_col].map(mapper)
        s = tmp.groupby("image_id", sort=False)["value"].sum()
        return df["image_id"].map(s).fillna(0) 

    def calculate(self):
        """Compute Q-scores for *metric*."""
        df_ = self.get_summarize_matches().copy()
        print(f" Processing {df_.shape[0]} comparisons")
        
        id_to_win  = df_.set_index("image_id")["win_rate"].to_dict()
        id_to_draw = df_.set_index("image_id")["draw_rate"].to_dict()
        id_to_lose = df_.set_index("image_id")["lose_rate"].to_dict()

        # Strength-of-schedule weights
        df_["win_weight"]  = self.get_weight(df_, "win_against", id_to_win)
        df_["draw_weight"] = self.get_weight(df_, "draw_against", id_to_draw)
        df_["lose_weight"] = self.get_weight(df_, "lose_against", id_to_lose)

        df_["W"] = np.where(df_["wins"] > 0, df_["win_weight"] / df_["wins"], 0.0)
        df_["D"] = np.where(df_["draws"] > 0, df_["draw_weight"] / df_["draws"], 0.0)
        df_["L"] = np.where(df_["loses"] > 0, df_["lose_weight"] / df_["loses"], 0.0)

        df_["Qscore"] = (10/3) * (df_["win_rate"] + df_["W"] - df_["L"] + 1)

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

