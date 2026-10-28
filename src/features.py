"""Feature construction and named feature sets."""
import numpy as np
import pandas as pd

from .config import PLAYER_NUMERIC, TEAM_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    # Team context is measured RELATIVE to the league that year. Raw team 3-point rate rises
    # with the calendar (about .22 in 2011 to .39 in 2023), so left raw it would act as a
    # draft-year proxy and masquerade as team fit. Centering uses only same-season data
    # that is public on draft night.
    for c in TEAM_FEATURES:
        if c in X and X[c].notna().any():
            X[c] = X[c] - X.groupby("draft_year")[c].transform("mean")
    X["log_pick"] = np.log(X["draft_pick"].astype(float))
    # Reference levels: position B (bigs), context ncaa.
    X["pos_G"] = (X["position"] == "G").astype(float)
    X["pos_W"] = (X["position"] == "W").astype(float)
    X["ctx_intl"] = (X["pre_nba_context"] == "intl").astype(float)
    X["ctx_gleague"] = (X["pre_nba_context"] == "gleague").astype(float)
    return X


_PLAYER_BASE = PLAYER_NUMERIC + ["pos_G", "pos_W", "ctx_intl", "ctx_gleague", "log_pick"]

FEATURE_SETS = {
    "slot": ["log_pick"],
    "player": _PLAYER_BASE,
    "player_team": _PLAYER_BASE + TEAM_FEATURES,
}


def drop_empty_features(X: pd.DataFrame) -> list:
    """Remove feature columns that are entirely missing from every feature list, in place.

    Real exports cannot always supply every column (e.g. team pace). An all-missing column
    carries no information, so it is dropped and reported rather than silently imputed.
    """
    from . import config

    empty = [c for c in set(PLAYER_NUMERIC) | set(TEAM_FEATURES) if X[c].isna().all()]
    for lst in [config.PLAYER_NUMERIC, config.TEAM_FEATURES, *FEATURE_SETS.values()]:
        lst[:] = [c for c in lst if c not in empty]
    return sorted(empty)
