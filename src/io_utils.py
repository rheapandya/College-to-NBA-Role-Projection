"""Load and validate the two input CSVs."""
from pathlib import Path

import pandas as pd

from .config import (
    CATEGORICAL,
    MIN_NBA_MINUTES,
    PLAYER_NUMERIC,
    TARGETS,
    TEAM_FEATURES,
    UNDRAFTED_PICK,
)

REQUIRED_PROSPECT_COLS = (
    ["player_id", "name", "draft_year", "draft_pick", "pre_nba_context", "position"]
    + PLAYER_NUMERIC
    + TEAM_FEATURES
)
REQUIRED_OUTCOME_COLS = ["player_id", "nba_minutes_first3"] + list(TARGETS)


def _check_columns(df: pd.DataFrame, required: list, label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}. "
            "See data/README.md for the expected schema."
        )


def load_data(data_dir: str) -> pd.DataFrame:
    """Return one row per prospect with pre-draft features and NBA role outcomes."""
    d = Path(data_dir)
    prospects = pd.read_csv(d / "prospects.csv")
    outcomes = pd.read_csv(d / "nba_outcomes.csv")
    _check_columns(prospects, REQUIRED_PROSPECT_COLS, "prospects.csv")
    _check_columns(outcomes, REQUIRED_OUTCOME_COLS, "nba_outcomes.csv")

    if prospects["player_id"].duplicated().any():
        raise ValueError("prospects.csv has duplicate player_id values.")

    for col, allowed in CATEGORICAL.items():
        bad = set(prospects[col].dropna().unique()) - set(allowed)
        if bad:
            raise ValueError(f"prospects.csv column '{col}' has unexpected values: {sorted(bad)}")

    df = prospects.merge(outcomes, on="player_id", how="left")
    df["nba_minutes_first3"] = df["nba_minutes_first3"].fillna(0)
    df["draft_pick"] = df["draft_pick"].fillna(UNDRAFTED_PICK)
    # "observed" = the player logged enough NBA minutes to have a meaningful role.
    df["observed"] = df["nba_minutes_first3"] >= MIN_NBA_MINUTES
    df["synthetic"] = (d / "SYNTHETIC.txt").exists()
    return df.reset_index(drop=True)
