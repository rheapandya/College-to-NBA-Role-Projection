"""Shared configuration: column names, role-outcome targets, feature groups.

Everything that depends on the data schema lives here so the rest of the
pipeline can stay generic.
"""

# Role outcomes: averaged over a player's first three NBA seasons.
# Keys are column names in nba_outcomes.csv, values are display labels.
TARGETS = {
    "nba_usg": "Usage rate (%)",
    "nba_3pa_per100": "3PA per 100 possessions",
    "nba_ast_pct": "Assist rate (%)",
    "nba_trb_pct": "Rebound rate (%)",
    "nba_stocks_pct": "Steal + block rate (%)",
}

# Pre-draft numeric features (college, international, or G League season
# immediately before the draft). Missing values are allowed and imputed
# inside each CV fold.
PLAYER_NUMERIC = [
    "age_at_draft",
    "height_in",
    "weight_lb",
    "mpg",
    "usg",
    "ts_pct",
    "three_par",
    "three_pct",
    "ftr",
    "ast_pct",
    "tov_pct",
    "orb_pct",
    "drb_pct",
    "stl_pct",
    "blk_pct",
    "sos",  # strength-of-schedule / competition-level adjustment
]

# Context about the drafting team, using the PRIOR season only so it is known
# on draft night (no leakage from the player's own NBA years).
TEAM_FEATURES = [
    "team_pace",
    "team_3par",
    "team_ast_rate",
    "team_open_minutes",  # share of prior-season minutes at the position group that left
]

CATEGORICAL = {
    "position": ["G", "W", "B"],  # guard / wing / big; B is the reference level
    "pre_nba_context": ["ncaa", "intl", "gleague"],  # ncaa is the reference level
}

UNDRAFTED_PICK = 61
MIN_NBA_MINUTES = 500  # a player needs this many first-three-season minutes to have a usable role
RANDOM_SEED = 0
