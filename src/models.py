"""Model zoo for role-outcome projection.

Every estimator is a Pipeline with an imputer first, so imputation statistics
are learned inside each cross-validation fold (no leakage across draft years).
The final step is always named "est" so sample weights can be passed uniformly
as est__sample_weight.
"""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED

# model name -> (estimator kind, feature set name)
MODEL_SPECS = {
    "mean_only": ("mean", "slot"),            # reference: predict the training-fold mean
    "slot_only": ("ridge", "slot"),           # how far does draft slot alone get you?
    "ridge_player": ("ridge", "player"),      # college/pre-draft profile + slot, linear
    "ridge_player_team": ("ridge", "player_team"),
    "gbm_player": ("gbm", "player"),          # same inputs, nonlinear
    "gbm_player_team": ("gbm", "player_team"),
}

# pairs used to test whether team context adds predictive value: (without, with)
TEAM_FIT_PAIRS = [
    ("ridge_player", "ridge_player_team"),
    ("gbm_player", "gbm_player_team"),
]

# models allowed to be "the" player-only model in the residual analysis
PLAYER_ONLY_MODELS = ["ridge_player", "gbm_player"]


def make_estimator(kind: str) -> Pipeline:
    if kind == "mean":
        est = DummyRegressor(strategy="mean")
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("est", est)])
    if kind == "ridge":
        est = RidgeCV(alphas=np.logspace(-2, 3, 20))
        return Pipeline(
            [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("est", est)]
        )
    if kind == "gbm":
        # Classic (non-histogram) boosting is much faster than HistGradientBoosting on a few
        # hundred rows, and shallow trees plus subsampling keep it from memorizing small classes.
        est = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
        )
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("est", est)])
    raise ValueError(f"unknown estimator kind: {kind}")
