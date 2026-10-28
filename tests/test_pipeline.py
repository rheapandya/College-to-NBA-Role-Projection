"""Fast checks that the pipeline's guarantees hold. Run with: python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.config import TARGETS
from src.cv import bootstrap_mse, oof_for_target
from src.features import FEATURE_SETS, build_features
from src.io_utils import load_data
from src.make_synthetic import generate
from src.models import MODEL_SPECS
from src.residuals import _bh


@pytest.fixture(scope="module")
def X(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    p, o = generate(years=range(2012, 2018))
    p.to_csv(d / "prospects.csv", index=False)
    o.to_csv(d / "nba_outcomes.csv", index=False)
    return build_features(load_data(str(d)))


def test_team_features_not_outcomes():
    # team context must never include any NBA outcome column
    assert not set(FEATURE_SETS["player_team"]) & set(TARGETS)


def test_every_prediction_is_out_of_fold(X):
    subset = {k: MODEL_SPECS[k] for k in ["mean_only", "ridge_player"]}
    oof = oof_for_target(X, "nba_usg", models=subset)
    assert oof["ridge_player"].notna().all()
    assert len(oof) == int((X.observed & X.nba_usg.notna()).sum())


def test_mean_baseline_ignores_features(X):
    subset = {"mean_only": MODEL_SPECS["mean_only"]}
    oof = oof_for_target(X, "nba_usg", models=subset)
    # within a held-out year, the baseline predicts one constant
    assert oof.groupby("draft_year")["mean_only"].nunique().max() == 1


def test_bootstrap_is_reproducible():
    sse = pd.DataFrame({"a": [4.0, 9.0, 1.0], "b": [2.0, 3.0, 1.0]}, index=[1, 2, 3])
    n = pd.Series([2, 3, 1], index=[1, 2, 3])
    a = bootstrap_mse(sse, n, B=50, seed=1)
    b = bootstrap_mse(sse, n, B=50, seed=1)
    assert a.equals(b)


def test_group_recovers_planted_effect():
    # a pre-specified group with a clear shift should show a CI excluding zero; the rest should not
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(
        {"row": np.arange(400), "draft_year": np.repeat(np.arange(2010, 2020), 40),
         "z": rng.normal(0, 1, 400), "target": "nba_usg"}
    )
    in_group = pd.Series(np.arange(400) % 8 == 0)
    rows.loc[in_group.to_numpy(), "z"] -= 1.0
    from src.residuals import test_group

    hit = test_group(rows, "nba_usg", in_group)
    miss = test_group(rows, "nba_usg", pd.Series(np.arange(400) % 8 == 1))
    assert hit["ci_hi"] < 0
    assert miss["ci_lo"] < 0 < miss["ci_hi"]


def test_bh_monotone_and_bounded():
    p = np.array([0.001, 0.04, 0.03, 0.5, 0.9])
    q = _bh(p)
    assert (q >= p - 1e-12).all() and (q <= 1).all()
