"""Leave-one-draft-year-out cross-validation, year-level bootstrap, team-fit test.

Why leave-one-draft-year-out: a draft class is the natural unit of independence
(same market, same rules, same era of play). Random k-fold would let the model see
neighbours from the same class and overstate accuracy. It also gives ~14 folds to
reason about uncertainty instead of one lucky split.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_SETS
from .models import MODEL_SPECS, TEAM_FIT_PAIRS, make_estimator


def played_propensity(X: pd.DataFrame) -> np.ndarray:
    """Out-of-fold probability that a prospect logs enough NBA minutes to have a role.

    Used only for the optional inverse-probability weighting that partially corrects
    for the fact that role outcomes exist only for players who actually play.
    Pre-draft features only, so there is no leakage.
    """
    feats = FEATURE_SETS["player"]
    obs = X["observed"].astype(int).to_numpy()
    groups = X["draft_year"].to_numpy()
    p = np.zeros(len(X))
    for tr, te in LeaveOneGroupOut().split(X, obs, groups):
        clf = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(C=0.5, max_iter=2000)),
            ]
        )
        clf.fit(X.iloc[tr][feats], obs[tr])
        p[te] = clf.predict_proba(X.iloc[te][feats])[:, 1]
    return p


def oof_for_target(X: pd.DataFrame, target: str, models=None, ipw: bool = False) -> pd.DataFrame:
    """Out-of-fold predictions for one role target, for every model in `models`.

    Only players with an observed role outcome are used. Training can be weighted by
    1 / P(played); evaluation is always unweighted on the observed players.
    """
    models = models or MODEL_SPECS
    sub = X[X["observed"] & X[target].notna()].copy()
    sub.insert(0, "row", sub.index)
    sub = sub.reset_index(drop=True)
    y = sub[target].to_numpy(float)
    groups = sub["draft_year"].to_numpy()

    if ipw:
        w = 1.0 / np.clip(sub["p_played"].to_numpy(), 0.15, 1.0)
        w = w / w.mean()
    else:
        w = np.ones(len(sub))

    preds = {m: np.full(len(sub), np.nan) for m in models}
    for tr, te in LeaveOneGroupOut().split(sub, y, groups):
        for m, (kind, fs) in models.items():
            cols = FEATURE_SETS[fs]
            est = make_estimator(kind)
            est.fit(sub.iloc[tr][cols], y[tr], est__sample_weight=w[tr])
            preds[m][te] = est.predict(sub.iloc[te][cols])

    out = sub[["row", "player_id", "name", "draft_year", "draft_pick", target]].rename(
        columns={target: "y"}
    )
    for m in models:
        out[m] = preds[m]
    return out


def run_cv(X: pd.DataFrame, targets, ipw: bool = False) -> dict:
    X = X.reset_index(drop=True).copy()
    X["p_played"] = played_propensity(X)
    return {t: oof_for_target(X, t, ipw=ipw) for t in targets}


# --------------------------------------------------------------------------- metrics


def _year_tables(oof: pd.DataFrame, models):
    se = pd.DataFrame({m: (oof["y"] - oof[m]) ** 2 for m in models})
    se["draft_year"] = oof["draft_year"].to_numpy()
    sse = se.groupby("draft_year")[list(models)].sum()
    n = se.groupby("draft_year").size()
    return sse, n


def bootstrap_mse(sse: pd.DataFrame, n: pd.Series, B: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Resample draft years with replacement; return B bootstrap pooled-MSE draws per model."""
    rng = np.random.default_rng(seed)
    years = len(n)
    idx = rng.integers(0, years, size=(B, years))
    denom = n.to_numpy()[idx].sum(axis=1)
    return pd.DataFrame(
        {m: sse[m].to_numpy()[idx].sum(axis=1) / denom for m in sse.columns}
    )


def summarize(oof_by_target: dict, B: int = 2000, seed: int = 0):
    """Return (metrics, team_fit) DataFrames.

    metrics: per target x model pooled RMSE, skill vs the training-mean baseline (an
        out-of-sample R^2), a draft-year bootstrap CI on that skill, and the share of
        draft classes where the model beat the baseline.
    team_fit: whether adding drafting-team context lowers out-of-sample MSE, with a
        draft-year bootstrap CI on the percent change in MSE.
    """
    metric_rows, team_rows = [], []
    for t, oof in oof_by_target.items():
        models = [m for m in MODEL_SPECS]
        sse, n = _year_tables(oof, models)
        boot = bootstrap_mse(sse, n, B=B, seed=seed)
        mse = sse.sum() / n.sum()
        base = mse["mean_only"]
        per_year_mse = sse.div(n, axis=0)

        for m in models:
            skill = 1 - mse[m] / base
            skill_boot = 1 - boot[m] / boot["mean_only"]
            metric_rows.append(
                {
                    "target": t,
                    "model": m,
                    "n": int(n.sum()),
                    "rmse": float(np.sqrt(mse[m])),
                    "skill_r2": float(skill),
                    "skill_lo": float(np.percentile(skill_boot, 2.5)),
                    "skill_hi": float(np.percentile(skill_boot, 97.5)),
                    "years_beating_mean": float((per_year_mse[m] < per_year_mse["mean_only"]).mean()),
                }
            )

        for without, with_ in TEAM_FIT_PAIRS:
            delta = (mse[with_] / mse[without] - 1) * 100
            delta_boot = (boot[with_] / boot[without] - 1) * 100
            team_rows.append(
                {
                    "target": t,
                    "without_team": without,
                    "with_team": with_,
                    "mse_change_pct": float(delta),
                    "lo": float(np.percentile(delta_boot, 2.5)),
                    "hi": float(np.percentile(delta_boot, 97.5)),
                    "p_team_helps": float((delta_boot < 0).mean()),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(team_rows)


# --------------------------------------------------------------------------- sanity


def permutation_check(X: pd.DataFrame, target: str, seed: int = 0) -> pd.DataFrame:
    """Shuffle the target within each draft year and re-run the CV.

    If the pipeline leaks, skill stays well above zero even though the labels are noise.
    A healthy pipeline gives skill at or below ~0.
    """
    rng = np.random.default_rng(seed)
    X2 = X.reset_index(drop=True).copy()
    mask = (X2["observed"] & X2[target].notna()).to_numpy()
    for yr, idx in X2[mask].groupby("draft_year").groups.items():
        vals = X2.loc[idx, target].to_numpy().copy()
        rng.shuffle(vals)
        X2.loc[idx, target] = vals
    subset = {k: MODEL_SPECS[k] for k in ["mean_only", "ridge_player", "gbm_player"]}
    oof = oof_for_target(X2, target, models=subset)
    mse = {m: float(((oof["y"] - oof[m]) ** 2).mean()) for m in subset}
    rows = [
        {"model": m, "shuffled_skill_r2": 1 - mse[m] / mse["mean_only"]}
        for m in subset
        if m != "mean_only"
    ]
    return pd.DataFrame(rows)
