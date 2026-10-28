"""Residual analysis: where does college-to-NBA role projection fail?

Residual = actual NBA role outcome - out-of-fold prediction from a player-only model.
We standardize residuals per target, group players into interpretable segments, and ask
whether any segment is systematically over- or under-projected.

Honesty guards built in:
- Confidence intervals come from a cluster bootstrap over draft years (players in the
  same class are not independent).
- We run many segment x target tests, so we report Benjamini-Hochberg q-values and flag
  only what survives. Unadjusted "significant" cells are expected by chance alone.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

from .config import PLAYER_NUMERIC, RANDOM_SEED
from .features import FEATURE_SETS
from .models import PLAYER_ONLY_MODELS

# natural display order for segment levels (anything not listed sorts alphabetically after)
LEVEL_ORDER = {
    "position": ["G", "W", "B"],
    "context": ["ncaa", "intl", "gleague"],
    "age": ["youngest third", "middle third", "oldest third"],
    "college_usage": ["low usage", "mid usage", "high usage"],
    "three_pt_volume": ["low 3pt volume", "mid 3pt volume", "high 3pt volume"],
    "competition": ["weaker schedule", "mid schedule", "stronger schedule", "n/a (no schedule data)"],
    "draft_slot": ["picks 1-5", "picks 6-14", "picks 15-30", "picks 31+"],
}

SEGMENT_VARS = [
    "position",
    "context",
    "age",
    "college_usage",
    "three_pt_volume",
    "competition",
    "draft_slot",
    "archetype",
]


def _tertile(s: pd.Series, labels) -> pd.Series:
    out = pd.qcut(s, 3, labels=labels, duplicates="drop").astype(object)
    return out.where(s.notna(), "missing")


def _archetypes(X: pd.DataFrame, k: int = 6):
    """Cluster prospects on their pre-draft profile and give each cluster a readable name."""
    feats = [c for c in PLAYER_NUMERIC if c != "sos"] + ["log_pick"]
    Z = SimpleImputer(strategy="median").fit_transform(X[feats])
    Z = StandardScaler().fit_transform(Z)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(Z)
    names = {}
    for c in range(k):
        center = km.cluster_centers_[c]
        order = np.argsort(-np.abs(center))[:3]
        parts = [f"{feats[i]}{'+' if center[i] > 0 else '-'}" for i in order]
        names[c] = f"A{c}: " + " ".join(parts)
    return pd.Series([names[c] for c in km.labels_], index=X.index)


def add_segments(X: pd.DataFrame) -> pd.DataFrame:
    seg = pd.DataFrame(index=X.index)
    seg["position"] = X["position"]
    seg["context"] = X["pre_nba_context"]
    seg["age"] = _tertile(X["age_at_draft"], ["youngest third", "middle third", "oldest third"])
    seg["college_usage"] = _tertile(X["usg"], ["low usage", "mid usage", "high usage"])
    seg["three_pt_volume"] = _tertile(X["three_par"], ["low 3pt volume", "mid 3pt volume", "high 3pt volume"])
    if X["sos"].nunique() <= 3:  # conference-tier proxy (0/1/2) rather than a continuous rating
        tiers = {0.0: "weaker schedule", 1.0: "mid schedule", 2.0: "stronger schedule"}
        comp = X["sos"].map(tiers).astype(object)
    else:
        comp = _tertile(X["sos"], ["weaker schedule", "mid schedule", "stronger schedule"])
    seg["competition"] = comp.where(X["sos"].notna(), "n/a (no schedule data)")
    seg["draft_slot"] = pd.cut(
        X["draft_pick"], [0, 5, 14, 30, 100], labels=["picks 1-5", "picks 6-14", "picks 15-30", "picks 31+"]
    ).astype(object)
    seg["archetype"] = _archetypes(X)
    return seg


def residual_frame(oof_by_target: dict, metrics: pd.DataFrame) -> pd.DataFrame:
    """Stack standardized residuals from the better player-only model for each target."""
    frames = []
    for t, oof in oof_by_target.items():
        sub = metrics[(metrics["target"] == t) & (metrics["model"].isin(PLAYER_ONLY_MODELS))]
        best = sub.sort_values("rmse").iloc[0]["model"]
        d = oof[["row", "player_id", "name", "draft_year", "draft_pick", "y"]].copy()
        d["pred"] = oof[best]
        d["resid"] = d["y"] - d["pred"]
        d["z"] = d["resid"] / d["resid"].std()
        d["target"] = t
        d["model"] = best
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _bh(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


def segment_tests(resid: pd.DataFrame, seg: pd.DataFrame, min_n: int = 25, B: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for t, d in resid.groupby("target"):
        d = d.merge(seg, left_on="row", right_index=True)
        years = np.sort(d["draft_year"].unique())
        idx = rng.integers(0, len(years), size=(B, len(years)))
        for var in SEGMENT_VARS:
            for level, g in d.groupby(var):
                if len(g) < min_n:
                    continue
                s = g.groupby("draft_year")["z"].sum().reindex(years, fill_value=0.0).to_numpy()
                c = g.groupby("draft_year").size().reindex(years, fill_value=0).to_numpy()
                num, den = s[idx].sum(axis=1), c[idx].sum(axis=1)
                boot = num[den > 0] / den[den > 0]
                p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
                rows.append(
                    {
                        "target": t,
                        "segment": var,
                        "level": level,
                        "n": len(g),
                        "mean_z": g["z"].mean(),
                        "ci_lo": np.percentile(boot, 2.5),
                        "ci_hi": np.percentile(boot, 97.5),
                        "p_boot": max(p, 1.0 / B),
                    }
                )
    res = pd.DataFrame(rows)
    res["q_bh"] = _bh(res["p_boot"].to_numpy())
    res["flag_fdr10"] = res["q_bh"] < 0.10
    return res.sort_values("q_bh").reset_index(drop=True)


def residual_structure(resid: pd.DataFrame, X: pd.DataFrame, depth: int = 3, min_leaf: int = 12):
    """Is there any learnable structure left in the misses?

    Single-variable segments cannot see interactions (e.g. "high-usage guards from weaker
    schedules"). Here a shallow decision tree tries to predict each standardized residual
    from the pre-draft profile, scored leave-one-draft-year-out against a no-structure
    baseline of zero. Out-of-sample R^2 at or below zero means the misses look like noise
    given these features. Above zero means something systematic is left.

    The printed rules come from a tree fit on all data. They are in-sample and exploratory:
    read them as hypotheses to check with case studies, not as results.
    """
    feats = FEATURE_SETS["player"]
    rows, rule_text = [], []
    for t, d in resid.groupby("target"):
        d = d.reset_index(drop=True)
        M = X.loc[d["row"].to_numpy(), feats].reset_index(drop=True)
        z = d["z"].to_numpy()
        pred = np.zeros(len(d))
        for tr, te in LeaveOneGroupOut().split(M, z, d["draft_year"].to_numpy()):
            pipe = Pipeline(
                [("imp", SimpleImputer(strategy="median")),
                 ("tree", DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=RANDOM_SEED))]
            )
            pipe.fit(M.iloc[tr], z[tr])
            pred[te] = pipe.predict(M.iloc[te])
        r2 = 1 - ((z - pred) ** 2).sum() / (z ** 2).sum()
        rows.append({"target": t, "oos_r2_of_residuals": float(r2), "n": len(d)})

        pipe = Pipeline(
            [("imp", SimpleImputer(strategy="median")),
             ("tree", DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=RANDOM_SEED))]
        ).fit(M, z)
        rule_text.append(f"=== {t}  (out-of-sample R^2 of residuals = {r2:+.3f}) ===")
        rule_text.append(export_text(pipe.named_steps["tree"], feature_names=feats, decimals=2))
    header = (
        "EXPLORATORY. Trees fit on all data, so these rules are in-sample. Values are standardized "
        "residuals (+ = player did more than projected). Trust a rule only if its target's "
        "out-of-sample R^2 above is clearly positive.\n"
    )
    return pd.DataFrame(rows), header + "\n".join(rule_text)


def test_group(resid: pd.DataFrame, target: str, mask: pd.Series, B: int = 2000, seed: int = 0) -> dict:
    """Test one named subgroup's mean standardized residual with a draft-class bootstrap.

    `mask` is a boolean Series indexed like the feature table (row ids). Use this for a
    hypothesis you wrote down BEFORE looking at the residuals, e.g.
        test_group(resid, "nba_usg", (X.position == "G") & (X.sos < 0) & (X.usg > 24))
    A group chosen after scanning the residuals needs the multiple-testing correction in
    segment_tests, or this p-value will be too optimistic.
    """
    rng = np.random.default_rng(seed)
    d = resid[resid["target"] == target].copy()
    d["in_group"] = mask.reindex(d["row"]).fillna(False).to_numpy(bool)
    years = np.sort(d["draft_year"].unique())
    g = d[d["in_group"]]
    if len(g) < 5:
        return {"n": int(len(g)), "mean_z": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_boot": np.nan}
    s = g.groupby("draft_year")["z"].sum().reindex(years, fill_value=0.0).to_numpy()
    c = g.groupby("draft_year").size().reindex(years, fill_value=0).to_numpy()
    idx = rng.integers(0, len(years), size=(B, len(years)))
    num, den = s[idx].sum(axis=1), c[idx].sum(axis=1)
    boot = num[den > 0] / den[den > 0]
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "n": int(len(g)),
        "mean_z": float(g["z"].mean()),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
        "p_boot": float(max(p, 1.0 / B)),
    }


def top_misses(resid: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    rows = []
    for t, d in resid.groupby("target"):
        over = d.nsmallest(k, "resid").assign(kind="over-projected (did less than expected)")
        under = d.nlargest(k, "resid").assign(kind="under-projected (did more than expected)")
        rows += [over, under]
    out = pd.concat(rows, ignore_index=True)
    return out[["target", "kind", "name", "draft_year", "draft_pick", "y", "pred", "resid", "model"]]
