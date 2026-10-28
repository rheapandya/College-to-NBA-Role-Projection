"""Harder baselines than draft slot, same leave-one-draft-year-out scoring as metrics.csv.

  position_height : position + height only
  same_stat       : the college version of the same stat (+ position), nothing else
  ridge_player    : the full model (copied from metrics.csv for reference)

Run from the project root:  python scripts/extra_baselines.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import TARGETS
from src.features import build_features
from src.io_utils import load_data

df = load_data("data/real")
X = build_features(df)
X["college_trb"] = X["orb_pct"] + X["drb_pct"]
X["college_stocks"] = X["stl_pct"] + X["blk_pct"]
SETS = {
    "position_height": ["pos_G", "pos_W", "height_in"],
    "same_stat": {
        "nba_usg": ["usg", "pos_G", "pos_W"],
        "nba_3pa_per100": ["three_par", "pos_G", "pos_W"],
        "nba_ast_pct": ["ast_pct", "pos_G", "pos_W"],
        "nba_trb_pct": ["college_trb", "pos_G", "pos_W"],
        "nba_stocks_pct": ["college_stocks", "pos_G", "pos_W"],
    },
}
rng = np.random.default_rng(0)
rows = []
obs = X[X.observed].reset_index(drop=True)
years = np.sort(obs.draft_year.unique())
for tgt in TARGETS:
    for name, spec in SETS.items():
        cols = spec if isinstance(spec, list) else spec[tgt]
        sse_m, sse_b = {}, {}
        for y in years:
            tr, te = obs[obs.draft_year != y], obs[obs.draft_year == y]
            mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(tr[cols], tr[tgt])
            pred = mdl.predict(te[cols])
            sse_m[y] = ((te[tgt] - pred) ** 2).sum()
            sse_b[y] = ((te[tgt] - tr[tgt].mean()) ** 2).sum()
        skill = lambda ys: 1 - sum(sse_m[y] for y in ys) / sum(sse_b[y] for y in ys)
        boots = [skill(rng.choice(years, len(years))) for _ in range(2000)]
        rows.append(dict(target=tgt, baseline=name, skill_r2=skill(years),
                         lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5)))
out = pd.DataFrame(rows)
full = pd.read_csv("outputs/real/metrics.csv")
full = full[full.model == "ridge_player"][["target", "skill_r2", "skill_lo", "skill_hi"]]
full.columns = ["target", "skill_r2", "lo", "hi"]
full.insert(1, "baseline", "ridge_player (full model)")
out = pd.concat([out, full]).sort_values(["target", "skill_r2"]).round(3)
out.to_csv("outputs/real/extra_baselines.csv", index=False)
print(out.to_string(index=False))
