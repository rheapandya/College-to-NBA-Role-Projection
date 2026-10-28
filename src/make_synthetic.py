"""Generate a SYNTHETIC stand-in dataset with the same schema as the real inputs.

This exists only so the pipeline can be tested end to end without real data. The
numbers are invented. Nothing the pipeline reports on this data is a finding about
real players. Two effects are deliberately planted so we can check that the pipeline
is able to detect them:

  1. Team fit: a drafting team's prior-season 3PA rate raises a player's NBA 3PA volume.
  2. A hidden miss: guards with high college usage against weaker schedules lose
     usage in the NBA by more than any pre-draft feature explains.

Usage:  python -m src.make_synthetic --out data/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def generate(seed: int = 7, years=range(2010, 2024), per_year: int = 65):
    rng = np.random.default_rng(seed)
    prospects, outcomes = [], []
    pid = 0

    for year in years:
        n = per_year
        pos = rng.choice(["G", "W", "B"], size=n, p=[0.38, 0.32, 0.30])
        ctx = rng.choice(["ncaa", "intl", "gleague"], size=n, p=[0.78, 0.15, 0.07])
        t = rng.normal(0, 1, n)  # latent talent

        score = t + rng.normal(0, 0.6, n)
        rank = (-score).argsort().argsort() + 1
        pick = rank.astype(float)
        pick[rank > 60] = np.nan  # undrafted

        age = 19.0 + rng.gamma(1.6, 0.9, n) - 0.25 * t
        age = _clip(age + np.where(ctx == "gleague", 1.0, 0.0), 18.6, 25.0)
        h_mu = np.select([pos == "G", pos == "W"], [76.0, 79.0], 82.5)
        w_mu = np.select([pos == "G", pos == "W"], [195.0, 215.0], 245.0)
        height = h_mu + rng.normal(0, 2.0, n)
        weight = w_mu + rng.normal(0, 12.0, n)

        sos = np.where(ctx == "ncaa", 0.5 * t + rng.normal(0, 1, n), np.nan)

        usg = _clip(22 + 3 * t + np.select([pos == "G", pos == "W"], [1.5, 0.0], -1.0) + rng.normal(0, 3, n), 12, 38)
        mpg = _clip(28 + 2 * t - np.where(ctx == "intl", 6, 0) + rng.normal(0, 4, n), 8, 38)
        ts = _clip(0.54 + 0.02 * t + np.where(pos == "B", 0.03, 0) + rng.normal(0, 0.04, n), 0.40, 0.70)
        tpar = _clip(np.select([pos == "G", pos == "W"], [0.42, 0.36], 0.10) + 0.03 * t + rng.normal(0, 0.10, n), 0.0, 0.70)
        tpct = _clip(0.33 + 0.02 * (tpar > 0.2) + 0.01 * t + rng.normal(0, 0.06, n), 0.15, 0.50)
        ftr = _clip(0.32 + 0.05 * t + np.where(pos == "B", 0.08, 0) + rng.normal(0, 0.09, n), 0.05, 0.9)
        ast = _clip(np.select([pos == "G", pos == "W"], [22.0, 12.0], 8.0) + 2 * t + rng.normal(0, 4, n), 2, 45)
        tov = _clip(17 - 1.0 * t + rng.normal(0, 3, n), 7, 30)
        orb = _clip(np.select([pos == "G", pos == "W"], [2.5, 5.0], 11.0) + rng.normal(0, 2.0, n), 0.5, 20)
        drb = _clip(np.select([pos == "G", pos == "W"], [11.0, 16.0], 24.0) + rng.normal(0, 3.0, n), 5, 35)
        stl = _clip(2.0 + rng.normal(0, 0.6, n), 0.3, 4.5)
        blk = _clip(np.select([pos == "G", pos == "W"], [0.6, 1.4], 4.5) + rng.normal(0, 0.8, n), 0.0, 9)

        # drafting-team context (prior season)
        pace = rng.normal(99.5, 2.2, n)
        t3 = rng.normal(0.38, 0.04, n)
        tast = rng.normal(0.60, 0.04, n)
        topen = rng.uniform(0.0, 0.5, n)

        # NBA outcomes ---------------------------------------------------------
        p_play = 1 / (1 + np.exp(-(1.2 * t - 0.1 + np.where(pos == "B", 0.25, 0))))
        played = rng.random(n) < p_play
        minutes = np.where(
            played,
            np.exp(np.log(2800) + 0.5 * t + rng.normal(0, 0.7, n)),
            rng.uniform(0, 350, n),
        )
        noise_scale = _clip(np.sqrt(2500 / np.maximum(minutes, 300)), 0.7, 2.0)

        slot = 1 - np.where(np.isnan(pick), 65, pick) / 65
        hidden = (pos == "G") & (sos < 0.0) & (usg > 24)  # NaN comparisons are False
        nba_usg = (
            18 + 0.45 * (usg - 22) + 3.0 * (slot - 0.5) + 0.6 * t + 0.10 * (pace - 99.5)
            - np.where(hidden, 3.5, 0.0) + rng.normal(0, 2.2, n) * noise_scale
        )
        nba_3pa = _clip(
            6 + 26 * tpar + 80 * (t3 - 0.38) + 15 * (tpct - 0.33) - np.where(ctx == "intl", 1.0, 0.0)
            + rng.normal(0, 2.5, n) * noise_scale, 0, 20)
        nba_ast = _clip(
            11 + 0.55 * (ast - 14) + 12 * (tast - 0.60) + rng.normal(0, 2.5, n) * noise_scale, 2, 40)
        nba_trb = _clip(
            1.0 + 0.8 * (0.35 * orb + 0.65 * drb) + 0.06 * (height - 79) + rng.normal(0, 1.5, n) * noise_scale, 3, 30)
        nba_stocks = _clip(
            0.8 + 0.6 * (stl + blk) + 0.03 * (height - 79) + rng.normal(0, 0.7, n) * noise_scale, 0.5, 9)

        for i in range(n):
            pid += 1
            pid_s = f"P{pid:05d}"
            prospects.append(
                {
                    "player_id": pid_s, "name": f"Synthetic Player {pid}", "draft_year": year,
                    "draft_pick": pick[i], "pre_nba_context": ctx[i], "position": pos[i],
                    "age_at_draft": age[i], "height_in": height[i], "weight_lb": weight[i],
                    "mpg": mpg[i], "usg": usg[i], "ts_pct": ts[i], "three_par": tpar[i],
                    "three_pct": tpct[i], "ftr": ftr[i], "ast_pct": ast[i], "tov_pct": tov[i],
                    "orb_pct": orb[i], "drb_pct": drb[i], "stl_pct": stl[i], "blk_pct": blk[i],
                    "sos": sos[i], "team_pace": pace[i], "team_3par": t3[i],
                    "team_ast_rate": tast[i], "team_open_minutes": topen[i],
                }
            )
            if minutes[i] >= 1:  # players who never appeared are simply absent, as in real data
                outcomes.append(
                    {
                        "player_id": pid_s, "nba_minutes_first3": minutes[i],
                        "nba_usg": nba_usg[i], "nba_3pa_per100": nba_3pa[i],
                        "nba_ast_pct": nba_ast[i], "nba_trb_pct": nba_trb[i],
                        "nba_stocks_pct": nba_stocks[i],
                    }
                )
    return pd.DataFrame(prospects), pd.DataFrame(outcomes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prospects, outcomes = generate()
    prospects.to_csv(out / "prospects.csv", index=False)
    outcomes.to_csv(out / "nba_outcomes.csv", index=False)
    (out / "SYNTHETIC.txt").write_text(
        "This dataset is SYNTHETIC. Values were generated by src/make_synthetic.py and "
        "describe no real players. Delete this file only when the CSVs hold real data.\n"
    )
    print(f"wrote {len(prospects)} prospects, {len(outcomes)} outcome rows to {out}")
