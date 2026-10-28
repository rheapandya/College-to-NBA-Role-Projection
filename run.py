"""End-to-end run: load data, cross-validate, test team fit, analyze residuals, make figures.

    python run.py --data-dir data/real          # your real CSVs
    python run.py --data-dir data/synthetic     # pipeline test on generated data
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from src.config import TARGETS
from src.cv import permutation_check, run_cv, summarize
from src.features import build_features, drop_empty_features
from src.io_utils import load_data
from src.plots import make_all
from src.report import write_summary
from src.residuals import add_segments, residual_frame, residual_structure, segment_tests, top_misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/real")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--ipw", action="store_true", help="weight training rows by 1/P(played)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap draws")
    args = ap.parse_args()

    t0 = time.time()
    df = load_data(args.data_dir)
    synthetic = bool(df["synthetic"].iloc[0])
    X = build_features(df)
    dropped = drop_empty_features(X)
    if dropped:
        print(f"[data] dropped all-missing features: {dropped}")
    out = Path(args.out) / ("synthetic_TEST_ONLY" if synthetic else "real")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[data] {len(df)} prospects, {int(df.observed.sum())} with a role outcome, "
          f"draft years {int(df.draft_year.min())}-{int(df.draft_year.max())}"
          f"{'  (SYNTHETIC)' if synthetic else ''}")

    print("[cv] leave-one-draft-year-out, all models x all targets ...")
    oof = run_cv(X, list(TARGETS), ipw=args.ipw)
    pd.concat([d.assign(target=t) for t, d in oof.items()]).to_csv(out / "oof_predictions.csv", index=False)

    print("[metrics] bootstrap over draft classes ...")
    metrics, team = summarize(oof, B=args.boot)
    metrics.to_csv(out / "metrics.csv", index=False)
    team.to_csv(out / "team_fit.csv", index=False)

    print("[residuals] segment tests ...")
    seg = add_segments(X.reset_index(drop=True))
    resid = residual_frame(oof, metrics)
    tests = segment_tests(resid, seg, B=args.boot)
    tests.to_csv(out / "residual_segments.csv", index=False)
    top_misses(resid).to_csv(out / "top_misses.csv", index=False)
    struct, rules = residual_structure(resid, X)
    struct.to_csv(out / "residual_structure.csv", index=False)
    (out / "residual_rules_EXPLORATORY.txt").write_text(rules + "\n")

    print("[sanity] shuffled-target leakage check ...")
    perm = permutation_check(X, "nba_usg")
    perm.to_csv(out / "leakage_check.csv", index=False)

    print("[figures] ...")
    make_all(out, metrics, team, tests, resid, synthetic)
    write_summary(out / "SUMMARY.md", df, metrics, team, tests, struct, perm, args.ipw, synthetic)
    print(f"[done] {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
