"""Auto-generate SUMMARY.md from the result tables, so the numbers in the writeup are
always the numbers the code produced."""
import pandas as pd

from .config import MIN_NBA_MINUTES, TARGETS

BANNER = (
    "> **SYNTHETIC DATA.** This run used generated stand-in data to test the pipeline. "
    "Nothing below is a finding about real players. Two effects were planted on purpose "
    "(a team-fit effect on 3-point volume and a hidden usage miss for guards from weaker "
    "schedules), so the checks here show whether the pipeline *can* detect such effects.\n"
)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def write_summary(path, df, metrics, team, seg_tests, struct, perm, ipw, synthetic):
    n_total = len(df)
    n_obs = int(df["observed"].sum())
    years = f"{int(df.draft_year.min())}-{int(df.draft_year.max())}"
    out = ["# Results summary (auto-generated)\n"]
    if synthetic:
        out.append(BANNER)
    out.append(
        f"**Data:** {n_total} prospects, draft classes {years}. {n_obs} "
        f"({n_obs / n_total:.0%}) logged at least {MIN_NBA_MINUTES} NBA minutes in their first "
        f"three seasons and so have a role outcome; the rest are excluded from fitting and scoring. "
        f"Inverse-probability weighting for that selection: {'on' if ipw else 'off'}.\n"
    )

    # model skill
    show = metrics[metrics.model.isin(["slot_only", "ridge_player", "gbm_player"])].copy()
    show["skill (95% CI)"] = show.apply(lambda r: f"{r.skill_r2:.2f} ({r.skill_lo:.2f} to {r.skill_hi:.2f})", axis=1)
    show["rmse"] = show["rmse"].round(2)
    show["classes beating avg"] = (show["years_beating_mean"] * 100).round(0).astype(int).astype(str) + "%"
    show["target"] = show["target"].map(TARGETS)
    out.append("## 1. How predictable is each role outcome?\n")
    out.append(
        "Skill is out-of-sample R² against predicting the training-fold average, leave-one-draft-year-out. "
        "Intervals come from resampling draft classes.\n"
    )
    out.append(_md(show[["target", "model", "rmse", "skill (95% CI)", "classes beating avg"]]))

    # team fit
    t = team.copy()
    t["target"] = t["target"].map(TARGETS)
    t["mse change % (95% CI)"] = t.apply(lambda r: f"{r.mse_change_pct:+.1f} ({r.lo:+.1f} to {r.hi:+.1f})", axis=1)
    t["P(team helps)"] = t["p_team_helps"].round(2)
    out.append("\n## 2. Does drafting-team context add predictive value?\n")
    out.append("Negative means lower error with team context. An interval that spans zero means no clear evidence.\n")
    out.append(_md(t[["target", "without_team", "mse change % (95% CI)", "P(team helps)"]]))

    # residual segments
    n_tests = len(seg_tests)
    flagged = seg_tests[seg_tests.flag_fdr10]
    out.append("\n## 3. Where do projections systematically miss?\n")
    out.append(
        f"{n_tests} segment x outcome cells were tested. By chance alone, roughly {int(0.05 * n_tests)} would "
        "look 'significant' at 5% without any correction, so only cells surviving a 10% false-discovery "
        "correction are listed.\n"
    )
    if len(flagged):
        f = flagged.copy()
        f["target"] = f["target"].map(TARGETS)
        f["mean resid (z)"] = f.mean_z.round(2)
        f["95% CI"] = f.apply(lambda r: f"{r.ci_lo:+.2f} to {r.ci_hi:+.2f}", axis=1)
        f["q"] = f.q_bh.round(3)
        out.append(_md(f[["target", "segment", "level", "n", "mean resid (z)", "95% CI", "q"]]))
    else:
        out.append("No segment survived the correction.")

    # interaction structure
    s = struct.copy()
    s["target"] = s["target"].map(TARGETS)
    s["out-of-sample R² of residuals"] = s["oos_r2_of_residuals"].round(3)
    out.append("\n### Is any learnable structure left in the misses?\n")
    out.append(
        "Single-variable segments cannot see interactions, so a shallow tree tries to predict each residual "
        "from the pre-draft profile, scored leave-one-draft-year-out. At or below zero means the misses look "
        "like noise given these features. The tree's rules are in `residual_rules_EXPLORATORY.txt` and are "
        "in-sample, so treat them as hypotheses.\n"
    )
    out.append(_md(s[["target", "n", "out-of-sample R² of residuals"]]))

    # sanity
    out.append("\n## 4. Leakage check\n")
    out.append(
        "Target shuffled within each draft class, then the same CV re-run. Skill should be at or below zero. "
        "A clearly positive number would signal leakage.\n"
    )
    p = perm.copy()
    p["shuffled_skill_r2"] = p["shuffled_skill_r2"].round(3)
    out.append(_md(p))

    out.append(
        "\n## Limits to state alongside any result\n"
        "- Role outcomes exist only for players who got minutes, so results describe players who made it onto the floor.\n"
        "- Draft classes are small; every interval above reflects only ~14 independent classes.\n"
        "- Draft slot is an input and already encodes what teams believed, so skill beyond slot-only is the interesting number.\n"
        "- Team context uses the prior season's numbers, so it measures team tendencies, not coaching changes after the draft.\n"
        "- The blind residual searches (segments and the tree) have little power for narrow groups of a few dozen players. "
        "Absence of a flag is not evidence of absence. Test a specific hypothesis directly with `test_group` in `src/residuals.py`.\n"
    )
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
