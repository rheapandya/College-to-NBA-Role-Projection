"""Figures. Colors follow a small validated palette: blue and orange for the two model
families, gray for the baseline, and a blue-to-red diverging ramp with a gray midpoint
for residuals. Every figure has a legend or direct labels, and the underlying numbers
are written to CSV next to the figures so nothing depends on color alone."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from .config import TARGETS
from .residuals import LEVEL_ORDER, SEGMENT_VARS

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIV = LinearSegmentedColormap.from_list("div", ["#6fa3e3", "#f0efec", "#ee7d7c"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 10,
    }
)

BANNER = "SYNTHETIC DATA - pipeline test only, these are not findings about real players"


def _clean(ax, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _title(fig, text, synthetic, y=0.985):
    fig.suptitle(text, x=0.01, y=y, ha="left", fontsize=13, fontweight="bold", color=INK)
    if synthetic:
        fig.text(0.01, y - 0.045, BANNER, ha="left", va="top", fontsize=9, color=INK2, style="italic")


def fig_model_skill(metrics, outdir, synthetic):
    models = [
        ("slot_only", "Draft slot only", MUTED),
        ("ridge_player", "Ridge: pre-draft profile + slot", BLUE),
        ("gbm_player", "Gradient boosting: pre-draft profile + slot", ORANGE),
    ]
    targets = list(TARGETS)
    fig, axes = plt.subplots(len(targets), 1, figsize=(8.5, 1.15 * len(targets) + 1.9), sharex=True)
    for ax, t in zip(axes, targets):
        for j, (m, _, color) in enumerate(models):
            r = metrics[(metrics.target == t) & (metrics.model == m)].iloc[0]
            ax.errorbar(
                r.skill_r2, j, xerr=[[r.skill_r2 - r.skill_lo], [r.skill_hi - r.skill_r2]],
                fmt="o", color=color, lw=2, ms=8, mec=SURFACE, mew=2, capsize=0,
            )
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_ylim(-0.7, len(models) - 0.3)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.set_ylabel(TARGETS[t], rotation=0, ha="right", va="center", labelpad=10, color=INK)
        _clean(ax)
    axes[-1].set_xlabel("Out-of-sample skill vs. predicting the training average (R², leave-one-draft-year-out)")
    handles = [Line2D([0], [0], marker="o", color=c, lw=2, ms=8, label=l) for _, l, c in models]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.0))
    _title(fig, "How much of a player's NBA role can be projected before the draft?", synthetic)
    fig.text(0.01, 0.905 if synthetic else 0.945, "Dots are pooled out-of-sample R²; bars are 95% intervals from resampling draft classes.",
             fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.13, 1, 0.88 if synthetic else 0.92))
    fig.savefig(outdir / "fig1_model_skill.png", dpi=170)
    plt.close(fig)


def fig_team_fit(team, outdir, synthetic):
    targets = list(TARGETS)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    series = [("ridge_player", "Ridge", BLUE, -0.15), ("gbm_player", "Gradient boosting", ORANGE, 0.15)]
    for key, label, color, off in series:
        d = team[team.without_team == key].set_index("target").loc[targets]
        y = np.arange(len(targets)) + off
        ax.errorbar(
            d.mse_change_pct, y, xerr=[d.mse_change_pct - d.lo, d.hi - d.mse_change_pct],
            fmt="o", color=color, lw=2, ms=8, mec=SURFACE, mew=2, label=label,
        )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([TARGETS[t] for t in targets], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Change in out-of-sample error after adding drafting-team context (%)\nLeft of zero = team context helps; bars are 95% intervals from resampling draft classes")
    _clean(ax)
    ax.legend(frameon=False, loc="upper left")
    _title(fig, "Does the team a player lands on change what he becomes?", synthetic)
    fig.tight_layout(rect=(0, 0, 1, 0.90 if synthetic else 0.94))
    fig.savefig(outdir / "fig2_team_fit.png", dpi=170)
    plt.close(fig)


def fig_residual_segments(seg_tests, outdir, synthetic):
    targets = list(TARGETS)
    order = {v: i for i, v in enumerate(SEGMENT_VARS)}
    def _lvl(row):
        levels = LEVEL_ORDER.get(row.segment, [])
        return levels.index(row.level) if row.level in levels else len(levels)

    keys = seg_tests[["segment", "level"]].drop_duplicates().copy()
    keys["o"] = keys.segment.map(order)
    keys["l"] = keys.apply(_lvl, axis=1)
    keys = keys.sort_values(["o", "l", "level"])
    labels = [f"{s}: {l}" for s, l in zip(keys.segment, keys.level)]
    Z = np.full((len(keys), len(targets)), np.nan)
    F = np.zeros_like(Z, dtype=bool)
    lookup = seg_tests.set_index(["segment", "level", "target"])
    for i, (s, l) in enumerate(zip(keys.segment, keys.level)):
        for j, t in enumerate(targets):
            if (s, l, t) in lookup.index:
                r = lookup.loc[(s, l, t)]
                Z[i, j], F[i, j] = r.mean_z, bool(r.flag_fdr10)
    vmax = max(0.4, np.nanmax(np.abs(Z)))
    fig, ax = plt.subplots(figsize=(9, 0.26 * len(labels) + 2.6))
    im = ax.imshow(np.ma.masked_invalid(Z), cmap=DIV, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([TARGETS[t].replace(" (%)", "").replace(" per 100 possessions", "/100") for t in targets], rotation=0, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if not np.isnan(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:+.2f}{'*' if F[i, j] else ''}", ha="center", va="center",
                        fontsize=8, color=INK, fontweight="bold" if F[i, j] else "normal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("mean standardized residual (+ = did more than projected)", color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    _title(fig, "Where do projections miss? (* = survives 10% false-discovery control)", synthetic)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if synthetic else 0.96))
    fig.savefig(outdir / "fig3_residual_segments.png", dpi=170)
    plt.close(fig)


def fig_calibration(resid, outdir, synthetic):
    targets = list(TARGETS)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    for ax, t in zip(axes.ravel(), targets):
        d = resid[resid.target == t].copy()
        d["bin"] = pd.qcut(d["pred"], 10, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(pred=("pred", "mean"), y=("y", "mean"))
        lo, hi = min(g.pred.min(), g.y.min()), max(g.pred.max(), g.y.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
        ax.plot(g.pred, g.y, "o-", color=BLUE, lw=2, ms=7, mec=SURFACE, mew=1.5)
        ax.set_title(TARGETS[t], fontsize=9.5, color=INK, loc="left")
        _clean(ax, "both")
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0, 0.6, "Each dot is a tenth of players,\ngrouped by predicted value.\nOn the gray line = well calibrated.",
                          fontsize=9, color=INK2, va="top")
    fig.supxlabel("Average out-of-sample prediction", color=INK2, fontsize=9)
    fig.supylabel("Average actual first-three-year outcome", color=INK2, fontsize=9)
    _title(fig, "Calibration of the pre-draft projections", synthetic)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.9 if synthetic else 0.94))
    fig.savefig(outdir / "fig4_calibration.png", dpi=170)
    plt.close(fig)


def make_all(outdir, metrics, team, seg_tests, resid, synthetic):
    outdir = Path(outdir)
    fig_model_skill(metrics, outdir, synthetic)
    fig_team_fit(team, outdir, synthetic)
    fig_residual_segments(seg_tests, outdir, synthetic)
    fig_calibration(resid, outdir, synthetic)
