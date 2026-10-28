"""Build data/real/prospects.csv and data/real/nba_outcomes.csv from the raw exports.

Run from the project root:  python -I scripts/build_real.py

Inputs (all under data/raw/):
  draft/draft_<year>.csv      Basketball Reference draft pages (two header rows, footer notes)
  players/players_<x>.csv     Basketball Reference player index (From/To are season END years)
  torvik/torvik_<year>.csv    Bart Torvik player-season export, 67 columns, no header
  nba/<year>_advanced.csv     league advanced tables, year = season END year
  nba/<year>_per_poss.csv     league per-100-possession tables

Design choices (also written to data/real/BUILD_REPORT.md):
  * A draftee's pre-draft season is the Torvik season whose year equals the draft year.
  * Team context comes from the drafting team's PRIOR season, i.e. the NBA table whose
    year equals the draft year. Tables before 2011 were not exported, so the 2010 class is dropped.
  * Outcomes use each player's first three NBA seasons in which he appears, starting no earlier
    than the season after the draft. Totals rows (TOT/2TM/...) are used for traded players.
  * Only players matched to a Torvik college season are kept (draft-only, college-only sample).
"""
import csv
import glob
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "real")
FIRST_CLASS = 2011  # prior-season team tables start at 2011

# Torvik columns (no header): positional map.
T = dict(name=0, team=1, conf=2, gp=3, min_pct=4, usg=6, efg=7, ts=8, orb=9, drb=10, ast=11,
         tov=12, ftr=24, two_a=17, three_a=20, three_pct=21, blk=22, stl=23, cls=25, ht=26,
         role=64, birth=66)

ROLE_TO_POS = {"Pure PG": "G", "Scoring PG": "G", "Combo G": "G", "Wing G": "G",
               "Wing F": "W", "Stretch 4": "W", "PF/C": "B", "C": "B"}
POWER = {"ACC", "B10", "B12", "BE", "SEC", "P12", "P10"}
HIGH = {"A10", "Amer", "MWC", "WCC", "MVC", "CUSA"}
TOT_TEAMS = {"TOT", "2TM", "3TM", "4TM", "5TM"}
TEAM_FIX = {(2014, "CHO"): "CHA"}  # draft page uses current franchise code for 2014


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("*", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", " ", s)
    return re.sub(r"[^a-z]", "", s)


def norm_school(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = s.replace("university", " ").replace("college", " ").replace("state", "st")
    s = s.replace("saint", "st").replace("&", " ")
    return re.sub(r"[^a-z]", "", s)


def name_tokens(s: str):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", " ", s.replace("*", ""))
    return [w for w in re.sub(r"[^a-z ]", " ", s).split() if w]


def fuzzy_candidates(name: str, college: str, rows):
    """Same college + same last name; if several, require the first two letters of the first name to agree."""
    tk = name_tokens(name)
    if not tk:
        return []
    out = [r for r in rows if (lambda t: t and t[-1] == tk[-1])(name_tokens(r[T["name"]]))
           and school_match(r[T["team"]], college)]
    if len(out) > 1:
        out = [r for r in out if name_tokens(r[T["name"]])[0][:2] == tk[0][:2]]
    return out


def school_match(a: str, b: str) -> bool:
    a, b = norm_school(a), norm_school(b)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def parse_height(h: str):
    m = re.fullmatch(r"\s*(\d)-(\d{1,2})\s*", h or "")
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read_draft():
    rows_out = []
    for f in sorted(glob.glob(os.path.join(RAW, "draft", "draft_*.csv"))):
        year = int(os.path.basename(f)[6:10])
        rows = list(csv.reader(open(f, encoding="utf-8")))
        hdr = rows[1]
        i_pk, i_tm, i_pl, i_col = hdr.index("Pk"), hdr.index("Tm"), hdr.index("Player"), hdr.index("College")
        for r in rows[2:]:
            if len(r) > i_col and r[i_pk].strip().isdigit():
                rows_out.append(dict(draft_year=year, draft_pick=int(r[i_pk]), tm=r[i_tm],
                                     name=r[i_pl], college=r[i_col]))
    return pd.DataFrame(rows_out)


def read_players():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "players", "players_*.csv"))):
        rows += list(csv.DictReader(open(f, encoding="utf-8")))
    idx = defaultdict(list)
    for r in rows:
        idx[norm_name(r["Player"])].append(r)
    return idx


def read_torvik():
    by_year = {}
    for f in sorted(glob.glob(os.path.join(RAW, "torvik", "torvik_*.csv"))):
        year = int(os.path.basename(f)[7:11])
        d = defaultdict(list)
        for r in csv.reader(open(f, encoding="utf-8")):
            if len(r) == 67 and int(r[31]) == year:
                d[norm_name(r[T["name"]])].append(r)
        by_year[year] = d
    return by_year


def read_nba():
    adv, per = {}, {}
    for f in glob.glob(os.path.join(RAW, "nba", "*_advanced.csv")):
        y = int(os.path.basename(f)[:4])
        adv[y] = pd.read_csv(f)
    for f in glob.glob(os.path.join(RAW, "nba", "*_per_poss.csv")):
        y = int(os.path.basename(f)[:4])
        per[y] = pd.read_csv(f)
    return adv, per


def season_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season: the totals row if the player was traded, else the lone row."""
    d = df[df["Player-additional"].notna()].copy()
    d["is_tot"] = d["Team"].isin(TOT_TEAMS)
    d = d.sort_values(["Player-additional", "is_tot"], ascending=[True, False])
    return d.drop_duplicates("Player-additional", keep="first")


def team_context(per: dict):
    """Minutes-weighted team 3PA/FGA and AST/FGM for each (season, team), from player-team rows."""
    out = {}
    for y, df in per.items():
        d = df[df["Player-additional"].notna() & ~df["Team"].isin(TOT_TEAMS)].copy()
        for c in ["MP", "FGA", "3PA", "FG", "AST"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["MP", "FGA", "3PA", "FG", "AST"])
        for tm, g in d.groupby("Team"):
            w = g["MP"]
            out[(y, tm)] = dict(team_3par=(g["3PA"] * w).sum() / (g["FGA"] * w).sum(),
                                team_ast_rate=(g["AST"] * w).sum() / (g["FG"] * w).sum())
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    draft = read_draft()
    players = read_players()
    torvik = read_torvik()
    torvik_rows = {y: [r for v in d.values() for r in v] for y, d in torvik.items()}
    adv, per = read_nba()
    ctx = team_context(per)
    report = []

    n_all = len(draft)
    draft = draft[draft.draft_year >= FIRST_CLASS].reset_index(drop=True)
    report.append(f"Draft rows read: {n_all}. Classes before {FIRST_CLASS} dropped (no prior-season team table): "
                  f"{n_all - len(draft)} rows removed; {len(draft)} remain.")

    # ---- match draftees to a Torvik pre-draft season
    prospects, unmatched = [], []
    for _, d in draft.iterrows():
        y = d.draft_year
        if not d.college:
            unmatched.append((y, d["name"], "no college listed (intl / G League)"))
            continue
        cands = torvik.get(y, {}).get(norm_name(d["name"]), [])
        if len(cands) > 1:
            good = [c for c in cands if school_match(c[T["team"]], d.college)]
            cands = good or cands
        fuzzy = False
        if not cands:
            cands = fuzzy_candidates(d["name"], d.college, torvik_rows.get(y, []))
            fuzzy = len(cands) == 1
            cands = cands if fuzzy else []
        if not cands:
            unmatched.append((y, d["name"], f"college '{d.college}' but no Torvik row in {y}"))
            continue
        # Same name, different school, and neither school looks alike: keep but flag.
        t = max(cands, key=lambda c: fnum(c[T["min_pct"]]))
        flag = "" if school_match(t[T["team"]], d.college) else f"school differs: BR '{d.college}' vs Torvik '{t[T['team']]}'"
        if fuzzy:
            flag = (flag + "; " if flag else "") + f"fuzzy name match: BR '{d['name']}' -> Torvik '{t[T['name']]}'"
        prospects.append((d, t, flag))
    report.append(f"Matched to Torvik: {len(prospects)}. Unmatched: {len(unmatched)}.")

    # ---- match to the NBA player index (for outcomes) by name + rookie-year window + college
    rows_p, id_map, school_flags, amb = [], {}, [], 0
    for d, t, flag in prospects:
        y = d.draft_year
        cands = [r for r in players.get(norm_name(d["name"]), [])
                 if r["From"].isdigit() and y + 1 <= int(r["From"]) <= y + 4]
        if len(cands) > 1:
            good = [r for r in cands if any(school_match(s, d.college) for s in [r["Colleges"]]
                    ) or school_match(r["Colleges"], d.college)]
            cands = good or cands
            if len(cands) > 1:
                amb += 1
                cands = sorted(cands, key=lambda r: int(r["From"]))[:1]
        nba_id = cands[0]["Player-additional"] if cands else None
        weight = fnum(cands[0]["Wt"]) if cands else np.nan
        pid = f"{y}_{d.draft_pick:02d}"
        id_map[pid] = nba_id
        if flag:
            school_flags.append((y, d["name"], flag))

        birth = t[T["birth"]]
        try:
            by, bm, bd = map(int, birth.split("-"))
            age = (date(y, 6, 25) - date(by, bm, bd)).days / 365.25
        except Exception:
            age = np.nan
        two_a, three_a = fnum(t[T["two_a"]]), fnum(t[T["three_a"]])
        conf = t[T["conf"]]
        rows_p.append(dict(
            player_id=pid, name=d["name"], draft_year=y, draft_pick=d.draft_pick,
            pre_nba_context="ncaa", position=ROLE_TO_POS.get(t[T["role"]], np.nan),
            age_at_draft=age, height_in=parse_height(t[T["ht"]]), weight_lb=weight,
            mpg=fnum(t[T["min_pct"]]) * 0.4,  # Min% of team minutes -> approx minutes per 40-minute game
            usg=fnum(t[T["usg"]]), ts_pct=fnum(t[T["ts"]]) / 100,
            three_par=three_a / (two_a + three_a) if (two_a + three_a) > 0 else np.nan,
            three_pct=fnum(t[T["three_pct"]]), ftr=fnum(t[T["ftr"]]) / 100,
            ast_pct=fnum(t[T["ast"]]), tov_pct=fnum(t[T["tov"]]),
            orb_pct=fnum(t[T["orb"]]), drb_pct=fnum(t[T["drb"]]),
            stl_pct=fnum(t[T["stl"]]), blk_pct=fnum(t[T["blk"]]),
            sos=2.0 if conf in POWER else 1.0 if conf in HIGH else 0.0,  # conference-tier proxy
            college_team=t[T["team"]], conf=conf, torvik_min_pct=fnum(t[T["min_pct"]]),
            tm=TEAM_FIX.get((y, d.tm), d.tm),
        ))
    P = pd.DataFrame(rows_p)
    report.append(f"Ambiguous NBA-index name matches resolved by earliest rookie year: {amb}.")

    # team context from the drafting team's prior season (table year == draft year)
    P["team_3par"] = [ctx.get((y, tm), {}).get("team_3par", np.nan) for y, tm in zip(P.draft_year, P.tm)]
    P["team_ast_rate"] = [ctx.get((y, tm), {}).get("team_ast_rate", np.nan) for y, tm in zip(P.draft_year, P.tm)]
    P["team_pace"] = np.nan
    P["team_open_minutes"] = np.nan
    miss_team = P[P.team_3par.isna()][["draft_year", "name", "tm"]]
    report.append(f"Drafting-team context missing for {len(miss_team)} prospects.")

    # ---- outcomes: first three NBA seasons per player
    adv_s = {y: season_rows(df).set_index("Player-additional") for y, df in adv.items()}
    per_s = {y: season_rows(df).set_index("Player-additional") for y, df in per.items()}
    out_rows = []
    for _, r in P.iterrows():
        nid = id_map[r.player_id]
        if not nid:
            continue
        seasons = [y for y in range(r.draft_year + 1, 2027) if nid in adv_s.get(y, pd.DataFrame()).index]
        seasons = seasons[:3]
        if not seasons:
            continue
        mp = np.array([fnum(adv_s[y].loc[nid, "MP"]) for y in seasons])
        if mp.sum() <= 0:
            continue

        def wavg(tab, col):
            v = np.array([fnum(tab[y].loc[nid, col]) for y in seasons])
            ok = ~np.isnan(v)
            return float((v[ok] * mp[ok]).sum() / mp[ok].sum()) if ok.any() else np.nan

        stocks = np.array([fnum(adv_s[y].loc[nid, "STL%"]) + fnum(adv_s[y].loc[nid, "BLK%"]) for y in seasons])
        ok = ~np.isnan(stocks)
        out_rows.append(dict(
            player_id=r.player_id, nba_minutes_first3=float(mp.sum()), n_seasons=len(seasons),
            nba_usg=wavg(adv_s, "USG%"),
            nba_3pa_per100=wavg(per_s, "3PA") if all(nid in per_s.get(y, pd.DataFrame()).index for y in seasons) else np.nan,
            nba_ast_pct=wavg(adv_s, "AST%"), nba_trb_pct=wavg(adv_s, "TRB%"),
            nba_stocks_pct=float((stocks[ok] * mp[ok]).sum() / mp[ok].sum()) if ok.any() else np.nan,
        ))
    O = pd.DataFrame(out_rows)

    # classes whose third NBA season (draft_year + 3) is not in the exported tables are incomplete
    max_year = max(adv)
    incomplete = sorted(y for y in P.draft_year.unique() if y + 3 > max_year)
    report.append(f"Classes with fewer than 3 possible NBA seasons in the exports (through {max_year}): {incomplete or 'none'}.")

    keep = [c for c in P.columns if c not in ("college_team", "conf", "torvik_min_pct", "tm")]
    P[keep].to_csv(os.path.join(OUT, "prospects.csv"), index=False)
    O.drop(columns=["n_seasons"]).to_csv(os.path.join(OUT, "nba_outcomes.csv"), index=False)
    P.to_csv(os.path.join(OUT, "prospects_with_audit_cols.csv"), index=False)

    n_obs = int((O.nba_minutes_first3 >= 500).sum())
    report.append(f"prospects.csv: {len(P)} rows. nba_outcomes.csv: {len(O)} rows ({n_obs} with >= 500 minutes).")
    report.append("Classes: " + ", ".join(f"{y}: {n}" for y, n in sorted(Counter(P.draft_year).items())))
    report.append("Position mix (from Torvik role): " + str(P.position.value_counts(dropna=False).to_dict()))
    nan_cols = P[keep].isna().mean().round(3)
    report.append("Share missing by column: " + str({k: v for k, v in nan_cols.items() if v > 0}))

    with open(os.path.join(OUT, "BUILD_REPORT.md"), "w") as fh:
        fh.write("# Build report (data/real)\n\n" + "\n".join(f"- {x}" for x in report) + "\n\n")
        fh.write("## Unmatched draftees\n\n")
        for y, n, why in unmatched:
            fh.write(f"- {y} {n}: {why}\n")
        fh.write("\n## Matched with a flag: school differs or fuzzy name (check by hand)\n\n")
        for y, n, why in school_flags:
            fh.write(f"- {y} {n}: {why}\n")
        fh.write("\n## Missing drafting-team context\n\n")
        fh.write(miss_team.to_string(index=False) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    sys.exit(main())
