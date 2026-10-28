# Data you need to supply

Put two CSVs in `data/real/` and run `python run.py --data-dir data/real`.
The loader checks every required column and stops with a clear message if one is missing.

Real run uses draft classes 2011 through 2023 (2010 needs a 2010 team table). Later classes do not yet have three full NBA seasons.

## prospects.csv: one row per prospect (drafted or undrafted)

| column | meaning | likely source |
|---|---|---|
| player_id | any unique id you create | you |
| name | player name | draft page |
| draft_year | year of the draft | draft page |
| draft_pick | overall pick; leave blank if undrafted | Basketball Reference draft page |
| pre_nba_context | `ncaa`, `intl`, or `gleague` | you |
| position | `G`, `W`, or `B` (guard / wing / big) | draft page or roster, then bucket it |
| age_at_draft | age on draft night | player page |
| height_in, weight_lb | measurements in inches and pounds | draft combine or player page |
| mpg, usg, ts_pct, three_par, three_pct, ftr | minutes, usage, true shooting, 3PA/FGA, 3P%, FTA/FGA for the last pre-draft season (use fractions for the last four, e.g. 0.57) | Bart Torvik (via toRvik or cbbdata) |
| ast_pct, tov_pct, orb_pct, drb_pct, stl_pct, blk_pct | same season, in percent units | Bart Torvik |
| sos | a strength-of-schedule or conference-strength number; leave blank for non-NCAA players | Bart Torvik |
| team_pace, team_3par, team_ast_rate, team_open_minutes | the drafting team's PRIOR-season pace, 3PA/FGA, assists/FGM, and the share of that team's prior-season minutes at the player's position group that left in the offseason | Basketball Reference team pages |

Missing values are fine in the numeric columns. They are imputed inside each cross-validation fold.

`team_*` columns must come from the season before the draft, not from the player's own NBA years.
Otherwise the team-fit test leaks the answer.

## nba_outcomes.csv: one row per player who appeared in an NBA game

Everything is averaged (minutes-weighted) over the player's first three NBA seasons.

| column | meaning | likely source |
|---|---|---|
| player_id | matches prospects.csv | you |
| nba_minutes_first3 | total minutes in the first three seasons | Basketball Reference |
| nba_usg | usage % | advanced table |
| nba_3pa_per100 | 3-point attempts per 100 possessions | per-100-possessions table |
| nba_ast_pct, nba_trb_pct | assist % and total rebound % | advanced table |
| nba_stocks_pct | steal % + block % | advanced table |

Prospects who never appear in this file are treated as 0 minutes. Players under 500 minutes are
excluded from fitting and scoring because their rates are too noisy to be a "role".

## Matching players across sites

Player ids differ between sites. Join on a cleaned name (lowercase, strip punctuation and suffixes
like Jr.) plus draft year, then check the unmatched rows by hand.

## Terms of use

Export tables by hand where a site offers it (Basketball Reference has a "Get table as CSV" option)
and check each site's terms before scraping. Bulk scraping often violates them and gets you blocked.

## Testing without real data

`python -m src.make_synthetic --out data/synthetic` writes generated data with the same schema.
A `SYNTHETIC.txt` file marks it, and every output from that data is labeled as a pipeline test.
