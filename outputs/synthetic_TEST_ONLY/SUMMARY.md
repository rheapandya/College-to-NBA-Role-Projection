# Results summary (auto-generated)

> **SYNTHETIC DATA.** This run used generated stand-in data to test the pipeline. Nothing below is a finding about real players. Two effects were planted on purpose (a team-fit effect on 3-point volume and a hidden usage miss for guards from weaker schedules), so the checks here show whether the pipeline *can* detect such effects.

**Data:** 910 prospects, draft classes 2010-2023. 420 (46%) logged at least 500 NBA minutes in their first three seasons and so have a role outcome; the rest are excluded from fitting and scoring. Inverse-probability weighting for that selection: off.

## 1. How predictable is each role outcome?

Skill is out-of-sample R² against predicting the training-fold average, leave-one-draft-year-out. Intervals come from resampling draft classes.

| target | model | rmse | skill (95% CI) | classes beating avg |
|---|---|---|---|---|
| Usage rate (%) | slot_only | 2.93 | 0.33 (0.25 to 0.38) | 86% |
| Usage rate (%) | ridge_player | 2.37 | 0.56 (0.48 to 0.62) | 100% |
| Usage rate (%) | gbm_player | 2.34 | 0.57 (0.48 to 0.63) | 100% |
| 3PA per 100 possessions | slot_only | 5.12 | 0.01 (-0.02 to 0.03) | 57% |
| 3PA per 100 possessions | ridge_player | 3.71 | 0.48 (0.41 to 0.54) | 100% |
| 3PA per 100 possessions | gbm_player | 3.7 | 0.48 (0.42 to 0.54) | 100% |
| Assist rate (%) | slot_only | 4.77 | 0.01 (-0.02 to 0.04) | 71% |
| Assist rate (%) | ridge_player | 2.77 | 0.67 (0.62 to 0.72) | 100% |
| Assist rate (%) | gbm_player | 2.83 | 0.65 (0.60 to 0.70) | 100% |
| Rebound rate (%) | slot_only | 4.37 | -0.00 (-0.01 to 0.00) | 71% |
| Rebound rate (%) | ridge_player | 1.43 | 0.89 (0.88 to 0.91) | 100% |
| Rebound rate (%) | gbm_player | 1.45 | 0.89 (0.88 to 0.90) | 100% |
| Steal + block rate (%) | slot_only | 1.47 | -0.00 (-0.00 to -0.00) | 21% |
| Steal + block rate (%) | ridge_player | 0.74 | 0.74 (0.70 to 0.78) | 100% |
| Steal + block rate (%) | gbm_player | 0.78 | 0.72 (0.67 to 0.76) | 100% |

## 2. Does drafting-team context add predictive value?

Negative means lower error with team context. An interval that spans zero means no clear evidence.

| target | without_team | mse change % (95% CI) | P(team helps) |
|---|---|---|---|
| Usage rate (%) | ridge_player | +0.4 (-1.4 to +2.3) | 0.34 |
| Usage rate (%) | gbm_player | +1.5 (-1.0 to +3.8) | 0.11 |
| 3PA per 100 possessions | ridge_player | -52.6 (-58.6 to -45.8) | 1.0 |
| 3PA per 100 possessions | gbm_player | -52.9 (-58.8 to -46.2) | 1.0 |
| Assist rate (%) | ridge_player | -3.3 (-7.6 to +1.8) | 0.9 |
| Assist rate (%) | gbm_player | -3.4 (-6.8 to +0.5) | 0.96 |
| Rebound rate (%) | ridge_player | +1.5 (+0.1 to +3.0) | 0.02 |
| Rebound rate (%) | gbm_player | +1.0 (-1.0 to +2.9) | 0.16 |
| Steal + block rate (%) | ridge_player | +1.7 (+1.3 to +2.3) | 0.0 |
| Steal + block rate (%) | gbm_player | +2.0 (+0.2 to +4.2) | 0.02 |

## 3. Where do projections systematically miss?

140 segment x outcome cells were tested. By chance alone, roughly 7 would look 'significant' at 5% without any correction, so only cells surviving a 10% false-discovery correction are listed.

No segment survived the correction.

### Is any learnable structure left in the misses?

Single-variable segments cannot see interactions, so a shallow tree tries to predict each residual from the pre-draft profile, scored leave-one-draft-year-out. At or below zero means the misses look like noise given these features. The tree's rules are in `residual_rules_EXPLORATORY.txt` and are in-sample, so treat them as hypotheses.

| target | n | out-of-sample R² of residuals |
|---|---|---|
| 3PA per 100 possessions | 420 | -0.224 |
| Assist rate (%) | 420 | -0.076 |
| Steal + block rate (%) | 420 | -0.173 |
| Rebound rate (%) | 420 | -0.127 |
| Usage rate (%) | 420 | -0.159 |

## 4. Leakage check

Target shuffled within each draft class, then the same CV re-run. Skill should be at or below zero. A clearly positive number would signal leakage.

| model | shuffled_skill_r2 |
|---|---|
| ridge_player | -0.014 |
| gbm_player | -0.051 |

## Limits to state alongside any result
- Role outcomes exist only for players who got minutes, so results describe players who made it onto the floor.
- Draft classes are small; every interval above reflects only ~14 independent classes.
- Draft slot is an input and already encodes what teams believed, so skill beyond slot-only is the interesting number.
- Team context uses the prior season's numbers, so it measures team tendencies, not coaching changes after the draft.
- The blind residual searches (segments and the tree) have little power for narrow groups of a few dozen players. Absence of a flag is not evidence of absence. Test a specific hypothesis directly with `test_group` in `src/residuals.py`.

