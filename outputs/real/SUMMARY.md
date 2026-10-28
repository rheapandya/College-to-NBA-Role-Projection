# Results summary (auto-generated)

**Data:** 615 prospects, draft classes 2011-2023. 477 (78%) logged at least 500 NBA minutes in their first three seasons and so have a role outcome; the rest are excluded from fitting and scoring. Inverse-probability weighting for that selection: off.

## 1. How predictable is each role outcome?

Skill is out-of-sample R² against predicting the training-fold average, leave-one-draft-year-out. Intervals come from resampling draft classes.

| target | model | rmse | skill (95% CI) | classes beating avg |
|---|---|---|---|---|
| Usage rate (%) | slot_only | 3.53 | 0.21 (0.15 to 0.25) | 100% |
| Usage rate (%) | ridge_player | 3.13 | 0.38 (0.31 to 0.44) | 100% |
| Usage rate (%) | gbm_player | 3.22 | 0.34 (0.28 to 0.40) | 100% |
| 3PA per 100 possessions | slot_only | 3.17 | -0.00 (-0.00 to -0.00) | 23% |
| 3PA per 100 possessions | ridge_player | 1.9 | 0.64 (0.56 to 0.69) | 100% |
| 3PA per 100 possessions | gbm_player | 1.92 | 0.63 (0.56 to 0.68) | 100% |
| Assist rate (%) | slot_only | 6.93 | 0.02 (-0.01 to 0.04) | 69% |
| Assist rate (%) | ridge_player | 4.26 | 0.63 (0.57 to 0.69) | 100% |
| Assist rate (%) | gbm_player | 4.0 | 0.67 (0.62 to 0.71) | 100% |
| Rebound rate (%) | slot_only | 3.98 | 0.01 (-0.01 to 0.02) | 69% |
| Rebound rate (%) | ridge_player | 1.82 | 0.79 (0.77 to 0.81) | 100% |
| Rebound rate (%) | gbm_player | 1.83 | 0.79 (0.75 to 0.82) | 100% |
| Steal + block rate (%) | slot_only | 1.52 | 0.00 (-0.00 to 0.00) | 69% |
| Steal + block rate (%) | ridge_player | 0.87 | 0.67 (0.59 to 0.72) | 100% |
| Steal + block rate (%) | gbm_player | 0.91 | 0.64 (0.53 to 0.70) | 92% |

## 2. Does drafting-team context add predictive value?

Negative means lower error with team context. An interval that spans zero means no clear evidence.

| target | without_team | mse change % (95% CI) | P(team helps) |
|---|---|---|---|
| Usage rate (%) | ridge_player | +0.1 (-0.9 to +1.4) | 0.47 |
| Usage rate (%) | gbm_player | +0.5 (-0.7 to +1.7) | 0.21 |
| 3PA per 100 possessions | ridge_player | +0.4 (-1.0 to +2.2) | 0.3 |
| 3PA per 100 possessions | gbm_player | +0.0 (-1.1 to +1.2) | 0.46 |
| Assist rate (%) | ridge_player | +0.5 (+0.0 to +0.9) | 0.02 |
| Assist rate (%) | gbm_player | +1.9 (+0.7 to +3.3) | 0.0 |
| Rebound rate (%) | ridge_player | +0.6 (+0.0 to +1.2) | 0.02 |
| Rebound rate (%) | gbm_player | -1.0 (-1.8 to +0.1) | 0.96 |
| Steal + block rate (%) | ridge_player | +0.7 (-0.5 to +2.4) | 0.19 |
| Steal + block rate (%) | gbm_player | +1.5 (-0.1 to +3.2) | 0.03 |

## 3. Where do projections systematically miss?

130 segment x outcome cells were tested. By chance alone, roughly 6 would look 'significant' at 5% without any correction, so only cells surviving a 10% false-discovery correction are listed.

No segment survived the correction.

### Is any learnable structure left in the misses?

Single-variable segments cannot see interactions, so a shallow tree tries to predict each residual from the pre-draft profile, scored leave-one-draft-year-out. At or below zero means the misses look like noise given these features. The tree's rules are in `residual_rules_EXPLORATORY.txt` and are in-sample, so treat them as hypotheses.

| target | n | out-of-sample R² of residuals |
|---|---|---|
| 3PA per 100 possessions | 477 | -0.095 |
| Assist rate (%) | 477 | -0.179 |
| Steal + block rate (%) | 477 | -0.168 |
| Rebound rate (%) | 477 | -0.085 |
| Usage rate (%) | 477 | -0.114 |

## 4. Leakage check

Target shuffled within each draft class, then the same CV re-run. Skill should be at or below zero. A clearly positive number would signal leakage.

| model | shuffled_skill_r2 |
|---|---|
| ridge_player | -0.004 |
| gbm_player | -0.008 |

## Limits to state alongside any result
- Role outcomes exist only for players who got minutes, so results describe players who made it onto the floor.
- Draft classes are small; every interval above reflects only ~14 independent classes.
- Draft slot is an input and already encodes what teams believed, so skill beyond slot-only is the interesting number.
- Team context uses the prior season's numbers, so it measures team tendencies, not coaching changes after the draft.
- The blind residual searches (segments and the tree) have little power for narrow groups of a few dozen players. Absence of a flag is not evidence of absence. Test a specific hypothesis directly with `test_group` in `src/residuals.py`.

