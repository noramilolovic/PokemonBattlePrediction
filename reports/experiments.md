# Experiment log

All results: accuracy on train.jsonl with RepeatedStratifiedKFold (5 folds × 3 repeats,
random_state=42) unless noted otherwise. Rows below are appended by `experiments/cv_compare.py`.

| Date | Feature set | Model | Mean acc. | Std |
|---|---|---|---|---|
| 2026-10-29 | v1 team+timeline (253 features) | logreg | 0.6730 | 0.0063 |
| 2026-10-29 | v1 team+timeline (253 features) | rf | 0.6706 | 0.0076 |
| 2026-10-29 | v1 team+timeline (253 features) | hgb15 | 0.6752 | 0.0047 |
| 2026-10-30 | v2 team+timeline+species (1053 features) | logreg | 0.6575 | 0.0079 |
| 2026-10-30 | v2 team+timeline+species (1053 features) | rf | 0.6706 | 0.0056 |
| 2026-10-30 | v2 team+timeline+species (1053 features) | hgb15 | 0.6771 | 0.0033 |
| 2026-10-30 | v3 team+timeline+species+types (1057 features) | logreg | 0.6571 | 0.0078 |
| 2026-10-30 | v3 team+timeline+species+types (1057 features) | rf | 0.6698 | 0.0076 |
| 2026-10-30 | v3 team+timeline+species+types (1057 features) | hgb15 | 0.6770 | 0.0049 |
| 2026-10-30 | v3 tuned HGB (1057 features) | hgb_tuned | 0.6771 | 0.0063 |


### V3 Hyperparameter Tuning

Model: HistGradientBoostingClassifier

Best parameters:
- max_iter: 400
- learning_rate: 0.05
- max_leaf_nodes: 31
- l2_regularization: 5.0

RandomizedSearchCV best score: 0.6805
Repeated 5-fold CV (3 repeats): 0.6771
Standard deviation: 0.0063

Conclusion:
The tuned V3 model tied the V2 baseline in mean accuracy.
V2 remains the preferred simpler model.

### Extra feature blocks and ensembles (paired CV)

Paired comparison on identical folds (5 folds × 3 repeats), Nadeau-Bengio corrected t-test.
Team block here is the 56-feature version (with type matchup D and lead matchup F).
Scripts: `experiments/check_matchup.py`, `check_momentum.py`, `check_stall.py`.

| Date | Comparison | Model | Before | After | Diff | Folds won | Corrected p |
|---|---|---|---|---|---|---|---|
| 2026-10-31 | team+timeline → +matchup (317) | HistGB | 0.6716 | 0.6741 | +0.0025 | 10/15 | 0.386 |
| 2026-10-31 | team+timeline → +matchup (317) | LogReg | 0.6733 | 0.6710 | -0.0023 | 5/15 | 0.567 |
| 2026-10-31 | team+timeline → +matchup (317) | Ensemble LR+HGB | 0.6764 | 0.6793 | +0.0029 | 13/15 | 0.288 |
| 2026-11-02 | +matchup → +momentum (334) | Ensemble LR+HGB | 0.6793 | 0.6786 | -0.0007 | 5/15 | 0.747 |
| 2026-11-02 | 334 features, add Extra Trees | Ensemble LR+HGB → LR+HGB+ET | 0.6786 | 0.6799 | +0.0013 | 9/15 | 0.519 |
| 2026-11-03 | +momentum → +stall (370) | Ensemble LR+HGB+ET | 0.6799 | 0.6825 | +0.0026 | 8/15 | 0.361 |

Out-of-fold errors (5-fold, 334 features): LogReg 0.6757, HGB 0.6748, Extra Trees 0.6680.
Error correlation: LogReg/HGB 0.63, LogReg/ET 0.61, HGB/ET 0.75.

### Error analysis (OOF, ensemble LR+HGB+ET, 334 features, OOF acc. 0.6818)

- The model is well calibrated: in every probability bin the predicted p matches the real win rate.
- 36% of battles have p between 0.4 and 0.6, with only ~55% accuracy: they are still open at turn 30.
- Accuracy is ~0.60 when the hp difference at turn 30 is close to 0 and ~0.83 when it is large.
- No species or lead with a clearly lower accuracy. Battles with stall moves (rest, recover,
  reflect, rapid spin) are 3-5 points harder, but stall features did not help reliably (see above).

Conclusion: none of the extra blocks gives a significant gain. The limit is the information in the
first 30 turns, not the model.

### Submissions

| Date | Configuration (`make_submissions.py`) | CV | Public score |
|---|---|---|---|
| 2026-10-30 | `hgb_v1`: HGB default, team+timeline | 0.673 | 0.68066 |
| 2026-10-31 | `ensemble2`: LR+HGB, team+timeline+matchup | 0.6793 | – |
| 2026-10-31 | `hgb_reg_extra`: regularized HGB, team+timeline+extra | 0.6863 | 0.688 |
| 2026-11-03 | `ensemble3`: LR+HGB+ET, team+timeline+matchup+momentum | 0.6799 | 0.6880 |
| 2026-11-03 | `ensemble4_extra`: HGB+HGB_reg+LR+RF, team+timeline+extra (394 features) | 0.6878 | 0.698 |

CV for the two `extra` configurations: 5-fold, single run, seed 42 (`experiments/ensemble_tune.py`).
The other CV values come from repeated 5-fold (3 repeats), so the two protocols are not strictly comparable.
Public score = Kaggle public leaderboard.
