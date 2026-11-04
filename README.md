# PokemonBattlePrediction
Kaggle competition for the course Fundamentals of Data Science, the task is to train a machine learning model that can predict the final outcome of a Pokémon battle.

Each battle (Gen 2 singles) gives Player 1's full team, Player 2's lead Pokémon and the first
30 turns of the battle. The target is `player_won` (did Player 1 win). Metric: accuracy.

## How to run

```
pip install -r requirements.txt
```

Put the competition files in `data/` (`train.jsonl`, `test.jsonl`, `sample_submission.csv`,
`type_chart_gen2.csv`). On Kaggle, `src/data.py` finds them in `/kaggle/input` automatically.
Run everything from the repo root.

```
python make_submissions.py --list        # show the submission configurations
python make_submissions.py ensemble3     # write submissions/submission_ensemble3.csv
python make_submissions.py --all         # write all of them
```

Experiments (cross-validation, feature checks) are run as modules, for example:

```
python -m experiments.check_matchup
```

## Structure

```
src/
  data.py                loading train/test, target, writing submissions
  features_team.py       team block: stats, speed, items, types vs P2 lead, lead vs lead
  features_timeline.py   timeline block: hp, faints, status, boosts, moves over 30 turns
  features_matchup.py    surviving strength, active duel at turn 30, real move effectiveness
  features_momentum.py   volatility of the hp difference, recency of faints
  features_species.py    one-hot species (v3)
  features_types.py      type effectiveness vs leads (v3)
  features_stall.py      stall potential after turn 30 (experiment, not used in submissions)
  features_extra.py      per-species end hp + team target encoding (uses labels, built per fold)
  pipeline.py            build_X(battles, feature_set): named feature sets
  models.py              models for experiments and for the submissions
make_submissions.py      one named configuration (feature set + model) per submission
experiments/             CV comparisons, tuning, feature sanity checks
reports/experiments.md   log of all CV results and submissions
submissions/             generated CSV files (not in git)
```

## Feature sets

| Name | Blocks | Features |
|---|---|---|
| `v1` | team + timeline | 272 |
| `v3` | team + timeline + species + types | 1076 |
| `matchup` | team + timeline + matchup | 317 |
| `best` | team + timeline + matchup + momentum | 334 |

## Results

Accuracy with repeated stratified 5-fold CV on the training set (details in `reports/experiments.md`):

| Configuration | Features | Model | CV | Public |
|---|---|---|---|---|
| baseline | – | always majority class | 0.500 | |
| team only | team | HistGB | 0.541 | |
| timeline only | timeline | HistGB | 0.666 | |
| `hgb_v1` | `v1` | HistGB | 0.673 | 0.681 |
| `hgb_tuned_v3` | `v3` | tuned HistGB | 0.677 | |
| `ensemble2` | `matchup` | LogReg + HistGB (soft voting) | 0.679 | |
| `ensemble3` | `best` | LogReg + HistGB + Extra Trees (soft voting) | 0.680 | 0.688 |
| `hgb_reg_extra`   | `v1` + extra | regularized HistGB | 0.686 | 0.688 |
| `ensemble4_extra` | `v1` + extra | HistGB + regularized HistGB + LogReg + RF (mean probability) | 0.688 | 0.698 |

"extra" = per-species hp at turn 30 for the 60 most common species and a smoothed win rate
of the exact P1 team (target encoding, out-of-fold on train). See `src/features_extra.py`.

The timeline block carries almost all of the signal. Extra blocks (matchup, momentum, stall)
and extra models give gains of 0.001–0.003, which are not statistically significant
(corrected paired t-test). The model is well calibrated; the remaining errors are battles that
are still open after 30 turns.
