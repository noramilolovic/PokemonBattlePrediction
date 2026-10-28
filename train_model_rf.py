"""Train a tree-based model (gradient boosting or random forest) on team + timeline features.

Run from the repo root:
    py train_model.py

What it does
  1. builds the feature table (team block + timeline block, joined on battle_id);
  2. estimates accuracy with 5-fold stratified cross-validation on train.jsonl ONLY;
  3. fits on the whole train set and writes predictions for test.jsonl
     in the sample_submission.csv format (battle_id, player_won).

The test set is never used to choose features or parameters.
Needs: pandas, scikit-learn  ->  py -m pip install pandas scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import get_target, load_test, load_train, save_submission
from src.features_team import build_team_frame
from src.features_timeline import build_timeline_frame

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
MODEL_NAME = "rf"          # "hgb" = gradient boosting, "rf" = random forest
RUN_CV = True               # estimate accuracy with cross-validation
MAKE_SUBMISSION = True      # fit on all of train and write test predictions
SUBMISSION_PATH = f"submissions/{MODEL_NAME}_submission.csv"
SEED = 42                   # same seed in every script => comparable folds across the team
N_FOLDS = 5


def build_features(battles):
    """Team block + timeline block, one row per battle, indexed by battle_id."""
    return build_team_frame(battles).join(build_timeline_frame(battles))


def make_model(name):
    if name == "hgb":
        # many small, shallow trees added one after another; the settings below keep
        # it from overfitting (only 10,000 battles but ~270 columns)
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=SEED,
        )
    if name == "rf":
        # many independent deep-ish trees voting; averaging reduces variance
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        )
    raise ValueError(f"unknown model: {name}")


def main():
    train = load_train()
    X = build_features(train)
    y = get_target(train).loc[X.index]
    print(f"train table: {X.shape[0]} battles x {X.shape[1]} features")

    if RUN_CV:
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        scores = cross_val_score(make_model(MODEL_NAME), X, y, cv=cv, scoring="accuracy")
        print(f"[{MODEL_NAME}] {N_FOLDS}-fold CV accuracy: {scores.mean():.4f} +- {scores.std():.4f}")
        print("per fold:", np.round(scores, 4))

    if MAKE_SUBMISSION:
        test = load_test()
        X_test = build_features(test)[X.columns]      # same columns, same order as train
        model = make_model(MODEL_NAME).fit(X, y)
        preds = model.predict(X_test)
        sub = save_submission(X_test.index, preds, SUBMISSION_PATH)
        print(f"wrote {SUBMISSION_PATH}: {len(sub)} rows, share of 1s = {preds.mean():.3f}")


if __name__ == "__main__":
    main()
