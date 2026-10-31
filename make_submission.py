"""Final fit on ALL of train (base + extra features) and write the Kaggle submission.

Run from the repo root:
    py make_submission.py quick   # ~20 s on a small sample, only checks that it runs
    py make_submission.py         # real run, writes submissions/<name>.csv

Choose the models in MODEL_SET below. With several models the predicted probabilities are
averaged (same thing ensemble_tune.py measured). Models are defined in ensemble_tune.py.

How the extra features are built here (same logic as in the cross-validation):
  - species end-HP columns: species list decided on train only;
  - team target encoding: train rows get out-of-fold values, test rows get the value learned
    from all of train. The test labels are never used (we do not have them).
"""

import sys

import pandas as pd

from ensemble_tune import NEEDS_NO_NAN, make_model
from src.data import get_target, load_test, load_train, save_submission
from src.features_extra import add_team_te, species_end_hp_frame, team_keys, top_species
from src.features_team import build_team_frame
from src.features_timeline import build_timeline_frame

QUICK = "quick" in sys.argv
MODEL_SET = ["hgb_reg"]                      # e.g. ["hgb", "hgb_reg", "lr"] for an ensemble
USE_EXTRA = True                             # False = old 272-column table
NAME = "_".join(MODEL_SET) + ("_extra" if USE_EXTRA else "_base")
OUT = f"submissions/{NAME}{'_quick' if QUICK else ''}.csv"


def base_table(battles):
    return build_team_frame(battles).join(build_timeline_frame(battles))


def main():
    train, test = load_train(limit=1500 if QUICK else None), load_test()
    if QUICK:
        test = test[:500]
    X = base_table(train)
    y = get_target(train).loc[X.index]
    X_test = base_table(test)[X.columns]                       # same columns, same order
    print(f"train: {X.shape} | test: {X_test.shape}", flush=True)

    if USE_EXTRA:
        species = top_species(train)
        X = X.join(species_end_hp_frame(train, species).loc[X.index])
        X_test = X_test.join(species_end_hp_frame(test, species).loc[X_test.index])
        k_tr, k_te = team_keys(train).loc[X.index], team_keys(test).loc[X_test.index]
        print(f"share of test teams that also appear in train: {k_te.isin(set(k_tr)).mean():.3f}")
        X, X_test = add_team_te(X, X_test, k_tr, y, k_te, seed=0)
        assert list(X.columns) == list(X_test.columns)

    probs = []
    for name in MODEL_SET:
        A, B = (X.fillna(-1), X_test.fillna(-1)) if name in NEEDS_NO_NAN else (X, X_test)
        probs.append(make_model(name).fit(A, y).predict_proba(B)[:, 1])
        print(f"fitted {name}", flush=True)
    p = pd.Series(sum(probs) / len(probs), index=X_test.index)
    preds = (p > 0.5).astype(int)
    save_submission(X_test.index, preds.to_numpy(), OUT)
    print(f"wrote {OUT}: {len(preds)} rows, share of 1s = {preds.mean():.3f}")


if __name__ == "__main__":
    main()
