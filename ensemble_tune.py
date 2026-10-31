"""Different models and their combinations, measured on the SAME folds.

Run from the repo root:
    py ensemble_tune.py quick        # ~1-2 min, only checks that everything runs
    py ensemble_tune.py              # full run, folds with seed 42
    py ensemble_tune.py 7            # full run on NEW folds (seed 7) = confirmation

What it does
  1. builds the base table (team + timeline, 272 columns) and the extra columns
     from src/features_extra.py (per-species end HP + team target encoding);
  2. for each feature set ("base", "base+extra") and each model it collects the
     out-of-fold probability of every battle (each battle is predicted by a model that
     never saw it, 5 folds);
  3. averages the probabilities of several models ("ensemble") and reports accuracy.
     Averaging out-of-fold probabilities needs no extra training, so all combinations
     are computed for free.

Models
  hgb      gradient boosting, current settings
  hgb_reg  gradient boosting, more cautious (smaller trees, lower learning rate, more l2)
  rf       random forest (500 trees, min_samples_leaf 5)
  lr       logistic regression on scaled columns, scaled values clipped to [-5, 5]
           so a column that is 1 in only a few battles cannot explode

Only train.jsonl is used. Nothing here touches Kaggle test labels.
Note: picking the best of many combinations on the same folds is slightly optimistic -
confirm the winner with `py ensemble_tune.py 7`.
"""

import itertools
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data import get_target, load_train
from src.features_extra import add_team_te, species_end_hp_frame, team_keys, top_species
from src.features_team import build_team_frame
from src.features_timeline import build_timeline_frame

QUICK = "quick" in sys.argv
_nums = [a for a in sys.argv[1:] if a.isdigit()]
SEED = int(_nums[0]) if _nums else 42
LIMIT = 1500 if QUICK else None
RF_TREES = 50 if QUICK else 500
MODELS = ["hgb", "hgb_reg", "lr"]       # remove "rf" if it is too slow
NEEDS_NO_NAN = {"rf", "lr"}                   # these two cannot take missing values
N_FOLDS = 5


def make_model(name):
    if name == "hgb":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
                                              min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    if name == "hgb_reg":
        return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=400, max_leaf_nodes=8,
                                              min_samples_leaf=60, l2_regularization=3.0, random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=5, max_features="sqrt",
                                      n_jobs=-1, random_state=SEED)
    if name == "lr":
        return make_pipeline(StandardScaler(),
                             FunctionTransformer(lambda Z: np.clip(Z, -5, 5)),
                             LogisticRegression(C=0.05, max_iter=3000))
    raise ValueError(name)


def oof_probabilities(X, y, hp, keys, folds, use_extra, models):
    """Out-of-fold P(player_won = 1) for every model; extras are built inside each fold."""
    oof = {m: np.zeros(len(y)) for m in models}
    for fi, (tr, va) in enumerate(folds):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        if use_extra:
            Xtr, Xva = Xtr.join(hp.iloc[tr]), Xva.join(hp.iloc[va])
            Xtr, Xva = add_team_te(Xtr, Xva, keys.iloc[tr], y.iloc[tr], keys.iloc[va], seed=fi)
        for m in models:
            A, B = (Xtr.fillna(-1), Xva.fillna(-1)) if m in NEEDS_NO_NAN else (Xtr, Xva)
            t0 = time.time()
            oof[m][va] = make_model(m).fit(A, y.iloc[tr]).predict_proba(B)[:, 1]
            print(f"   fold {fi + 1}/{len(folds)} {m:8s} {time.time() - t0:5.0f}s", flush=True)
    return oof


def accuracy_table(tag, oof, y):
    yv = y.to_numpy()
    rows = []
    for k in range(1, len(oof) + 1):
        for combo in itertools.combinations(oof, k):
            p = np.mean([oof[c] for c in combo], axis=0)
            rows.append(dict(features=tag, model="+".join(combo), acc=((p > 0.5) == yv).mean()))
    return pd.DataFrame(rows)


def main():
    train = load_train(limit=LIMIT)
    X = build_team_frame(train).join(build_timeline_frame(train))
    y = get_target(train).loc[X.index]
    hp = species_end_hp_frame(train, top_species(train)).loc[X.index]
    keys = team_keys(train).loc[X.index]
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    print(f"battles: {len(X)} | base columns: {X.shape[1]} | seed {SEED} | models: {MODELS}\n", flush=True)

    tables = []
    for tag, use_extra in (("base", False), ("base+extra", True)):
        print(f"== feature set: {tag}", flush=True)
        oof = oof_probabilities(X, y, hp, keys, folds, use_extra, MODELS)
        pd.DataFrame(oof, index=X.index).to_csv(f"oof_{tag.replace('+', '_')}_seed{SEED}.csv")
        tables.append(accuracy_table(tag, oof, y))
    res = pd.concat(tables, ignore_index=True)
    ref = res[(res.features == "base") & (res.model == "hgb")].acc.iloc[0]
    res["vs_base_hgb_pts"] = (res.acc - ref) * 100
    res = res.sort_values("acc", ascending=False)
    res.to_csv(f"ensemble_results_seed{SEED}.csv", index=False)

    print("\n--- single models ---")
    print(res[~res.model.str.contains(r"\+")].round(4).to_string(index=False))
    print("\n--- best 12 overall (single models and ensembles) ---")
    print(res.head(12).round(4).to_string(index=False))
    print(f"\nReference: base + hgb = {ref:.4f}. Differences below ~0.3 points are noise.")


if __name__ == "__main__":
    main()
