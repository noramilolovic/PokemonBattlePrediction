"""Sanity checks and paired CV for the momentum block and Extra Trees. Run from the repo root:

    python -m experiments.check_momentum

A: ensemble (LogReg + HGB) on team + timeline + matchup  vs  + momentum
B: on the full feature set, ensemble without vs with Extra Trees
C: correlation of out-of-fold errors between the three models
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import get_target, load_test, load_train
from src.features_matchup import SpeciesTable, build_matchup_frame
from src.features_momentum import build_momentum_frame
from src.features_team import build_team_frame

from src.features_timeline import build_timeline_frame

train, test = load_train(), load_test()
species = SpeciesTable(train, test)

# 1. Frames: shape, NaN, identical columns on train and test
mo_train = build_momentum_frame(train)
mo_test = build_momentum_frame(test)
print("momentum shape  train:", mo_train.shape, " test:", mo_test.shape)
print("NaN  train:", int(mo_train.isna().sum().sum()), " test:", int(mo_test.isna().sum().sum()))
print("same columns on train and test:", list(mo_train.columns) == list(mo_test.columns))
const = [c for c in mo_train.columns if mo_train[c].nunique() <= 1]
print("constant columns:", const)
print(mo_train.describe().T[["min", "mean", "max"]].round(2))

y = get_target(train)
base = (
    build_team_frame(train)
    .join(build_timeline_frame(train))
    .join(build_matchup_frame(train, species))
)
full = base.join(mo_train)
y = y.loc[base.index]
print("base:", base.shape, " full:", full.shape)

lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42)
et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt", random_state=42)
ens2 = VotingClassifier([("lr", lr), ("hgb", hgb)], voting="soft")
ens3 = VotingClassifier([("lr", lr), ("hgb", hgb), ("et", et)], voting="soft")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)


def paired(name, model_a, X_a, model_b, X_b):
    """Paired CV on identical folds with the Nadeau-Bengio corrected t-test."""
    a = cross_val_score(model_a, X_a, y, cv=cv, scoring="accuracy", n_jobs=-1)
    b = cross_val_score(model_b, X_b, y, cv=cv, scoring="accuracy", n_jobs=-1)
    d = b - a
    se = d.std(ddof=1) * np.sqrt(1 / len(d) + 0.25)  # test/train ratio = 1/4
    p = 2 * stats.t.sf(abs(d.mean() / se), df=len(d) - 1)
    print(
        f"{name:28s} {a.mean():.4f} -> {b.mean():.4f} | diff {d.mean():+.4f} "
        f"| wins {int((d > 0).sum())}/{len(d)} | corrected p = {p:.3f}"
    )


# 2. Paired comparisons
paired("A: ensemble +momentum", ens2, base, ens2, full)
paired("B: full, ensemble +ExtraTrees", ens2, full, ens3, full)

# 3. Out-of-fold errors: do the models make different mistakes?
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
errors = {}
for name, model in [("LogReg", lr), ("HGB", hgb), ("ExtraTrees", et)]:
    pred = cross_val_predict(model, full, y, cv=skf, n_jobs=-1)
    errors[name] = (pred != y.values).astype(int)
    print(f"{name:10s} OOF accuracy {1 - errors[name].mean():.4f}")
names = list(errors)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        r = np.corrcoef(errors[names[i]], errors[names[j]])[0, 1]
        print(f"error correlation {names[i]} / {names[j]}: {r:.3f}")
