"""Sanity checks and paired CV for the stall block. Run from the repo root:

    python -m experiments.check_stall

Model: the current best ensemble (LogReg + HGB + Extra Trees, make_submission_v3.py)
on team + timeline + matchup + momentum  vs  + stall, on identical folds.
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import get_target, load_test, load_train
from src.features_matchup import SpeciesTable, build_matchup_frame
from src.features_momentum import build_momentum_frame
from src.features_stall import build_stall_frame
from src.features_team import build_team_frame

from src.features_timeline import build_timeline_frame

train, test = load_train(), load_test()
species = SpeciesTable(train, test)

# 1. Frames: shape, NaN, identical columns on train and test
st_train = build_stall_frame(train)
st_test = build_stall_frame(test)
print("stall shape  train:", st_train.shape, " test:", st_test.shape)
print("NaN  train:", int(st_train.isna().sum().sum()), " test:", int(st_test.isna().sum().sum()))
print("same columns on train and test:", list(st_train.columns) == list(st_test.columns))
const = [c for c in st_train.columns if st_train[c].nunique() <= 1]
print("constant columns:", const)
print(st_train.describe().T[["min", "mean", "max"]].round(2))

y = get_target(train)
base = (
    build_team_frame(train)
    .join(build_timeline_frame(train))
    .join(build_matchup_frame(train, species))
    .join(build_momentum_frame(train))
)
full = base.join(st_train)
y = y.loc[base.index]
print("base:", base.shape, " full:", full.shape)

lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42)
et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt", random_state=42)
ens = VotingClassifier([("lr", lr), ("hgb", hgb), ("et", et)], voting="soft")

# 2. Paired CV on identical folds with the Nadeau-Bengio corrected t-test
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
a = cross_val_score(ens, base, y, cv=cv, scoring="accuracy", n_jobs=-1)
b = cross_val_score(ens, full, y, cv=cv, scoring="accuracy", n_jobs=-1)
d = b - a
se = d.std(ddof=1) * np.sqrt(1 / len(d) + 0.25)  # test/train ratio = 1/4
p = 2 * stats.t.sf(abs(d.mean() / se), df=len(d) - 1)
print(
    f"ensemble +stall  {a.mean():.4f} -> {b.mean():.4f} | diff {d.mean():+.4f} "
    f"| wins {int((d > 0).sum())}/{len(d)} | corrected p = {p:.3f}"
)
