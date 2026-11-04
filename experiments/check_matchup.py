"""Sanity checks and paired CV for the matchup block. Run from the repo root:

    python -m experiments.check_matchup
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import get_target, load_test, load_train
from src.features_matchup import SpeciesTable, build_matchup_frame
from src.features_team import build_team_frame

from src.features_timeline import build_timeline_frame

train, test = load_train(), load_test()
species = SpeciesTable(train, test)
print("species in lookup table:", len(species.table))


# 1. Coverage: P2 species that appear in a timeline but are missing from the table
def missing_species(battles):
    return {
        t["p2_pokemon_state"]["name"]
        for b in battles
        for t in b["battle_timeline"]
        if t["p2_pokemon_state"]["name"] not in species.table
    }


print("P2 species missing from table  train:", missing_species(train), " test:", missing_species(test))

# 2. Frames: shape, NaN, identical columns on train and test
mu_train = build_matchup_frame(train, species)
mu_test = build_matchup_frame(test, species)
print("matchup shape  train:", mu_train.shape, " test:", mu_test.shape)
print("NaN  train:", int(mu_train.isna().sum().sum()), " test:", int(mu_test.isna().sum().sum()))
print("same columns on train and test:", list(mu_train.columns) == list(mu_test.columns))

# 3. Paired CV: team + timeline versus team + timeline + matchup (identical folds)
y = get_target(train)
base = build_team_frame(train).join(build_timeline_frame(train))
full = base.join(mu_train)
y = y.loc[base.index]
print("base:", base.shape, " full:", full.shape)

lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42)
models = {
    "HistGB": hgb,
    "LogReg": lr,
    "Ensemble": VotingClassifier([("lr", lr), ("hgb", hgb)], voting="soft"),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
for name, model in models.items():
    a = cross_val_score(model, base, y, cv=cv, scoring="accuracy", n_jobs=-1)
    b = cross_val_score(model, full, y, cv=cv, scoring="accuracy", n_jobs=-1)
    d = b - a
    # Nadeau-Bengio corrected t-test (folds share data; test/train ratio = 1/4)
    se = d.std(ddof=1) * np.sqrt(1 / len(d) + 0.25)
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), df=len(d) - 1)
    print(
        f"{name:9s} base {a.mean():.4f} | +matchup {b.mean():.4f} | diff {d.mean():+.4f} "
        f"| wins {int((d > 0).sum())}/{len(d)} | corrected p = {p:.3f}"
    )
