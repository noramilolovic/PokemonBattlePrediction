"""Train on the full train set and write a submission file.

Model: soft-voting ensemble of HistGB and LogReg on team + timeline + matchup
features (best paired-CV result in check_matchup.py, 0.6793).
"""

from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_train, load_test, get_target, save_submission
from src.features_matchup import SpeciesTable, build_matchup_frame
from src.features_team import build_team_frame
from features_timeline import build_timeline_frame  # still in the repo root

OUT_PATH = "submissions/submission_ensemble_matchup.csv"


def build_X(battles, species):
    """Team + timeline + matchup features, one row per battle."""
    return (
        build_team_frame(battles)
        .join(build_timeline_frame(battles))
        .join(build_matchup_frame(battles, species))
    )


train, test = load_train(), load_test()
species = SpeciesTable(train, test)  # static Pokemon details only, no labels
X_train = build_X(train, species)
y = get_target(train).loc[X_train.index]
X_test = build_X(test, species)[X_train.columns]  # identical column order

# Same settings as in check_matchup.py
lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
hgb = HistGradientBoostingClassifier(
    learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42
)
model = VotingClassifier([("lr", lr), ("hgb", hgb)], voting="soft").fit(X_train, y)

preds = model.predict(X_test)
save_submission(X_test.index, preds, OUT_PATH)
print("written:", len(preds), "predictions to", OUT_PATH)
