from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)


def get_models():
    """Models compared in experiments/cv_compare.py."""
    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=3000)
        ),

        "rf": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=42
            )
        ),

        "hgb15": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42
            )
        )
    }


def get_final_models():
    """Models used for the submissions (see make_submissions.py)."""
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42
    )
    et = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1
    )
    return {
        "hgb_default": HistGradientBoostingClassifier(random_state=42),

        # best parameters from experiments/tune_hgb.py
        "hgb_tuned": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=5.0,
                random_state=42
            )
        ),

        "ensemble_lr_hgb": VotingClassifier(
            [("lr", lr), ("hgb", hgb)], voting="soft"
        ),

        "ensemble_lr_hgb_et": VotingClassifier(
            [("lr", lr), ("hgb", hgb), ("et", et)], voting="soft"
        ),
    }
