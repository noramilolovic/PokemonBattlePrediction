"""Train on the full train set and write submission files.

Every submission is one named configuration = feature set + model(s).
Run from the repo root:

    python make_submissions.py --list          # show the configurations
    python make_submissions.py ensemble3       # one submission
    python make_submissions.py --all           # all of them

Files are written to submissions/submission_<name>.csv.

Configurations with several models average their predicted probabilities.
The "extra" step (src/features_extra.py) adds per-species end hp and the team
target encoding. The encoding uses labels, so it is not part of build_X:
train rows get out-of-fold values, test rows get values learned on all of train.
"""

import argparse

import numpy as np

from src.data import get_target, load_test, load_train, save_submission
from src.features_extra import add_team_te, species_end_hp_frame, team_keys, top_species
from src.models import NEEDS_NO_NAN, get_final_models
from src.pipeline import FEATURE_SETS, align_test, build_X, make_species_table

# name: (feature set, models, add extra features, cross-validation result)
SUBMISSIONS = {
    "ensemble3": ("best", ["ensemble_lr_hgb_et"], False, "CV 0.6799, public 0.6880"),
    "ensemble2": ("matchup", ["ensemble_lr_hgb"], False, "CV 0.6793"),
    "hgb_tuned_v3": ("v3", ["hgb_tuned"], False, "CV 0.6771"),
    "hgb_v1": ("v1", ["hgb_default"], False, "CV 0.673"),
    "hgb_reg_extra": ("v1", ["hgb_reg"], True, "branch final-hgb-reg-extra"),
    "ensemble4_extra": ("v1", ["hgb", "hgb_reg", "lr_clip", "rf"], True, "branch final-ensemble-extra"),
}


def add_extra(train, test, X_train, X_test, y):
    """Per-species end hp (species list from train only) + team target encoding."""
    species = top_species(train)
    X_train = X_train.join(species_end_hp_frame(train, species).loc[X_train.index])
    X_test = X_test.join(species_end_hp_frame(test, species).loc[X_test.index])
    k_tr = team_keys(train).loc[X_train.index]
    k_te = team_keys(test).loc[X_test.index]
    X_train, X_test = add_team_te(X_train, X_test, k_tr, y.loc[X_train.index], k_te, seed=0)
    assert list(X_train.columns) == list(X_test.columns)
    return X_train, X_test


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="configurations to run")
    parser.add_argument("--all", action="store_true", help="run every configuration")
    parser.add_argument("--list", action="store_true", help="list the configurations")
    args = parser.parse_args()

    if args.list or not (args.all or args.names):
        for name, (fs, model_names, extra, note) in SUBMISSIONS.items():
            blocks = FEATURE_SETS[fs] + (("extra",) if extra else ())
            print(f"{name:16s} features={fs} {blocks}  models={'+'.join(model_names)}  ({note})")
        return

    names = list(SUBMISSIONS) if args.all else args.names
    unknown = [n for n in names if n not in SUBMISSIONS]
    if unknown:
        parser.error(f"unknown configuration(s): {unknown}")

    train, test = load_train(), load_test()
    y = get_target(train)
    species = make_species_table()
    models = get_final_models()
    frames = {}  # (feature set, extra) -> (X_train, X_test), each table is built only once

    for name in names:
        fs, model_names, extra, note = SUBMISSIONS[name]
        if (fs, False) not in frames:
            X_train = build_X(train, fs, species)
            X_test = align_test(build_X(test, fs, species), X_train.columns)
            frames[(fs, False)] = (X_train, X_test)
        if extra and (fs, True) not in frames:
            frames[(fs, True)] = add_extra(train, test, *frames[(fs, False)], y)
        X_train, X_test = frames[(fs, extra)]
        y_train = y.loc[X_train.index]

        if len(model_names) == 1:
            preds = models[model_names[0]].fit(X_train, y_train).predict(X_test)
        else:
            probs = []
            for m in model_names:
                A, B = (X_train.fillna(-1), X_test.fillna(-1)) if m in NEEDS_NO_NAN else (X_train, X_test)
                probs.append(models[m].fit(A, y_train).predict_proba(B)[:, 1])
            preds = (np.mean(probs, axis=0) > 0.5).astype(int)

        assert len(preds) == len(test) and X_test.index.is_unique
        path = f"submissions/submission_{name}.csv"
        save_submission(X_test.index, preds, path)
        print(f"{name:16s} {X_train.shape[1]:5d} features  -> {path}  ({note})")


if __name__ == "__main__":
    main()
