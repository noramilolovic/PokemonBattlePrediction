"""Train on the full train set and write submission files.

Every submission is one named configuration = feature set + model.
Run from the repo root:

    python make_submissions.py --list          # show the configurations
    python make_submissions.py ensemble3       # one submission
    python make_submissions.py --all           # all of them

Files are written to submissions/submission_<name>.csv.
"""

import argparse

from src.data import get_target, load_test, load_train, save_submission
from src.models import get_final_models
from src.pipeline import FEATURE_SETS, align_test, build_X, make_species_table

# name: (feature set, model, cross-validation result)
SUBMISSIONS = {
    "ensemble3": ("best", "ensemble_lr_hgb_et", "CV 0.6799, public 0.6880"),
    "ensemble2": ("matchup", "ensemble_lr_hgb", "CV 0.6793"),
    "hgb_tuned_v3": ("v3", "hgb_tuned", "CV 0.6771"),
    "hgb_v1": ("v1", "hgb_default", "CV 0.673"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="configurations to run")
    parser.add_argument("--all", action="store_true", help="run every configuration")
    parser.add_argument("--list", action="store_true", help="list the configurations")
    args = parser.parse_args()

    if args.list or not (args.all or args.names):
        for name, (fs, model, note) in SUBMISSIONS.items():
            print(f"{name:14s} features={fs} {FEATURE_SETS[fs]}  model={model}  ({note})")
        return

    names = list(SUBMISSIONS) if args.all else args.names
    unknown = [n for n in names if n not in SUBMISSIONS]
    if unknown:
        parser.error(f"unknown configuration(s): {unknown}")

    train, test = load_train(), load_test()
    y = get_target(train)
    species = make_species_table()
    models = get_final_models()
    frames = {}  # feature set -> (X_train, X_test), each set is built only once

    for name in names:
        fs, model_name, note = SUBMISSIONS[name]
        if fs not in frames:
            X_train = build_X(train, fs, species)
            X_test = align_test(build_X(test, fs, species), X_train.columns)
            frames[fs] = (X_train, X_test)
        X_train, X_test = frames[fs]

        model = models[model_name].fit(X_train, y.loc[X_train.index])
        preds = model.predict(X_test)

        assert len(preds) == len(test) and X_test.index.is_unique
        path = f"submissions/submission_{name}.csv"
        save_submission(X_test.index, preds, path)
        print(f"{name:14s} {X_train.shape[1]:5d} features  -> {path}  ({note})")


if __name__ == "__main__":
    main()
