"""Feature pipeline: one build_X for every feature set used in the project.

A feature set is a named list of blocks. Every block is built per battle and
the blocks are joined on battle_id. Column names of different blocks must not
overlap (join raises an error if they do).
"""

from pathlib import Path

import pandas as pd

from src.data import get_target, load_test, load_train
from src.features_matchup import SpeciesTable, build_matchup_frame
from src.features_momentum import build_momentum_frame
from src.features_species import build_species_frame
from src.features_team import build_team_frame
from src.features_timeline import build_timeline_frame
from src.features_types import build_types_frame

CACHE_DIR = Path("data")

FEATURE_SETS = {
    "v1": ("team", "timeline"),
    "v3": ("team", "timeline", "species", "types"),
    "matchup": ("team", "timeline", "matchup"),
    "best": ("team", "timeline", "matchup", "momentum"),
}

BUILDERS = {
    "team": build_team_frame,
    "timeline": build_timeline_frame,
    "species": build_species_frame,
    "types": build_types_frame,
    "momentum": build_momentum_frame,
}


def make_species_table():
    """Species lookup for the matchup block, built from train AND test (no labels)."""
    return SpeciesTable(load_train(), load_test())


def build_X(battles, feature_set, species=None):
    """Feature table for one feature set, one row per battle, indexed by battle_id.

    `species` (a SpeciesTable) is only needed when the set contains "matchup".
    """
    frames = []
    for block in FEATURE_SETS[feature_set]:
        if block == "matchup":
            if species is None:
                raise ValueError("feature set with 'matchup' needs a SpeciesTable")
            frames.append(build_matchup_frame(battles, species))
        else:
            frames.append(BUILDERS[block](battles))
    X = frames[0].join(frames[1:])
    return X.replace([float("inf"), float("-inf")], float("nan"))


def align_test(X_test, columns):
    """Test features in the training column order.

    One-hot species columns can differ between train and test: columns missing
    in test are filled with 0, columns seen only in test are dropped.
    """
    return X_test.reindex(columns=columns, fill_value=0)


def _cached(name, build, use_cache):
    """Optional pickle cache. Off by default: a stale cache silently trains on old features."""
    cache = CACHE_DIR / f"{name}.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)
    X = build()
    if use_cache:
        X.to_pickle(cache)
    return X


def get_train(feature_set="v3", use_cache=False):
    """Return X and y aligned by battle_id."""
    battles = load_train()
    y = get_target(battles)
    species = make_species_table() if "matchup" in FEATURE_SETS[feature_set] else None
    X = _cached(f"X_train_{feature_set}", lambda: build_X(battles, feature_set, species), use_cache)
    return X.loc[y.index], y


def get_test(columns, feature_set="v3", use_cache=False):
    """Return test features in the same column order as training."""
    battles = load_test()
    species = make_species_table() if "matchup" in FEATURE_SETS[feature_set] else None
    X = _cached(f"X_test_{feature_set}", lambda: build_X(battles, feature_set, species), use_cache)
    return align_test(X, columns)
