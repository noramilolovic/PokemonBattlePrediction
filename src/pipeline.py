
from pathlib import Path
import pandas as pd

from src.data import load_train, load_test, get_target
from src.features_team import build_team_frame
from features_timeline import build_timeline_frame
from src.features_species import build_species_frame
CACHE_DIR = Path("data")
from src.features_types import build_types_frame




def build_X(battles):
    team_df = build_team_frame(battles)
    timeline_df = build_timeline_frame(battles)
    species_df = build_species_frame(battles)
    types_df = build_types_frame(battles)

    X = team_df.join(timeline_df, rsuffix="_timeline")
    X = X.join(species_df)
    X = X.join(types_df)

    return X.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )




def get_train(use_cache=True):
    """Return X and y aligned by battle_id."""

    battles = load_train()
    y = get_target(battles)

    cache = CACHE_DIR / "X_train.pkl"

    if use_cache and cache.exists():
        X = pd.read_pickle(cache)
    else:
        X = build_X(battles)
        X.to_pickle(cache)

    return X.loc[y.index], y


def get_test(columns, use_cache=True):
    """Return test features in the same column order as training."""

    battles = load_test()
    cache = CACHE_DIR / "X_test.pkl"

    if use_cache and cache.exists():
        X = pd.read_pickle(cache)
    else:
        X = build_X(battles)
        X.to_pickle(cache)

    return X.reindex(columns=columns, fill_value=0)
