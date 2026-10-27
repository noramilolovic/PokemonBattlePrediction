"""Loading and saving helpers for the Pokémon battle prediction challenge.

Each line of train.jsonl / test.jsonl is one battle with the fields:
    battle_id, p1_team_details, p2_lead_details, battle_timeline
and, in train only, the target player_won.
"""

import json
from pathlib import Path

import pandas as pd

TARGET = "player_won"
ID_COL = "battle_id"

# Kaggle mounts competition data under /kaggle/input/
KAGGLE_DATA = Path("/kaggle/input/competitions/fds-pokemon-battles-prediction-2026")
KAGGLE_INPUT = Path("/kaggle/input")
LOCAL_DATA = Path(__file__).resolve().parent.parent / "data"


def find_data_dir(data_dir=None):
    """Return the folder containing train.jsonl.

    Uses `data_dir` if given, otherwise the repo's local data/ folder,
    otherwise searches /kaggle/input.
    """
    if data_dir is not None:
        return Path(data_dir)
    if (LOCAL_DATA / "train.jsonl").exists():
        return LOCAL_DATA
    if (KAGGLE_DATA / "train.jsonl").exists():
        return KAGGLE_DATA
    if KAGGLE_INPUT.exists():
        for path in KAGGLE_INPUT.rglob("train.jsonl"):
            return path.parent
    raise FileNotFoundError(
        "Could not find train.jsonl; pass data_dir explicitly or put the files in data/."
    )


def load_battles(path, limit=None):
    """Read a .jsonl file into a list of battle dicts (one per line).

    `limit` reads only the first n battles, handy for quick experiments.
    """
    battles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            battles.append(json.loads(line))
            if limit is not None and len(battles) >= limit:
                break
    return battles


def load_train(data_dir=None, limit=None):
    return load_battles(find_data_dir(data_dir) / "train.jsonl", limit=limit)


def load_test(data_dir=None, limit=None):
    return load_battles(find_data_dir(data_dir) / "test.jsonl", limit=limit)


def get_target(battles):
    """Target vector (0/1) aligned with `battles`, indexed by battle_id."""
    return pd.Series(
        [int(b[TARGET]) for b in battles],
        index=[b[ID_COL] for b in battles],
        name=TARGET,
    )


def p1_team_frame(battles):
    """Long table with one row per Player 1 Pokémon (6 per battle).

    Nested fields are flattened with json_normalize, so a dict like
    {"base_hp": 80, ...} becomes columns. A `slot` column (0-5) keeps the
    position of the Pokémon in the team.
    """
    rows = []
    for b in battles:
        for slot, mon in enumerate(b["p1_team_details"]):
            rows.append({ID_COL: b[ID_COL], "slot": slot, **mon})
    return pd.json_normalize(rows)


def p2_lead_frame(battles):
    """One row per battle describing Player 2's lead Pokémon."""
    rows = [{ID_COL: b[ID_COL], **b["p2_lead_details"]} for b in battles]
    return pd.json_normalize(rows)


def save_submission(battle_ids, predictions, path):
    """Write a submission in the sample_submission.csv format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID_COL: list(battle_ids), TARGET: [int(p) for p in predictions]})
    sub.to_csv(path, index=False)
    return sub
