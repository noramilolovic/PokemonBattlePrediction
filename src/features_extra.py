"""Two extra feature ideas that go beyond the aggregated team / timeline tables.

1. Per-species end HP  (species_end_hp_frame)
   For the most common Pokemon species we add two columns each:
       h1_<species>  HP fraction of that species on the p1 side at the end of turn 30
                     (1.0 = on the team but never hurt/never seen, NaN = not on the team)
       h2_<species>  HP fraction of that species on the p2 side at the end of turn 30
                     (NaN = never appeared on the p2 side)
   Idea: "Snorlax at 90% HP" and "Chansey at 90% HP" are not worth the same, and the
   aggregated tables cannot tell which Pokemon is the healthy one.

2. Team target encoding  (team_keys + target_encode + add_team_te)
   The same p1 team shows up in many battles (about 4,000 distinct teams in 10,000
   battles; ~70% of test teams also exist in train). We turn "this exact team"
   into the smoothed win rate that team had in the TRAINING battles.
   IMPORTANT: it uses the label, so it must be computed inside each CV fold:
     - rows of the training part get an out-of-fold value (no row sees its own label),
     - rows of the validation / test part get the value computed from the whole
       training part.
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

N_SPECIES = 60          # how many of the most common species get their own columns
TE_SMOOTHING = 10.0     # a team seen n times: (wins + 10*prior) / (n + 10)


# ----------------------------------------------------------------------
# Idea 1: per-species end HP
# ----------------------------------------------------------------------
def top_species(battles, k=N_SPECIES):
    """The k most frequent species (p1 team members + p2 Pokemon seen in the timeline).
    Decide this list on TRAIN only and reuse the same list for test."""
    cnt = Counter()
    for b in battles:
        cnt.update({m["name"] for m in b["p1_team_details"]})
        cnt.update({t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]})
    return [name for name, _ in cnt.most_common(k)]


def species_end_hp_frame(battles, species):
    """One row per battle (indexed by battle_id), columns h1_<name> and h2_<name>."""
    rows = {}
    for b in battles:
        last = {"p1": {m["name"]: 1.0 for m in b["p1_team_details"]}, "p2": {}}
        for turn in b["battle_timeline"]:
            for side in ("p1", "p2"):
                s = turn[f"{side}_pokemon_state"]
                last[side][s["name"]] = s["hp_pct"]
        row = {}
        for name in species:
            row[f"h1_{name}"] = last["p1"].get(name, np.nan)
            row[f"h2_{name}"] = last["p2"].get(name, np.nan)
        rows[b["battle_id"]] = row
    cols = [f"h{i}_{n}" for n in species for i in (1, 2)]
    return pd.DataFrame.from_dict(rows, orient="index")[cols]


# ----------------------------------------------------------------------
# Idea 2: team target encoding
# ----------------------------------------------------------------------
def team_keys(battles):
    """Identify the p1 team by its sorted species names (order of the six does not matter)."""
    return pd.Series(
        ["|".join(sorted(m["name"] for m in b["p1_team_details"])) for b in battles],
        index=[b["battle_id"] for b in battles],
    )


def target_encode(keys_fit, y_fit, keys_apply, smoothing=TE_SMOOTHING):
    """Smoothed win rate per key, learned on (keys_fit, y_fit), applied to keys_apply.
    Returns (encoding, n_battles_seen_for_that_key) as numpy arrays."""
    prior = float(np.mean(y_fit))
    g = pd.DataFrame({"k": np.asarray(keys_fit), "y": np.asarray(y_fit)}).groupby("k")["y"].agg(["sum", "count"])
    s = g["sum"].reindex(np.asarray(keys_apply)).fillna(0).to_numpy()
    n = g["count"].reindex(np.asarray(keys_apply)).fillna(0).to_numpy()
    return (s + smoothing * prior) / (n + smoothing), n


def add_team_te(X_fit, X_apply, keys_fit, y_fit, keys_apply, seed=0, n_inner=5):
    """Return copies of X_fit / X_apply with two extra columns: te_team, te_team_n.
    X_fit gets out-of-fold values, X_apply gets values learned on all of X_fit."""
    X_fit, X_apply = X_fit.copy(), X_apply.copy()
    enc = np.zeros(len(X_fit))
    cnt = np.zeros(len(X_fit))
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for a, b in kf.split(X_fit):
        enc[b], cnt[b] = target_encode(keys_fit.iloc[a], y_fit.iloc[a], keys_fit.iloc[b])
    X_fit["te_team"], X_fit["te_team_n"] = enc, cnt
    enc2, cnt2 = target_encode(keys_fit, y_fit, keys_apply)
    X_apply["te_team"], X_apply["te_team_n"] = enc2, cnt2
    return X_fit, X_apply
