"""Momentum features (extra block): how the advantage developed over the turns.

The timeline block already has the hp difference at turns 5, 10, ..., 30 and
its slope. This block adds what those snapshots cannot show:

  A. Volatility of the hp difference (stable lead versus a back-and-forth battle)
  B. Recency of faints (who lost a Pokemon last, and how long ago)
  C. Average hp of the Pokemon that are still alive

All features are prefixed with mo_. Every function takes ONE battle (dict)
and returns {feature_name: number}. player_won is never used.
"""

import pandas as pd

N_TURNS = 30
TEAM_SIZE = 6
RECENT = 5  # "recent" faints = the last 5 recorded turns


def _team_hp(last_hp):
    """Sum of the last known hp over 6 slots. Slots never seen count as 1.0."""
    return sum(last_hp.values()) + max(0, TEAM_SIZE - len(last_hp))


def momentum_features(battle, n_turns=N_TURNS):
    """ONE battle (dict) -> dict of numbers."""
    tl = battle["battle_timeline"][:n_turns]
    if not tl:
        raise ValueError(f"battle {battle.get('battle_id')} has an empty timeline")

    last_hp = {
        "p1": {p["name"]: 1.0 for p in battle["p1_team_details"]},
        "p2": {battle["p2_lead_details"]["name"]: 1.0},
    }
    fainted = {"p1": set(), "p2": set()}
    faint_turns = {"p1": [], "p2": []}
    diffs, turns = [], []

    # ---------------- replay the recorded turns ----------------
    for t in tl:
        for me in ("p1", "p2"):
            s = t[me + "_pokemon_state"]
            if s["status"] == "fnt":
                last_hp[me][s["name"]] = 0.0
                if s["name"] not in fainted[me]:
                    fainted[me].add(s["name"])
                    faint_turns[me].append(t["turn"])
            else:
                last_hp[me][s["name"]] = s["hp_pct"]
        diffs.append(_team_hp(last_hp["p1"]) - _team_hp(last_hp["p2"]))
        turns.append(t["turn"])

    feats = {}
    last_turn = turns[-1]

    # ---------------- A. volatility of the hp difference ----------------
    mean = sum(diffs) / len(diffs)
    signs = [1 if d > 0 else -1 for d in diffs if d != 0]  # ties do not change the lead
    changes = [i for i in range(1, len(signs)) if signs[i] != signs[i - 1]]
    i_max, i_min = diffs.index(max(diffs)), diffs.index(min(diffs))

    feats["mo_hp_diff_std"] = (sum((d - mean) ** 2 for d in diffs) / len(diffs)) ** 0.5
    feats["mo_n_lead_changes"] = len(changes)
    feats["mo_max_p1_lead"] = max(0.0, diffs[i_max])
    feats["mo_max_p2_lead"] = max(0.0, -diffs[i_min])
    feats["mo_turn_max_p1_lead"] = turns[i_max]
    feats["mo_turn_max_p2_lead"] = turns[i_min]
    feats["mo_frac_turns_p1_ahead"] = sum(d > 0 for d in diffs) / len(diffs)
    feats["mo_end_minus_best_p1"] = diffs[-1] - diffs[i_max]  # how much of its best lead P1 gave back

    # ---------------- B. recency of faints ----------------
    for me in ("p1", "p2"):
        ft = faint_turns[me]
        feats[f"mo_{me}_n_faint_last{RECENT}"] = sum(x > last_turn - RECENT for x in ft)
        feats[f"mo_{me}_turns_since_faint"] = last_turn - ft[-1] if ft else last_turn
    feats[f"mo_diff_faint_last{RECENT}"] = (
        feats[f"mo_p2_n_faint_last{RECENT}"] - feats[f"mo_p1_n_faint_last{RECENT}"]
    )
    last1 = faint_turns["p1"][-1] if faint_turns["p1"] else 0
    last2 = faint_turns["p2"][-1] if faint_turns["p2"] else 0
    # +1: P2 lost a Pokemon most recently (good for P1), -1: P1 did, 0: none / same turn
    feats["mo_last_faint_side"] = (last2 > last1) - (last1 > last2)

    # ---------------- C. hp per surviving Pokemon ----------------
    for me in ("p1", "p2"):
        alive = [h for n, h in last_hp[me].items() if n not in fainted[me]]
        alive += [1.0] * max(0, TEAM_SIZE - len(last_hp[me]))  # unseen slots
        feats[f"mo_{me}_hp_per_alive"] = sum(alive) / len(alive) if alive else 0.0
    feats["mo_hp_per_alive_diff"] = feats["mo_p1_hp_per_alive"] - feats["mo_p2_hp_per_alive"]
    return feats


def build_momentum_frame(battles, n_turns=N_TURNS):
    """Feature table with one row per battle, indexed by battle_id."""
    rows = [momentum_features(b, n_turns) for b in battles]
    return pd.DataFrame(rows, index=[b["battle_id"] for b in battles])
