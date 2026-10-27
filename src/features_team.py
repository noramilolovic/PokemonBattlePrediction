"""Team features (person 1): everything known before the first turn.

Each function takes ONE battle (dict) and returns a dict {feature_name: number}.
Keys must be identical for every battle. player_won must never be used.
"""

import pandas as pd

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]  # spd = special defense!


def _stat_features(battle):
    """Blocks A and B: P1 team stat aggregates, P2 lead stats and differences."""
    team = battle["p1_team_details"]
    lead = battle["p2_lead_details"]
    feats = {}
    for s in STATS:
        vals = [p[f"base_{s}"] for p in team]
        mean = sum(vals) / len(vals)
        feats[f"p1_mean_{s}"] = mean
        feats[f"p1_max_{s}"] = max(vals)
        feats[f"p1_min_{s}"] = min(vals)
        feats[f"p2_lead_{s}"] = lead[f"base_{s}"]
        feats[f"diff_mean_{s}"] = mean - lead[f"base_{s}"]
    feats["p1_bst_sum"] = sum(p[f"base_{s}"] for p in team for s in STATS)
    feats["p2_lead_bst"] = sum(lead[f"base_{s}"] for s in STATS)
    return feats


def _speed_features(battle):
    """Block C: speed decides who moves first."""
    team = battle["p1_team_details"]
    lead_spe = battle["p2_lead_details"]["base_spe"]
    spes = [p["base_spe"] for p in team]
    return {
        "p1_n_faster_than_lead": sum(s > lead_spe for s in spes),
        "p1_max_spe_minus_lead": max(spes) - lead_spe,
    }


def _item_features(battle):
    """Block E: items are only visible for P1 (the P2 lead has no item field)."""
    items = [p.get("item") for p in battle["p1_team_details"]]
    return {
        "p1_n_leftovers": sum(i == "leftovers" for i in items),
        "p1_n_distinct_items": len(set(items)),
    }


def _composition_features(battle):
    """Team diversity: number of distinct types, ignoring 'notype'."""
    types = set()
    for p in battle["p1_team_details"]:
        types.update(t for t in p["types"] if t != "notype")
    return {"p1_n_distinct_types": len(types)}


def team_features(battle):
    """Main entry point used by the other modules. Do not change the signature."""
    feats = {}
    feats.update(_stat_features(battle))
    feats.update(_speed_features(battle))
    feats.update(_item_features(battle))
    feats.update(_composition_features(battle))
    return feats


def build_team_frame(battles):
    """Feature table with one row per battle, indexed by battle_id."""
    rows = [team_features(b) for b in battles]
    return pd.DataFrame(rows, index=[b["battle_id"] for b in battles])
