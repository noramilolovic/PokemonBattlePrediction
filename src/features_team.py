"""Team features (person 1): everything known before the first turn.

Each function takes ONE battle (dict) and returns a dict {feature_name: number}.
Keys must be identical for every battle. player_won must never be used.
"""

from functools import lru_cache

import pandas as pd

from src.data import find_data_dir

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]  # spd = special defense!
NO_TYPE = "notype"  # placeholder for a missing second type
CHART_FILE = "type_chart_gen2.csv"


# ----------------------------------------------------------------------
# Type chart helpers (NEW). Person 2 can reuse these for real move types.
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_type_chart(data_dir=None):
    """Load the Gen 2 type chart as {(attacking_type, defending_type): multiplier}.

    The CSV only lists non-neutral pairs, so a missing pair means 1.0.
    """
    df = pd.read_csv(find_data_dir(data_dir) / CHART_FILE)
    return {
        (a, d): m
        for a, d, m in zip(df["attacking_type"], df["defending_type"], df["multiplier"])
    }


def type_multiplier(attack_type, defender_types, chart):
    """Damage multiplier of one attacking type against a (1 or 2 type) defender."""
    mult = 1.0
    for d in defender_types:
        if d != NO_TYPE:
            mult *= chart.get((attack_type, d), 1.0)
    return mult


def best_stab_multiplier(attacker_types, defender_types, chart):
    """Best multiplier an attacker reaches using one of its own types.

    Approximation: we do not know the real moves before the battle starts,
    so we assume the Pokemon attacks with a move of its own type (STAB).
    """
    mults = [
        type_multiplier(t, defender_types, chart)
        for t in attacker_types
        if t != NO_TYPE
    ]
    return max(mults) if mults else 1.0


# ----------------------------------------------------------------------
# Feature blocks
# ----------------------------------------------------------------------
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
        types.update(t for t in p["types"] if t != NO_TYPE)
    return {"p1_n_distinct_types": len(types)}


def _type_features(battle, chart):
    """Block D (NEW): type matchup of the whole P1 team against the P2 lead."""
    team = battle["p1_team_details"]
    lead_types = battle["p2_lead_details"]["types"]

    # How well each P1 Pokemon hits the P2 lead
    off = [best_stab_multiplier(p["types"], lead_types, chart) for p in team]
    # How well the P2 lead hits each P1 Pokemon
    deff = [best_stab_multiplier(lead_types, p["types"], chart) for p in team]

    off_mean = sum(off) / len(off)
    deff_mean = sum(deff) / len(deff)
    return {
        "type_off_mean": off_mean,
        "type_off_max": max(off),
        "type_n_super_vs_lead": sum(m >= 2.0 for m in off),
        "type_def_mean": deff_mean,
        "type_def_max": max(deff),
        "type_n_weak_to_lead": sum(m >= 2.0 for m in deff),
        "type_n_resist_lead": sum(m < 1.0 for m in deff),
        "type_edge_mean": off_mean - deff_mean,
    }


def _lead_features(battle, chart):
    """Block F (NEW): P1 lead (team[0]) versus P2 lead, the opening matchup."""
    p1_lead = battle["p1_team_details"][0]  # first Pokemon = the one P1 leads with
    p2_lead = battle["p2_lead_details"]
    feats = {}
    for s in STATS:
        feats[f"lead_diff_{s}"] = p1_lead[f"base_{s}"] - p2_lead[f"base_{s}"]
    feats["lead_p1_faster"] = int(p1_lead["base_spe"] > p2_lead["base_spe"])
    feats["lead_speed_tie"] = int(p1_lead["base_spe"] == p2_lead["base_spe"])
    off = best_stab_multiplier(p1_lead["types"], p2_lead["types"], chart)
    deff = best_stab_multiplier(p2_lead["types"], p1_lead["types"], chart)
    feats["lead_type_off"] = off
    feats["lead_type_def"] = deff
    feats["lead_type_edge"] = off - deff
    return feats


def team_features(battle, chart=None):
    """Main entry point used by the other modules. `chart` is optional."""
    if chart is None:
        chart = load_type_chart()
    feats = {}
    feats.update(_stat_features(battle))
    feats.update(_speed_features(battle))
    feats.update(_item_features(battle))
    feats.update(_composition_features(battle))
    feats.update(_type_features(battle, chart))
    feats.update(_lead_features(battle, chart))
    return feats


def build_team_frame(battles):
    """Feature table with one row per battle, indexed by battle_id."""
    chart = load_type_chart()
    rows = [team_features(b, chart) for b in battles]
    return pd.DataFrame(rows, index=[b["battle_id"] for b in battles])
