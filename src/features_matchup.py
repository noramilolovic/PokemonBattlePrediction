"""Matchup features (extra block): who is left, how strong, who wins the duel.

Idea taken from a simple 1v1 reference project: a duel is mostly decided by
the speed / stat difference and the type matchup of the two Pokemon that
really fight. Here we apply that to the Pokemon that are ACTIVE on the board
and to the moves that were really used in the recorded turns.

Three groups of features, all prefixed with mu_:
  A. Strength of the surviving Pokemon at the last recorded turn
  B. Matchup of the two active Pokemon at the last recorded turn
  C. Real move effectiveness and speed advantage over the whole window

Every function takes ONE battle (dict) and returns {feature_name: number}.
player_won is never used. Stats and types of P2 Pokemon come from a lookup
table built from the competition files (no external data, no labels).
"""

import pandas as pd

from src.features_team import (
    STATS,
    best_stab_multiplier,
    load_type_chart,
    type_multiplier,
)

N_TURNS = 30
TEAM_SIZE = 6
LAST_WINDOW = 10  # "end of the battle" = the last 10 recorded turns
PAR_SPEED_MULT = 0.25  # Gen 1-6 rule: paralysis cuts Speed to a quarter


# ----------------------------------------------------------------------
# Species lookup (P2 only shows its lead, but the same species appear in
# P1 teams of other battles, and base stats / types never change)
# ----------------------------------------------------------------------
def _record(p):
    """Convert a Pokemon details dict into {stat: value, types: [...], bst: int}."""
    rec = {s: p[f"base_{s}"] for s in STATS}
    rec["types"] = list(p["types"])
    rec["bst"] = sum(rec[s] for s in STATS)
    return rec


class SpeciesTable:
    """Lookup species name -> base stats and types, built from competition files.

    Pass every battle list you have (train and test). Only the static Pokemon
    details are read, never player_won. Unknown species fall back to the mean
    record, which has no types (so every type multiplier stays neutral).
    """

    def __init__(self, *battle_lists):
        self.table = {}
        for battles in battle_lists:
            for b in battles:
                for p in b["p1_team_details"] + [b["p2_lead_details"]]:
                    if p["name"] not in self.table:
                        self.table[p["name"]] = _record(p)
        n = len(self.table)
        self.mean = {
            s: sum(r[s] for r in self.table.values()) / n for s in STATS + ["bst"]
        }
        self.mean["types"] = []

    def get(self, name):
        return self.table.get(name, self.mean)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _stage_mult(stage):
    """Stat stage multiplier for stages -6..+6."""
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def _eff_speed(rec, state):
    """Effective base speed of an active Pokemon (boosts and paralysis included)."""
    spe = rec["spe"] * _stage_mult(state["boosts"].get("spe", 0))
    if state["status"] == "par":
        spe *= PAR_SPEED_MULT
    return spe


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _new_moves():
    return {
        "n": 0,
        "mult_sum": 0.0,
        "n_super": 0,
        "n_resisted": 0,
        "n_immune": 0,
        "power": 0.0,
        "power_last": 0.0,
    }


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def matchup_features(battle, species, chart, n_turns=N_TURNS):
    """ONE battle (dict) -> dict of numbers. `species` is a SpeciesTable."""
    tl = battle["battle_timeline"][:n_turns]
    if not tl:
        raise ValueError(f"battle {battle.get('battle_id')} has an empty timeline")

    p1_recs = {p["name"]: _record(p) for p in battle["p1_team_details"]}

    def rec_of(side, name):
        if side == "p1" and name in p1_recs:
            return p1_recs[name]
        return species.get(name)

    # last known hp per Pokemon; everybody starts at 100%
    hp = {
        "p1": {n: 1.0 for n in p1_recs},
        "p2": {battle["p2_lead_details"]["name"]: 1.0},
    }
    fainted = {"p1": set(), "p2": set()}
    moves = {"p1": _new_moves(), "p2": _new_moves()}
    n_faster = n_faster_last = 0
    act = {}

    # ---------------- replay the recorded turns ----------------
    for t in tl:
        in_last_window = t["turn"] > n_turns - LAST_WINDOW

        for me in ("p1", "p2"):
            s = t[me + "_pokemon_state"]
            name = s["name"]
            if s["status"] == "fnt":
                fainted[me].add(name)
                hp[me][name] = 0.0
            else:
                hp[me][name] = s["hp_pct"]
            act[me] = (rec_of(me, name), s)

        # who moves first among the two active Pokemon
        faster = _eff_speed(*act["p1"]) > _eff_speed(*act["p2"])
        n_faster += int(faster)
        n_faster_last += int(faster and in_last_window)

        # real move effectiveness: move type against the CURRENT defender
        for me, other in (("p1", "p2"), ("p2", "p1")):
            m = t[me + "_move_details"]
            if m is None or m["category"] == "status" or m["base_power"] <= 0:
                continue
            attacker, defender = act[me][0], act[other][0]
            mult = type_multiplier(m["type"], defender["types"], chart)
            stab = 1.5 if m["type"] in attacker["types"] else 1.0
            acc = m.get("accuracy")
            acc = 1.0 if acc is None else float(acc)
            power = m["base_power"] * mult * stab * acc

            d = moves[me]
            d["n"] += 1
            d["mult_sum"] += mult
            d["n_super"] += int(mult >= 2.0)
            d["n_resisted"] += int(0.0 < mult < 1.0)
            d["n_immune"] += int(mult == 0.0)
            d["power"] += power
            if in_last_window:
                d["power_last"] += power

    feats = {}

    # ---------------- A. strength of the survivors ----------------
    # P1: the whole team is known. P2: only seen Pokemon are known; the
    # unseen slots are at full hp and get the average species record.
    seen2 = list(hp["p2"])
    n_unseen = max(0, TEAM_SIZE - len(seen2))
    p1_alive = [r for n, r in p1_recs.items() if n not in fainted["p1"]]
    p2_alive = [species.get(n) for n in seen2 if n not in fainted["p2"]]
    p2_alive += [species.mean] * n_unseen

    p1_strength = sum(r["bst"] * hp["p1"][n] for n, r in p1_recs.items())
    p2_strength = sum(species.get(n)["bst"] * hp["p2"][n] for n in seen2)
    p2_strength += n_unseen * species.mean["bst"]

    feats["mu_p1_surv_strength"] = p1_strength
    feats["mu_p2_surv_strength"] = p2_strength
    feats["mu_surv_strength_diff"] = p1_strength - p2_strength
    feats["mu_surv_strength_share"] = p1_strength / max(p1_strength + p2_strength, 1e-9)
    feats["mu_p1_alive_bst"] = sum(r["bst"] for r in p1_alive)
    feats["mu_p2_alive_bst"] = sum(r["bst"] for r in p2_alive)
    feats["mu_alive_bst_diff"] = feats["mu_p1_alive_bst"] - feats["mu_p2_alive_bst"]
    feats["mu_alive_spe_diff"] = _mean([r["spe"] for r in p1_alive]) - _mean(
        [r["spe"] for r in p2_alive]
    )
    feats["mu_p2_n_unseen"] = n_unseen

    # ---------------- B. matchup of the two active Pokemon ----------------
    (r1, s1), (r2, s2) = act["p1"], act["p2"]
    e1, e2 = _eff_speed(r1, s1), _eff_speed(r2, s2)
    off = best_stab_multiplier(r1["types"], r2["types"], chart)
    deff = best_stab_multiplier(r2["types"], r1["types"], chart)

    feats["mu_act_spe_diff"] = e1 - e2
    feats["mu_act_p1_faster"] = int(e1 > e2)
    feats["mu_act_speed_tie"] = int(e1 == e2)
    for s in STATS:
        feats[f"mu_act_diff_{s}"] = r1[s] - r2[s]
    feats["mu_act_bst_diff"] = r1["bst"] - r2["bst"]
    feats["mu_act_type_off"] = off
    feats["mu_act_type_def"] = deff
    feats["mu_act_type_edge"] = off - deff
    feats["mu_act_p1_asleep"] = int(s1["status"] == "slp")
    feats["mu_act_p2_asleep"] = int(s2["status"] == "slp")

    # ---------------- C. real moves and speed over the window ----------------
    for me in ("p1", "p2"):
        d, pre = moves[me], f"mu_{me}_mv_"
        feats[pre + "n_dmg"] = d["n"]
        feats[pre + "mean_mult"] = d["mult_sum"] / d["n"] if d["n"] else 1.0
        feats[pre + "n_super"] = d["n_super"]
        feats[pre + "n_resisted"] = d["n_resisted"]
        feats[pre + "n_immune"] = d["n_immune"]
        feats[pre + "power"] = d["power"]
        feats[pre + "power_last10"] = d["power_last"]
    for k in ("mean_mult", "n_super", "n_resisted", "power", "power_last10"):
        feats[f"mu_mv_diff_{k}"] = feats[f"mu_p1_mv_{k}"] - feats[f"mu_p2_mv_{k}"]
    feats["mu_frac_p1_faster"] = n_faster / len(tl)
    feats["mu_frac_p1_faster_last10"] = n_faster_last / min(LAST_WINDOW, len(tl))
    return feats


def build_matchup_frame(battles, species, chart=None, n_turns=N_TURNS):
    """Feature table with one row per battle, indexed by battle_id.

    `species` must be a SpeciesTable built from train AND test battles, so the
    train and test tables are computed with the same lookup.
    """
    chart = load_type_chart() if chart is None else chart
    rows = [matchup_features(b, species, chart, n_turns) for b in battles]
    return pd.DataFrame(rows, index=[b["battle_id"] for b in battles])
