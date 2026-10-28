

from collections import defaultdict

import pandas as pd

N_TURNS = 30
TEAM_SIZE = 6
BAD_STATUSES = ["slp", "par", "tox", "psn", "brn", "frz"]   # 'nostatus' / 'fnt' handled separately
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe"]
# volatile effects with enough support in train
EFFECTS = ["confusion", "substitute", "leechseed", "protect", "perish", "trapped", "attract", "encore"]
# moves with >= ~700 uses in train that matter in Gen 2 (sleep, spikes, phazing, recovery, setup...)
KEY_MOVES = [
    "rest", "sleeptalk", "curse", "spikes", "rapidspin", "toxic", "thunderwave",
    "lovelykiss", "sleeppowder", "hypnosis", "spore", "roar", "whirlwind", "explosion",
    "selfdestruct", "reflect", "lightscreen", "healbell", "batonpass", "protect",
    "substitute", "swordsdance", "bellydrum", "recover", "milkdrink", "softboiled",
    "moonlight", "perishsong", "destinybond", "leechseed", "meanlook", "confuseray",
]
CHECKPOINTS = (5, 10, 15, 20, 25, 30)   # turns at which we snapshot the battle


def _new_side():
    return {
        "last_hp": {}, "last_status": {}, "fainted": set(), "seen": [],
        "dmg_taken": 0.0, "dmg_taken_last10": 0.0, "healed": 0.0,
        "switches": 0, "replacements": 0, "no_move_turns": 0,
        "status_turns": defaultdict(int), "status_inflicted": defaultdict(int),
        "effect_turns": defaultdict(int),
        "boost_total": 0, "boost_max": 0, "boost_min": 0,
        "n_moves": 0, "n_status_moves": 0, "n_phys": 0, "n_spec": 0, "n_priority": 0,
        "bp_sum": 0.0, "acc_sum": 0.0,
        "key_moves": defaultdict(int),
        "faint_turns": [],
    }


def _team_hp(side):
    """Sum of the last known hp over the 6 slots. Slots never seen count as healthy (1.0)."""
    return sum(side["last_hp"].values()) + max(0, TEAM_SIZE - len(side["last_hp"]))


def timeline_features(battle, n_turns=N_TURNS):
    """Main entry point. ONE battle (dict) -> dict of numbers. Do not change the signature."""
    tl = battle["battle_timeline"][:n_turns]
    S = {"p1": _new_side(), "p2": _new_side()}

    # before turn 1 everybody is at 100% hp; P1's full team is known from the start
    for p in battle["p1_team_details"]:
        S["p1"]["last_hp"][p["name"]] = 1.0
    S["p2"]["last_hp"][battle["p2_lead_details"]["name"]] = 1.0
    prev = {"p1": (battle["p1_team_details"][0]["name"], "nostatus"),
            "p2": (battle["p2_lead_details"]["name"], "nostatus")}

    hp_diff_series, snaps = [], {}

    for t in tl:
        turn = t["turn"]
        for me in ("p1", "p2"):
            s = t[me + "_pokemon_state"]
            m = t[me + "_move_details"]
            side, name = S[me], s["name"]
            if name not in side["seen"]:
                side["seen"].append(name)

            # switch (voluntary / forced) or replacement after a faint
            if name != prev[me][0]:
                side["replacements" if prev[me][1] == "fnt" else "switches"] += 1

            # hp change relative to the last time we saw this Pokémon
            delta = side["last_hp"].get(name, 1.0) - s["hp_pct"]
            if delta > 0:
                side["dmg_taken"] += delta
                if turn > n_turns - 10:
                    side["dmg_taken_last10"] += delta
            else:
                side["healed"] -= delta
            side["last_hp"][name] = s["hp_pct"]

            # status: turns spent under it, and how many times a new one appeared
            if s["status"] in BAD_STATUSES:
                side["status_turns"][s["status"]] += 1
                if side["last_status"].get(name, "nostatus") == "nostatus":
                    side["status_inflicted"][s["status"]] += 1   # includes self-inflicted sleep (Rest)
            side["last_status"][name] = s["status"]
            if s["status"] == "fnt" and name not in side["fainted"]:
                side["fainted"].add(name)
                side["faint_turns"].append(turn)

            for e in s["effects"]:
                if e in EFFECTS:
                    side["effect_turns"][e] += 1
            b = sum(s["boosts"].get(k, 0) for k in BOOST_KEYS)
            side["boost_total"] += b
            side["boost_max"] = max(side["boost_max"], b)
            side["boost_min"] = min(side["boost_min"], b)

            # the move of this turn
            if m is None:
                side["no_move_turns"] += 1
            else:
                side["n_moves"] += 1
                side["bp_sum"] += m["base_power"]
                side["acc_sum"] += m["accuracy"]
                side["n_priority"] += int(m["priority"] > 0)
                if m["name"] in KEY_MOVES:
                    side["key_moves"][m["name"]] += 1
                if m["category"] == "status":
                    side["n_status_moves"] += 1
                elif m["category"] == "physical":
                    side["n_phys"] += 1
                else:
                    side["n_spec"] += 1
            prev[me] = (name, s["status"])

        # snapshot at the end of the turn
        hp1, hp2 = _team_hp(S["p1"]), _team_hp(S["p2"])
        hp_diff_series.append(hp1 - hp2)
        if turn in CHECKPOINTS:
            snaps[turn] = (hp1 - hp2,
                           len(S["p2"]["fainted"]) - len(S["p1"]["fainted"]),
                           t["p1_pokemon_state"]["hp_pct"] - t["p2_pokemon_state"]["hp_pct"])

    # ------------------------------------------------------------------ assemble
    f = {}
    last = tl[-1]
    nt = float(len(tl))
    for me in ("p1", "p2"):
        side, pre = S[me], f"tl_{me}_"
        a = last[me + "_pokemon_state"]
        f[pre + "n_fainted"] = len(side["fainted"])
        f[pre + "n_alive"] = TEAM_SIZE - len(side["fainted"])
        f[pre + "team_hp_total"] = _team_hp(side)
        f[pre + "n_seen"] = len(side["seen"])
        f[pre + "first_faint_turn"] = side["faint_turns"][0] if side["faint_turns"] else n_turns + 1
        f[pre + "last_faint_turn"] = side["faint_turns"][-1] if side["faint_turns"] else 0
        f[pre + "n_faint_last10"] = sum(1 for ft in side["faint_turns"] if ft > n_turns - 10)
        f[pre + "dmg_taken"] = side["dmg_taken"]
        f[pre + "dmg_taken_last10"] = side["dmg_taken_last10"]
        f[pre + "healed"] = side["healed"]
        # healthy = alive, hp >= 50%, no major status (unseen slots count as healthy)
        f[pre + "n_healthy"] = (
            sum(1 for n_, hp_ in side["last_hp"].items()
                if hp_ >= 0.5 and side["last_status"].get(n_, "nostatus") not in BAD_STATUSES + ["fnt"])
            + max(0, TEAM_SIZE - len(side["last_hp"])))
        f[pre + "n_switches"] = side["switches"]
        f[pre + "n_replacements"] = side["replacements"]
        f[pre + "no_move_turns"] = side["no_move_turns"]
        for st in BAD_STATUSES:
            f[pre + f"turns_{st}"] = side["status_turns"][st]
            f[pre + f"inflicted_{st}"] = side["status_inflicted"][st]
        f[pre + "n_statused_now"] = sum(
            1 for n_, st in side["last_status"].items() if st in BAD_STATUSES and n_ not in side["fainted"])
        f[pre + "n_asleep_now"] = sum(
            1 for n_, st in side["last_status"].items() if st == "slp" and n_ not in side["fainted"])
        for e in EFFECTS:
            f[pre + f"eff_{e}"] = side["effect_turns"][e]
        f[pre + "boost_mean"] = side["boost_total"] / nt
        f[pre + "boost_max"] = side["boost_max"]
        f[pre + "boost_min"] = side["boost_min"]
        f[pre + "boost_end"] = sum(a["boosts"].get(k, 0) for k in BOOST_KEYS)
        for k in BOOST_KEYS:
            f[pre + f"boost_{k}_end"] = a["boosts"].get(k, 0)
        f[pre + "active_hp_end"] = a["hp_pct"]
        f[pre + "active_fainted_end"] = int(a["status"] == "fnt")
        f[pre + "active_status_end"] = BAD_STATUSES.index(a["status"]) + 1 if a["status"] in BAD_STATUSES else 0
        f[pre + "n_moves"] = side["n_moves"]
        f[pre + "frac_status_moves"] = side["n_status_moves"] / max(1, side["n_moves"])
        f[pre + "n_phys"] = side["n_phys"]
        f[pre + "n_spec"] = side["n_spec"]
        f[pre + "n_priority"] = side["n_priority"]
        f[pre + "mean_base_power"] = side["bp_sum"] / max(1, side["n_moves"])
        f[pre + "mean_accuracy"] = side["acc_sum"] / max(1, side["n_moves"])
        for k in KEY_MOVES:
            f[pre + f"mv_{k}"] = side["key_moves"][k]

    # trajectory: how the battle looked at turns 5, 10, ... 30 (from Player 1's point of view)
    for cp in CHECKPOINTS:
        hp_d, faint_d, act_d = snaps[cp]
        f[f"tl_diff_hp_t{cp}"] = hp_d
        f[f"tl_diff_faint_t{cp}"] = faint_d      # p2 faints - p1 faints (positive = good for P1)
        f[f"tl_diff_active_hp_t{cp}"] = act_d
    hs = hp_diff_series
    f["tl_diff_hp_mean"] = sum(hs) / len(hs)
    f["tl_diff_hp_min"] = min(hs)
    f["tl_diff_hp_max"] = max(hs)
    f["tl_diff_hp_slope_last10"] = hs[-1] - hs[-11]
    f["tl_diff_hp_slope_last5"] = hs[-1] - hs[-6]

    # explicit p1 - p2 differences of the main aggregates
    for name in ["n_fainted", "n_alive", "team_hp_total", "n_seen", "dmg_taken", "dmg_taken_last10",
                 "n_faint_last10", "n_healthy", "healed", "n_switches", "n_replacements", "no_move_turns",
                 "n_statused_now", "n_asleep_now", "boost_mean", "boost_end", "frac_status_moves",
                 "active_hp_end", "first_faint_turn"]:
        f[f"tl_diff_{name}"] = f[f"tl_p1_{name}"] - f[f"tl_p2_{name}"]
    return f


def build_timeline_frame(battles, n_turns=N_TURNS):
    """Feature table with one row per battle, indexed by battle_id."""
    rows = [timeline_features(b, n_turns) for b in battles]
    return pd.DataFrame(rows, index=[b["battle_id"] for b in battles])
