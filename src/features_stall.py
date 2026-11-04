"""Stall features (extra block): how much stalling power is left after turn 30.

Error analysis showed that battles with rest / recover / reflect / rapid spin
are the hardest ones: they last long and are decided after the recorded
window. The timeline block counts how often such moves were used; this block
asks which of the Pokemon that showed them are STILL ALIVE at the end, and
whether hazards are still on the field.

All features are prefixed with st_. Every function takes ONE battle (dict)
and returns {feature_name: number}. player_won is never used. A move is
credited to the Pokemon that was active in the turn it was used.
"""

import pandas as pd

N_TURNS = 30
RECOVERY = {"rest", "recover", "milkdrink", "softboiled", "moonlight", "morningsun", "synthesis"}
PHAZING = {"roar", "whirlwind"}
SCREENS = {"reflect", "lightscreen"}
ROLES = {
    "recovery": RECOVERY,
    "rest": {"rest"},
    "sleeptalk": {"sleeptalk"},
    "spinner": {"rapidspin"},
    "phazer": PHAZING,
    "screens": SCREENS,
    "healbell": {"healbell"},
}


def stall_features(battle, n_turns=N_TURNS):
    """ONE battle (dict) -> dict of numbers."""
    tl = battle["battle_timeline"][:n_turns]
    if not tl:
        raise ValueError(f"battle {battle.get('battle_id')} has an empty timeline")

    used = {"p1": {}, "p2": {}}  # Pokemon name -> set of moves it used
    status = {"p1": {}, "p2": {}}  # Pokemon name -> last known status
    spikes_on = {"p1": False, "p2": False}  # spikes lying on THIS side of the field
    spikes_turns = {"p1": 0, "p2": 0}  # turns this side spent with spikes on it

    # ---------------- replay the recorded turns ----------------
    for t in tl:
        for me, other in (("p1", "p2"), ("p2", "p1")):
            s = t[me + "_pokemon_state"]
            status[me][s["name"]] = s["status"]
            m = t[me + "_move_details"]
            if m is None:
                continue
            used[me].setdefault(s["name"], set()).add(m["name"])
            if m["name"] == "spikes":
                spikes_on[other] = True
            elif m["name"] == "rapidspin":
                spikes_on[me] = False
        for me in ("p1", "p2"):
            spikes_turns[me] += int(spikes_on[me])

    feats = {}
    last = tl[-1]
    for me in ("p1", "p2"):
        alive = {n for n, st in status[me].items() if st != "fnt"}
        for role, moves in ROLES.items():
            feats[f"st_{me}_n_alive_{role}"] = sum(
                1 for n in alive if used[me].get(n, set()) & moves
            )
        # sleeping Pokemon that put themselves to sleep with rest wake up healed;
        # the others were put to sleep by the opponent
        asleep = [n for n in alive if status[me][n] == "slp"]
        feats[f"st_{me}_n_asleep_rest"] = sum("rest" in used[me].get(n, set()) for n in asleep)
        feats[f"st_{me}_n_asleep_forced"] = sum("rest" not in used[me].get(n, set()) for n in asleep)

        act = last[me + "_pokemon_state"]["name"]
        act_moves = used[me].get(act, set())
        feats[f"st_{me}_active_has_recovery"] = int(bool(act_moves & RECOVERY))
        feats[f"st_{me}_active_n_moves_seen"] = len(act_moves)
        feats[f"st_{me}_spikes_on_side"] = int(spikes_on[me])
        feats[f"st_{me}_spikes_turns"] = spikes_turns[me]

    for k in list(ROLES) + ["asleep_rest", "asleep_forced"]:
        name = f"n_alive_{k}" if k in ROLES else f"n_{k}"
        feats[f"st_diff_{name}"] = feats[f"st_p1_{name}"] - feats[f"st_p2_{name}"]
    # spikes hurt the side they lie on, so P1's advantage is P2's spikes minus P1's
    feats["st_diff_spikes_on_side"] = feats["st_p2_spikes_on_side"] - feats["st_p1_spikes_on_side"]
    return feats


def build_stall_frame(battles, n_turns=N_TURNS):
    """Feature table with one row per battle, indexed by battle_id."""
    rows = [stall_features(b, n_turns) for b in battles]
    return pd.DataFrame(rows, index=[b["battle_id"] for b in battles])
