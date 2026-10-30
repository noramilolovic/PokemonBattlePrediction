
from pathlib import Path
import pandas as pd

CHART_PATH = Path(__file__).resolve().parent.parent / "data" / "type_chart_gen2.csv"

chart = pd.read_csv(CHART_PATH)

# Create a dictionary for looking up damage multipliers
type_chart = {}

for _, row in chart.iterrows():
    attack = row["attacking_type"]
    defense = row["defending_type"]
    type_chart[(attack, defense)] = float(row["multiplier"])


def effectiveness(attacking_type, defending_types):
    multiplier = 1.0

    for defending_type in defending_types:
        if defending_type == "notype":
            continue

        multiplier *= type_chart.get(
            (attacking_type, defending_type), 1.0
        )

    return multiplier


def build_types_frame(battles):
    rows = []

    for battle in battles:
        features = {}

        team = battle["p1_team_details"]
        p2_lead = battle["p2_lead_details"]

        p2_types = p2_lead["types"]

        # How effective each P1 Pokemon's types are
        # against P2's lead
        attack_scores = []

        for pokemon in team:
            scores = []

            for pokemon_type in pokemon["types"]:
                if pokemon_type == "notype":
                    continue

                score = effectiveness(pokemon_type, p2_types)
                scores.append(score)

            if scores:
                attack_scores.append(max(scores))

        features["p1_best_type_attack_vs_p2_lead"] = max(attack_scores)
        features["p1_mean_type_attack_vs_p2_lead"] = (
            sum(attack_scores) / len(attack_scores)
        )

        # How effective P2's lead types are against P1's lead
        p1_lead_types = team[0]["types"]
        defense_scores = []

        for pokemon_type in p2_types:
            if pokemon_type == "notype":
                continue

            score = effectiveness(pokemon_type, p1_lead_types)
            defense_scores.append(score)

        features["p2_best_type_attack_vs_p1_lead"] = max(defense_scores)

        # Difference between the two lead matchups
        p1_lead_scores = []

        for pokemon_type in p1_lead_types:
            if pokemon_type == "notype":
                continue

            score = effectiveness(pokemon_type, p2_types)
            p1_lead_scores.append(score)

        features["lead_type_advantage"] = (
            max(p1_lead_scores) - max(defense_scores)
        )

        rows.append(features)

    return pd.DataFrame(
        rows,
        index=[battle["battle_id"] for battle in battles]
    )
