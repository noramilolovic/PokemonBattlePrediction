
import pandas as pd


def build_species_frame(battles):
    rows = []

    for battle in battles:
        features = {}

        # All 6 Pokemon in Player 1's team
        for pokemon in battle["p1_team_details"]:
            name = pokemon["name"]
            features["p1_has_" + name] = 1

        # Player 1's lead Pokemon
        p1_lead = battle["p1_team_details"][0]["name"]
        features["p1_lead_" + p1_lead] = 1

        # Player 2's lead Pokemon
        p2_lead = battle["p2_lead_details"]["name"]
        features["p2_lead_" + p2_lead] = 1

        # Pokemon revealed by Player 2 in the timeline
        for turn in battle["battle_timeline"][:30]:
            name = turn["p2_pokemon_state"]["name"]
            features["p2_seen_" + name] = 1

        rows.append(features)

    df = pd.DataFrame(
        rows,
        index=[battle["battle_id"] for battle in battles]
    )

    # Missing means the Pokemon was not present
    df = df.fillna(0)

    return df
