"""Strength-weighted HP features.

The plain `hp_diff` counts every Pokemon the same. Here each Pokemon's remaining HP is weighted by how
much that Pokemon matters. Player 1's whole team is known, so its weights come from
the team details; player 2's Pokemon come from its lead details or the lookup, and Pokemons never seen
get the average weight and full health.
"""
import numpy as np
import pandas as pd

from src.lookup import default_weight, species_weight

def _battle_strength(b, lookup, kind, default):
    p1_weights = {p["name"]: species_weight(p, kind) for p in b["p1_team_details"]}
    lead = b["p2_lead_details"]

    last_hp = {"p1": {b["p1_team_details"][0]["name"]: 1.0}, "p2": {lead["name"]: 1.0}}
    for t in b["battle_timeline"]:
        for side in ("p1", "p2"):
            s = t[f"{side}_pokemon_state"]
            last_hp[side][s["name"]] = s["hp_pct"]

    # player 1: the whole team is known; Pokemon never seen are at full health
    p1_hp = sum(w * last_hp["p1"].get(n, 1.0) for n, w in p1_weights.items())
    p1_faint = sum(w for n, w in p1_weights.items() if last_hp["p1"].get(n, 1.0) == 0)

    # player 2: weights from the lead details or the lookup; unknown species get the average weight
    p2_hp = p2_faint = 0.0
    unknown = 0
    for name, hp in last_hp["p2"].items():
        if name == lead["name"]:
            w = species_weight(lead, kind)
        elif name in lookup:
            w = species_weight(lookup[name], kind)
        else:
            w, unknown = default, unknown + 1
        p2_hp += w * hp
        p2_faint += w * (hp == 0)
    p2_hp += max(0, 6 - len(last_hp["p2"])) * default      # never seen: average Pokemon, full health

    return {
        "battle_id": b["battle_id"],
        "p1_whp": p1_hp, "p2_whp": p2_hp,
        "whp_diff": p1_hp - p2_hp,
        "whp_share": p1_hp / (p1_hp + p2_hp),
        "p1_wfaint": p1_faint, "p2_wfaint": p2_faint,
        "wfaint_diff": p2_faint - p1_faint,                # positive = good for player 1
        "p2_unknown": unknown,
    }

def strength_features(battles, lookup, kind="bst"):
    """Weighted-HP features; columns are prefixed with the weight kind (e.g. 'bst_whp_diff')."""
    default = default_weight(lookup, kind)
    df = pd.DataFrame([_battle_strength(b, lookup, kind, default) for b in battles]).set_index("battle_id")
    return df.add_prefix(f"{kind}_")
