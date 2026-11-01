"""Species lookup built from the competition data.

The lookup maps a species name to its types and base stats. It is built from the training inputs
(player-1 teams and player-2 leads), so it contains no information about who won. A species that is not in
the lookup (it can happen for player 2, whose team is only partly visible) is treated as unknown.
"""
import numpy as np

STAT_COLS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]


def build_lookup(battles):
    """name -> {'types': (type1, type2), base_hp: ..., ..., base_spe: ...}"""
    lookup = {}
    for b in battles:
        for p in list(b["p1_team_details"]) + [b["p2_lead_details"]]:
            if p["name"] not in lookup:
                lookup[p["name"]] = {"types": tuple(p["types"]), **{c: p[c] for c in STAT_COLS}}
    return lookup


def species_weight(stats, kind="bst"):
    """How much a Pokemon 'matters', from its base stats (a dict that has the six base_* keys).

    bst  : sum of the six base stats (overall strength)
    bulk : base_hp * (base_def + base_spd) / 100 (how much damage it can absorb)
    """
    if kind == "bst":
        return float(sum(stats[c] for c in STAT_COLS))
    if kind == "bulk":
        return stats["base_hp"] * (stats["base_def"] + stats["base_spd"]) / 100.0
    raise ValueError(f"unknown weight kind: {kind}")


def default_weight(lookup, kind="bst"):
    """Weight given to a Pokemon we know nothing about: the average over the lookup."""
    return float(np.mean([species_weight(s, kind) for s in lookup.values()]))
