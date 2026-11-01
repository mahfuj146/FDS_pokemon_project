"""Status and boost state at the end of the 30 turns.

A sleeping, frozen, paralysed or poisoned Pokemon is less useful than a healthy one with the same HP.
Status is tracked per Pokemon (the last status seen); fainted Pokemons are excluded, because their
status is just `fnt` and the faint information is already in the HP features.
"""
import pandas as pd

STATUS_GROUPS = {"slp": ("slp",), "frz": ("frz",), "par": ("par",), "poison": ("psn", "tox"), "brn": ("brn",)}
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe")

def _battle_status(b):
    f = {"battle_id": b["battle_id"]}
    bad = {}
    for side in ("p1", "p2"):
        last_status = {}
        for t in b["battle_timeline"]:
            s = t[f"{side}_pokemon_state"]
            last_status[s["name"]] = s["status"]
        total = 0
        for group, codes in STATUS_GROUPS.items():
            n = sum(st in codes for st in last_status.values())
            f[f"{side}_n_{group}"] = n
            total += n
        bad[side] = total
    f["status_burden_diff"] = bad["p2"] - bad["p1"]       # positive = good for player 1
    return f

def status_features(battles):
    """Number of Pokemons per side that are asleep, frozen, paralysed, poisoned or burned (not fainted)."""
    return pd.DataFrame([_battle_status(b) for b in battles]).set_index("battle_id")

def boost_features(battles):
    """Sum of the stat boosts of each side's active Pokemon after the last turn."""
    rows = []
    for b in battles:
        last = b["battle_timeline"][-1]
        p1 = sum(last["p1_pokemon_state"]["boosts"][k] for k in BOOST_KEYS)
        p2 = sum(last["p2_pokemon_state"]["boosts"][k] for k in BOOST_KEYS)
        rows.append({"battle_id": b["battle_id"], "p1_boost_sum": p1, "p2_boost_sum": p2, "boost_diff": p1 - p2})
    return pd.DataFrame(rows).set_index("battle_id")
