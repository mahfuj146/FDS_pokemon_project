"""Helpers for looking at the data (used in the eda notebooks, not in the final pipeline)."""
import pandas as pd

def turn_by_turn_battle(b):
    """One row per turn: active Pokemon, HP, status and move of both players."""
    rows = []
    for t in b["battle_timeline"]:
        p1, p2 = t["p1_pokemon_state"], t["p2_pokemon_state"]
        rows.append({
            "turn": t["turn"],
            "p1_pkmn": p1["name"], "p1_hp": round(p1["hp_pct"], 2), "p1_status": p1["status"],
            "p1_move": t["p1_move_details"]["name"] if t["p1_move_details"] else None,
            "p2_pkmn": p2["name"], "p2_hp": round(p2["hp_pct"], 2), "p2_status": p2["status"],
            "p2_move": t["p2_move_details"]["name"] if t["p2_move_details"] else None,
        })
    return pd.DataFrame(rows)

def turns_table(battles):
    """Flatten all timelines: one row per battle turn (sorted by battle_id as given, then turn)."""
    rows = []
    for b in battles:
        for t in b["battle_timeline"]:
            row = {"battle_id": b["battle_id"], "turn": t["turn"]}
            if "player_won" in b:
                row["player_won"] = int(b["player_won"])
            for side in ("p1", "p2"):
                s = t[f"{side}_pokemon_state"]
                row[f"{side}_name"] = s["name"]
                row[f"{side}_hp"] = s["hp_pct"]
                row[f"{side}_status"] = s["status"]
                row[f"{side}_effects"] = ",".join(s["effects"])
                for stat, v in s["boosts"].items():
                    row[f"{side}_boost_{stat}"] = v
                m = t[f"{side}_move_details"]
                row[f"{side}_move"] = m["name"] if m else None
                row[f"{side}_move_type"] = m["type"] if m else None
                row[f"{side}_move_cat"] = m["category"] if m else None
                row[f"{side}_move_power"] = m["base_power"] if m else None
            rows.append(row)
    return pd.DataFrame(rows)

def team_table_p1(battles):
    """One row per player-1 Pokemon (6 per battle), with its two types split into type1 / type2."""
    return pd.DataFrame([
        {"battle_id": b["battle_id"], "slot": i,
         **{k: v for k, v in p.items() if k != "types"},
         "type1": p["types"][0], "type2": p["types"][1]}
        for b in battles for i, p in enumerate(b["p1_team_details"])
    ])

def fainted_per_battle(turns, side):
    """Number of distinct Pokemon of `side` that show status `fnt` during the 30 turns."""
    fnt = turns[turns[f"{side}_status"] == "fnt"]
    counts = fnt.groupby("battle_id")[f"{side}_name"].nunique()
    return counts.reindex(turns["battle_id"].unique(), fill_value=0)

def exits_table(turns, side):
    """Every time the active Pokemon of `side` changes: was the previous one fainted, and does it return?"""
    d = turns[["battle_id", "turn", f"{side}_name", f"{side}_status"]].copy()
    d.columns = ["battle_id", "turn", "name", "status"]
    d["last_turn"] = d.groupby(["battle_id", "name"])["turn"].transform("max")
    prev = d.groupby("battle_id")[["turn", "name", "status", "last_turn"]].shift(1)
    changed = prev["name"].notna() & (d["name"] != prev["name"])
    out = pd.DataFrame({
        "battle_id": d.loc[changed, "battle_id"],
        "exit_turn": prev.loc[changed, "turn"],                      # last turn the old Pokemon was active
        "after_visible_faint": prev.loc[changed, "status"] == "fnt",
        "returns": prev.loc[changed, "last_turn"] > prev.loc[changed, "turn"],
    })
    return out
