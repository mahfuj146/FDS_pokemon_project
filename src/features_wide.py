"""Wide feature set: the state of the Pokemons on the field, type matchups, move-level damage, speed,
HP flow (damage taken and healed), species flags and other flags.

Ideas follow the earlier pipeline, implemented group by group (damaging moves only for the
type-effectiveness counts; fainted Pokemons excluded from status counts).

Uses the species lookup (built from train data) and the type chart. A species that is not in the
lookup has unknown stats and types: the features that need them are NaN (neutral type multiplier 1.0 in
the move aggregates). All functions return DataFrames indexed by battle_id.
"""
import math

import numpy as np
import pandas as pd

from src.features import summary_features, team_features
from src.features_status import boost_features, status_features

SIDES = ("p1", "p2")
STATUS_FLAGS = {"slp": ("slp",), "par": ("par",), "frz": ("frz",), "poison": ("psn", "tox"), "brn": ("brn",)} # the remaining 2 are "fnt" and "nostatus"
EFFECT_FLAGS = ("confusion", "substitute", "leechseed", "trapped")
STAT_KEYS = ("base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe")
ACTIVE_KEYS = (["hp_pct"] + [f"st_{k}" for k in STATUS_FLAGS] + ["n_effects"] + [f"fx_{k}" for k in EFFECT_FLAGS]
               + ["bst"] + [k.replace("base_", "") for k in STAT_KEYS])
MOVE_KEYS = ["n_dmg", "n_se", "n_res", "n_imm", "n_stab", "pot", "pot_last10",
             "n_physical", "n_special", "n_status", "n_priority", "n_distinct"]

ACTIVE_COLS = [f"{s}_act_{k}" for s in SIDES for k in ACTIVE_KEYS] + ["act_hp_diff", "act_bst_diff", "act_spe_diff"]
MATCHUP_COLS = ["p1_act_best_mult", "p2_act_best_mult", "matchup_mult_diff"]
SPEED_COLS = ["eff_spe_logratio", "speed_adv_turns_diff"]
MOVE_COLS = [f"{s}_mv_{k}" for s in SIDES for k in MOVE_KEYS] + ["mv_pot_diff", "mv_pot_last10_diff", "mv_se_diff"]
FLOW_KEYS = ["taken", "taken_last10", "healed", "bighits"]
FLOW_COLS = [f"{s}_flow_{k}" for s in SIDES for k in FLOW_KEYS] + ["flow_taken_diff", "flow_taken_last10_diff"]
MISC_COLS = ["p1_n_low_level", "p1_n_other_item", "p2_lead_low_level"]
BIG_HIT = 0.35           # a drop of at least 35% of max HP in one turn counts as a big hit (our choice)
LAST_TURNS_FROM = 20      # "last 10 turns" = turns 21-30

# ---------------------------------------------------------------- type chart
def type_multipliers(type_chart):
    """DataFrame (attacking_type, defending_type, multiplier) -> dict; unlisted pairs are neutral (1.0)."""
    return {(r.attacking_type, r.defending_type): float(r.multiplier) for r in type_chart.itertuples()}

def type_mult(chart, attack_type, defend_types):
    """Damage multiplier of a move type against a Pokemon with up to two types (multipliers multiply)."""
    if attack_type in ("notype", "threequestionmarks"):
        return 1.0
    m = 1.0
    for d in defend_types:
        if d != "notype":
            m *= chart.get((attack_type, d), 1.0)
    return m

def boost_multiplier(stage):
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)

def effective_speed(info, state):
    """Base speed adjusted for speed boosts and paralysis (quartered); None if the species is unknown."""
    if info is None:
        return None
    return info["base_spe"] * boost_multiplier(state["boosts"]["spe"]) * (0.25 if state["status"] == "par" else 1.0)

# ---------------------------------------------------------------- one battle
def _real_types(info):
    return None if info is None else [t for t in info["types"] if t != "notype"]

def _wide_battle(b, lookup, chart):
    lead2 = b["p2_lead_details"]
    team1 = {p["name"]: p for p in b["p1_team_details"]}

    def info(side, name):
        if side == "p1":
            return team1.get(name)
        return lead2 if name == lead2["name"] else lookup.get(name)

    mv = {s: {k: 0.0 for k in MOVE_KEYS} for s in SIDES}
    distinct = {s: set() for s in SIDES}
    speed_adv = 0
    flow = {s: {k: 0.0 for k in FLOW_KEYS} for s in SIDES}
    flow_last = {s: {} for s in SIDES}          # last known HP of every Pokemon (unseen = full health)

    for t in b["battle_timeline"]:
        st = {s: t[f"{s}_pokemon_state"] for s in SIDES}
        for s in SIDES:                         # HP lost and regained, including on switch-in
            delta = st[s]["hp_pct"] - flow_last[s].get(st[s]["name"], 1.0)
            flow_last[s][st[s]["name"]] = st[s]["hp_pct"]
            if delta < 0:
                flow[s]["taken"] += -delta
                flow[s]["taken_last10"] += -delta if t["turn"] > LAST_TURNS_FROM else 0.0
                flow[s]["bighits"] += -delta >= BIG_HIT
            elif delta > 0:
                flow[s]["healed"] += delta
        inf = {s: info(s, st[s]["name"]) for s in SIDES}
        eff = {s: effective_speed(inf[s], st[s]) for s in SIDES}
        if eff["p1"] is not None and eff["p2"] is not None:
            speed_adv += (eff["p1"] > eff["p2"]) - (eff["p2"] > eff["p1"])

        for s, o in (("p1", "p2"), ("p2", "p1")):
            m = t[f"{s}_move_details"]
            if m is None:
                continue
            distinct[s].add(m["name"])
            if m["category"] == "physical":
                mv[s]["n_physical"] += 1
            elif m["category"] == "special":
                mv[s]["n_special"] += 1
            else:
                mv[s]["n_status"] += 1
            if m["priority"] > 0:
                mv[s]["n_priority"] += 1
            if m["category"] != "status" and m["base_power"] > 0:        # damaging moves only
                att, dfn = _real_types(inf[s]), _real_types(inf[o])
                mult = type_mult(chart, m["type"], dfn) if dfn is not None else 1.0
                stab = 1.5 if (att and m["type"] in att) else 1.0
                acc = m["accuracy"] / 100 if m["accuracy"] > 1 else m["accuracy"]
                pot = m["base_power"] * mult * stab * acc
                mv[s]["n_dmg"] += 1
                mv[s]["n_se"] += mult > 1
                mv[s]["n_res"] += 0 < mult < 1
                mv[s]["n_imm"] += mult == 0
                mv[s]["n_stab"] += stab > 1
                mv[s]["pot"] += pot
                if t["turn"] > LAST_TURNS_FROM:
                    mv[s]["pot_last10"] += pot

    f = {}
    # state of the Pokemons on the field after the last turn
    last = b["battle_timeline"][-1]
    act_info, act_state = {}, {}
    for s in SIDES:
        state = last[f"{s}_pokemon_state"]
        inf_s = info(s, state["name"])
        act_info[s], act_state[s] = inf_s, state
        f[f"{s}_act_hp_pct"] = state["hp_pct"]
        for k, codes in STATUS_FLAGS.items():
            f[f"{s}_act_st_{k}"] = float(state["status"] in codes)
        f[f"{s}_act_n_effects"] = float(sum(e != "noeffect" for e in state["effects"]))
        for k in EFFECT_FLAGS:
            f[f"{s}_act_fx_{k}"] = float(k in state["effects"])
        for k in STAT_KEYS:
            f[f"{s}_act_{k.replace('base_', '')}"] = float(inf_s[k]) if inf_s is not None else np.nan
        f[f"{s}_act_bst"] = float(sum(inf_s[k] for k in STAT_KEYS)) if inf_s is not None else np.nan
    f["act_hp_diff"] = f["p1_act_hp_pct"] - f["p2_act_hp_pct"]
    f["act_bst_diff"] = f["p1_act_bst"] - f["p2_act_bst"]
    f["act_spe_diff"] = f["p1_act_spe"] - f["p2_act_spe"]

    # type matchup between the two active Pokemons (best STAB-type multiplier each way)
    t1, t2 = _real_types(act_info["p1"]), _real_types(act_info["p2"])
    if t1 and t2:
        f["p1_act_best_mult"] = max(type_mult(chart, t, t2) for t in t1)
        f["p2_act_best_mult"] = max(type_mult(chart, t, t1) for t in t2)
        f["matchup_mult_diff"] = f["p1_act_best_mult"] - f["p2_act_best_mult"]
    else:
        f["p1_act_best_mult"] = f["p2_act_best_mult"] = f["matchup_mult_diff"] = np.nan

    # speed
    e1 = effective_speed(act_info["p1"], act_state["p1"])
    e2 = effective_speed(act_info["p2"], act_state["p2"])
    f["eff_spe_logratio"] = math.log(e1 / e2) if (e1 and e2) else np.nan
    f["speed_adv_turns_diff"] = float(speed_adv)

    # moves over the whole timeline
    for s in SIDES:
        mv[s]["n_distinct"] = len(distinct[s])
        for k in MOVE_KEYS:
            f[f"{s}_mv_{k}"] = float(mv[s][k])
    f["mv_pot_diff"] = f["p1_mv_pot"] - f["p2_mv_pot"]
    f["mv_pot_last10_diff"] = f["p1_mv_pot_last10"] - f["p2_mv_pot_last10"]
    f["mv_se_diff"] = (f["p1_mv_n_se"] - f["p1_mv_n_res"] - f["p1_mv_n_imm"]) - (f["p2_mv_n_se"] - f["p2_mv_n_res"] - f["p2_mv_n_imm"])

    for s in SIDES:
        for k in FLOW_KEYS:
            f[f"{s}_flow_{k}"] = float(flow[s][k])
    f["flow_taken_diff"] = f["p2_flow_taken"] - f["p1_flow_taken"]                # positive = good for player 1
    f["flow_taken_last10_diff"] = f["p2_flow_taken_last10"] - f["p1_flow_taken_last10"]

    # other flags from the team details
    f["p1_n_low_level"] = float(sum(p["level"] < 100 for p in b["p1_team_details"]))
    f["p1_n_other_item"] = float(sum(p["item"] != "leftovers" for p in b["p1_team_details"]))
    f["p2_lead_low_level"] = float(lead2["level"] < 100)
    return f

def wide_features(battles, lookup, chart):
    """Active-state, matchup, speed, move and misc features (one row per battle)."""
    rows = []
    for b in battles:
        row = _wide_battle(b, lookup, chart)
        row["battle_id"] = b["battle_id"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("battle_id")

# ---------------------------------------------------------------- species flags
def top_species(battles, k=40):
    """The k species that appear most often in player-1 teams (built from the training data)."""
    counts = {}
    for b in battles:
        for name in {p["name"] for p in b["p1_team_details"]}:
            counts[name] = counts.get(name, 0) + 1
    return [n for n, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]

def species_flags(battles, vocab):
    """p1_has_<species> (in player 1's team) and p2_seen_<species> (shown by player 2) for each species in vocab."""
    rows = []
    for b in battles:
        team = {p["name"] for p in b["p1_team_details"]}
        seen2 = {b["p2_lead_details"]["name"]} | {t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]}
        row = {"battle_id": b["battle_id"]}
        for n in vocab:
            row[f"p1_has_{n}"] = float(n in team)
            row[f"p2_seen_{n}"] = float(n in seen2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("battle_id")

# ---------------------------------------------------------------- the whole table
def build_wide_table(battles, lookup, chart, vocab):
    """All feature groups in one table, plus a dict group -> list of columns."""
    parts = {
        "summary": summary_features(battles),
        "team": team_features(battles),
        "status": status_features(battles),
        "boosts": boost_features(battles),
        "species": species_flags(battles, vocab),
    }
    wide = wide_features(battles, lookup, chart)
    parts["active"] = wide[ACTIVE_COLS]
    parts["matchup"] = wide[MATCHUP_COLS]
    parts["speed"] = wide[SPEED_COLS]
    parts["moves"] = wide[MOVE_COLS]
    parts["flow"] = wide[FLOW_COLS]
    parts["misc"] = wide[MISC_COLS]
    X = pd.concat(parts.values(), axis=1)
    groups = {g: list(df.columns) for g, df in parts.items()}
    assert X.columns.is_unique
    return X, groups
