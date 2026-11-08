# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:04:29 2026

@author: mashi
"""

import os
import glob
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

SEED = 42
N_TURNS = 30
BIG_HIT_PCT = 35.0        # >= 35% HP drop in one turn (on 0-100 scale)
LAST_TURNS_FROM = 20      # "last 10 turns" = turns 21..30
TOP_K_SPECIES = 40        # Top 40 species from train.jsonl

STAT_KEYS = ("base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe")
CHECKPOINTS = (5, 10, 15, 20, 25)
STATUS_FLAGS = {
    "slp": ("slp", "sleep"),
    "par": ("par", "paralysis"),
    "frz": ("frz", "freeze"),
    "poison": ("psn", "tox", "poison", "toxic"),
    "brn": ("brn", "burn")
}
EFFECT_FLAGS = ("confusion", "substitute", "leechseed", "trapped")

# New cache filenames for the merged ~210-feature dataset
TRAIN_FEATURES_CSV = "train_features_merged.csv"
TEST_FEATURES_CSV = "test_features_merged.csv"
REGIME_PLOT_PNG = "regime_feature_analysis.png"

# Precomputed vector for fast OLS linear regression slope across 30 turns
_T_CENTERED = np.arange(N_TURNS, dtype=float) - ((N_TURNS - 1) / 2.0)
_T_DENOM = float(np.sum(_T_CENTERED ** 2))


def resolve_data_path(filename: str) -> str:
    """
    Finds train.jsonl, test.jsonl, and type_chart_gen2.csv automatically
    in current folder, Kaggle (/kaggle/input), or local Windows (C:\\FDS).
    """
    if os.path.exists(filename):
        return filename

    kaggle_matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if kaggle_matches:
        return kaggle_matches[0]

    local_win_path = os.path.join(r"C:\FDS", filename)
    if os.path.exists(local_win_path):
        return local_win_path

    raise FileNotFoundError(f"Could not find {filename} in current dir, /kaggle/input, or C:\\FDS")


def load_type_lookup(type_chart_path: str) -> dict:
    type_chart = pd.read_csv(type_chart_path)
    return type_chart.set_index(["attacking_type", "defending_type"])["multiplier"].to_dict()


def build_internal_pokedex_and_vocab(train_jsonl_path: str, k_species: int = TOP_K_SPECIES):
    """
    Scans strictly train.jsonl to build:
      1. pokedex: species -> types, 6 base stats, and Base Stat Total (bst)
      2. top_vocab: list of top K most frequent species in Player 1 teams
      3. default_bst: average BST across known species for unseen Pokémon fallback
    """
    pokedex = {}
    species_counts = {}

    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            battle = json.loads(line)
            p1_team = battle.get("p1_team_details", [])

            for name in {p.get("name") for p in p1_team if p.get("name")}:
                species_counts[name] = species_counts.get(name, 0) + 1

            for pkmn in p1_team:
                name = pkmn.get("name")
                if name and name not in pokedex:
                    stats = {s: float(pkmn.get(s, 80)) for s in STAT_KEYS}
                    pokedex[name] = {
                        "types": pkmn.get("types", ["notype"]),
                        **stats,
                        "bst": sum(stats.values())
                    }

            p2_lead = battle.get("p2_lead_details", {})
            lead_name = p2_lead.get("name")
            if lead_name and lead_name not in pokedex:
                stats = {s: float(p2_lead.get(s, 80)) for s in STAT_KEYS}
                pokedex[lead_name] = {
                    "types": p2_lead.get("types", ["notype"]),
                    **stats,
                    "bst": sum(stats.values())
                }

    top_vocab = [
        n for n, _ in sorted(species_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k_species]
    ]
    default_bst = float(np.mean([v["bst"] for v in pokedex.values()])) if pokedex else 480.0
    return pokedex, top_vocab, default_bst


def calculate_move_multiplier(attacking_type: str, defending_types: list, type_lookup: dict) -> float:
    if not attacking_type or attacking_type in ("notype", "threequestionmarks"):
        return 1.0
    total_multiplier = 1.0
    for def_type in defending_types:
        if def_type and def_type != "notype":
            total_multiplier *= type_lookup.get((attacking_type, def_type), 1.0)
    return total_multiplier


def get_pokemon_info(name: str, local_roster: dict, global_pokedex: dict, default_bst: float) -> dict:
    if name in local_roster:
        info = local_roster[name]
        if "bst" not in info:
            info["bst"] = sum(float(info.get(s, 80)) for s in STAT_KEYS)
        return info
    if name in global_pokedex:
        return global_pokedex[name]
    return {
        "types": ["notype"],
        "base_hp": 80.0, "base_atk": 80.0, "base_def": 80.0,
        "base_spa": 80.0, "base_spd": 80.0, "base_spe": 80.0,
        "bst": default_bst,
        "unknown": True
    }


def get_boost_multiplier(stage: int) -> float:
    stage = max(-6, min(6, int(stage)))
    return (2.0 + stage) / 2.0 if stage >= 0 else 2.0 / (2.0 - stage)


def normalize_hp_pct(raw_hp, is_fractional_scale: bool) -> float:
    if raw_hp is None:
        return 0.0
    val = float(raw_hp)
    if is_fractional_scale and val <= 1.0:
        return val * 100.0
    return val


def extract_battle_features(
    battle_json: dict,
    type_lookup: dict,
    global_pokedex: dict,
    top_vocab: list,
    default_bst: float
) -> dict:
    features = {
        "battle_id": battle_json["battle_id"],
        "p1_super_effective_hits": 0,
        "p2_super_effective_hits": 0,
        "p1_resisted_hits": 0,
        "p2_resisted_hits": 0,
        "p1_immune_hits": 0,
        "p2_immune_hits": 0,
        "p1_weighted_damage_potential": 0.0,
        "p2_weighted_damage_potential": 0.0,
        "p1_stat_matched_damage_score": 0.0,
        "p2_stat_matched_damage_score": 0.0,
        "p1_stab_moves_used": 0,
        "p2_stab_moves_used": 0,
        "p1_physical_moves_used": 0,
        "p1_special_moves_used": 0,
        "p2_physical_moves_used": 0,
        "p2_special_moves_used": 0,
        "p1_status_moves_used": 0,
        "p2_status_moves_used": 0,
        "p1_late_damage_score": 0.0,
        "p2_late_damage_score": 0.0,
        "p1_mv_pot_last10": 0.0,
        "p2_mv_pot_last10": 0.0,
        "p1_switch_count": 0,
        "p2_switch_count": 0,
        "p1_null_move_turns": 0,
        "p2_null_move_turns": 0,
        "p1_status_turns": 0,
        "p2_status_turns": 0,
        "p1_volatile_turns": 0,
        "p2_volatile_turns": 0,
        "p1_speed_advantage_turns": 0,
        "speed_adv_turns_diff": 0,
        "p1_true_first_strike_turns": 0,
        "p1_priority_moves_used": 0,
        "p2_priority_moves_used": 0,
        "p1_active_turn30_hp": 100.0,
        "p2_active_turn30_hp": 100.0,
        "turn30_matchup_mult_diff": 0.0,
        # Merged HP Flow features (damage taken, healing, big hits >= 35%)
        "p1_flow_taken": 0.0,
        "p2_flow_taken": 0.0,
        "p1_flow_taken_last10": 0.0,
        "p2_flow_taken_last10": 0.0,
        "p1_flow_healed": 0.0,
        "p2_flow_healed": 0.0,
        "p1_flow_bighits": 0,
        "p2_flow_bighits": 0,
    }

    p1_team = battle_json.get("p1_team_details", [])
    p2_lead = battle_json.get("p2_lead_details", {})
    timeline = battle_json.get("battle_timeline", [])

    # Detect whether raw hp_pct is fractional (0.0-1.0) or percentage (0-100)
    max_raw_hp = 0.0
    for t in timeline:
        s1 = (t.get("p1_pokemon_state") or {}).get("hp_pct")
        s2 = (t.get("p2_pokemon_state") or {}).get("hp_pct")
        if s1 is not None:
            max_raw_hp = max(max_raw_hp, float(s1))
        if s2 is not None:
            max_raw_hp = max(max_raw_hp, float(s2))
    is_fractional = (max_raw_hp <= 1.0001)

    local_roster = {}
    p1_hp_tracker = {}
    p1_status_tracker = {}

    for pkmn in p1_team:
        name = pkmn.get("name")
        if name:
            pkmn_copy = dict(pkmn)
            pkmn_copy["bst"] = sum(float(pkmn_copy.get(s, 80)) for s in STAT_KEYS)
            local_roster[name] = pkmn_copy
            p1_hp_tracker[name] = 100.0
            p1_status_tracker[name] = None

    p2_hp_tracker = {}
    p2_status_tracker = {}
    if p2_lead.get("name"):
        p2_name = p2_lead["name"]
        lead_copy = dict(p2_lead)
        lead_copy["bst"] = sum(float(lead_copy.get(s, 80)) for s in STAT_KEYS)
        local_roster[p2_name] = lead_copy
        p2_hp_tracker[p2_name] = 100.0
        p2_status_tracker[p2_name] = None

    # Static Team & Misc Level/Item Features
    if p1_team:
        p1_lead = p1_team[0]
        features["lead_spe_diff"] = p1_lead.get("base_spe", 0) - p2_lead.get("base_spe", 0)
        features["lead_hp_diff"] = p1_lead.get("base_hp", 0) - p2_lead.get("base_hp", 0)
        for s_col in STAT_KEYS:
            features[f"p1_team_mean_{s_col}"] = float(np.mean([p.get(s_col, 80) for p in p1_team]))
        features["p1_team_mean_spe"] = features["p1_team_mean_base_spe"]
        features["p1_team_mean_bulk"] = float(np.mean([
            p.get("base_hp", 0) + p.get("base_def", 0) + p.get("base_spd", 0) for p in p1_team
        ]))
        features["p1_leftovers_count"] = sum(1 for p in p1_team if p.get("item") == "leftovers")
        features["p1_n_low_level"] = float(sum(p.get("level", 100) < 100 for p in p1_team))
        features["p1_n_other_item"] = float(sum(p.get("item") != "leftovers" for p in p1_team))
    else:
        features["lead_spe_diff"] = 0.0
        features["lead_hp_diff"] = 0.0
        for s_col in STAT_KEYS:
            features[f"p1_team_mean_{s_col}"] = 80.0
        features["p1_team_mean_spe"] = 80.0
        features["p1_team_mean_bulk"] = 240.0
        features["p1_leftovers_count"] = 0
        features["p1_n_low_level"] = 0.0
        features["p1_n_other_item"] = 0.0

    features["p2_lead_low_level"] = float(p2_lead.get("level", 100) < 100)

    # Trackers for 30-turn trajectory and move diversity
    p1_seen_timeline = {p1_team[0]["name"]: 100.0} if p1_team and p1_team[0].get("name") else {}
    p2_seen_timeline = {p2_lead["name"]: 100.0} if p2_lead.get("name") else {}
    flow_last = {"p1": {}, "p2": {}}
    distinct_moves = {"p1": set(), "p2": set()}

    hp_diff_series = []
    faint_diff_series = []

    p1_active_prev = None
    p2_active_prev = None
    p1_boost_sum_last = 0
    p2_boost_sum_last = 0
    hp_diff_turn15 = None

    last_p1_state, last_p2_state = {}, {}
    last_p1_info, last_p2_info = {}, {}
    last_p1_eff_spe, last_p2_eff_spe = 80.0, 80.0

    for idx, turn_data in enumerate(timeline):
        turn_num = turn_data.get("turn", idx + 1)
        p1_state = turn_data.get("p1_pokemon_state") or {}
        p2_state = turn_data.get("p2_pokemon_state") or {}
        p1_move = turn_data.get("p1_move_details")
        p2_move = turn_data.get("p2_move_details")

        p1_curr_name = p1_state.get("name")
        p2_curr_name = p2_state.get("name")

        p1_info = get_pokemon_info(p1_curr_name, local_roster, global_pokedex, default_bst)
        p2_info = get_pokemon_info(p2_curr_name, local_roster, global_pokedex, default_bst)

        p1_types = p1_info.get("types", ["notype"])
        p2_types = p2_info.get("types", ["notype"])

        if p1_active_prev is not None and p1_curr_name != p1_active_prev:
            features["p1_switch_count"] += 1
        if p2_active_prev is not None and p2_curr_name != p2_active_prev:
            features["p2_switch_count"] += 1

        p1_active_prev = p1_curr_name
        p2_active_prev = p2_curr_name

        p1_cur_hp = normalize_hp_pct(p1_state.get("hp_pct", 0.0), is_fractional)
        p2_cur_hp = normalize_hp_pct(p2_state.get("hp_pct", 0.0), is_fractional)

        # HP Flow (Damage Taken, Healing, and Big Hits >= 35%)
        for side, c_name, c_hp in (("p1", p1_curr_name, p1_cur_hp), ("p2", p2_curr_name, p2_cur_hp)):
            if c_name:
                prev_hp = flow_last[side].get(c_name, 100.0)
                delta = c_hp - prev_hp
                flow_last[side][c_name] = c_hp
                if delta < 0:
                    dmg_drop = -delta
                    features[f"{side}_flow_taken"] += dmg_drop
                    if turn_num > LAST_TURNS_FROM:
                        features[f"{side}_flow_taken_last10"] += dmg_drop
                    if dmg_drop >= BIG_HIT_PCT:
                        features[f"{side}_flow_bighits"] += 1
                elif delta > 0:
                    features[f"{side}_flow_healed"] += delta

        if p1_curr_name:
            p1_hp_tracker[p1_curr_name] = p1_cur_hp
            p1_seen_timeline[p1_curr_name] = p1_cur_hp
            p1_status_tracker[p1_curr_name] = p1_state.get("status")
        if p2_curr_name:
            p2_hp_tracker[p2_curr_name] = p2_cur_hp
            p2_seen_timeline[p2_curr_name] = p2_cur_hp
            p2_status_tracker[p2_curr_name] = p2_state.get("status")

        # Turn-by-turn trajectory series (0-600 scale)
        p1_t_hp = sum(p1_hp_tracker.values())
        p2_t_hp = sum(p2_hp_tracker.values()) + max(0, 6 - len(p2_hp_tracker)) * 100.0
        hp_diff_series.append(p1_t_hp - p2_t_hp)

        p1_t_fnt = sum(1 for v in p1_hp_tracker.values() if v <= 0.001)
        p2_t_fnt = sum(1 for v in p2_hp_tracker.values() if v <= 0.001)
        faint_diff_series.append(p2_t_fnt - p1_t_fnt)

        if turn_num == 15 or idx == 14:
            hp_diff_turn15 = p1_t_hp - p2_t_hp

        p1_status_val = str(p1_state.get("status") or "").lower()
        p2_status_val = str(p2_state.get("status") or "").lower()

        if p1_status_val not in ("", "none", "nostatus", "fnt"):
            features["p1_status_turns"] += 1
        if p2_status_val not in ("", "none", "nostatus", "fnt"):
            features["p2_status_turns"] += 1

        p1_fx = p1_state.get("effects") or p1_state.get("volatile_effects") or []
        p2_fx = p2_state.get("effects") or p2_state.get("volatile_effects") or []
        features["p1_volatile_turns"] += sum(1 for e in p1_fx if e != "noeffect")
        features["p2_volatile_turns"] += sum(1 for e in p2_fx if e != "noeffect")

        p1_boosts = p1_state.get("boosts") or p1_state.get("stat_boosts") or {}
        p2_boosts = p2_state.get("boosts") or p2_state.get("stat_boosts") or {}
        if not isinstance(p1_boosts, dict):
            p1_boosts = {}
        if not isinstance(p2_boosts, dict):
            p2_boosts = {}

        p1_boost_sum_last = sum(p1_boosts.values())
        p2_boost_sum_last = sum(p2_boosts.values())

        if p1_info.get("base_spe", 80) > p2_info.get("base_spe", 80):
            features["p1_speed_advantage_turns"] += 1

        p1_prio = (p1_move.get("priority") or 0) if p1_move else 0
        p2_prio = (p2_move.get("priority") or 0) if p2_move else 0

        if p1_prio > 0:
            features["p1_priority_moves_used"] += 1
        if p2_prio > 0:
            features["p2_priority_moves_used"] += 1

        p1_eff_spe = p1_info.get("base_spe", 80) * get_boost_multiplier(p1_boosts.get("spe", 0))
        p2_eff_spe = p2_info.get("base_spe", 80) * get_boost_multiplier(p2_boosts.get("spe", 0))
        if p1_status_val == "par":
            p1_eff_spe *= 0.25
        if p2_status_val == "par":
            p2_eff_spe *= 0.25

        features["speed_adv_turns_diff"] += int(p1_eff_spe > p2_eff_spe) - int(p2_eff_spe > p1_eff_spe)

        if (p1_prio > p2_prio) or (p1_prio == p2_prio and p1_eff_spe > p2_eff_spe):
            features["p1_true_first_strike_turns"] += 1

        # Move Analysis for P1
        if p1_move and p1_move.get("type"):
            if p1_move.get("name"):
                distinct_moves["p1"].add(p1_move["name"])
            m_type = p1_move["type"]
            m_cat = str(p1_move.get("category") or "").lower()
            base_power = p1_move.get("base_power") or 0

            if "phys" in m_cat:
                features["p1_physical_moves_used"] += 1
                atk_stat = max(1.0, p1_info.get("base_atk", 80) * get_boost_multiplier(p1_boosts.get("atk", 0)))
                def_stat = max(1.0, p2_info.get("base_def", 80) * get_boost_multiplier(p2_boosts.get("def", 0)))
                stat_ratio = atk_stat / def_stat
            elif "spec" in m_cat:
                features["p1_special_moves_used"] += 1
                spa_stat = max(1.0, p1_info.get("base_spa", 80) * get_boost_multiplier(p1_boosts.get("spa", 0)))
                spd_stat = max(1.0, p2_info.get("base_spd", 80) * get_boost_multiplier(p2_boosts.get("spd", 0)))
                stat_ratio = spa_stat / spd_stat
            else:
                features["p1_status_moves_used"] += 1
                stat_ratio = 1.0

            if "stat" not in m_cat and base_power > 0:
                mult = calculate_move_multiplier(m_type, p2_types, type_lookup)
                if mult > 1.0:
                    features["p1_super_effective_hits"] += 1
                elif mult == 0.0:
                    features["p1_immune_hits"] += 1
                elif mult < 1.0:
                    features["p1_resisted_hits"] += 1

                stab = 1.5 if (m_type != "notype" and m_type in p1_types) else 1.0
                if stab > 1.0:
                    features["p1_stab_moves_used"] += 1

                accuracy = p1_move.get("accuracy") or 100
                acc_factor = accuracy / 100.0 if accuracy > 1.0 else accuracy

                raw_dmg = base_power * mult * acc_factor
                pot_dmg = raw_dmg * stab
                true_dmg = pot_dmg * stat_ratio

                features["p1_weighted_damage_potential"] += raw_dmg
                features["p1_stat_matched_damage_score"] += true_dmg
                if turn_num > 15:
                    features["p1_late_damage_score"] += true_dmg
                if turn_num > LAST_TURNS_FROM:
                    features["p1_mv_pot_last10"] += pot_dmg
        else:
            features["p1_null_move_turns"] += 1

        # Move Analysis for P2
        if p2_move and p2_move.get("type"):
            if p2_move.get("name"):
                distinct_moves["p2"].add(p2_move["name"])
            m_type = p2_move["type"]
            m_cat = str(p2_move.get("category") or "").lower()
            base_power = p2_move.get("base_power") or 0

            if "phys" in m_cat:
                features["p2_physical_moves_used"] += 1
                atk_stat = max(1.0, p2_info.get("base_atk", 80) * get_boost_multiplier(p2_boosts.get("atk", 0)))
                def_stat = max(1.0, p1_info.get("base_def", 80) * get_boost_multiplier(p1_boosts.get("def", 0)))
                stat_ratio = atk_stat / def_stat
            elif "spec" in m_cat:
                features["p2_special_moves_used"] += 1
                spa_stat = max(1.0, p2_info.get("base_spa", 80) * get_boost_multiplier(p2_boosts.get("spa", 0)))
                spd_stat = max(1.0, p1_info.get("base_spd", 80) * get_boost_multiplier(p1_boosts.get("spd", 0)))
                stat_ratio = spa_stat / spd_stat
            else:
                features["p2_status_moves_used"] += 1
                stat_ratio = 1.0

            if "stat" not in m_cat and base_power > 0:
                mult = calculate_move_multiplier(m_type, p1_types, type_lookup)
                if mult > 1.0:
                    features["p2_super_effective_hits"] += 1
                elif mult == 0.0:
                    features["p2_immune_hits"] += 1
                elif mult < 1.0:
                    features["p2_resisted_hits"] += 1

                stab = 1.5 if (m_type != "notype" and m_type in p2_types) else 1.0
                if stab > 1.0:
                    features["p2_stab_moves_used"] += 1

                accuracy = p2_move.get("accuracy") or 100
                acc_factor = accuracy / 100.0 if accuracy > 1.0 else accuracy

                raw_dmg = base_power * mult * acc_factor
                pot_dmg = raw_dmg * stab
                true_dmg = pot_dmg * stat_ratio

                features["p2_weighted_damage_potential"] += raw_dmg
                features["p2_stat_matched_damage_score"] += true_dmg
                if turn_num > 15:
                    features["p2_late_damage_score"] += true_dmg
                if turn_num > LAST_TURNS_FROM:
                    features["p2_mv_pot_last10"] += pot_dmg
        else:
            features["p2_null_move_turns"] += 1

        features["p1_active_turn30_hp"] = p1_cur_hp
        features["p2_active_turn30_hp"] = p2_cur_hp

        p1_best_stab = max([calculate_move_multiplier(t, p2_types, type_lookup) for t in p1_types])
        p2_best_stab = max([calculate_move_multiplier(t, p1_types, type_lookup) for t in p2_types])
        features["p1_act_best_mult"] = p1_best_stab
        features["p2_act_best_mult"] = p2_best_stab
        features["turn30_matchup_mult_diff"] = p1_best_stab - p2_best_stab

        last_p1_state, last_p2_state = p1_state, p2_state
        last_p1_info, last_p2_info = p1_info, p2_info
        last_p1_eff_spe, last_p2_eff_spe = p1_eff_spe, p2_eff_spe

    # Active Turn-30 Field State, Status/Effect Flags & Effective Speed Log-Ratio
    for side, st, inf in (("p1", last_p1_state, last_p1_info), ("p2", last_p2_state, last_p2_info)):
        st_code = str(st.get("status") or "").lower()
        for k, codes in STATUS_FLAGS.items():
            features[f"{side}_act_st_{k}"] = float(st_code in codes)
        fx_list = st.get("effects") or st.get("volatile_effects") or []
        features[f"{side}_act_n_effects"] = float(sum(1 for e in fx_list if e != "noeffect"))
        for k in EFFECT_FLAGS:
            features[f"{side}_act_fx_{k}"] = float(k in fx_list)
        for k in STAT_KEYS:
            features[f"{side}_act_{k.replace('base_', '')}"] = float(inf.get(k, 80.0))
        features[f"{side}_act_bst"] = float(inf.get("bst", default_bst))

    features["act_hp_diff"] = features["p1_active_turn30_hp"] - features["p2_active_turn30_hp"]
    features["act_bst_diff"] = features["p1_act_bst"] - features["p2_act_bst"]
    features["act_spe_diff"] = features["p1_act_spe"] - features["p2_act_spe"]
    features["eff_spe_logratio"] = (
        math.log(max(1.0, last_p1_eff_spe) / max(1.0, last_p2_eff_spe))
    )

    # Move & Flow Differentials
    features["p1_mv_n_distinct"] = float(len(distinct_moves["p1"]))
    features["p2_mv_n_distinct"] = float(len(distinct_moves["p2"]))
    features["mv_pot_last10_diff"] = features["p1_mv_pot_last10"] - features["p2_mv_pot_last10"]
    features["mv_se_diff"] = (
        (features["p1_super_effective_hits"] - features["p1_resisted_hits"] - features["p1_immune_hits"]) -
        (features["p2_super_effective_hits"] - features["p2_resisted_hits"] - features["p2_immune_hits"])
    )
    features["flow_taken_diff"] = features["p2_flow_taken"] - features["p1_flow_taken"]
    features["flow_taken_last10_diff"] = features["p2_flow_taken_last10"] - features["p1_flow_taken_last10"]
    features["flow_healed_diff"] = features["p1_flow_healed"] - features["p2_flow_healed"]

    # Trajectory Checkpoints (Turns 5, 10, 15, 20, 25) & Linear Slope
    if not hp_diff_series:
        hp_diff_series = [0.0] * N_TURNS
        faint_diff_series = [0] * N_TURNS
    while len(hp_diff_series) < N_TURNS:
        hp_diff_series.append(hp_diff_series[-1])
        faint_diff_series.append(faint_diff_series[-1])

    hp_arr = np.array(hp_diff_series[:N_TURNS], dtype=float)
    faint_arr = np.array(faint_diff_series[:N_TURNS], dtype=float)

    for c in CHECKPOINTS:
        features[f"hp_diff_t{c}"] = float(hp_arr[c - 1])
    features["hp_diff_delta_mid"] = float(hp_arr[19] - hp_arr[9])     # Turns 10 -> 20
    features["hp_diff_delta_last"] = float(hp_arr[29] - hp_arr[19])   # Turns 20 -> 30
    features["faint_diff_t10"] = float(faint_arr[9])
    features["faint_diff_t20"] = float(faint_arr[19])

    features["hp_diff_mean"] = float(hp_arr.mean())
    features["hp_diff_min"] = float(hp_arr.min())
    features["hp_diff_max"] = float(hp_arr.max())
    features["hp_diff_std"] = float(hp_arr.std())
    features["hp_diff_slope"] = float(np.sum(_T_CENTERED * (hp_arr - hp_arr.mean())) / _T_DENOM)

    # Turn-30 Totals, Knockouts & BST Strength-Weighted HP
    features["p1_turn30_boost_sum"] = p1_boost_sum_last
    features["p2_turn30_boost_sum"] = p2_boost_sum_last
    features["stat_matched_damage_diff"] = (
        features["p1_stat_matched_damage_score"] - features["p2_stat_matched_damage_score"]
    )
    features["late_damage_diff"] = features["p1_late_damage_score"] - features["p2_late_damage_score"]

    p1_seen_count = len(p1_seen_timeline)
    p2_revealed_count = len(p2_hp_tracker)
    p2_unrevealed_count = max(0, 6 - p2_revealed_count)

    p1_total_hp = sum(p1_hp_tracker.values())
    p2_total_hp_est = sum(p2_hp_tracker.values()) + (p2_unrevealed_count * 100.0)

    features["p1_seen_pokemon_count"] = p1_seen_count
    features["p2_revealed_pokemon_count"] = p2_revealed_count
    features["p1_total_hp_remaining"] = p1_total_hp
    features["p2_total_hp_remaining_est"] = p2_total_hp_est
    features["total_hp_diff_est"] = p1_total_hp - p2_total_hp_est

    if hp_diff_turn15 is None:
        hp_diff_turn15 = features["total_hp_diff_est"]
    features["hp_diff_turn15_vs_turn30"] = features["total_hp_diff_est"] - hp_diff_turn15

    features["p1_fainted_count"] = sum(1 for hp in p1_hp_tracker.values() if hp <= 0.001)
    features["p2_fainted_count"] = sum(1 for hp in p2_hp_tracker.values() if hp <= 0.001)
    features["fainted_diff"] = features["p2_fainted_count"] - features["p1_fainted_count"]

    # BST Strength-Weighted HP (whp) and Weighted Faints (wfaint)
    p1_whp, p1_wfaint = 0.0, 0.0
    for name, hp in p1_hp_tracker.items():
        inf = get_pokemon_info(name, local_roster, global_pokedex, default_bst)
        w = float(inf.get("bst", default_bst))
        p1_whp += w * (hp / 100.0)
        if hp <= 0.001:
            p1_wfaint += w

    p2_whp, p2_wfaint, p2_unknown = 0.0, 0.0, 0
    for name, hp in p2_hp_tracker.items():
        inf = get_pokemon_info(name, local_roster, global_pokedex, default_bst)
        w = float(inf.get("bst", default_bst))
        if inf.get("unknown"):
            p2_unknown += 1
        p2_whp += w * (hp / 100.0)
        if hp <= 0.001:
            p2_wfaint += w
    p2_whp += p2_unrevealed_count * default_bst

    features["bst_p1_whp"] = p1_whp
    features["bst_p2_whp"] = p2_whp
    features["bst_whp_diff"] = p1_whp - p2_whp
    features["bst_whp_share"] = p1_whp / max(1.0, p1_whp + p2_whp)
    features["bst_p1_wfaint"] = p1_wfaint
    features["bst_p2_wfaint"] = p2_wfaint
    features["bst_wfaint_diff"] = p2_wfaint - p1_wfaint
    features["bst_p2_unknown"] = float(p2_unknown)

    # Surviving Defensive Bulk
    p1_surviving_bulk = 0.0
    for name, hp in p1_hp_tracker.items():
        if hp > 0.001:
            info = get_pokemon_info(name, local_roster, global_pokedex, default_bst)
            bulk = info.get("base_hp", 80) * (info.get("base_def", 80) + info.get("base_spd", 80))
            p1_surviving_bulk += (hp / 100.0) * bulk

    p2_surviving_bulk = 0.0
    for name, hp in p2_hp_tracker.items():
        if hp > 0.001:
            info = get_pokemon_info(name, local_roster, global_pokedex, default_bst)
            bulk = info.get("base_hp", 80) * (info.get("base_def", 80) + info.get("base_spd", 80))
            p2_surviving_bulk += (hp / 100.0) * bulk
    p2_surviving_bulk += p2_unrevealed_count * (80.0 * 160.0)

    features["p1_surviving_bulk_score"] = p1_surviving_bulk
    features["p2_surviving_bulk_score_est"] = p2_surviving_bulk
    features["surviving_bulk_ratio"] = p1_surviving_bulk / max(1.0, p2_surviving_bulk)

    # Roster Status Counts (excluding fainted Pokémon)
    def count_status_group(hp_tr: dict, st_tr: dict, target_codes: set) -> int:
        return sum(
            1 for n, st in st_tr.items()
            if hp_tr.get(n, 0.0) > 0.001 and st and str(st).lower() in target_codes
        )

    features["p1_statused_pokemon_count"] = sum(
        1 for n, s in p1_status_tracker.items()
        if p1_hp_tracker.get(n, 0.0) > 0.001 and s not in (None, "none", "nostatus", "fnt", "")
    )
    features["p2_statused_pokemon_count"] = sum(
        1 for n, s in p2_status_tracker.items()
        if p2_hp_tracker.get(n, 0.0) > 0.001 and s not in (None, "none", "nostatus", "fnt", "")
    )
    features["p1_sleep_freeze_count"] = count_status_group(p1_hp_tracker, p1_status_tracker, {"slp", "sleep", "frz", "freeze"})
    features["p2_sleep_freeze_count"] = count_status_group(p2_hp_tracker, p2_status_tracker, {"slp", "sleep", "frz", "freeze"})
    features["p1_par_count"] = count_status_group(p1_hp_tracker, p1_status_tracker, {"par", "paralysis"})
    features["p2_par_count"] = count_status_group(p2_hp_tracker, p2_status_tracker, {"par", "paralysis"})
    features["p1_burn_poison_count"] = count_status_group(
        p1_hp_tracker, p1_status_tracker, {"brn", "burn", "psn", "poison", "tox", "toxic"}
    )
    features["p2_burn_poison_count"] = count_status_group(
        p2_hp_tracker, p2_status_tracker, {"brn", "burn", "psn", "poison", "tox", "toxic"}
    )

    # Surviving Pairwise Bench Type Coverage & Speed Control
    p1_surviving_names = [n for n, hp in p1_hp_tracker.items() if hp > 0.001]
    p2_surviving_names = [n for n, hp in p2_hp_tracker.items() if hp > 0.001]

    coverage_advantages = 0
    speed_advantages = 0
    total_pairs = 0

    for p1_n in p1_surviving_names:
        p1_inf = get_pokemon_info(p1_n, local_roster, global_pokedex, default_bst)
        p1_t = p1_inf.get("types", ["notype"])
        p1_s = p1_inf.get("base_spe", 80)

        for p2_n in p2_surviving_names:
            p2_inf = get_pokemon_info(p2_n, local_roster, global_pokedex, default_bst)
            p2_t = p2_inf.get("types", ["notype"])
            p2_s = p2_inf.get("base_spe", 80)

            total_pairs += 1
            best_mult = max([calculate_move_multiplier(t, p2_t, type_lookup) for t in p1_t])
            if best_mult > 1.0:
                coverage_advantages += 1
            if p1_s > p2_s:
                speed_advantages += 1

    features["p1_surviving_type_coverage_ratio"] = coverage_advantages / max(1, total_pairs)
    features["p1_surviving_speed_control_ratio"] = speed_advantages / max(1, total_pairs)

    # Top-40 Meta Species One-Hot Flags
    p1_team_set = set(p1_hp_tracker.keys())
    p2_seen_set = set(p2_hp_tracker.keys())
    for sp_name in top_vocab:
        features[f"p1_has_{sp_name}"] = float(sp_name in p1_team_set)
        features[f"p2_seen_{sp_name}"] = float(sp_name in p2_seen_set)

    if "player_won" in battle_json:
        features["player_won"] = int(battle_json["player_won"])

    return features


def add_regime_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies whether Player 1 is in the Leading Role (1) or Underdog Role (0)
    at Turn 30 and creates 13 specialized regime interaction features
    (combining our original regime features with the new BST-WHP, Healing, and Slope features).
    """
    df = df.copy()

    is_leading = (
        (df["fainted_diff"] > 0) |
        ((df["fainted_diff"] == 0) & (df["total_hp_diff_est"] >= 0))
    ).astype(int)

    is_underdog = 1 - is_leading
    df["is_p1_leading_turn30"] = is_leading

    # Case A Features (Active when P1 is Leading at Turn 30: Front-Runner Consolidation)
    df["leader_bulk_security"] = is_leading * df["surviving_bulk_ratio"]
    df["leader_whp_security"] = is_leading * df["bst_whp_share"]
    df["leader_item_sustain"] = is_leading * df["p1_leftovers_count"]
    df["leader_healing_sustain"] = is_leading * df["flow_healed_diff"]
    df["leader_status_suppression"] = is_leading * (
        df["p2_sleep_freeze_count"] * 2.0 + df["p2_par_count"] + df["p2_burn_poison_count"]
    )
    df["leader_opponent_boost_risk"] = is_leading * df["p2_turn30_boost_sum"]

    # Case B Features (Active when P1 is Underdog at Turn 30: Comeback Potential)
    df["underdog_momentum_surge"] = is_underdog * (
        df["hp_diff_turn15_vs_turn30"] + 0.1 * df["late_damage_diff"]
    )
    df["underdog_slope_comeback"] = is_underdog * df["hp_diff_slope"]
    df["underdog_last10_potency"] = is_underdog * df["mv_pot_last10_diff"]
    df["underdog_sweep_boosts"] = is_underdog * (
        df["p1_turn30_boost_sum"] - df["p2_turn30_boost_sum"]
    )
    df["underdog_bench_coverage_advantage"] = is_underdog * (
        df["p1_surviving_type_coverage_ratio"] * df["p1_surviving_speed_control_ratio"]
    )
    df["underdog_turn30_counter_matchup"] = is_underdog * df["turn30_matchup_mult_diff"]
    df["underdog_surviving_p1_roster"] = is_underdog * (6 - df["p1_fainted_count"])

    return df


def load_or_build_features():
    """
    Loads cached merged feature CSVs if available; otherwise extracts all ~210 merged
    features from train.jsonl and test.jsonl.
    """
    if os.path.exists(TRAIN_FEATURES_CSV) and os.path.exists(TEST_FEATURES_CSV):
        train_df = pd.read_csv(TRAIN_FEATURES_CSV)
        test_df = pd.read_csv(TEST_FEATURES_CSV)
        if "hp_diff_slope" in train_df.columns and "bst_whp_diff" in train_df.columns:
            print(f"📂 Loaded cached merged datasets ({train_df.shape[1]} columns) from {TRAIN_FEATURES_CSV}")
            return train_df, test_df

    train_jsonl_path = resolve_data_path("train.jsonl")
    test_jsonl_path = resolve_data_path("test.jsonl")
    type_chart_path = resolve_data_path("type_chart_gen2.csv")

    print("⚙️ Extracting merged Pokémon feature matrix (Trajectory + BST-WHP + Flow + Species + Regime)...")
    type_lookup = load_type_lookup(type_chart_path)
    pokedex, top_vocab, default_bst = build_internal_pokedex_and_vocab(train_jsonl_path, k_species=TOP_K_SPECIES)

    train_rows = []
    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            train_rows.append(
                extract_battle_features(json.loads(line), type_lookup, pokedex, top_vocab, default_bst)
            )
    train_df = add_regime_specific_features(pd.DataFrame(train_rows))

    test_rows = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            test_rows.append(
                extract_battle_features(json.loads(line), type_lookup, pokedex, top_vocab, default_bst)
            )
    test_df = add_regime_specific_features(pd.DataFrame(test_rows))

    train_df.to_csv(TRAIN_FEATURES_CSV, index=False)
    test_df.to_csv(TEST_FEATURES_CSV, index=False)
    print(f"✅ Saved merged datasets ({train_df.shape[1]} columns) to {TRAIN_FEATURES_CSV} and {TEST_FEATURES_CSV}\n")
    return train_df, test_df


def analyze_and_plot_regime_features(train_df: pd.DataFrame, save_path: str = REGIME_PLOT_PNG):
    """
    Splits train_df into Case A (P1 Leading at Turn 30) and Case B (P1 Underdog at Turn 30)
    and plots the top 15 features driving wins in each case.
    """
    feature_cols = [
        c for c in train_df.columns
        if c not in ("battle_id", "player_won", "fold", "is_p1_leading_turn30")
    ]

    leading_df = train_df[train_df["is_p1_leading_turn30"] == 1]
    underdog_df = train_df[train_df["is_p1_leading_turn30"] == 0]

    print("=" * 75)
    print(f"🔍 TURN-30 REGIME BREAKDOWN ({len(feature_cols)} MERGED FEATURES)")
    print("=" * 75)
    print(
        f"Case A (P1 Leading at Turn 30) : {len(leading_df)} battles | "
        f"P1 Win Rate = {leading_df['player_won'].mean():.2%}"
    )
    print(
        f"Case B (P1 Underdog at Turn 30): {len(underdog_df)} battles | "
        f"P1 Comeback Win Rate = {underdog_df['player_won'].mean():.2%}"
    )

    rf_lead = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1)
    rf_lead.fit(leading_df[feature_cols].fillna(0), leading_df["player_won"].astype(int))
    imp_lead = pd.Series(rf_lead.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)

    rf_under = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1)
    rf_under.fit(underdog_df[feature_cols].fillna(0), underdog_df["player_won"].astype(int))
    imp_under = pd.Series(rf_under.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].barh(imp_lead.index, imp_lead.values, color="#2b8cbe", edgecolor="black")
    axes[0].set_title(
        f"Case A: Top 15 Merged Features When P1 Leads at Turn 30\n"
        f"({len(leading_df)} Battles | Win Rate: {leading_df['player_won'].mean():.1%})"
    )
    axes[0].set_xlabel("Relative Importance (Protecting the Lead)")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(imp_under.index, imp_under.values, color="#e34a33", edgecolor="black")
    axes[1].set_title(
        f"Case B: Top 15 Merged Features When P1 is Underdog at Turn 30\n"
        f"({len(underdog_df)} Battles | Comeback Win Rate: {underdog_df['player_won'].mean():.1%})"
    )
    axes[1].set_xlabel("Relative Importance (Staging a Comeback)")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved Regime Feature Comparison plot to: {save_path}\n")