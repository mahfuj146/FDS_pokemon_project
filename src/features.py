"""Feature builders.

Every `*_features(battles)` function takes a list of battle dictionaries and returns a DataFrame
indexed by `battle_id`, with a fixed set of columns. All functions are *stateless*: they look at one
battle at a time and fit nothing, so applying them to the test set is safe.

Feature groups are registered in FEATURE_GROUPS, so experiments can add one group at a time:
    X = build_features(train, groups=("summary", "team"))
"""
import numpy as np
import pandas as pd

N_TURNS = 30
STAT_COLS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]

SUMMARY_COLS = [
    "p1_seen", "p1_fainted", "p1_total_hp", "p1_null_moves", "p1_name_changes",
    "p2_seen", "p2_fainted", "p2_total_hp", "p2_null_moves", "p2_name_changes",
    "hp_diff", "faint_diff",
]
TEAM_COLS = [f"p1_team_mean_{c}" for c in STAT_COLS]
CHECKPOINTS = (5, 10, 15, 20, 25)
CHECKPOINT_COLS = [f"hp_diff_t{c}" for c in CHECKPOINTS] + [
    "hp_diff_delta_mid", "hp_diff_delta_last", "faint_diff_t10", "faint_diff_t20"]
TRAJ_SUMMARY_COLS = ["hp_diff_mean", "hp_diff_min", "hp_diff_max", "hp_diff_std", "hp_diff_slope"]


# ----- end-of-timeline summaries -----
def battle_summary(b):
    """Whole-team summary at the end of the 30 turns.

    The timeline shows only each side's *active* Pokemon, so the team is reconstructed by keeping the
    last known HP of every Pokemon seen. Pokemon never seen count as full health (1.0). The listed lead
    of each side is counted as seen from the start, even if it switched out on turn 1.
    """
    f = {"battle_id": b["battle_id"]}
    leads = {"p1": b["p1_team_details"][0]["name"], "p2": b["p2_lead_details"]["name"]}
    for side in ("p1", "p2"):
        last_hp = {leads[side]: 1.0}
        prev, name_changes, null_moves = None, 0, 0
        for t in b["battle_timeline"]:
            s = t[f"{side}_pokemon_state"]
            last_hp[s["name"]] = s["hp_pct"]
            name_changes += prev is not None and s["name"] != prev
            null_moves += t[f"{side}_move_details"] is None
            prev = s["name"]
        f[f"{side}_seen"] = len(last_hp)
        f[f"{side}_fainted"] = sum(v == 0 for v in last_hp.values())
        f[f"{side}_total_hp"] = sum(last_hp.values()) + 6 - len(last_hp)
        f[f"{side}_null_moves"] = null_moves
        f[f"{side}_name_changes"] = int(name_changes)
    f["hp_diff"] = f["p1_total_hp"] - f["p2_total_hp"]
    f["faint_diff"] = f["p2_fainted"] - f["p1_fainted"]  # positive = good for player 1
    return f

def summary_features(battles):
    df = pd.DataFrame([battle_summary(b) for b in battles]).set_index("battle_id")
    return df[SUMMARY_COLS]

# ----- static team strength ----------
def team_features(battles):
    """Mean base stats of player 1's six Pokemon (independent of the timeline)."""
    rows = []
    for b in battles:
        row = {"battle_id": b["battle_id"]}
        for c in STAT_COLS:
            row[f"p1_team_mean_{c}"] = float(np.mean([p[c] for p in b["p1_team_details"]]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("battle_id")[TEAM_COLS]

# ----- trajectory --------------------
def hp_faint_series(b):
    """After each turn: (p1 - p2) total last-known HP, and faint difference (p2 fainted - p1 fainted)."""
    last = {"p1": {b["p1_team_details"][0]["name"]: 1.0},
            "p2": {b["p2_lead_details"]["name"]: 1.0}}
    hp_series, faint_series = [], []
    for t in b["battle_timeline"]:
        for side in ("p1", "p2"):
            s = t[f"{side}_pokemon_state"]
            last[side][s["name"]] = s["hp_pct"]
        total = {side: sum(h.values()) + 6 - len(h) for side, h in last.items()}
        fainted = {side: sum(v == 0 for v in h.values()) for side, h in last.items()}
        hp_series.append(total["p1"] - total["p2"])
        faint_series.append(fainted["p2"] - fainted["p1"])
    return hp_series, faint_series

def hp_faint_matrices(battles):
    """(battle_ids, hp matrix, faint matrix); matrices have shape (n_battles, 30)."""
    series = [hp_faint_series(b) for b in battles]
    ids = np.array([b["battle_id"] for b in battles])
    hp_mat = np.array([s[0] for s in series])
    faint_mat = np.array([s[1] for s in series])
    assert hp_mat.shape[1] == N_TURNS, "expected 30 turns per battle"
    return ids, hp_mat, faint_mat

def trajectory_checkpoints(battles):
    """HP / faint difference at fixed turns and the changes between them (turn 30 is in the summary)."""
    ids, hp, faint = hp_faint_matrices(battles)
    f = pd.DataFrame(index=pd.Index(ids, name="battle_id"))
    for c in CHECKPOINTS:
        f[f"hp_diff_t{c}"] = hp[:, c - 1]
    f["hp_diff_delta_mid"] = hp[:, 19] - hp[:, 9]      # turns 10 -> 20
    f["hp_diff_delta_last"] = hp[:, 29] - hp[:, 19]    # turns 20 -> 30
    f["faint_diff_t10"] = faint[:, 9]
    f["faint_diff_t20"] = faint[:, 19]
    return f[CHECKPOINT_COLS]

def trajectory_summary(battles):
    """Mean, min, max, spread and slope of the HP difference over all 30 turns."""
    ids, hp, _ = hp_faint_matrices(battles)
    f = pd.DataFrame(index=pd.Index(ids, name="battle_id"))
    f["hp_diff_mean"] = hp.mean(axis=1)
    f["hp_diff_min"] = hp.min(axis=1)
    f["hp_diff_max"] = hp.max(axis=1)
    f["hp_diff_std"] = hp.std(axis=1)
    f["hp_diff_slope"] = np.polyfit(np.arange(hp.shape[1]), hp.T, 1)[0]
    return f[TRAJ_SUMMARY_COLS]

# ----- registry ----------------------
FEATURE_GROUPS = {
    "summary": summary_features,
    "team": team_features,
    "traj_checkpoints": trajectory_checkpoints,
    "traj_summary": trajectory_summary,
}

def build_features(battles, groups=("summary",)):
    """Concatenate the requested feature groups (aligned on battle_id)."""
    return pd.concat([FEATURE_GROUPS[g](battles) for g in groups], axis=1)
