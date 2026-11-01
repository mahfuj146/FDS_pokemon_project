"""Loading the data.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "player_won"
N_TURNS = 30  # every battle is cut at 30 turns

def find_data_dir():
    """Return the folder that contains train.jsonl."""
    root = Path("/kaggle/input")
    if root.exists():
        hits = sorted(root.rglob("train.jsonl"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(
        "train.jsonl not found. Add the competition data to the notebook."
    )

def load_jsonl(path):
    """Read a .jsonl file: one JSON object (one battle) per line."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def load_train(data_dir=None):
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    return load_jsonl(data_dir / "train.jsonl")

def load_test(data_dir=None):
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    return load_jsonl(data_dir / "test.jsonl")

def load_battles(data_dir=None):
    """Return (train, test) as lists of battle dictionaries."""
    return load_train(data_dir), load_test(data_dir)

def load_type_chart(data_dir=None):
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    return pd.read_csv(data_dir / "type_chart_gen2.csv")

def load_sample_submission(data_dir=None):
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    return pd.read_csv(data_dir / "sample_submission.csv")

def get_target(battles):
    """Target as a 0/1 array, in the order of `battles` (train only)."""
    return np.array([int(b[TARGET]) for b in battles])
