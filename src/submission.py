"""Writing a submission file aligned to sample_submission.csv."""
import pandas as pd

def write_submission(battle_ids, predictions, sample_submission_path, out_path):
    """Map predictions to battle_ids, check completeness, and write the CSV (0/1 per battle)."""
    sub = pd.read_csv(sample_submission_path)
    assert list(sub.columns) == ["battle_id", "player_won"], "unexpected sample_submission columns"
    sub["player_won"] = sub["battle_id"].map(pd.Series(list(predictions), index=list(battle_ids)))
    assert not sub["player_won"].isna().any(), "some battle_ids have no prediction"
    sub["player_won"] = sub["player_won"].astype(int)
    sub.to_csv(out_path, index=False)
    return sub
