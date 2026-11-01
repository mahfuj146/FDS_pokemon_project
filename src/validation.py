"""Cross-validation and logging experiments.

Every feature set is scored with repeated stratified k-fold (default 3 x 5 = 15 splits). The first
repeat (seed 42) uses the same folds as StratifiedKFold(5, shuffle=True, random_state=42), so results
are comparable with scripts that use a single 5-fold split. Per-split scores are kept so that two
feature sets can be compared on *identical* splits (a paired comparison).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models import baseline_models

class Experiments:
    def __init__(self, seeds=(42, 43, 44), n_folds=5, model_seed=42, models=None):
        self.seeds = tuple(seeds)
        self.n_folds = n_folds
        self.models = models if models is not None else baseline_models(model_seed)
        self.log = []
        self.scores = {}

    def run(self, name, X, y):
        """Cross-validate every model on X, log the result and keep the per-split scores."""
        rows = []
        for model_name, model in self.models.items():
            scores = []
            for seed in self.seeds:
                cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
                scores.extend(cross_val_score(model, X, y, cv=cv, scoring="accuracy"))
            scores = np.array(scores)
            self.scores[(name, model_name)] = scores
            rows.append({
                "features": name, "n_features": X.shape[1], "model": model_name,
                "cv_acc_mean": round(scores.mean(), 4),
                "cv_acc_std": round(scores.std(), 4),
                "acc_first_5_folds": round(scores[:self.n_folds].mean(), 4),
            })
        self.log.extend(rows)
        return pd.DataFrame(rows)

    def compare(self, new, old):
        """Paired comparison on identical splits."""
        rows = []
        for model_name in self.models:
            d = self.scores[(new, model_name)] - self.scores[(old, model_name)]
            rows.append({
                "model": model_name,
                "mean_gain": round(d.mean(), 4),
                "splits_improved": f"{int((d > 0).sum())}/{len(d)}",
                "paired_std": round(d.std(), 4),
            })
        return pd.DataFrame(rows)

    def table(self):
        return pd.DataFrame(self.log)

    def save(self, path):
        self.table().to_csv(path, index=False)
