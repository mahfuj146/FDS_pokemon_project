"""Model definitions used in the experiments."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def baseline_models(seed=42):
    """A linear and a tree-based model."""
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "hist_gb": HistGradientBoostingClassifier(random_state=seed),
    }
