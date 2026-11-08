# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:04:40 2026

@author: mashi
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SEED = 42


def get_lr_builder():
    return lambda seed: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1500, C=0.25, random_state=seed))
    ])


def get_svm_builder():
    return lambda seed: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, cache_size=1000, random_state=seed))
    ])


def get_xgb_builder():
    return lambda seed: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=450,
            learning_rate=0.035,
            max_depth=4,
            min_child_weight=6,
            subsample=0.80,
            colsample_bytree=0.70,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1
        ))
    ])


def run_kfold_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_fn,
    n_splits: int = 5,
    random_state: int = SEED
):
    """
    Runs Stratified K-Fold cross-validation on train_df, tracks overall and
    regime-specific (Leading vs. Underdog) validation Accuracy, and averages
    test set probabilities across folds.
    """
    feature_cols = [
        c for c in train_df.columns
        if c not in ("battle_id", "player_won", "fold")
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df["player_won"].astype(int).values
    X_test = test_df[feature_cols].fillna(0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(len(train_df))
    test_pred_probs = np.zeros(len(test_df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = model_fn(random_state)
        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        test_pred_probs += model.predict_proba(X_test)[:, 1] / n_splits

        val_binary = (val_probs >= 0.5).astype(int)
        fold_acc = accuracy_score(y_val, val_binary)
        print(f"  Fold {fold} Accuracy: {fold_acc:.4f}")

    overall_binaries = (oof_preds >= 0.5).astype(int)
    overall_oof_acc = accuracy_score(y, overall_binaries)

    lead_mask = (train_df["is_p1_leading_turn30"].values == 1)
    lead_acc = accuracy_score(y[lead_mask], overall_binaries[lead_mask])
    under_acc = accuracy_score(y[~lead_mask], overall_binaries[~lead_mask])

    print(f"--> Overall Out-Of-Fold Validation Accuracy: {overall_oof_acc:.4f}")
    print(f"    [Case A Leading Acc: {lead_acc:.4f} | Case B Underdog Acc: {under_acc:.4f}]\n")

    return oof_preds, test_pred_probs, overall_oof_acc


def save_submission(test_df: pd.DataFrame, pred_probs: np.ndarray, filename: str):
    """
    Formats binary predictions (1 -> Player 1 wins, 0 -> Player 1 loses)
    paired with battle_id to match sample_submission.csv.
    """
    submission = pd.DataFrame({
        "battle_id": test_df["battle_id"].astype(int),
        "player_won": (pred_probs >= 0.5).astype(int)
    })
    submission.to_csv(filename, index=False)
    print(f"✅ Saved submission to {filename} (Shape: {submission.shape})")