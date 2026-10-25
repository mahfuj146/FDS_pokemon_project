# -*- coding: utf-8 -*-
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

from features import load_or_build_features, analyze_and_plot_regime_features, REGIME_PLOT_PNG
from models import (
    get_lr_builder,
    get_svm_builder,
    get_rf_builder,
    run_kfold_experiment,
    save_submission
)

warnings.filterwarnings("ignore")

EVALUATION_PLOT_PNG = "simple_ensemble_evaluation.png"


def main():
    # 1. Load or extract all features (including Turn-30 Regime features)
    train_df, test_df = load_or_build_features()
    y_true = train_df["player_won"].astype(int).values

    # 2. Run Turn-30 Regime Breakdown & Plot Top 15 Features for Case A vs. Case B
    analyze_and_plot_regime_features(train_df, save_path=REGIME_PLOT_PNG)

    # 3. Candidate Model 1: Logistic Regression
    print("=== Model 1: Logistic Regression ===")
    lr_oof, lr_test_probs, lr_acc = run_kfold_experiment(train_df, test_df, get_lr_builder())
    save_submission(test_df, lr_test_probs, "submission_logistic_regression.csv")

    # 4. Candidate Model 2: Standard Support Vector Machine (SVM)
    print("\n=== Model 2: Support Vector Machine (SVM) ===")
    svm_oof, svm_test_probs, svm_acc = run_kfold_experiment(train_df, test_df, get_svm_builder())
    save_submission(test_df, svm_test_probs, "submission_svm.csv")

    # 5. Candidate Model 3: Standard Random Forest
    print("\n=== Model 3: Standard Random Forest ===")
    rf_oof, rf_test_probs, rf_acc = run_kfold_experiment(train_df, test_df, get_rf_builder())
    save_submission(test_df, rf_test_probs, "submission_random_forest.csv")

    # 6. Ensemble (Soft-Voting Blend of Logistic Regression + SVM + Random Forest)
    print("\n=== Model 4: Ensemble (Logistic Regression + SVM + Random Forest) ===")
    w_lr, w_svm, w_rf = 0.25, 0.35, 0.40

    ensemble_oof = (w_lr * lr_oof) + (w_svm * svm_oof) + (w_rf * rf_oof)
    ensemble_preds = (ensemble_oof >= 0.5).astype(int)
    ensemble_acc = accuracy_score(y_true, ensemble_preds)

    lead_mask = (train_df["is_p1_leading_turn30"].values == 1)
    ens_lead_acc = accuracy_score(y_true[lead_mask], ensemble_preds[lead_mask])
    ens_under_acc = accuracy_score(y_true[~lead_mask], ensemble_preds[~lead_mask])

    print(f"--> Ensemble Out-Of-Fold Validation Accuracy: {ensemble_acc:.4f}")
    print(f"    [Case A Leading Acc: {ens_lead_acc:.4f} | Case B Underdog Acc: {ens_under_acc:.4f}]\n")
    print("Classification Report (Ensemble OOF):")
    print(classification_report(y_true, ensemble_preds, target_names=["P1 Loses (0)", "P1 Wins (1)"]))

    ensemble_test_probs = (w_lr * lr_test_probs) + (w_svm * svm_test_probs) + (w_rf * rf_test_probs)
    save_submission(test_df, ensemble_test_probs, "submission_ensemble.csv")

    # 7. Summary Evaluation Plots for Methodology Report
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    model_names = ["Logistic Reg", "SVM", "Random Forest", "Ensemble"]
    accuracies = [lr_acc * 100, svm_acc * 100, rf_acc * 100, ensemble_acc * 100]
    bars = axes[0].bar(
        model_names, accuracies,
        color=["#74a9cf", "#4b729f", "#2b8cbe", "#5e3c99"],
        edgecolor="black"
    )
    axes[0].set_ylim(max(50, min(accuracies) - 3), max(accuracies) + 2)
    axes[0].set_ylabel("5-Fold OOF Accuracy (%)")
    axes[0].set_title("Model Comparison (Validation Accuracy)")
    axes[0].grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.2f}%", ha="center", fontsize=9)

    cm = confusion_matrix(y_true, ensemble_preds)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=axes[1],
        xticklabels=["P1 Loses (0)", "P1 Wins (1)"],
        yticklabels=["P1 Loses (0)", "P1 Wins (1)"]
    )
    axes[1].set_title(f"Ensemble Confusion Matrix (Acc: {ensemble_acc:.2%})")
    axes[1].set_xlabel("Predicted Outcome")
    axes[1].set_ylabel("Actual Outcome")

    for name, probs, col in [
        ("Logistic Reg", lr_oof, "#74a9cf"),
        ("SVM", svm_oof, "#4b729f"),
        ("Random Forest", rf_oof, "#2b8cbe"),
        ("Ensemble", ensemble_oof, "#5e3c99")
    ]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        axes[2].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})", color=col)
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (0.500)")
    axes[2].set_title("Out-of-Fold ROC Curves")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].legend(loc="lower right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(EVALUATION_PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\n📊 Saved evaluation plots to {REGIME_PLOT_PNG} and {EVALUATION_PLOT_PNG}")


if __name__ == "__main__":
    main()

