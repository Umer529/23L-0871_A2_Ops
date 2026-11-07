"""Task 3.5 - compare the VisionGuard runs straight from the MLflow server.

  python src/compare_runs.py
Output: reports/experiment_comparison.csv, reports/figures/experiment_comparison.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

import tracking

COLS = {
    "tags.mlflow.runName": "run", "params.model": "backbone", "params.img_size": "img",
    "params.ft_lr": "ft_lr", "params.unfreeze": "unfreeze", "params.no_aug": "no_aug",
    "tags.tracking_mode": "tracking",
    "metrics.best_val_loss": "val_loss", "metrics.best_val_auc": "val_auc",
    "metrics.best_val_accuracy": "val_acc", "metrics.test_roc_auc": "test_auc",
    "metrics.test_f1": "test_f1", "metrics.test_precision": "test_precision",
    "metrics.test_recall": "test_recall", "metrics.test_fpr_real_called_ai": "test_fpr",
}


def main():
    tracking.setup("visionguard")
    df = mlflow.search_runs(experiment_names=["visionguard"], order_by=["metrics.best_val_auc DESC"])
    df = df[[c for c in COLS if c in df.columns]].rename(columns=COLS)
    df.to_csv("reports/experiment_comparison.csv", index=False)
    print(df.round(3).to_string(index=False))

    d = df.sort_values("run")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = range(len(d))
    for off, col, label in [(-0.2, "val_auc", "validation AUC (selection metric)"),
                            (0.2, "test_auc", "test AUC (reported once, not used to choose)")]:
        bars = ax.bar([i + off for i in x], d[col], width=0.4, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(list(x), d["run"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("VisionGuard runs on DagsHub MLflow - champion = highest validation AUC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("reports/figures/experiment_comparison.png", dpi=110)
    print("saved reports/figures/experiment_comparison.png")


if __name__ == "__main__":
    main()
