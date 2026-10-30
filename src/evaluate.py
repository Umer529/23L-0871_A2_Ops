"""Task 2.5 / 2.6 - threshold on validation, evaluate on test ONCE, error analysis.

  python src/evaluate.py --runs cnn_224 mobilenetv2_224 --error-run mobilenetv2_224

Positive class = AI-generated (label 1). The costlier mistake for VisionGuard is a
false positive: calling a real photo fake accuses a real person/photographer of
faking it. So the threshold is the one that keeps val FPR <= MAX_FPR while catching
as many AI images as possible.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from tensorflow import keras

from data import make_dataset, read_split

MAX_FPR = 0.10
FIG_DIR = Path("reports/figures")


def predict(model, split, img_size):
    ds = make_dataset(split, img_size)
    y = np.concatenate([yb.numpy() for _, yb in ds]).astype(int)
    p = model.predict(ds, verbose=0).ravel()
    return y, p


def rates(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return fp / (fp + tn), tp / (tp + fn), f1_score(y, pred, zero_division=0)


def choose_threshold(y, p):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    table = pd.DataFrame([(t, *rates(y, p, t)) for t in grid], columns=["threshold", "fpr", "tpr", "f1"])
    ok = table[table["fpr"] <= MAX_FPR]
    if len(ok):
        # most AI images caught under the FPR cap; ties -> higher (more conservative) threshold
        row = ok.sort_values(["tpr", "threshold"], ascending=[False, False]).iloc[0]
        rule = f"max TPR with val FPR <= {MAX_FPR:.0%}"
    else:
        row = table.sort_values(["fpr", "tpr"], ascending=[True, False]).iloc[0]
        rule = "lowest val FPR (cap unreachable)"
    youden = table.iloc[(table["tpr"] - table["fpr"]).argmax()]
    compare = pd.DataFrame([table[table["threshold"] == 0.5].iloc[0], youden, row],
                           index=["default 0.5", "Youden J (max TPR-FPR)", f"chosen: {rule}"])
    return float(row["threshold"]), rule, compare


def test_metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {"threshold": t,
            "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
            "fpr_real_called_ai": rates(y, p, t)[0]}


def save_eval_figures(run, y, p, t, out_dir):
    """Per-run confusion matrix + ROC (used by train.py so every MLflow run carries them)."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay(confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]),
                           display_labels=["Real", "AI"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{run} - test (t={t:.2f})")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter(*rates(y, p, t)[:2], s=40, color="red", zorder=3, label=f"threshold {t:.2f}")
    ax.set(xlabel="FPR (real called AI)", ylabel="TPR (AI caught)", title=f"{run} - test ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "roc_curve.png", dpi=110)
    plt.close(fig)


def misclassified_grid(run, y, p, t, out):
    paths, _ = read_split("test")
    audit = pd.read_csv("reports/audit.csv").set_index("path")
    wrong = np.where((p >= t).astype(int) != y)[0]
    # most confident mistakes first: these are the interesting ones
    wrong = wrong[np.argsort(-np.abs(p[wrong] - t))][:10]
    fig, axes = plt.subplots(2, 5, figsize=(17, 8))
    rows = []
    for ax, i in zip(axes.flat, wrong):
        a = audit.loc[paths[i]]
        with Image.open(paths[i]) as img:
            ax.imshow(img.convert("RGB"))
        true = "AI" if y[i] else "Real"
        ax.set_title(f"true={true}  p(AI)={p[i]:.2f}\n{a['width']}x{a['height']} {a['format']} {a['file_size_kb']:.0f}KB",
                     fontsize=9, color="darkred")
        ax.axis("off")
        rows.append({"path": paths[i], "true": true, "p_ai": round(float(p[i]), 3),
                     "width": a["width"], "height": a["height"], "format": a["format"]})
    for ax in axes.flat[len(wrong):]:
        ax.axis("off")
    fig.suptitle(f"{run}: 10 most confident test mistakes (threshold={t:.2f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["cnn_224", "mobilenetv2_224"])
    ap.add_argument("--error-run", default="mobilenetv2_224")
    a = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results, roc_fig = {}, plt.figure(figsize=(6, 5.5))
    for run in a.runs:
        img_size = json.loads((Path("reports") / run / "summary.json").read_text())["img_size"]
        model = keras.models.load_model(f"models/{run}.keras")

        yv, pv = predict(model, "val", img_size)
        t, rule, compare = choose_threshold(yv, pv)
        print(f"\n=== {run}: threshold candidates on VALIDATION ===\n{compare.round(3).to_string()}")

        yt, pt = predict(model, "test", img_size)   # the one and only look at the test set
        m = test_metrics(yt, pt, t)
        m["threshold_rule"] = rule
        results[run] = m
        out_dir = Path("reports") / run
        compare.round(4).to_csv(out_dir / "threshold_choice.csv")
        (out_dir / "test_metrics.json").write_text(json.dumps(m, indent=2))
        np.save(out_dir / "test_probs.npy", pt)

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ConfusionMatrixDisplay(confusion_matrix(yt, (pt >= t).astype(int)),
                               display_labels=["Real", "AI"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{run} - test (t={t:.2f})")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"confusion_{run}.png", dpi=110)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(yt, pt)
        plt.figure(roc_fig.number)
        plt.plot(fpr, tpr, label=f"{run} (AUC={m['roc_auc']:.3f})")
        plt.scatter([m["fpr_real_called_ai"]], [m["recall"]], s=40, zorder=3)

        if run == a.error_run:
            errs = misclassified_grid(run, yt, pt, t, FIG_DIR / f"misclassified_{run}.png")
            errs.to_csv(out_dir / "misclassified.csv", index=False)

    plt.figure(roc_fig.number)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    plt.xlabel("False positive rate (real called AI)")
    plt.ylabel("True positive rate (AI caught)")
    plt.title("ROC on test set (dots = chosen threshold)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_test.png", dpi=110)

    table = pd.DataFrame(results).T.drop(columns="threshold_rule")
    table.round(4).to_csv("reports/test_metrics_comparison.csv")
    print("\n=== TEST metrics (evaluated once) ===\n" + table.astype(float).round(3).to_string())


if __name__ == "__main__":
    main()
