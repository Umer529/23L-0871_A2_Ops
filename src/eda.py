"""Task 1.3 - Visual EDA.

Reads reports/audit.csv (from src/audit.py) and draws:
  - a grid of random sample images per class
  - histograms of width, height and file size per class

Usage:  python src/eda.py
Output: reports/figures/sample_grid.png, reports/figures/histograms.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from PIL import Image

AUDIT_CSV = Path("reports/audit.csv")
FIG_DIR = Path("reports/figures")
SEED = 42
N_PER_CLASS = 8


def sample_grid(df: pd.DataFrame):
    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(len(classes), N_PER_CLASS,
                             figsize=(2.2 * N_PER_CLASS, 2.6 * len(classes)))
    for r, label in enumerate(classes):
        picks = df[df["class"] == label].sample(N_PER_CLASS, random_state=SEED)
        for c, (_, row) in enumerate(picks.iterrows()):
            ax = axes[r, c]
            with Image.open(row["path"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{row['width']}x{row['height']}", fontsize=8)
            ax.axis("off")
        axes[r, 0].text(-0.1, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, ha="right", va="center", fontsize=12, weight="bold")
    fig.suptitle(f"Random samples per class (seed={SEED})", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "sample_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


def histograms(df: pd.DataFrame):
    cols = [("width", "Width (px)"), ("height", "Height (px)"), ("file_size_kb", "File size (KB)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (col, xlabel) in zip(axes, cols):
        # log-spaced bins: sizes span two orders of magnitude, linear bins hide the bulk
        lo, hi = df[col].min(), df[col].max()
        bins = np.logspace(np.log10(lo), np.log10(hi), 40)
        for label, grp in df.groupby("class"):
            ax.hist(grp[col], bins=bins, alpha=0.55, label=f"{label} (n={len(grp)})")
            ax.axvline(grp[col].median(), linestyle="--", linewidth=1,
                       color=ax.patches[-1].get_facecolor()[:3])
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(xlabel + " - log scale")
        ax.set_ylabel("Images")
        ax.legend(fontsize=8)
    fig.suptitle("Width, height and file size per class (dashed line = median)", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "histograms.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


def main():
    df = pd.read_csv(AUDIT_CSV)
    df = df[df["readable"] & ~df["is_duplicate"]]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sample_grid(df)
    histograms(df)


if __name__ == "__main__":
    main()
