"""Make train / val / test split lists from the audit table.

Drops unreadable files and exact duplicates, then does a stratified
70 / 15 / 15 split with a fixed seed so every run gets the same lists.

Usage:  python src/split.py
Output: data/splits/{train,val,test}.csv  (columns: path, label)
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

AUDIT_CSV = Path("reports/audit.csv")
SPLIT_DIR = Path("data/splits")
SEED = 42
# AiArtData -> 1 (AI-generated, the "positive" we want to detect), RealArt -> 0
LABELS = {"AiArtData": 1, "RealArt": 0}


def main():
    df = pd.read_csv(AUDIT_CSV)
    df = df[df["readable"] & ~df["is_duplicate"]]
    df = pd.DataFrame({"path": df["path"], "label": df["class"].map(LABELS)})

    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values("path").to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        counts = part["label"].value_counts().to_dict()
        print(f"{name:5s}: {len(part):4d} images  (AI={counts.get(1, 0)}, real={counts.get(0, 0)})")


if __name__ == "__main__":
    main()
