"""Task 1.2 - Dataset audit.

Walks data/raw/<class>/*, records per-image metadata and a SHA-256 content hash,
flags unreadable files and exact duplicates, and prints summary tables.

Usage:  python src/audit.py
Output: reports/audit.csv
"""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

RAW_DIR = Path("data/raw")
OUT_CSV = Path("reports/audit.csv")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: Path, label: str) -> dict:
    row = {
        "path": path.as_posix(),
        "class": label,
        "extension": path.suffix.lower() or "(none)",
        "file_size_kb": round(path.stat().st_size / 1024, 2),
        "width": None, "height": None, "mode": None, "format": None,
        "sha256": sha256_of(path),
        "readable": True, "error": "",
    }
    try:
        # verify() checks file integrity but leaves the image unusable,
        # so reopen it to read the size/mode and force a full decode.
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            row.update(width=img.width, height=img.height,
                       mode=img.mode, format=img.format)
    except Exception as e:  # any decode failure = corrupt/unreadable
        row.update(readable=False, error=f"{type(e).__name__}: {e}")
    return row


def main():
    rows = [inspect(p, class_dir.name)
            for class_dir in sorted(RAW_DIR.iterdir()) if class_dir.is_dir()
            for p in sorted(class_dir.rglob("*")) if p.is_file()]
    df = pd.DataFrame(rows)

    # exact duplicates: same bytes -> same hash. keep="first" marks the extra copies.
    df["is_duplicate"] = df.duplicated("sha256", keep="first")
    cross = df.groupby("sha256")["class"].nunique()
    df["dup_across_classes"] = df["sha256"].map(cross) > 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 140)
    print("\n=== Images per class ===")
    print(df["class"].value_counts().to_string())

    ok = df[df["readable"]]
    print("\n=== Width / height / file size (KB) per class: min, median, max ===")
    print(ok.groupby("class")[["width", "height", "file_size_kb"]]
            .agg(["min", "median", "max"]).round(1).to_string())

    print("\n=== Colour mode per class ===")
    print(pd.crosstab(ok["class"], ok["mode"]).to_string())
    print("\n=== Format per class ===")
    print(pd.crosstab(ok["class"], ok["format"]).to_string())

    bad = df[~df["readable"]]
    print(f"\nCorrupt / unreadable files: {len(bad)}")
    for _, r in bad.iterrows():
        print(f"  {r['path']}  ->  {r['error']}")

    print(f"Exact duplicate files (extra copies): {int(df['is_duplicate'].sum())}")
    print(f"Hashes appearing in BOTH classes: {int((cross > 1).sum())}")
    print(f"\nFull table saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
