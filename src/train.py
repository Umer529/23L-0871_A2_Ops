"""Task 2.2 - 2.4 - train the baseline CNN or a transfer-learning model.

  python src/train.py --model cnn
  python src/train.py --model mobilenetv2            # stage 1 frozen, stage 2 last block unfrozen

Writes models/<run>.keras (best val_loss) and reports/<run>/ (curves, summary,
layer table, trainable layers, history).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from data import SEED, make_dataset
from models import build_baseline_cnn, build_transfer_model, unfreeze_last

def metrics():
    return [keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn", choices=["cnn", "mobilenetv2", "efficientnetb0"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ft-lr", type=float, default=1e-5, help="stage-2 learning rate (transfer only)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--ft-epochs", type=int, default=15)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--unfreeze", type=int, default=None,
                   help="unfreeze the last N backbone layers instead of the last block")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def callbacks(ckpt_path, patience=5, best_so_far=None):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1,
                                        initial_value_threshold=best_so_far),  # stage 2 must beat stage 1
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]


def plot_curves(hist, title, out, epoch_offset=0):
    epochs = np.arange(1, len(hist["loss"]) + 1) + epoch_offset
    best = epochs[int(np.argmin(hist["val_loss"]))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, hist[key], "o-", ms=3, label=f"train {key}")
        ax.plot(epochs, hist[f"val_{key}"], "o-", ms=3, label=f"val {key}")
        ax.axvline(best, color="red", linestyle="--", label=f"best epoch = {best}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return int(best)


def save_model_tables(model, out_dir):
    lines = []
    model.summary(print_fn=lambda s, **_: lines.append(s), expand_nested=False)
    (out_dir / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    rows = [{"layer": l.name, "type": type(l).__name__,
             "output_shape": str(tuple(l.output.shape)), "params": l.count_params(),
             "trainable": l.trainable} for l in model.layers]
    pd.DataFrame(rows).to_csv(out_dir / "layer_table.csv", index=False)
    print("\n".join(lines))


def main():
    a = parse_args()
    keras.utils.set_random_seed(SEED)  # python, numpy and tensorflow seeds in one call
    run = a.run_name or f"{a.model}_{a.img_size}"
    out_dir = Path("reports") / run
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    ckpt = f"models/{run}.keras"

    train_ds = make_dataset("train", a.img_size, a.batch_size, augment=not a.no_aug, shuffle=True)
    val_ds = make_dataset("val", a.img_size, a.batch_size)
    loss = keras.losses.BinaryCrossentropy()
    summary = {"run": run, **vars(a)}

    if a.model == "cnn":
        model = build_baseline_cnn(a.img_size, a.dropout if a.dropout is not None else 0.4)
        model.compile(optimizer=keras.optimizers.Adam(a.lr), loss=loss, metrics=metrics())
        save_model_tables(model, out_dir)
        h = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=callbacks(ckpt, patience=7))
        hist = h.history
        summary["best_epoch"] = plot_curves(hist, f"{run}: baseline CNN", out_dir / "curves.png")
    else:
        model, base = build_transfer_model(a.model, a.img_size, a.dropout if a.dropout is not None else 0.3)

        # stage 1: frozen backbone, train only the new head
        model.compile(optimizer=keras.optimizers.Adam(a.lr), loss=loss, metrics=metrics())
        save_model_tables(model, out_dir)
        stage1 = [l.name for l in model.layers if l.trainable and l.weights]
        n_train = lambda m: sum(int(np.prod(w.shape)) for w in m.trainable_weights)
        stage1_params = n_train(model)
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=callbacks(ckpt))
        summary["stage1_best_epoch"] = plot_curves(h1.history, f"{run} - stage 1 (backbone frozen, lr={a.lr})",
                                                   out_dir / "curves_stage1.png")

        # stage 2: unfreeze the last block, much smaller lr so pretrained features are nudged, not destroyed
        stage2_base = unfreeze_last(base, a.model, a.unfreeze)
        model.compile(optimizer=keras.optimizers.Adam(a.ft_lr), loss=loss, metrics=metrics())  # recompile so the change takes effect
        stage2 = [l.name for l in model.layers if l.trainable and l.weights]
        n1 = len(h1.history["loss"])
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=n1 + a.ft_epochs, initial_epoch=n1,
                       callbacks=callbacks(ckpt, best_so_far=min(h1.history["val_loss"])))
        summary["stage2_best_epoch"] = plot_curves(h2.history, f"{run} - stage 2 (last block unfrozen, lr={a.ft_lr})",
                                                   out_dir / "curves_stage2.png", epoch_offset=n1)
        hist = {k: h1.history[k] + h2.history[k] for k in h2.history if k in h1.history}
        summary["best_epoch"] = plot_curves(hist, f"{run}: both stages (stage 2 starts after epoch {n1})",
                                            out_dir / "curves.png")

        (out_dir / "trainable_layers.txt").write_text(
            f"STAGE 1 - trainable layers ({stage1_params:,} params), whole {base.name} frozen:\n  "
            + "\n  ".join(stage1)
            + f"\n\nSTAGE 2 - trainable top-level layers ({n_train(model):,} params):\n  " + "\n  ".join(stage2)
            + f"\n\nSTAGE 2 - trainable layers inside {base.name} ({len(stage2_base)}, BatchNorm kept frozen):\n  "
            + "\n  ".join(stage2_base) + "\n",
            encoding="utf-8")

    # checkpoint on disk = best val_loss over the whole run
    best = keras.models.load_model(ckpt)
    val_metrics = best.evaluate(val_ds, return_dict=True, verbose=0)
    summary["val_metrics_best"] = {k: round(float(v), 4) for k, v in val_metrics.items()}
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index_label="epoch")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
