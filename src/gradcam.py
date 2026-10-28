"""Bonus - Grad-CAM for 4 correct and 4 wrong test predictions.

Heat map = ReLU(sum_k  mean(dP/dA_k) * A_k) over the backbone's last conv
feature maps A (7x7x1280 for MobileNetV2 at 224), upsampled onto the image.
Usage:  python src/gradcam.py --run mobilenetv2_224
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data import load_image, read_split


def gradcam(model, img):
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    x = model.layers[1](img[None])           # the model's own preprocessing layer
    with tf.GradientTape() as tape:
        feats = base(x, training=False)
        tape.watch(feats)
        h = model.get_layer("gap")(feats)
        p = model.get_layer("prob_ai")(h)[:, 0]
    grads = tape.gradient(p, feats)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], img.shape[:2])[..., 0]
    return cam.numpy(), float(p[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mobilenetv2_224")
    a = ap.parse_args()
    out_dir = Path("reports") / a.run
    t = json.loads((out_dir / "test_metrics.json").read_text())["threshold"]
    probs = np.load(out_dir / "test_probs.npy")
    paths, labels = read_split("test")
    y = np.array(labels, dtype=int)
    pred = (probs >= t).astype(int)
    conf = np.abs(probs - t)
    right = [i for i in np.argsort(-conf) if pred[i] == y[i]]
    # 2 correct per class so both kinds of decision are shown
    picks = [i for i in right if y[i] == 1][:2] + [i for i in right if y[i] == 0][:2]
    picks += [i for i in np.argsort(-conf) if pred[i] != y[i]][:4]

    model = keras.models.load_model(f"models/{a.run}.keras")
    fig, axes = plt.subplots(2, 8, figsize=(22, 6.4))
    for c, i in enumerate(picks):
        img = load_image(paths[i]).numpy()
        cam, p = gradcam(model, img)
        axes[0, c].imshow(img.astype("uint8"))
        axes[1, c].imshow(img.astype("uint8"))
        axes[1, c].imshow(cam, cmap="jet", alpha=0.45)
        tag = "CORRECT" if pred[i] == y[i] else "WRONG"
        axes[0, c].set_title(f"{tag}\ntrue={'AI' if y[i] else 'Real'} p(AI)={p:.2f}", fontsize=9,
                             color="green" if tag == "CORRECT" else "darkred")
        for ax in axes[:, c]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM ({a.run}, threshold={t:.2f}): red = regions that push p(AI) up", fontsize=13)
    fig.tight_layout()
    out = Path("reports/figures") / f"gradcam_{a.run}.png"
    fig.savefig(out, dpi=100)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
