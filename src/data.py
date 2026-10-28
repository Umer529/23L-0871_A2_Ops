"""Task 2.1 - tf.data input pipeline.

Stages (per image):
  1. read raw bytes from the path in the split CSV
  2. decode JPEG/PNG to 3-channel uint8 (palette, RGBA and greyscale all become RGB)
  3. centre-crop to a square, then resize to IMG_SIZE x IMG_SIZE
  4. cache the small uint8 tensors in RAM (decoding 500 MB of large JPEGs every epoch is the bottleneck)
  5. train only: shuffle + random augmentation
  6. batch + prefetch so the CPU prepares the next batch while the model trains

Pixel values leave the pipeline as float32 in [0, 255]. Model-specific scaling
(/255 for the CNN, [-1, 1] for MobileNetV2) is a layer inside each model, so the
Flask app can feed the exact same tensor that load_image() produces.

Usage (figure of one augmented batch):  python src/data.py
"""
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras

SPLIT_DIR = Path("data/splits")
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = {0: "Real", 1: "AI-generated"}


def decode_and_resize(image_bytes, img_size=IMG_SIZE):
    """Bytes -> (img_size, img_size, 3) float32 in [0, 255]. Shared with the Flask app."""
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    # centre square crop: no stretching, and the model cannot learn the
    # aspect ratio (which differs between the two classes) as a shortcut
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    side = tf.minimum(h, w)
    img = tf.image.crop_to_bounding_box(img, (h - side) // 2, (w - side) // 2, side, side)
    img = tf.image.resize(img, [img_size, img_size], antialias=True)
    return tf.clip_by_value(img, 0.0, 255.0)


def load_image(path, img_size=IMG_SIZE):
    return decode_and_resize(tf.io.read_file(path), img_size)


def augmenter(seed=SEED):
    # mild on purpose: heavy blur/noise/colour jitter would wipe out the
    # fine texture artefacts that separate AI images from real ones
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=seed),
        keras.layers.RandomZoom((-0.1, 0.0), fill_mode="reflect", seed=seed),
        keras.layers.RandomContrast(0.1, seed=seed),
    ], name="augmentation")


def read_split(name):
    df = pd.read_csv(SPLIT_DIR / f"{name}.csv")
    return df["path"].tolist(), df["label"].astype("float32").tolist()


def make_dataset(name, img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=False, shuffle=False):
    paths, labels = read_split(name)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (tf.cast(load_image(p, img_size), tf.uint8), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug = augmenter()
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def save_augmented_batch_figure(out="reports/figures/augmented_batch.png", n=16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = next(iter(make_dataset("train", augment=True, shuffle=True)))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10.5))
    for i, ax in enumerate(axes.flat[:n]):
        ax.imshow(x[i].numpy().astype("uint8"))
        ax.set_title(CLASS_NAMES[int(y[i])], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"One augmented training batch ({IMG_SIZE}x{IMG_SIZE}, centre-cropped)", fontsize=12)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}  batch shape={tuple(x.shape)} dtype={x.dtype} "
          f"min={float(tf.reduce_min(x)):.1f} max={float(tf.reduce_max(x)):.1f}")


if __name__ == "__main__":
    keras.utils.set_random_seed(SEED)
    save_augmented_batch_figure()
