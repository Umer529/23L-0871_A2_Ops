"""Image preprocessing shared by training (src/data.py), the Flask app and the cloud model API.

Pure Pillow + NumPy on purpose: the serverless model API runs LiteRT (TFLite)
without TensorFlow, so a TF-based resize could not be reproduced there. If
training and serving preprocess differently, the model sees inputs at serving
time that it never saw in training and its accuracy silently drops - so every
stage imports this one function instead of re-implementing it.
"""
import io

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224


def decode_and_resize(image_bytes, img_size=IMG_SIZE):
    """Bytes -> (img_size, img_size, 3) float32 array in [0, 255]."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = ImageOps.exif_transpose(img)   # phone photos: respect the camera's rotation flag
        img = img.convert("RGB")             # palette / greyscale / RGBA -> 3 channels (alpha dropped)
        # centre square crop: no stretching, and the model cannot learn the
        # aspect ratio (which differs between the two classes) as a shortcut
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        # Pillow's BILINEAR filter widens its kernel when shrinking, i.e. it is antialiased
        img = img.resize((img_size, img_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)
