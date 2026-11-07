"""Task 4.1 / Task 8 - inference module.

LocalPredictor loads the champion Keras model ONCE when the app starts. Loading
per request would re-read ~10 MB of weights and rebuild the graph every time
(seconds of latency, memory churn, and a crash if the file moves mid-traffic).

RemotePredictor sends the image to the model API running in the cloud instead;
the Flask app then needs no TensorFlow at all.
"""
import json
import time
from pathlib import Path

import requests


def _verdict(p_ai, threshold):
    is_ai = p_ai >= threshold
    # confidence = probability of the class we announce, not of "AI" every time
    return ("AI-generated" if is_ai else "Real"), (p_ai if is_ai else 1.0 - p_ai)


class LocalPredictor:
    mode = "local"

    def __init__(self, model_path, threshold=None):
        import numpy as np
        from tensorflow import keras
        from preprocess import decode_and_resize  # the exact function used in training

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path.resolve()} (set MODEL_PATH)")
        meta_file = path.with_suffix(".json")
        meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}

        self._np, self._decode = np, decode_and_resize
        self.model = keras.models.load_model(path, compile=False)
        self.img_size = int(self.model.input_shape[1])
        self.threshold = threshold if threshold is not None else float(meta.get("threshold", 0.5))
        self.model_name = meta.get("model_name", path.stem)
        self.model_version = str(meta.get("version", "local"))
        self.model.predict(np.zeros((1, self.img_size, self.img_size, 3), "float32"), verbose=0)  # warm-up

    def predict(self, image_bytes):
        t0 = time.perf_counter()
        x = self._decode(image_bytes, self.img_size)[None]
        p_ai = float(self.model(x, training=False).numpy()[0, 0])
        label, conf = _verdict(p_ai, self.threshold)
        return {"label": label, "confidence": conf, "p_ai": p_ai, "threshold": self.threshold,
                "model_name": self.model_name, "model_version": self.model_version,
                "latency_ms": (time.perf_counter() - t0) * 1000}


class RemotePredictor:
    mode = "remote"

    def __init__(self, url, api_key, timeout=60):
        if not url:
            raise ValueError("INFERENCE_MODE=remote needs MODEL_API_URL")
        self.url, self.timeout = url.rstrip("/"), timeout
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key
        self.threshold, self.model_name, self.model_version = None, "remote", "?"

    def info(self):
        r = self.session.get(f"{self.url}/health", timeout=self.timeout)
        r.raise_for_status()
        info = r.json()
        self.threshold, self.model_name, self.model_version = (info.get("threshold"), info.get("model_name"),
                                                               str(info.get("model_version")))
        return info

    def predict(self, image_bytes, filename="upload"):
        t0 = time.perf_counter()
        r = self.session.post(f"{self.url}/predict", files={"image": (filename, image_bytes)},
                              timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"model API returned {r.status_code}: {r.text[:200]}")
        out = r.json()
        out["latency_ms"] = (time.perf_counter() - t0) * 1000   # includes the network round trip
        self.threshold, self.model_name, self.model_version = (out["threshold"], out["model_name"],
                                                               out["model_version"])
        return out


def make_predictor(cfg):
    if cfg.inference_mode == "remote":
        return RemotePredictor(cfg.model_api_url, cfg.api_key)
    return LocalPredictor(cfg.model_path, cfg.threshold)
