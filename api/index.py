"""VisionGuard model API (Task 7) - deployed as a Vercel Python function.

  GET  /health    -> status + which model is loaded (no key needed)
  POST /predict   -> multipart field "image"; header X-API-Key required

Runs the champion as TFLite on the LiteRT runtime and preprocesses with the same
preprocess.py as training. The interpreter is created once per warm instance.
"""
import hmac
import json
import os
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import UnidentifiedImageError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from preprocess import decode_and_resize  # noqa: E402

try:
    from ai_edge_litert.interpreter import Interpreter          # Linux / Vercel
except ImportError:                                              # local Windows dev: TF ships the same interpreter
    from tensorflow.lite.python.interpreter import Interpreter

MODEL_DIR = Path(os.getenv("MODEL_DIR", HERE / "model"))
META = json.loads((MODEL_DIR / "champion.json").read_text())
THRESHOLD = float(os.getenv("THRESHOLD") or META.get("threshold", 0.5))
MAX_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "4")) * 1024 * 1024)   # Vercel rejects bodies > 4.5 MB

interpreter = Interpreter(model_path=str(MODEL_DIR / "champion.tflite"))
interpreter.allocate_tensors()
_IN, _OUT = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
IMG_SIZE = int(_IN["shape"][1])

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


def _error(status, message):
    return jsonify({"error": message}), status


def _authorised():
    expected = os.getenv("API_KEY", "")
    given = request.headers.get("X-API-Key", "")
    # fail closed if the key is not configured; constant-time compare
    return bool(expected) and hmac.compare_digest(given.encode(), expected.encode())


@app.get("/")
@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "visionguard-model-api", "runtime": "litert",
                    "model_name": META.get("model_name"), "model_version": str(META.get("version")),
                    "img_size": IMG_SIZE, "threshold": THRESHOLD})


@app.post("/predict")
def predict():
    if not _authorised():
        return _error(401, "missing or wrong X-API-Key")
    f = request.files.get("image")
    data = f.read() if f else request.get_data()
    if not data:
        return _error(400, "send the image as multipart field 'image'")
    t0 = time.perf_counter()
    try:
        x = decode_and_resize(data, IMG_SIZE)[None]
    except (UnidentifiedImageError, OSError, ValueError):
        return _error(422, "file could not be decoded as an image")
    interpreter.set_tensor(_IN["index"], x)
    interpreter.invoke()
    p_ai = float(interpreter.get_tensor(_OUT["index"])[0, 0])
    is_ai = p_ai >= THRESHOLD
    return jsonify({"label": "AI-generated" if is_ai else "Real",
                    "confidence": p_ai if is_ai else 1.0 - p_ai, "p_ai": p_ai, "threshold": THRESHOLD,
                    "model_name": META.get("model_name"), "model_version": str(META.get("version")),
                    "inference_ms": (time.perf_counter() - t0) * 1000})


@app.errorhandler(413)
def too_large(_):
    return _error(413, f"image larger than {MAX_BYTES // (1024 * 1024)} MB")


if __name__ == "__main__":
    app.run(port=8000)   # local testing only; Vercel imports `app`
