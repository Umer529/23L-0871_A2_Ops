"""Task 7 - convert the champion Keras model to TFLite for the serverless model API.

TensorFlow (~600 MB installed) does not fit a Vercel function; the LiteRT runtime
(~11 MB) does. The .tflite graph holds the same weights and the same in-model
scaling layer, and the API uses the same src/preprocess.py, so predictions match.

  python src/export_tflite.py      # models/champion.keras -> api/model/champion.tflite (+ .json)
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from preprocess import decode_and_resize

SRC = Path("models/champion.keras")
OUT_DIR = Path("api/model")


def main():
    model = keras.models.load_model(SRC, compile=False)
    with tempfile.TemporaryDirectory() as d:
        model.export(d, format="tf_saved_model", verbose=False)
        tflite = tf.lite.TFLiteConverter.from_saved_model(d).convert()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "champion.tflite").write_bytes(tflite)
    meta = json.loads(SRC.with_suffix(".json").read_text())
    (OUT_DIR / "champion.json").write_text(json.dumps(meta, indent=2))
    shutil.copyfile("src/preprocess.py", "api/preprocess.py")
    print(f"wrote {OUT_DIR / 'champion.tflite'} ({len(tflite) / 1e6:.1f} MB), metadata {meta}")

    # parity check: Keras vs TFLite on 30 validation images
    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    size = int(model.input_shape[1])
    diffs = []
    for path in pd.read_csv("data/splits/val.csv")["path"][:30]:
        x = decode_and_resize(Path(path).read_bytes(), size)[None]
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        diffs.append(abs(float(interp.get_tensor(out["index"])[0, 0]) - float(model(x)[0, 0])))
    print(f"Keras vs TFLite on 30 val images: max |diff p_ai| = {max(diffs):.2e}, mean = {np.mean(diffs):.2e}")


if __name__ == "__main__":
    main()
