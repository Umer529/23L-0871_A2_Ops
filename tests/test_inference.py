import json

import numpy as np
from conftest import png_bytes


def test_preprocess_shape_and_range():
    from preprocess import decode_and_resize
    x = decode_and_resize(png_bytes(size=(300, 120)), 224)
    assert x.shape == (224, 224, 3)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 255.0


def test_local_predictor_loads_once_and_predicts(tmp_path):
    from tensorflow import keras
    from inference import LocalPredictor

    inp = keras.Input((32, 32, 3))
    out = keras.layers.Dense(1, activation="sigmoid")(keras.layers.GlobalAveragePooling2D()(inp))
    path = tmp_path / "tiny.keras"
    keras.Model(inp, out).save(path)
    (tmp_path / "tiny.json").write_text(json.dumps({"model_name": "tiny", "version": 3, "threshold": 0.7}))

    pred = LocalPredictor(str(path))
    assert pred.img_size == 32 and pred.threshold == 0.7 and pred.model_version == "3"
    r = pred.predict(png_bytes())
    assert r["label"] in {"AI-generated", "Real"}
    assert 0.5 <= r["confidence"] <= 1.0
    assert abs(r["confidence"] - (r["p_ai"] if r["label"] == "AI-generated" else 1 - r["p_ai"])) < 1e-9


def test_threshold_env_overrides_metadata(tmp_path):
    from tensorflow import keras
    from inference import LocalPredictor

    inp = keras.Input((16, 16, 3))
    keras.Model(inp, keras.layers.Dense(1, activation="sigmoid")(keras.layers.Flatten()(inp))).save(
        tmp_path / "m.keras")
    assert LocalPredictor(str(tmp_path / "m.keras"), threshold=0.3).threshold == 0.3
