import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "app", ROOT / "src"):
    sys.path.insert(0, str(p))

from config import Config  # noqa: E402
from db import PredictionStore  # noqa: E402
from main import create_app  # noqa: E402


class FakePredictor:
    mode = "local"
    model_name, model_version, threshold = "fake", "0", 0.5

    def predict(self, image_bytes):
        return {"label": "AI-generated", "confidence": 0.9, "p_ai": 0.9, "threshold": 0.5,
                "model_name": "fake", "model_version": "0", "latency_ms": 1.0}


def png_bytes(size=(64, 48), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def store(tmp_path):
    return PredictionStore(f"sqlite:///{tmp_path / 'test.db'}")


@pytest.fixture
def client(store):
    cfg = Config(model_path="unused", threshold=None, database_url="unused", inference_mode="local",
                 model_api_url="", api_key="", max_upload_mb=1, port=0)
    app = create_app(cfg, predictor=FakePredictor(), store=store)
    app.testing = True
    return app.test_client()
