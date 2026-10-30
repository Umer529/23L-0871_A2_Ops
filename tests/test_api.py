import importlib.util
import io
from pathlib import Path

import pytest
from conftest import png_bytes

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "api" / "model" / "champion.tflite"


def test_api_preprocess_is_identical_copy():
    # the API ships its own copy; it must never drift from the one used in training
    assert (ROOT / "api" / "preprocess.py").read_bytes().replace(b"\r\n", b"\n") == \
        (ROOT / "src" / "preprocess.py").read_bytes().replace(b"\r\n", b"\n")


@pytest.fixture
def api(monkeypatch):
    if not MODEL.is_file():
        pytest.skip("api/model/champion.tflite not exported (run src/export_tflite.py)")
    monkeypatch.setenv("API_KEY", "test-key")
    spec = importlib.util.spec_from_file_location("vg_api", ROOT / "api" / "index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app.test_client()


def post(client, data, key="test-key"):
    return client.post("/predict", data={"image": (io.BytesIO(data), "x.png")},
                       headers={"X-API-Key": key}, content_type="multipart/form-data")


def test_api_health(api):
    assert api.get("/health").get_json()["status"] == "ok"


def test_api_rejects_missing_or_wrong_key(api):
    assert post(api, png_bytes(), key="").status_code == 401
    assert post(api, png_bytes(), key="wrong").status_code == 401


def test_api_predicts(api):
    r = post(api, png_bytes())
    assert r.status_code == 200
    body = r.get_json()
    assert body["label"] in {"AI-generated", "Real"} and 0.5 <= body["confidence"] <= 1


def test_api_rejects_garbage(api):
    assert post(api, b"not an image at all").status_code == 422
