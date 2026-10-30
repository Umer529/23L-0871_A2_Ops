import io

from conftest import png_bytes


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"
    assert r.get_json()["database"] == "ok"


def test_upload_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Analyse image" in r.data


def test_valid_upload_is_predicted_and_logged(client, store):
    r = upload(client, png_bytes(), "cat.png")
    assert r.status_code == 200
    assert b"AI-generated" in r.data and b"90.0%" in r.data
    rows = store.recent()
    assert len(rows) == 1
    assert rows[0]["filename"] == "cat.png" and rows[0]["model_name"] == "fake"
    h = client.get("/history")
    assert b"cat.png" in h.data


def test_jpeg_accepted(client):
    assert upload(client, png_bytes(fmt="JPEG"), "photo.jpg").status_code == 200


def test_wrong_extension_rejected(client, store):
    r = upload(client, b"hello", "notes.txt")
    assert r.status_code == 415
    assert b"Unsupported file type" in r.data
    assert store.count() == 0


def test_too_large_rejected(client):
    r = upload(client, b"0" * (2 * 1024 * 1024), "big.png")   # limit is 1 MB in the fixture
    assert r.status_code == 413
    assert b"File too large" in r.data


def test_undecodable_image_rejected(client, store):
    r = upload(client, b"\x89PNG\r\n\x1a\n" + b"garbage" * 50, "broken.png")
    assert r.status_code == 422
    assert b"could not be read" in r.data
    assert store.count() == 0


def test_disguised_format_rejected(client):
    import io as _io
    from PIL import Image
    buf = _io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, "GIF")
    assert upload(client, buf.getvalue(), "sneaky.png").status_code == 415


def test_filename_sanitized(client, store):
    upload(client, png_bytes(), "../../etc/passwd.png")
    assert store.recent()[0]["filename"] == "etc_passwd.png"


def test_missing_file(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_unknown_route_friendly_404(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert b"Error 404" in r.data
