"""VisionGuard Flask app - Task 4 / 5 / 8.

  python main.py        # production WSGI server (waitress), debug off

Debug mode stays off everywhere: the Werkzeug debugger lets anyone who sees an
error page run Python on the server, it leaks source and settings in tracebacks,
and the auto-reloader would load the model twice.
"""
import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

# app/ and src/ side by side in the repo; in the Docker image both live in /app
HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "src"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import Config  # noqa: E402
from db import PredictionStore  # noqa: E402
from inference import make_predictor  # noqa: E402
from validation import UploadError, thumbnail_data_uri, validate_upload  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("visionguard")

CAVEAT = ("This verdict is a statistical guess, not proof. On held-out test images the model "
          "caught about two thirds of AI-generated images and wrongly flagged about 1 in 13 real ones. "
          "Paintings, heavy editing, filters and new generators fool it. Never use it alone to "
          "accuse anyone of faking an image.")


def create_app(cfg=None, predictor=None, store=None):
    cfg = cfg or Config.from_env()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(cfg.max_upload_mb * 1024 * 1024)
    # registry versions are numbers ("v3"); anything else (e.g. "pre-registry") is shown as-is
    app.jinja_env.filters["ver"] = lambda v: f"v{v}" if str(v).isdigit() else f"({v})"

    # loaded ONCE here, shared by every request
    predictor = predictor or make_predictor(cfg)
    store = store or PredictionStore(cfg.database_url)
    log.info("model ready: mode=%s name=%s version=%s threshold=%s", predictor.mode,
             predictor.model_name, predictor.model_version, predictor.threshold)

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=cfg.max_upload_mb)

    @app.post("/predict")
    def predict():
        name, data, sha = validate_upload(request.files.get("image"))
        if predictor.mode == "remote":
            out = predictor.predict(data, name)
        else:
            out = predictor.predict(data)
        row_id = store.add(filename=name, image_sha256=sha, label=out["label"],
                           confidence=out["confidence"], p_ai=out["p_ai"], threshold=out["threshold"],
                           model_name=out["model_name"], model_version=str(out["model_version"]),
                           inference_mode=predictor.mode, latency_ms=out["latency_ms"])
        log.info("prediction #%d %s -> %s (%.1f%%, p_ai=%.3f, %.0f ms)", row_id, name, out["label"],
                 out["confidence"] * 100, out["p_ai"], out["latency_ms"])
        return render_template("result.html", r=out, filename=name, row_id=row_id,
                               preview=thumbnail_data_uri(data), caveat=CAVEAT)

    @app.get("/history")
    def history():
        rows = store.recent(int(request.args.get("limit", 100)))
        return render_template("history.html", rows=rows, total=store.count())

    @app.get("/health")
    def health():
        body = {"status": "ok", "inference_mode": predictor.mode}
        try:
            store.ping()
            body["database"] = "ok"
        except Exception as e:
            body.update(status="degraded", database=f"error: {type(e).__name__}")
        if predictor.mode == "remote":
            try:
                body["model_api"] = predictor.info().get("status", "?")
            except Exception as e:
                body.update(status="degraded", model_api=f"error: {type(e).__name__}")
        body.update(model_name=predictor.model_name, model_version=predictor.model_version,
                    threshold=predictor.threshold)
        return jsonify(body), 200 if body["status"] == "ok" else 503

    @app.errorhandler(UploadError)
    def upload_error(e):
        log.warning("rejected upload: %s - %s", e.title, e.message)
        return render_template("error.html", title=e.title, message=e.message, status=e.status), e.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(e):
        return render_template("error.html", status=413, title="File too large",
                               message=f"Images must be smaller than {cfg.max_upload_mb:g} MB."), 413

    @app.errorhandler(HTTPException)
    def http_error(e):
        return render_template("error.html", status=e.code, title=e.name, message=e.description), e.code

    @app.errorhandler(Exception)
    def server_error(e):
        log.exception("unhandled error")
        return render_template("error.html", status=500, title="Something went wrong",
                               message="The image could not be analysed right now. Please try again."), 500

    return app


if __name__ == "__main__":
    from waitress import serve
    cfg = Config.from_env()
    log.info("starting waitress on 0.0.0.0:%d (debug off)", cfg.port)
    serve(create_app(cfg), host="0.0.0.0", port=cfg.port, threads=int(os.getenv("THREADS", "4")))
