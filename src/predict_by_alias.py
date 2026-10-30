"""Task 3.6 - load the registered model by alias and classify one image.

  python src/predict_by_alias.py path/to/image.jpg
  python src/predict_by_alias.py path/to/image.jpg --export models/champion.keras

--export also writes the champion to a plain .keras file. That file is what the
Flask app / model API loads at start-up, so serving never calls DagsHub per request.
"""
import argparse
import json

import mlflow
import mlflow.keras

import tracking
from data import load_image

MODEL_NAME = "visionguard"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--export", default=None)
    a = ap.parse_args()

    tracking.setup("visionguard")
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, a.alias)
    run = client.get_run(mv.run_id)
    img_size = int(run.data.params.get("img_size", 224))
    threshold = float(run.data.metrics.get("test_threshold", 0.5))

    model = mlflow.keras.load_model(f"models:/{MODEL_NAME}@{a.alias}")
    print(f"loaded models:/{MODEL_NAME}@{a.alias} -> version {mv.version} "
          f"(run {run.info.run_name}, img_size={img_size}, threshold={threshold})")

    p = float(model.predict(load_image(a.image, img_size)[None], verbose=0)[0, 0])
    label = "AI-generated" if p >= threshold else "Real"
    print(json.dumps({"image": a.image, "p_ai": round(p, 4), "threshold": threshold, "label": label}))

    if a.export:
        model.save(a.export)
        meta = {"model_name": MODEL_NAME, "version": mv.version, "alias": a.alias,
                "run_name": run.info.run_name, "img_size": img_size, "threshold": threshold}
        with open(a.export.rsplit(".", 1)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"exported {a.export} (+ metadata json)")


if __name__ == "__main__":
    main()
