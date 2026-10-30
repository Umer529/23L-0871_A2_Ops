"""Task 3.3 / 3.4 - MLflow instrumentation for src/train.py.

Two modes:
  manual  - our own Keras callback logs every epoch; we log params, tags,
            test metrics and all artifacts ourselves
  autolog - mlflow.tensorflow.autolog() logs params, epoch metrics and the model;
            we only add the tags and test metrics it cannot know about

Tracking URI/credentials come from .env (DagsHub). Without them, runs go to a
local sqlite store so the code still works offline.
"""
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mlflow
import mlflow.keras
import mlflow.tensorflow
import yaml
from dotenv import load_dotenv
from tensorflow import keras

from data import SEED


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def dvc_data_version(dvc_file="data/raw.dvc"):
    out = yaml.safe_load(Path(dvc_file).read_text())["outs"][0]
    return out["md5"]


class MlflowEpochLogger(keras.callbacks.Callback):
    """Manual per-epoch logging. `epoch` is absolute, so stage 2 continues the curve."""

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        logs["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


def setup(experiment):
    load_dotenv()
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment)
    print(f"MLflow tracking -> {mlflow.get_tracking_uri()}")


@contextmanager
def tracked_run(mode, experiment, run_name, params):
    """Yields the extra Keras callbacks the training loop should use."""
    if mode == "none":
        yield []
        return
    setup(experiment)
    if mode == "autolog":
        mlflow.tensorflow.autolog(log_models=True, log_datasets=False, checkpoint=False)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
            "dvc_data_version": dvc_data_version(),
            "seed": SEED,
            "tracking_mode": mode,
            "model_family": params["model"],
        })
        if mode == "manual":
            mlflow.log_params(params)
            yield [MlflowEpochLogger()]
        else:
            yield []
    if mode == "autolog":
        mlflow.tensorflow.autolog(disable=True)


def log_results(mode, test_metrics, val_metrics, summary, out_dir, model):
    if mode == "none":
        return
    mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items() if isinstance(v, float)})
    mlflow.log_metrics({f"best_val_{k}": float(v) for k, v in val_metrics.items()})
    mlflow.log_metric("best_epoch", summary["best_epoch"])
    mlflow.set_tag("threshold_rule", test_metrics["threshold_rule"])
    if mode == "manual":
        mlflow.log_artifacts(str(out_dir), artifact_path="reports")     # curves, CM, ROC, tables
        mlflow.log_artifacts("data/splits", artifact_path="splits")     # exact train/val/test lists
        with tempfile.TemporaryDirectory() as d:
            req = Path(d) / "pip_freeze.txt"
            req.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
            mlflow.log_artifact(str(req), artifact_path="environment")
        mlflow.keras.log_model(model, name="model")
