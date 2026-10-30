"""Task 3.1 / 3.2 - MLflow tool lab.

Tiny Keras MLP on MNIST, two runs that differ only in learning rate. Logs
params, per-epoch metrics, one plot and the model.

  python labs/mlflow_toy.py            # local store  -> mlflow ui --backend-store-uri sqlite:///mlflow.db
  python labs/mlflow_toy.py --remote   # DagsHub, URI + credentials read from .env
"""
import argparse
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from tensorflow import keras

ap = argparse.ArgumentParser()
ap.add_argument("--remote", action="store_true")
args = ap.parse_args()

if args.remote:
    load_dotenv()  # MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD - never written in code
    assert os.getenv("MLFLOW_TRACKING_URI"), "set MLFLOW_TRACKING_URI in .env"
else:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("toy-mnist")

(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x, xt = x[:10000] / 255.0, xt[:2000] / 255.0
y, yt = y[:10000], yt[:2000]


class MlflowEpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


for lr in [1e-3, 1e-2]:
    keras.utils.set_random_seed(42)
    with mlflow.start_run(run_name=f"mlp_lr_{lr}"):
        params = {"learning_rate": lr, "units": 64, "epochs": 5, "batch_size": 64}
        mlflow.log_params(params)
        mlflow.set_tags({"lab": "3.1", "dataset": "mnist-10k", "author": "Umer529"})

        model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                                  keras.layers.Dense(params["units"], activation="relu"),
                                  keras.layers.Dense(10, activation="softmax")])
        model.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(x, y, validation_split=0.2, epochs=params["epochs"], batch_size=params["batch_size"],
                      callbacks=[MlflowEpochLogger()], verbose=0)
        test_loss, test_acc = model.evaluate(xt, yt, verbose=0)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(h.history["loss"], label="train")
        ax.plot(h.history["val_loss"], label="val")
        ax.set(title=f"loss, lr={lr}", xlabel="epoch")
        ax.legend()
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.keras")
            model.save(path)
            mlflow.log_artifact(path, artifact_path="model")
        print(f"lr={lr}: test_accuracy={test_acc:.4f}  run_id={mlflow.active_run().info.run_id}")
