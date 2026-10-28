"""Task 2.7 (toy part) - Keras Tuner on Fashion-MNIST.

Random search over learning rate and number of hidden units of a tiny MLP.
Usage:  python labs/keras_tuner_toy.py
Output: reports/tuner_toy_results.csv
"""
from pathlib import Path

import keras_tuner as kt
import pandas as pd
from tensorflow import keras

keras.utils.set_random_seed(42)
(x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
x_train = x_train[:20000] / 255.0   # subset keeps every trial under a few seconds on CPU
y_train = y_train[:20000]


def build(hp):
    model = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=8, seed=42,
                        directory="tuner_dir", project_name="fashion_mnist_toy", overwrite=True)
tuner.search(x_train, y_train, validation_split=0.2, epochs=3, verbose=0)

rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score}
        for t in tuner.oracle.trials.values()]
df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
Path("reports").mkdir(exist_ok=True)
df.to_csv("reports/tuner_toy_results.csv", index=False)
print(df.to_string(index=False))
print("\nBest settings:", tuner.get_best_hyperparameters(1)[0].values)
