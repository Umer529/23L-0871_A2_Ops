"""Task 2.7 (apply part) - Keras Tuner on VisionGuard (MobileNetV2).

Searches learning rate, head dropout and number of unfrozen backbone layers,
8 random trials, each a short single-stage run judged on val_loss.
Usage:  python src/tune.py
Output: reports/tuner_visionguard_results.csv
"""
import keras_tuner as kt
import pandas as pd
from tensorflow import keras

from data import SEED, make_dataset
from models import build_transfer_model, unfreeze_last

TRIALS = 8
EPOCHS = 5


def build(hp):
    model, base = build_transfer_model("mobilenetv2", dropout=hp.Float("dropout", 0.1, 0.5, step=0.1))
    n = hp.Choice("unfrozen_layers", [0, 10, 25, 50])
    if n:
        unfreeze_last(base, "mobilenetv2", n_layers=n)
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", [1e-3, 3e-4, 1e-4, 3e-5])),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


def main():
    keras.utils.set_random_seed(SEED)
    train_ds = make_dataset("train", augment=True, shuffle=True)
    val_ds = make_dataset("val")
    tuner = kt.RandomSearch(build, objective="val_loss", max_trials=TRIALS, seed=SEED,
                            directory="tuner_dir", project_name="visionguard", overwrite=True)
    tuner.search(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

    rows = []
    for t in tuner.oracle.trials.values():
        m = t.metrics
        rows.append({"trial": t.trial_id, **t.hyperparameters.values,
                     "val_loss": t.score,
                     "val_accuracy": m.get_best_value("val_accuracy"),
                     "val_auc": m.get_best_value("val_auc")})
    df = pd.DataFrame(rows).sort_values("val_loss")
    df.to_csv("reports/tuner_visionguard_results.csv", index=False)
    print(df.round(4).to_string(index=False))
    # crude importance: how much does the mean val_loss move across each setting's values?
    for hp in ["learning_rate", "dropout", "unfrozen_layers"]:
        g = df.groupby(hp)["val_loss"].mean()
        print(f"\n{hp}: mean val_loss per value (spread={g.max() - g.min():.4f})\n{g.round(4).to_string()}")
    print("\nBest:", tuner.get_best_hyperparameters(1)[0].values)


if __name__ == "__main__":
    main()
