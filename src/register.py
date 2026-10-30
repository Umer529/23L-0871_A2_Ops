"""Task 3.6 - register the best runs and alias the winner as `champion`.

Ranks the experiment's manual runs by best_val_auc (validation, not test, so
the test set stays an unbiased final check), registers the top N as versions
of one registered model and points the `champion` alias at the best one.

  python src/register.py               # top 2 runs -> versions 1, 2; champion = best
"""
import argparse

import mlflow

import tracking

MODEL_NAME = "visionguard"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="visionguard")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--metric", default="best_val_auc")
    a = ap.parse_args()

    tracking.setup(a.experiment)
    exp = mlflow.get_experiment_by_name(a.experiment)
    runs = mlflow.search_runs([exp.experiment_id], filter_string="tags.tracking_mode = 'manual'",
                              order_by=[f"metrics.{a.metric} DESC"], max_results=a.top)
    client = mlflow.MlflowClient()
    # register worst-first so the best run ends up as the newest version
    versions = []
    for _, r in runs.iloc[::-1].iterrows():
        model = mlflow.search_logged_models(experiment_ids=[exp.experiment_id], output_format="list",
                                            filter_string=f"source_run_id='{r.run_id}'")[0]
        v = mlflow.register_model(model.model_uri, MODEL_NAME, tags={"run_name": r["tags.mlflow.runName"]})
        client.set_model_version_tag(MODEL_NAME, v.version, a.metric, f"{r[f'metrics.{a.metric}']:.4f}")
        client.set_model_version_tag(MODEL_NAME, v.version, "img_size", r["params.img_size"])
        versions.append((v.version, r["tags.mlflow.runName"], r[f"metrics.{a.metric}"]))
        print(f"registered {MODEL_NAME} v{v.version} <- run {r['tags.mlflow.runName']} "
              f"({a.metric}={r[f'metrics.{a.metric}']:.4f})")

    best_version = versions[-1][0]
    client.set_registered_model_alias(MODEL_NAME, "champion", best_version)
    if len(versions) > 1:
        client.set_registered_model_alias(MODEL_NAME, "challenger", versions[-2][0])
    print(f"alias champion -> {MODEL_NAME} v{best_version} ({versions[-1][1]})")


if __name__ == "__main__":
    main()
