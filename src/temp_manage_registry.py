"""Helper for working with the local MLflow Model Registry.

    python src/temp_manage_registry.py list            # show every version and its stage
    python src/temp_manage_registry.py archive-all     # archive all versions (clean slate)
    python src/temp_manage_registry.py reset           # DELETE the registered model entirely
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

os.chdir(Path(__file__).resolve().parents[1])
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["list", "archive-all", "reset"])
    action = ap.parse_args().action

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    name = config["registry"]["model_name"]
    metric = config["registry"]["decision_metric"]

    versions = client.search_model_versions(f"name='{name}'")
    if action == "list":
        if not versions:
            print(f"[INFO] No versions registered for '{name}'.")
        for mv in sorted(versions, key=lambda v: int(v.version)):
            score = client.get_run(mv.run_id).data.metrics.get(metric)
            print(f"Version {mv.version:>3} | {mv.current_stage:<10} | {metric}={score}")
    elif action == "archive-all":
        for mv in versions:
            if mv.current_stage != "Archived":
                client.transition_model_version_stage(name, mv.version, "Archived")
        print("[SUCCESS] All versions archived.")
    elif action == "reset":
        if versions:
            client.delete_registered_model(name)
        print(f"[SUCCESS] Registered model '{name}' deleted.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
