"""Lab 8: run the CI job locally. Run from anywhere:  python pipelines/run_lab8_ci.py

Mirrors .github/workflows/lab8_ci.yml:
  1. check the files that trigger CI exist and the dependencies import
  2. run the Churn Prediction MLOps pipeline (the CI job's main step)
  3. check the quality gate passed and the evidence the CI uploads was produced
"""
import importlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Lab 8: Continuous Integration (local run)"

REQUIRED_FILES = ["config.yaml", "requirements.txt", "data/raw/churn.csv",
                  ".github/workflows/lab8_ci.yml", "src/mlops_pipeline/churn_prediction_pipeline.py"]
MODULES = ["pandas", "numpy", "sklearn", "yaml", "joblib", "mlflow", "pandera"]


def get_model_path():
    try:
        with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("paths", {}).get("model", "models/churn_model.pkl")
    except Exception:
        return "models/churn_model.pkl"


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip())
    return {"check": name, "status": "PASSED" if ok else "FAILED", "detail": detail}


def run():
    print(f"[INFO] ===== {TITLE} =====")
    log = {"pipeline": TITLE, "started": datetime.now().isoformat(timespec="seconds"), "steps": []}

    # 1. pre-flight
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    log["steps"].append(check("required files present", not missing, ", ".join(missing)))
    bad = []
    for m in MODULES:
        try:
            importlib.import_module(m)
        except ImportError:
            bad.append(m)
    log["steps"].append(check("dependencies importable", not bad, ", ".join(bad)))
    ok = not missing and not bad

    # 2. the CI job's main step
    if ok:
        print("\n[INFO] ---> Executing src/mlops_pipeline/churn_prediction_pipeline.py")
        code = subprocess.run([sys.executable, "src/mlops_pipeline/churn_prediction_pipeline.py"], cwd=ROOT).returncode
        log["steps"].append(check("churn_prediction_pipeline.py", code == 0, f"exit code {code}"))
        ok = code == 0

    # 3. post-flight
    if ok:
        gate = json.load(open(ROOT / "reports/quality_gate_report.json"))
        log["steps"].append(check("quality gate", gate["status"] == "PASSED",
                                  f"{gate['metric']}={gate['actual_value']} (min {gate['minimum_value']})"))
        
        model_path = get_model_path()
        evidence_files = [
            "reports/evaluation_report.json",
            "reports/quality_gate_report.json",
            "logs/churn_prediction_pipeline_run_log.json",
            "outputs/inference_predictions.csv",
            model_path
        ]
        absent = [f for f in evidence_files if not (ROOT / f).exists()]
        log["steps"].append(check("CI evidence produced", not absent, ", ".join(absent)))
        ok = gate["status"] == "PASSED" and not absent

    log["overall_status"] = "PASSED" if ok else "FAILED"
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "logs" / "pipeline_run_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4)
    print(f"\n[{'SUCCESS' if ok else 'ERROR'}] {TITLE}: {log['overall_status']} (log: logs/pipeline_run_log.json)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
