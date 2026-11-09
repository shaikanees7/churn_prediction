"""Lab 12: Drift Monitoring & Retraining Pipeline runner. Run from anywhere: python pipelines/run_lab12_monitoring.py"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Lab 12: Drift Monitoring & Retraining Pipeline"
STEPS = [
    "src/mlops_pipeline/12_data_drift.py",
    "src/mlops_pipeline/12_prediction_drift.py",
    "src/mlops_pipeline/12_concept_drift.py",
    "src/mlops_pipeline/12_drift_analysis.py",
    "src/mlops_pipeline/12_model_comparison.py",
    "src/mlops_pipeline/12_monitor_trigger.py",
]


def run():
    print(f"[INFO] ===== {TITLE} =====")
    log = {"pipeline": TITLE, "started": datetime.now().isoformat(timespec="seconds"), "steps": []}
    ok = True
    for step in STEPS:
        cmd = [step] if isinstance(step, str) else list(step)
        print(f"\n[INFO] ---> Executing {' '.join(cmd)}")
        code = subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode
        status = "PASSED" if code == 0 else "FAILED"
        log["steps"].append({"command": " ".join(cmd), "exit_code": code, "status": status})
        if code != 0:
            print(f"[ERROR] Step failed: {' '.join(cmd)} (exit code {code}).")
            ok = False
            break

    log["overall_status"] = "PASSED" if ok else "FAILED"
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "logs" / "pipeline_run_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4)
    print(f"\n[{'SUCCESS' if ok else 'ERROR'}] {TITLE}: {log['overall_status']} (log: logs/pipeline_run_log.json)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
