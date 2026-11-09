"""Lab 9: Failure Handling and Recovery runner. Run from anywhere:  python pipelines/run_lab9_failure.py"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Lab 9: Failure Handling and Recovery"
STEPS = [
    "src/mlops_pipeline/churn_prediction_pipeline.py",
    "src/mlops_pipeline/10_failure_injection.py",
]


def run():
    print(f"[INFO] ===== {TITLE} =====")
    log = {"pipeline": TITLE, "started": datetime.now().isoformat(timespec="seconds"), "steps": []}
    ok = True
    for step in STEPS:
        cmd = [step] if isinstance(step, str) else list(step)
        print(f"\n[INFO] ---> Executing {' '.join(cmd)}")
        code = subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode
        log["steps"].append({"command": " ".join(cmd), "exit_code": code,
                             "status": "PASSED" if code == 0 else "FAILED"})
        if code != 0:
            print(f"[ERROR] Pipeline halted: {' '.join(cmd)} failed (exit code {code}).")
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
