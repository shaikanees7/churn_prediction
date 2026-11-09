"""Lab 10: API Test Runner. Run from anywhere:  python pipelines/run_lab10_api.py"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Lab 10: FastAPI Inference & Serving Validation"


def run():
    print(f"[INFO] ===== {TITLE} =====")
    log = {"pipeline": TITLE, "started": datetime.now().isoformat(timespec="seconds"), "steps": []}
    
    cmd = [sys.executable, "-m", "pytest", "tests/test_api.py", "-v"]
    print(f"\n[INFO] ---> Executing {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    code = result.returncode
    
    status = "PASSED" if code == 0 else "FAILED"
    log["steps"].append({"command": " ".join(cmd), "exit_code": code, "status": status})
    log["overall_status"] = status
    
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "logs" / "pipeline_run_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4)
        
    print(f"\n[{'SUCCESS' if code == 0 else 'ERROR'}] {TITLE}: {status} (log: logs/pipeline_run_log.json)")
    return code == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)

