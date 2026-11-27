"""Execute the authorized local experiment to its terminal report, not just smoke."""
import argparse
import fcntl
import os
from pathlib import Path
import traceback

from .src.artifacts import ROOT, now, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Complete authorized exploratory campaign")
    parser.add_argument("--output", default=str(ROOT / "runs/full_exploratory_001"))
    args = parser.parse_args(argv)
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "process.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(destination / "process.json", {"pid": os.getpid(), "started_at": now(), "status": "running"})
        try:
            from .src.prepare import prepare_exploration
            from .src.full_campaign import run_full
            context = prepare_exploration(destination)
            result = run_full(context)
            write_json(destination / "process.json", {"pid": os.getpid(), "completed_at": now(), "status": "executed", "result": result})
            return 0
        except BaseException as exc:
            write_json(destination / "process.json", {"pid": os.getpid(), "failed_at": now(), "status": "failed",
                       "error": repr(exc), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
