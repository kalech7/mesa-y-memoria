"""Export existing SFT/GRPO journals to a local, live TensorBoard dashboard."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

PRUNE = {"source_snapshot", "states", "sqlite", "audit_work", "teacher_work",
         "environment_rollouts", "teacher_sqlite", "__pycache__"}
DETAILS = ("loss", "policy_loss", "kl", "kl_loss", "grad_norm", "gradient_norm_pre_clip",
           "gradient_norm_post_clip", "ratio_min", "ratio_max", "ratio_mean", "clip_fraction_phase_mean",
           "zero_task_signal_fraction", "zero_signal_fraction", "sampled_token_surprisal_mean")


class Exporter:
    def __init__(self, source: Path, output: Path):
        self.source = source
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.writer = EventFileWriter(str(output), max_queue_size=1, flush_secs=1)
        self.readers = {}
        self.emitted = set()
        self.summary_mtime = None
        self.groups_seen = defaultdict(int)
        self.optimization_passes = None

    def scalar(self, tag, value, step, wall_time=None):
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return
        key = (tag, int(step))
        if key in self.emitted:
            return
        event = Event(wall_time=wall_time or time.time(), step=int(step),
                      summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(value))]))
        self.writer.add_event(event)
        self.emitted.add(key)

    def jsonl(self, name):
        path = self.source / name
        if not path.is_file():
            return []
        offset, inode = self.readers.get(name, (0, None))
        stat = path.stat()
        if inode != stat.st_ino or stat.st_size < offset:
            offset = 0
        rows = []
        with path.open("rb") as stream:
            stream.seek(offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line.endswith(b"\n"):
                    stream.seek(start)
                    break
                try:
                    rows.append(json.loads(line))
                except ValueError as error:
                    raise ValueError(f"Invalid complete JSON record in {path} at byte {start}") from error
            self.readers[name] = (stream.tell(), stat.st_ino)
        return rows

    def details(self, row):
        step = row["step"]
        for key in DETAILS:
            if key in row:
                self.scalar("optimization/" + key, row[key], step)
        for key, value in row.get("resource_counters", {}).items():
            self.scalar("resources/" + key, value, step)

    def pass_details(self, row):
        step = row["outer_update"]
        pass_index = row["optimization_pass"]
        prefix = f"optimization/pass_{pass_index}/"
        for key in DETAILS:
            if key in row:
                self.scalar(prefix + key, row[key], step)
        self.scalar(prefix + "optimizer_step", row["optimizer_step"], step)
        for phase, loss in row.get("phase_losses", {}).items():
            self.scalar(prefix + "phase/" + phase + "/loss", loss, step)
        if pass_index == self.optimization_passes:
            self.details({**row, "step": step})

    def validation(self, row):
        for key, value in row.get("metrics", row.get("validation", {})).items():
            self.scalar("validation/" + key, value, row["step"])

    def poll(self):
        for row in self.jsonl("progress.jsonl"):
            if row["stage"] == "started":
                self.optimization_passes = row.get("plan", {}).get("rollout_optimization_passes")
            elif row["stage"] == "training":
                self.details(row)
            elif row["stage"] == "validation_completed":
                self.validation(row)
            elif row["stage"] == "completed":
                self.scalar("status/completed", 1, row["step"])
                self.scalar("status/optimizer_selected_step", row.get("selected_step", 0), row["step"])
        # The pass journal is the sole source of pass-specific series. The
        # duplicate optimization_pass_completed progress event is ignored.
        for row in self.jsonl("optimizer_passes.jsonl"):
            self.pass_details(row)
        for row in self.jsonl("updates.jsonl"):
            self.details(row)
        for row in self.jsonl("group_diagnostics.jsonl"):
            step = row["optimizer_update"]
            self.groups_seen[step] += 1
            # One series per batch position avoids collapsing groups that share
            # the same optimizer step in runs with multiple roots per batch.
            position = self.groups_seen[step]
            prefix = f"reward/group_{position}/"
            for key, values in (("total", row["returns"]), ("task", row["task_returns"])):
                self.scalar(prefix + key + "_mean", statistics.fmean(values), step)
                self.scalar(prefix + key + "_min", min(values), step)
                self.scalar(prefix + key + "_max", max(values), step)
            for key in ("std_task", "std_total", "semantic_action_diversity", "empirical_action_entropy_nats"):
                self.scalar(prefix + key, row[key], step)
            for phase, values in row.get("phase_returns", {}).items():
                phase_prefix = f"credit/group_{position}/{phase}/"
                self.scalar(phase_prefix + "return_mean", statistics.fmean(values), step)
                self.scalar(phase_prefix + "return_min", min(values), step)
                self.scalar(phase_prefix + "return_max", max(values), step)
                if phase in row.get("phase_std", {}):
                    self.scalar(phase_prefix + "return_std", row["phase_std"][phase], step)
        for row in self.jsonl("training_metrics.jsonl"):
            for tag, value in row["values"].items():
                self.scalar(tag, value, row["step"])
        # Detailed GRPO telemetry is persisted in training.json at completion.
        # Live loss and return curves above come directly from flushed journals.
        summary = self.source / "training.json"
        if summary.is_file() and summary.stat().st_mtime_ns != self.summary_mtime:
            try:
                result = json.loads(summary.read_text())
            except ValueError:
                return  # A concurrent atomic/partial writer may still be finishing.
            for row in result.get("history", []):
                if row["step"]:
                    self.details(row)
                if "validation" in row:
                    self.validation(row)
            self.summary_mtime = summary.stat().st_mtime_ns
        self.writer.flush()

    def close(self):
        self.writer.close()


def branches(root, patterns):
    for run in sorted(root.iterdir()):
        if not run.is_dir() or not any(run.match(pattern) for pattern in patterns):
            continue
        for directory, folders, files in os.walk(run):
            folders[:] = [name for name in folders if name not in PRUNE and not name.startswith("step_") and name != "last"]
            if "progress.jsonl" in files:
                yield Path(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("research/runs"))
    parser.add_argument("--match", nargs="+", default=["weak_sft_natural_*", "reward_review_*", "ai_feedback_*", "joint_reward_*"])
    parser.add_argument("--output", type=Path, default=Path("reports/training_monitor"))
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.once and args.serve:
        parser.error("--serve needs a live watcher, omit --once")
    root = args.runs.resolve()
    if not root.is_dir():
        parser.error("--runs must be an existing directory")
    # A fresh session prevents old and newly replayed events from being mixed.
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output.resolve() / session
    output.mkdir(parents=True)
    processes, exporters = [], {}
    metadata = {"pid": os.getpid(), "runs_root": str(root), "match": args.match,
                "logdir": str(output), "refresh_seconds": args.interval,
                "url": f"http://127.0.0.1:{args.port}" if args.serve else None,
                "scalar_identity": ["tag", "outer_step"],
                "duplicate_policy": "Emit each tag/step once per fresh export session",
                "optimization_pass_series_source": "optimizer_passes.jsonl only; duplicate progress events ignored",
                "optimization_pass_step": "outer_update, separated by optimization/pass_N tag",
                "legacy_optimization_series": "Last completed pass at each outer step",
                "phase_credit_series": "credit/group_N/PHASE/return_mean|min|max|std"}
    args.output.resolve().joinpath("CURRENT.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        while True:
            for source in branches(root, args.match):
                if source not in exporters:
                    exporters[source] = Exporter(source, output / source.relative_to(root))
                    print("Monitoring", source.relative_to(root), flush=True)
            for exporter in exporters.values():
                exporter.poll()
            if args.serve and not processes:
                process = subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", str(output),
                    "--host", "127.0.0.1", "--port", str(args.port), "--reload_interval", str(args.interval)])
                processes.append(process)
                print("Dashboard", metadata["url"], flush=True)
            if processes and processes[0].poll() is not None:
                raise RuntimeError("TensorBoard exited")
            if args.once:
                print("Exported", output, flush=True)
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        for exporter in exporters.values():
            exporter.close()
        for process in processes:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
