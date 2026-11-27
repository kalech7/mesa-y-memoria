"""One frozen test campaign. No training, tuning or checkpoint selection API."""
from __future__ import annotations

from pathlib import Path
from ..protocol import require_gate
from ..artifacts import ROOT, canonical, file_hash, now, write_json
from .runner import run_attempts


def run_closed_test(episodes, policy_factories, destination, *, root=None, seeds=(101, 202, 303), reader=None):
    require_gate("G4", root)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    marker = Path(root or ROOT) / "manifests/test_opening.json"
    if marker.exists():
        raise RuntimeError("TEST_ALREADY_OPENED: preserve the original campaign")
    if any(not episode["split"].startswith("test") for episode in episodes):
        raise ValueError("Closed evaluation only accepts test partitions")
    for factory in policy_factories.values():
        # Factories must return frozen controllers; the runner has no optimizer.
        controller = factory(seed=seeds[0])
        backend = getattr(controller, "backend", None)
        if backend is not None and any(p.requires_grad for p in backend.model.parameters()):
            raise ValueError("TEST_TIME_WEIGHT_UPDATES_FORBIDDEN: freeze all model parameters")
        del controller, backend
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x") as stream:
        stream.write(canonical({"opened_at": now(), "status": "opened", "methods": list(policy_factories),
                                "seeds": list(seeds), "destination": str(destination.resolve())}) + "\n")
    path = destination / "TRAZAS.jsonl"
    with path.open("x") as stream:
        for seed in seeds:
            for method, factory in policy_factories.items():
                controller = factory(seed=seed)
                backend = getattr(controller, "backend", None)
                if backend is not None and any(p.requires_grad for p in backend.model.parameters()):
                    raise ValueError("TEST_TIME_WEIGHT_UPDATES_FORBIDDEN")
                for episode in episodes:
                    row = run_attempts(episode, lambda: controller, method=method, rollout_seed=seed, reader=reader, root=root)
                    stream.write(canonical(row) + "\n")
                    stream.flush()
                del controller, backend
    write_json(destination / "test_completion.json", {"status": "executed", "completed_at": now(), "trace_sha256": file_hash(path)})
    return path
