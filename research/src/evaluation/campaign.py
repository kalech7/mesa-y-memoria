"""Frozen test factory matrix. Each training seed loads its own selected adapter."""
from pathlib import Path
import json

from ..artifacts import ROOT, load_config
from ..protocol import require_gate
from ..policy.llm_controller import SpecLocalPolicy, LLMController
from ..policy.rule_controller import RuleController
from .run_closed_test import run_closed_test


def closed_campaign(campaign, destination, *, root=None):
    root = Path(root or ROOT)
    require_gate("G4", root)
    manifest = load_config(root / "manifests/selected_checkpoints.json")
    checkpoints = {(c["method"], c["seed"]): (root / c["adapter_file"]).parent for c in manifest["checkpoints"]}
    factories = {"B4": lambda *, seed: RuleController()}
    config = load_config(root / "configs/protocol.yaml")
    device = campaign.get("device", config.get("runtime", {}).get("device", "cuda"))

    def base(variant=None):
        def factory(*, seed):
            backend = SpecLocalPolicy(campaign["model_path"], device=device,
                                      precision=config["model"]["precision"], lora=False, seed=seed)
            return LLMController(backend, prompt_variant=variant)
        return factory
    factories["B0"], factories["B0-P"] = base(), base("B0-P")
    for method in ("B1", "B2-L", "B3", "B2-I", "C-NEVER", "C-ALWAYS"):
        def factory(*, seed, _method=method):
            checkpoint_method = "B1" if _method.startswith("C-") else _method
            backend = SpecLocalPolicy.from_checkpoint(checkpoints[(checkpoint_method, seed)], trainable=False, device=device)
            return LLMController(backend)
        factories[method] = factory
    episodes = []
    for path in campaign["test_banks"]:
        episodes.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line)
    traces = run_closed_test(episodes, factories, destination, root=root)
    from ..reporting import export_results
    summary = export_results(traces, Path(destination) / "RESULTADOS.csv")
    return {"status": "executed", "traces": str(traces), **summary}
