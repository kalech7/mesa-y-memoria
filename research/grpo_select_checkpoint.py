"""Load a verified copy of the pilot's base model without altering checkpoints."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_local_001.model import artifact_hashes
from research.legal_selection_policy import LegalSelectionPolicy
from research.src.artifacts import write_json


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _without_empty_download_locks(hashes):
    return {name: value for name, value in hashes.items()
            if not (name.startswith(".cache/huggingface/download/") and name.endswith(".lock")
                    and value == _EMPTY_SHA256)}


def load_selector(checkpoint, *, device, trainable=False, model_path=None):
    """Route a verified relocation through the existing audited loader.

    Older manifests hash transient empty Hugging Face download locks. A copy may
    lack those locks, while all other files must keep their recorded hashes.
    The temporary manifest stays valid for the standard loader and the model's
    origin still identifies the untouched original checkpoint.
    """
    if model_path is None:
        return LegalSelectionPolicy.from_checkpoint(checkpoint, device=device, trainable=trainable)

    checkpoint, model_path = Path(checkpoint).resolve(), Path(model_path).resolve()
    manifest = json.loads((checkpoint / "local_checkpoint.json").read_text(encoding="utf-8"))
    if manifest["parameters"]["training_scope"] != "lora_adapters_integrated_in_llm":
        raise ValueError("A relocated base is only valid for integrated LoRA")
    adapter_hashes = artifact_hashes(checkpoint)
    if adapter_hashes != manifest["artifacts_sha256"]:
        raise ValueError("Checkpoint artifact hash mismatch")
    actual_base = artifact_hashes(model_path)
    expected_base = manifest["base_artifacts_sha256"]
    if not actual_base or _without_empty_download_locks(actual_base) != _without_empty_download_locks(expected_base):
        raise ValueError("Relocated model differs from the recorded base artifacts")

    manifest["config"]["model_path"] = str(model_path)
    manifest["base_artifacts_sha256"] = actual_base
    with TemporaryDirectory(prefix="grpo-selector-checkpoint-") as work:
        view = Path(work)
        for name in adapter_hashes:
            artifact = view / name
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.symlink_to(checkpoint / name)
        write_json(view / "local_checkpoint.json", manifest)
        policy = LegalSelectionPolicy.from_checkpoint(view, device=device, trainable=trainable)
    policy.origin_checkpoint = str(checkpoint)
    return policy
