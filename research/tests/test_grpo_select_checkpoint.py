import json
from types import SimpleNamespace

import pytest

from llm_local_001.model import artifact_hashes
from research.grpo_select_checkpoint import LegalSelectionPolicy, load_selector


def test_relocation_ignores_only_empty_download_locks(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.mkdir()
    (original / "model.safetensors").write_bytes(b"weights")
    lock_dir = original / ".cache/huggingface/download"
    lock_dir.mkdir(parents=True)
    (lock_dir / "config.json.lock").touch()
    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / "model.safetensors").write_bytes(b"weights")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    manifest = {
        "config": {"model_path": str(original)},
        "parameters": {"training_scope": "lora_adapters_integrated_in_llm"},
        "artifacts_sha256": artifact_hashes(checkpoint),
        "base_artifacts_sha256": artifact_hashes(original),
    }
    manifest_path = checkpoint / "local_checkpoint.json"
    manifest_path.write_text(json.dumps(manifest))

    def from_checkpoint(view, **options):
        loaded = json.loads((view / "local_checkpoint.json").read_text())
        assert loaded["config"]["model_path"] == str(copied)
        assert loaded["base_artifacts_sha256"] == artifact_hashes(copied)
        assert loaded["artifacts_sha256"] == artifact_hashes(view)
        assert options == {"device": "cpu", "trainable": True}
        return SimpleNamespace(temporary_view=view)

    monkeypatch.setattr(LegalSelectionPolicy, "from_checkpoint", from_checkpoint)
    model = load_selector(checkpoint, device="cpu", trainable=True, model_path=copied)
    assert model.origin_checkpoint == str(checkpoint)
    assert not model.temporary_view.exists()
    assert json.loads(manifest_path.read_text()) == manifest

    (copied / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Relocated model differs"):
        load_selector(checkpoint, device="cpu", model_path=copied)
    (copied / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"changed adapter")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_selector(checkpoint, device="cpu", model_path=copied)
