"""Public training entrypoints reject leakage before creating a model or run."""
from copy import deepcopy
import json

import pytest

from research.src.closure_environment import _episode
from research.src.data.semantic_partition_v2 import SemanticSplitLeakage
from research.src.training import campaign
from research.src.closure_training import _read_training_banks


def aliases():
    train, validation = _episode(), _episode()
    train.update(split="train", root_id="training_root")
    validation.update(split="validation", root_id="different_validation_root", latent_world_id="renamed_world")
    return train, validation


@pytest.mark.parametrize("entrypoint", ["sft", "continuations"])
def test_campaign_rejects_content_leakage_before_loading_model_or_writing_run(entrypoint, tmp_path, monkeypatch):
    train, validation = aliases()
    monkeypatch.setattr(campaign, "require_gate", lambda *a: None)
    monkeypatch.setattr(campaign, "load_config", lambda *a: {})
    monkeypatch.setattr(campaign, "load_bank", lambda path, split: [train if split == "train" else validation])
    def forbidden(*a, **kw):
        raise AssertionError("A leaking bank must be rejected before loading a model")
    monkeypatch.setattr(campaign, "SpecLocalPolicy", forbidden)
    destination = tmp_path / "must_not_exist"
    with pytest.raises(SemanticSplitLeakage):
        if entrypoint == "sft":
            campaign.run_sft_campaign("model", "train", "validation", destination, root=tmp_path)
        else:
            campaign.run_continuations("sft.json", "train", "validation", destination, root=tmp_path)
    assert not destination.exists()


def test_pilot_checks_all_validation_roots_before_selecting_a_subset(tmp_path):
    train, leaked = aliases()
    safe = deepcopy(leaked)
    safe["root_id"] = "safe_root_first"
    safe["user_messages"][0]["text"] = "Ahora prefiero arroz. ¿Qué ceno hoy?"
    safe["truth"].update(current_value="arroz", future_value="arroz")
    train_path, validation_path = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train_path.write_text(json.dumps(train) + "\n")
    validation_path.write_text(json.dumps(safe) + "\n" + json.dumps(leaked) + "\n")
    with pytest.raises(SemanticSplitLeakage):
        _read_training_banks(train_path, validation_path, validation_roots=1)


def test_data_audit_cannot_certify_renamed_historical_content_as_disjoint():
    from research.src.data.audit_splits import audit_splits
    from research.src.data.build_bank import build_bank
    train = deepcopy(build_bank()[0])
    validation = deepcopy(train)
    train.update(split="train", root_id="train_root")
    validation.update(split="validation", root_id="another_root")
    with pytest.raises(SemanticSplitLeakage):
        audit_splits({"train": [train], "validation": [validation]})
