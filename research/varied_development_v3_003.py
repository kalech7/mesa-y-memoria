"""Neutralize nine assistant proposals in frozen v3.1 development revision002.

Only assistant text changes. Episode metadata/IDs remain source provenance;
the new manifest identifies revision003 and computes identity from actual text.
This replacement revision is AI-authored development, never an independent test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from .src.data.semantic_partition_v3 import assert_semantic_split_isolation_v3, semantic_episode_id_v3

VERSION = "varied_scripted_development_v3.2_neutral_assistant"
NAMESPACE = "development_varied_v3_20261025_003"
NEUTRAL_QUESTIONS = (
    "¿Quieres que revisemos tus opciones?",
    "¿Te puedo orientar con lo que ya me contaste?",
    "¿Qué te gustaría consultar sobre tus preferencias?",
    "¿Quieres algunas sugerencias?",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def neutralize_assistant_proposals(episodes):
    revised = deepcopy(episodes)
    changes = []
    for index, episode in enumerate(revised):
        if episode["authoring"]["stratum"] != "unrelated_attribute_control":
            continue
        matches = [m for m in episode["user_messages"]
                   if m["role"] == "assistant" and m["text"].startswith("¿Quieres que anote «")]
        if not matches:
            continue
        if len(matches) != 1 or "prefiero" not in matches[0]["text"]:
            raise ValueError("Unrecognized parent proposal; require an explicit reviewed recipe")
        message = matches[0]
        previous = message["text"]
        message["text"] = NEUTRAL_QUESTIONS[len(changes) % len(NEUTRAL_QUESTIONS)]
        changes.append({"episode_index": index, "root_id": episode["root_id"],
                        "turn_id": message["turn_id"], "role": "assistant",
                        "old_text": previous, "new_text": message["text"]})
    if len(changes) != 9 or len({c["root_id"] for c in changes}) != 9:
        raise ValueError("Frozen002 must contain exactly nine assistant preference proposals")
    return revised, changes


def prepare_revision(parent_directory, data_directory, report_directory, *,
                     historical_directory="research/data/bank_1_1",
                     previous_development="research/data/development_v2_20261024_001/validation.jsonl",
                     parent_generator="research/varied_development_v3_002.py"):
    parent_directory, data_directory, report_directory = map(Path, (parent_directory, data_directory, report_directory))
    parent_bank, parent_manifest_path = parent_directory / "validation.jsonl", parent_directory / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if sha(parent_bank) != parent_manifest["bank_sha256"] or sha(parent_generator) != parent_manifest["producer_sha256"]:
        raise ValueError("Parent002 bank or producer differs from its frozen manifest")
    data_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    if (data_directory / "validation.jsonl").exists() or (data_directory / "manifest.json").exists():
        raise FileExistsError("Preserve existing revisions; select a new output version")
    inputs = {f"legacy_{s}": Path(historical_directory) / f"{s}.jsonl"
              for s in ("dev_smoke", "train", "validation", "test_id", "test_ood", "test_external")}
    inputs["development_v2"] = Path(previous_development)
    protected = {**inputs, "parent_bank": parent_bank, "parent_manifest": parent_manifest_path,
                 "parent_generator": Path(parent_generator),
                 "ancestor001_bank": Path(parent_manifest["parent_bank_path"]),
                 "ancestor001_generator": Path("research/varied_development.py")}
    before = {name: sha(path) for name, path in protected.items()}
    snapshots = report_directory / "source_snapshot"
    snapshots.mkdir(exist_ok=True)
    for name, path in {**protected, "revision_generator": Path(__file__)}.items():
        target = snapshots / (name + path.suffix)
        if target.exists() and sha(target) != sha(path):
            raise ValueError("Source snapshot already contains another version")
        if not target.exists():
            shutil.copyfile(path, target)
    original_lines = parent_bank.read_text().splitlines(keepends=True)
    original = [json.loads(s) for s in original_lines]
    revised, changes = neutralize_assistant_proposals(original)
    changed_indices = {c["episode_index"] for c in changes}
    for index, (old, new) in enumerate(zip(original, revised)):
        restored = deepcopy(new)
        restored["user_messages"] = deepcopy(old["user_messages"])
        assert restored == old
        for before_message, after_message in zip(old["user_messages"], new["user_messages"]):
            assert {k:v for k,v in before_message.items() if k != "text"} == {k:v for k,v in after_message.items() if k != "text"}
            if before_message["role"] == "user":
                assert before_message == after_message
        assert (old != new) == (index in changed_indices)
    historical = {name:[json.loads(line) for line in path.open()] for name,path in inputs.items()}
    isolation = assert_semantic_split_isolation_v3({"validation":revised}, historical_splits=historical)
    signatures = {semantic_episode_id_v3(e) for e in original}
    overlap = sum(semantic_episode_id_v3(e) in signatures for e in revised)
    assert overlap == 51
    ancestor_rows = [json.loads(s) for s in protected["ancestor001_bank"].open()]
    ancestor_signatures = {semantic_episode_id_v3(e) for e in ancestor_rows}
    ancestor_overlap = sum(semantic_episode_id_v3(e) in ancestor_signatures for e in revised)
    bank = data_directory / "validation.jsonl"
    with bank.open("x") as stream:
        for index, episode in enumerate(revised):
            # Preserve the untouched 51 physical lines exactly, not just JSON equality.
            stream.write(original_lines[index] if index not in changed_indices else json.dumps(episode, ensure_ascii=False, sort_keys=True) + "\n")
    after = {name:sha(path) for name,path in protected.items()}
    assert before == after
    manifest = deepcopy(parent_manifest)
    manifest.update(version=VERSION, data_namespace=NAMESPACE, bank_path=str(bank.resolve()), bank_sha256=sha(bank),
        producer_sha256=sha(__file__), parent_bank_path=str(parent_bank.resolve()), parent_bank_sha256=before["parent_bank"],
        parent_generator_sha256=before["parent_generator"], parent_manifest_sha256=before["parent_manifest"],
        lineage={"parent_bank":str(parent_bank.resolve()), "parent_bank_sha256":before["parent_bank"],
                 "relationship":"revision_of_same_corpus_not_independent", "model_inference_calls_before_revision":0,
                 "ancestor_chain":[deepcopy(parent_manifest["lineage"])]},
        replaces_namespace=parent_manifest["data_namespace"], independent_of_parent=False,
        revision_changes=9, changed_roots=9, changed_fields="user_messages[i].text for nine assistant turns only",
        unchanged_physical_jsonl_lines=51, expected_parent_semantic_overlap_episodes=overlap,
        ancestor001_semantic_overlap_episodes=ancestor_overlap, root_overlap_with_parent=50,
        isolation=isolation, protected_sources_before=before, protected_sources_after=after,
        episode_metadata_policy="All episode metadata and IDs retain parent002 provenance unchanged; manifest identifies revision003. Observable/semantic identities are recomputed from content by the v3 guard, not taken from stored IDs.",
        revision_selection_basis="Remove audited pragmatic ambiguity of assistant proposals before any B1 varied-v3 inference; not selection on model results.",
        neutral_assistant_questions=list(NEUTRAL_QUESTIONS), new_Qwen_calls=0, training_steps=0)
    manifest["limits"] = [s for s in parent_manifest["limits"] if not s.startswith(("Revision002", "O1 support-loss"))] + [
        "Revision003 replaces002 and preserves its full root lineage; never pool versions as independent users.",
        "Only nine assistant questions become neutral. User statements, memories, dates, golds and all other episode fields are identical to002.",
        "B4 metamorphic verification checks contract invariance; it is not B1 evaluation or human validation."]
    (data_directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (report_directory / "revision_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (report_directory / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-directory", default="research/data/development_varied_v3_20261025_002")
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--report-directory", required=True)
    args = parser.parse_args(argv)
    manifest = prepare_revision(args.parent_directory, args.data_directory, args.report_directory)
    print(json.dumps({k:manifest[k] for k in ("version", "roots", "episodes", "bank_path", "bank_sha256", "changed_roots", "unchanged_physical_jsonl_lines")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
