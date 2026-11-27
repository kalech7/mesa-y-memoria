"""Published LoRA assets for the held-out recommendation comparison."""
from __future__ import annotations

from functools import lru_cache
import json

from research.src.artifacts import file_hash

from .cases import Model, ROOT


RUN = ROOT / "research/runs/tesis_meal_lora_v1"
VARIED_RUN = RUN.parent / "tesis_meal_lora_varied_v1"
QA_RUN = RUN.parent / "tesis_meal_qa_v1"
CLARIFICATION_RUN = RUN.parent / "tesis_meal_clarification_v1"
EXPLORATION_RUN = RUN.parent / "tesis_meal_exploration_v1"
LABELS = {"sft": "SFT con exclusiones", "grpo": "SFT + GRPO con exclusiones"}


@lru_cache
def _published(run, results_hash, audit_hash):
    selection_path = run / "SELECTION.json"
    selection = json.loads(selection_path.read_bytes())
    audit = json.loads((run / "AUDIT.json").read_bytes())
    results = json.loads((run / "RESULTS.json").read_bytes())
    if (audit["status"] != "verified" or audit["results_sha256"] != results_hash
            or results["selection_sha256"] != file_hash(selection_path)
            or selection["plan_sha256"] != file_hash(run / "PLAN.json")):
        raise ValueError("La comparación de recomendaciones no coincide con su auditoría.")
    models = {}
    for method, label in LABELS.items():
        path = run / method
        item = selection["methods"][method]
        if (file_hash(path / "local_checkpoint.json") != item["checkpoint_manifest_sha256"]
                or file_hash(path / "adapter_model.safetensors") != item["adapter_sha256"]):
            raise ValueError("El adaptador de recomendaciones no coincide con su selección.")
        model = Model(f"comidas-{method}", label, f"LoRA · paso {item['step']}", path, run)
        models[model.id] = model
    report = {"cases": results["evaluation_cases"],
              "selection_steps": {method: selection["methods"][method]["step"] for method in LABELS},
              "metrics": {key: value["summary"]["all"] for key, value in results["panels"].items()},
              "difference": audit["grpo_minus_sft_by_dialogue"]}
    return models, report


@lru_cache
def _published_varied(results_hash, audit_hash):
    run = VARIED_RUN
    result_path = run / "RESULTS.json"
    selection_path = run / "SELECTION.json"
    audit = json.loads((run / "AUDIT.json").read_bytes())
    result = json.loads(result_path.read_bytes())
    selection = json.loads(selection_path.read_bytes())
    if (audit["status"] != "verified" or audit["results_sha256"] != results_hash
            or audit["selection_sha256"] != file_hash(selection_path)
            or result["selection_sha256"] != audit["selection_sha256"]
            or selection["plan_sha256"] != file_hash(run / "PLAN.json")):
        raise ValueError("Los resultados del menú reordenado difieren de su auditoría.")
    models = {}
    for method, chosen in selection["methods"].items():
        folder = run / method
        if (file_hash(folder / "local_checkpoint.json") != chosen["manifest_sha256"]
                or file_hash(folder / "adapter_model.safetensors") != chosen["adapter_sha256"]):
            raise ValueError("El adaptador de menú reordenado cambió.")
        checkpoint = f"LoRA · época {chosen['epoch']}" if chosen["epoch"] else "LoRA · SFT común"
        label = f"{LABELS[method]} · menú variado"
        model = Model(f"comidas-variado-{method}", label, checkpoint, folder, run)
        models[model.id] = model
    report = {"cases": audit["cases"]["evaluation"], "dialogues": audit["dialogues"]["evaluation"],
                    "pairs": audit["paired_evaluation_groups"], "uniform": audit["uniform_accuracy"],
                    "metrics": {key: item["metrics"] for key, item in result["panels"].items()},
                    "difference": result["grpo_minus_sft"],
                    "selection": {key: item["epoch"] for key, item in selection["methods"].items()}}
    return models, report


@lru_cache
def _published_qa(results_hash, audit_hash):
    run = QA_RUN
    result = json.loads((run / "RESULTS.json").read_bytes())
    audit = json.loads((run / "AUDIT.json").read_bytes())
    selection_path = run / "SELECTION.json"
    selection = json.loads(selection_path.read_bytes())
    if (audit["status"] != "verified" or audit["results_sha256"] != results_hash
            or audit["selection_sha256"] != file_hash(selection_path)
            or result["selection_sha256"] != audit["selection_sha256"]
            or selection["plan_sha256"] != file_hash(run / "PLAN.json")):
        raise ValueError("El entrenamiento con conversaciones de QA difiere de su auditoría.")
    models = {}
    for method, chosen in selection["methods"].items():
        folder = run / method
        if (file_hash(folder / "local_checkpoint.json") != chosen["manifest_sha256"]
                or file_hash(folder / "adapter_model.safetensors") != chosen["adapter_sha256"]):
            raise ValueError("El adaptador de conversaciones de QA cambió.")
        checkpoint = f"LoRA · época {chosen['epoch']}" if chosen["epoch"] else "LoRA · SFT previo"
        label = f"{LABELS[method]} · cambios y erratas"
        model = Model(f"comidas-variado-qa-{method}", label, checkpoint, folder, run)
        models[model.id] = model
    selection_epochs = {method: item["epoch"] for method, item in selection["methods"].items()}
    plan = json.loads((run / "PLAN.json").read_bytes())
    return models, {"panels": result["panels"], "selection": selection_epochs, "seed": plan["seed"],
                    "device": plan["device"]}


@lru_cache
def _published_clarification(run, prefix, freeze_hash, audit_hash):
    freeze = json.loads((run / "FREEZE.json").read_bytes())
    if freeze["plan_sha256"] != file_hash(run / "PLAN.json"):
        raise ValueError("El plan de aclaración no coincide con los pesos congelados.")
    models = {}
    selected = {}
    for seed, selection_hash in freeze["selection_sha256"].items():
        selection_path = run / seed / "SELECTION.json"
        selection = json.loads(selection_path.read_bytes())
        if (file_hash(selection_path) != selection_hash or selection["split"] != "validation"
                or selection["plan_sha256"] != freeze["plan_sha256"]):
            raise ValueError("La selección de modelos de aclaración cambió.")
        selected.update({f"{seed}-{method}": spec["epoch"] for method, spec in selection["methods"].items()})
        methods = [("sft", "SFT"), ("grpo", "SFT + GRPO")]
        if "grpo_sampled" in selection["methods"]:
            methods.append(("grpo_sampled", "GRPO · muestreo"))
        for method, label in methods:
            chosen = selection["methods"][method]
            path = run / seed / method
            if (file_hash(path / "local_checkpoint.json") != chosen["manifest_sha256"]
                    or file_hash(path / "adapter_model.safetensors") != chosen["adapter_sha256"]):
                raise ValueError("Un adaptador de aclaración cambió.")
            model_id = f"comidas-variado-{prefix}-{method}-{seed}"
            weights = "pesos iniciales" if chosen["epoch"] == 0 else f"época {chosen['epoch']}"
            variant = {"grpo": " · seis acciones", "grpo_sampled": " · muestreo"}.get(method, "") if prefix == "exploracion" else ""
            model = Model(model_id, label, f"Semilla {seed} · {weights}{variant}", path, run, training_epoch=chosen["epoch"])
            models[model_id] = model
    report = None
    if audit_hash:
        audit = json.loads((run / "AUDIT.json").read_bytes())
        result_path = run / "RESULTS.json"
        result = json.loads(result_path.read_bytes())
        if (audit["status"] != "verified" or audit["results_sha256"] != file_hash(result_path)
                or result["freeze_sha256"] != freeze_hash):
            raise ValueError("La evaluación de aclaraciones difiere de su auditoría.")
        report = {"device": result["device"], "selection": selected,
                  "metrics": {key: item["reserved"]["live_order"] for key, item in result["profiles"].items()}}
    return models, report


def meal_models(run=RUN):
    output, audit = run / "RESULTS.json", run / "AUDIT.json"
    if not (output.is_file() and audit.is_file()):
        return {}, None
    models, report = _published(run, file_hash(output), file_hash(audit))
    position = run / "POSITION.json"
    if position.is_file():
        measured = json.loads(position.read_bytes())
        if measured["results_sha256"] != file_hash(output):
            raise ValueError("La prueba de rotación pertenece a otra comparación.")
        report = {**report, "position": measured["metrics"]}
    independent = run.parent / "tesis_meal_lora_v2"
    if run == RUN and (independent / "AUDIT.json").is_file():
        checked = json.loads((independent / "AUDIT.json").read_bytes())
        measured = json.loads((independent / "RESULTS.json").read_bytes())
        if checked["status"] != "verified" or checked["results_sha256"] != file_hash(independent / "RESULTS.json"):
            raise ValueError("La prueba por ingredientes difiere de su auditoría.")
        selection = json.loads((independent / "SELECTION.json").read_bytes())
        report = {**report, "independent": {"cases": measured["cases"],
                  "metrics": {key: item["metrics"] for key, item in measured["panels"].items()},
                  "selection_steps": {key: item["step"] for key, item in selection["methods"].items()},
                  "difference": measured["grpo_minus_sft"]}}
    if run == RUN and (VARIED_RUN / "AUDIT.json").is_file():
        result_hash = file_hash(VARIED_RUN / "RESULTS.json")
        varied, metrics = _published_varied(result_hash, file_hash(VARIED_RUN / "AUDIT.json"))
        shuffle = VARIED_RUN / "SHUFFLE.json"
        if shuffle.is_file():
            checked = json.loads(shuffle.read_bytes())
            if checked["results_sha256"] != result_hash:
                raise ValueError("Las permutaciones corresponden a otra evaluación.")
            metrics = {**metrics, "shuffle": checked["metrics"]}
        models = {**varied, **models}
        report = {**report, "varied": metrics}
    if run == RUN and (QA_RUN / "AUDIT.json").is_file():
        qa_models, metrics = _published_qa(file_hash(QA_RUN / "RESULTS.json"), file_hash(QA_RUN / "AUDIT.json"))
        models = {**qa_models, **models}
        report = {**report, "qa": metrics}
    if run == RUN:
        for directory, profile, key in ((CLARIFICATION_RUN, "aclaracion", "clarification"),
                                        (EXPLORATION_RUN, "exploracion", "exploration")):
            freeze, audit = directory / "FREEZE.json", directory / "AUDIT.json"
            if not freeze.is_file():
                continue
            advanced, metrics = _published_clarification(directory, profile, file_hash(freeze),
                                                         file_hash(audit) if audit.is_file() else None)
            models = {**advanced, **models}
            if metrics:
                report = {**report, key: metrics}
    return models, report
