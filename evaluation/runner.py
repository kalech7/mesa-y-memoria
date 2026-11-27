import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agents.client import API, TechnicalError, http_json
from agents.policies import LMPolicy, Heuristic, build_messages, NEUTRAL, GUIDED, normalize
from data_generation.build import make_dataset, write_dataset, VARIANTS
from data_generation.schema import digest, validate_dataset, validate_public
from evaluation.metrics import summarize
from evaluation.scoring import score_answer, JUDGE_PROTOCOL
from memory.extraction import Extractor, EXTRACTION
from memory.retrieval import BGE, retrieve, pair_recall
from memory.store import Store
from simulator.user import UserSimulator

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def append_jsonl(path, obj):
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")
    temporary.replace(path)


def balanced_select(cases, gold, split, limit, seed):
    indices = [i for i, c in enumerate(cases) if split == "all" or c["split"] == split]
    # Rotar historias entre variantes: un smoke de 5 casos no es cinco variantes de una historia.
    ordered, used = [], set()
    for variant in VARIANTS:
        group = sorted([i for i in indices if gold[i]["variant"] == variant], key=lambda i: digest([seed, cases[i]["case_id"]]))
        fresh = [i for i in group if cases[i]["person_id"] not in used]
        if fresh:
            ordered.append(fresh[0])
            used.add(cases[fresh[0]]["person_id"])
    ordered += sorted(set(indices) - set(ordered), key=lambda i: digest([seed, cases[i]["case_id"]]))
    ordered = ordered[:limit] if limit else ordered
    return [(cases[i], gold[i]) for i in ordered]


def run_case(case, annotation, policy, condition, directory, retrieval="full", topk=8,
             embedder=None, extractor=None, judge=None, two_turn=True, max_questions=2,
             persist_clarification=True, future_noise=True, memory_mode="raw", recorder=None):
    validate_public(case)
    dbpath = directory / "db.sqlite"
    directory.mkdir(parents=True, exist_ok=False)
    store = Store(dbpath, case["person_id"])
    case_start = time.monotonic()
    row = {"case_id": case["case_id"], "person_id": case["person_id"], "condition": condition,
        "kind": annotation["kind"], "variant": annotation["variant"], "split": case["split"],
        "historical_conflict": annotation["historical_conflict"], "unresolved_conflict": annotation["unresolved_conflict"],
        "acceptable_actions": annotation["acceptable_actions"], "human_gold_validated": annotation["annotation"]["validated"],
        "questions": [], "technical_errors": [], "extraction_audit": [], "api_calls": [],
        "confirmation_memory_saved": False, "clarification_resolved": False,
        "general_preference_confirmed": False, "end_to_end_success": False,
        "retrieval": retrieval, "topk": topk, "memory_mode": memory_mode,
        "history_messages": len(case["history"]), "history_months": round((datetime.fromisoformat(case["reference_time"]) - datetime.fromisoformat(case["history"][0]["at"])).days / 30.44, 1),
        "future_noise": future_noise, "persist_clarification": persist_clarification}
    stage = "memory"
    records, recent, last = [], [], None
    try:
        prior = ""
        for message in case["history"]:
            if extractor:
                if message["role"] == "user":
                    fragments, audit = extractor.extract(message, prior)
                    row["extraction_audit"].append({"origin_id": message["message_id"], **audit})
                    for n, text in enumerate(fragments):
                        store.add(message, text=text, event_id=digest([message["message_id"], n])[:24], source=message["source"] + ":extracted")
            else:
                store.add(message)
            prior = message["text"]
        row["snapshot_before_ask"] = store.snapshot()
        store.backup(directory / "before_ask.sqlite")
        stage = "retrieval"
        records = retrieve(store.all(), case["query"], retrieval, topk, embedder)
        row["initial_retrieved_memories"] = [{k: m[k] for k in ("at", "source", "role", "text", "origin_id", "status")} for m in records]
        row["retrieved_origin_ids"] = [r["origin_id"] for r in records]
        row["retrieval_pair_recall"] = pair_recall(records, annotation)
        row["memories_total"], row["memories_retrieved"] = len(store.all()), len(records)
        # La consulta actual es un mensaje real; puede comunicar una decisión duradera.
        # Se guarda DESPUÉS de recuperar para no duplicarla artificialmente en el prompt inicial.
        current_message = {"message_id": digest([case["case_id"], "current-query"])[:24],
            "session_id": digest([case["case_id"], "present-session"])[:16], "at": case["reference_time"],
            "role": "user", "channel": "chat", "source": "current_session", "text": case["query"]}
        if extractor:
            current_fragments, audit = extractor.extract(current_message)
            row["extraction_audit"].append({"origin_id": current_message["message_id"], **audit})
            for n, fragment in enumerate(current_fragments):
                store.add(current_message, text=fragment, event_id=digest([current_message["message_id"], n])[:24])
        else:
            store.add(current_message)
        stage = "initial"
        last, _ = policy.decide(case["query"], records, case["reference_time"])
        row.update({"initial_" + k: v for k, v in last.items()})
        row["action_correct"] = last["action"] in annotation["acceptable_actions"]
        known_public = "ahora quiero" in normalize(case["query"]) or any(
            "corrijo lo anterior" in normalize(m["text"]) for m in case["history"] if m["role"] == "user")
        simulator = UserSimulator(annotation["hidden_state"], preference_already_communicated=known_public)
        if last["action"] != "ASK":
            stage = "initial_scoring"
            # La verdad latente no autoriza evaluar ANSWER sin haber sido comunicada.
            if not annotation["unresolved_conflict"]:
                row["initial_score"] = score_answer(last["answer"], annotation["hidden_state"], case["query"], judge)
        turn = 0
        while two_turn and last["action"] == "ASK" and turn < max_questions:
            stage = "simulation"
            reply = simulator.respond(last["question"], case["query"])
            question_row = {"question": last["question"], **asdict(reply)}
            row["questions"].append(question_row)
            turn += 1
            recent.extend([{"role": "assistant", "content": json.dumps(last, ensure_ascii=False)},
                           {"role": "user", "content": reply.text}])
            if reply.resolved:
                row["clarification_resolved"] = True
                row["general_preference_confirmed"] |= reply.applicability == "current_scope"
                message = {"message_id": digest([case["case_id"], "clarification", turn])[:24],
                    "session_id": digest([case["case_id"], "present-session"])[:16],
                    "at": (datetime.fromisoformat(case["reference_time"]) + timedelta(minutes=turn)).isoformat(),
                    "role": "user", "channel": "chat", "source": "clarification", "text": reply.text}
                stage = "clarification_storage"
                fragments = [reply.text]
                if extractor:
                    fragments, audit = extractor.extract(message, last["question"])
                    row["extraction_audit"].append({"origin_id": message["message_id"], **audit})
                if persist_clarification and fragments:
                    # No usar evidencia gold para decidir vínculos/operaciones de memoria.
                    # Vínculos de procedencia con notas recuperadas; no se inactivan todas:
                    # no existe un reconciliador semántico validado que autorice hacerlo.
                    links = [m["event_id"] for m in records if m["role"] == "user"]
                    for n, fragment in enumerate(fragments):
                        store.add(message, text=fragment, event_id=digest([message["message_id"], n])[:24], linked_ids=links)
                    row["confirmation_memory_saved"] = True
            stage = "after_clarification"
            last, _ = policy.decide(case["query"], records, case["reference_time"], recent)
            row["after_action"], row["after_answer"] = last["action"], last["answer"]
            if row["clarification_resolved"] and last["action"] != "ASK":
                stage = "after_scoring"
                score = score_answer(last["answer"], annotation["hidden_state"], case["query"], judge)
                row["after_fidelity"], row["after_task_success"] = score["fidelity"], score["task_success"]
                row["after_score"] = score
        row["question_limit_reached"] = bool(last and last["action"] == "ASK" and turn == max_questions)
        if two_turn:
            stage = "future_session"
            if future_noise:
                for message in case["future_distractors"]:
                    if extractor:
                        fragments, audit = extractor.extract(message)
                        row["extraction_audit"].append({"origin_id": message["message_id"], **audit})
                        for n, text in enumerate(fragments):
                            store.add(message, text=text, event_id=digest([message["message_id"], n])[:24])
                    else:
                        store.add(message)
            # Reabrir la BD: verificar persistencia real, no solo un objeto Python vivo.
            store.close()
            store = Store(dbpath, case["person_id"])
            future_records = retrieve(store.all(), case["followup"], retrieval, topk, embedder)
            row["followup_retrieved_origin_ids"] = [m["origin_id"] for m in future_records]
            row["followup_confirmation_retrieved"] = any(m["source"] == "clarification" for m in future_records)
            future_reference = (datetime.fromisoformat(case["reference_time"]) + timedelta(days=3)).isoformat()
            row["followup_prompt_hash"] = digest(build_messages(case["followup"], future_records, future_reference))
            # Nueva llamada, recent=None: ni pregunta anterior ni respuesta del simulador en contexto.
            final, _ = policy.decide(case["followup"], future_records, future_reference)
            row["followup_action"], row["followup_answer"] = final["action"], final["answer"]
            if final["action"] != "ASK" and (row["general_preference_confirmed"] or not annotation["unresolved_conflict"]):
                stage = "future_scoring"
                score = score_answer(final["answer"], annotation["hidden_state"], case["followup"], judge)
                row["followup_fidelity"], row["followup_task_success"] = score["fidelity"], score["task_success"]
                row["followup_score"] = score
        row["end_to_end_success"] = bool(row["clarification_resolved"] and row["general_preference_confirmed"] and row["confirmation_memory_saved"]
            and row.get("after_fidelity") == "pass" and row.get("after_task_success") == "pass"
            and row.get("followup_fidelity") == "pass" and row.get("followup_task_success") == "pass"
            and row.get("followup_confirmation_retrieved") and row.get("followup_action") != "ASK")
    except TechnicalError as exc:
        row["technical_errors"].append({"stage": stage, "kind": exc.kind, "http_status": exc.status})
    except Exception as exc:
        row["technical_errors"].append({"stage": stage, "kind": "internal_pipeline_error", "exception_type": type(exc).__name__})
    finally:
        row["case_wall_seconds"] = time.monotonic() - case_start
        row["snapshot_after"] = store.snapshot()
        store.close()
    return row


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_reports(out, rows, public_by_id, seed):
    metrics = summarize(rows, seed)
    atomic_json(out / "metrics.json", metrics)
    errors = [{"case_id": r["case_id"], "condition": r["condition"], **e} for r in rows for e in r["technical_errors"]]
    write_csv(out / "errors.csv", errors, ["case_id", "condition", "stage", "kind", "http_status"])
    review, key = [], []
    for row in rows:
        case = public_by_id[row["case_id"]]
        review_id = digest([row["case_id"], row["condition"], "blind-v2"])[:20]
        review.append({"review_id": review_id, "history": json.dumps(case["history"], ensure_ascii=False),
            "query": case["query"], "action": row.get("initial_action"), "question": row.get("initial_question", ""),
            "retrieved_evidence": json.dumps(row.get("initial_retrieved_memories", []), ensure_ascii=False),
            "initial_answer": row.get("initial_answer", ""), "dialogue": json.dumps([
                {"question": q["question"], "reply": q["text"]} for q in row["questions"]], ensure_ascii=False),
            "after_answer": row.get("after_answer", ""), "followup": case["followup"],
            "followup_answer": row.get("followup_answer", ""), "question_relevance_human": "",
            "information_gain_human": "", "question_useful_human": "", "answer_grounded_human": "", "task_success_human": "",
            "after_grounded_human": "", "after_task_success_human": "", "followup_task_success_human": "",
            "followup_consistency_human": "", "reviewer": "", "uncertain": "", "notes": ""})
        key.append({"review_id": review_id, "case_id": row["case_id"], "condition": row["condition"]})
    import random
    random.Random(seed).shuffle(review)
    review_path = out / "human_review_blind.csv"
    if review_path.exists():
        # Preservar anotaciones humanas y actualizar solo registros nuevos.
        with review_path.open(encoding="utf8", newline="") as stream:
            previous = list(csv.DictReader(stream))
        existing = {item["review_id"]: item for item in previous}
        review = [existing.get(item["review_id"], item) for item in review]
    if review:
        write_csv(review_path, review, list(review[0]))
    atomic_json(out / "private" / "review_key.json", key)
    comparison = [{"condition": name, "n": m["n_attempted"], "initial_evaluable": m["initial_evaluable"],
                   "action_accuracy": m["action_accuracy"]["value"], "ask_recall_required": m["ASK_recall_required"]["value"],
                   "ask_false_positive_rate": m["ASK_false_positive_rate"]["value"],
                   "technical_errors": m["technical_errors"]} for name, m in metrics["conditions"].items()]
    if comparison:
        write_csv(out / "comparison.csv", comparison, list(comparison[0]))


def source_hash():
    sources = sorted(p for folder in ["agents", "memory", "simulator", "data_generation", "evaluation", "tests"] for p in (ROOT / folder).glob("*.py"))
    return digest({str(p.relative_to(ROOT)): digest(p.read_text()) for p in sources})


def snapshot_sources(out):
    import shutil
    destination = out / "source_snapshot"
    destination.mkdir(exist_ok=False)
    files = [p for folder in ["agents", "memory", "simulator", "data_generation", "evaluation", "tests", "original"] for p in (ROOT / folder).glob("*.py")]
    files += [ROOT / name for name in ("experimento.py", "scenarios.py", "pyproject.toml", "uv.lock", ".python-version") if (ROOT / name).exists()]
    for path in files:
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def check_gates(selected, full_gold, require_human=True):
    reasons = []
    if require_human:
        if any(not g["annotation"]["validated"] or not g["annotation"]["reviewer"].strip() for _, g in selected):
            reasons.append("Hay casos seleccionados sin validación humana identificada")
        topics = {g["topic_id"] for g in full_gold}
        validated = {g["topic_id"] for g in full_gold if g["annotation"]["validated"] and g["historical_conflict"] and g["annotation"]["reviewer"].strip()}
        if topics - validated:
            reasons.append(f"Falta una contradicción directa validada en {len(topics - validated)} temas")
    counts = {}
    for g in full_gold:
        counts.setdefault(g["person_id"], set()).add(g["variant"])
    if any(len(v) < 5 for v in counts.values()):
        reasons.append("Menos de cinco variantes por historia")
    return reasons


def verify_services(args):
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "inference_performed": False, "services": {}}
    for name, base, suffix, token_env in [("agent", args.url, "/models", args.token_env),
        ("judge_same_model_not_independent", args.judge_url, "/models", args.judge_token_env),
        ("embedding", args.ollama_url, "/api/tags", "OLLAMA_API_KEY")]:
        if not base:
            report["services"][name] = {"status": "not_configured"}
            continue
        try:
            data = http_json(base.rstrip("/") + suffix, timeout=min(args.timeout, 10), token=os.environ.get(token_env, ""))
            models = data.get("models", []) if name == "embedding" else data.get("data", [])
            report["services"][name] = {"status": "reachable", "models": [m.get("name", m.get("id")) for m in models]}
        except TechnicalError as exc:
            report["services"][name] = {"status": "unreachable", "kind": exc.kind, "http_status": exc.status}
    return report


def parser():
    p = argparse.ArgumentParser(description="Auditoría reproducible de clarificación; sin entrenamiento.")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify-services", action="store_true", help="Solo metadatos; no inferencia")
    p.add_argument("--technical-pilot", action="store_true", help="Piloto de 5–10 casos dev; exige la misma validación humana previa")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--two-turn", action="store_true")
    p.add_argument("--max-questions", type=int, default=2)
    p.add_argument("--memory-mode", choices=["raw", "oracle_full_history", "extracted"], default="raw")
    p.add_argument("--retrieval", choices=["full", "bge"], default="full")
    p.add_argument("--topk", type=int, choices=[4, 8, 16], default=8)
    p.add_argument("--policies", default="base,guided,always_ask,never_ask,recency_only,simple_conflict")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--url", default=os.environ.get("LLM_BASE_URL", ""))
    p.add_argument("--judge-url", default=os.environ.get("JUDGE_BASE_URL", ""))
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_BASE_URL", ""))
    p.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    p.add_argument("--judge-model", default="zai-org/GLM-5.3-Flash")
    p.add_argument("--embed-model", default="bge-m3:latest")
    p.add_argument("--token-env", default="VLLM_API_KEY")
    p.add_argument("--judge-token-env", default="VLLM_JUDGE_API_KEY")
    p.add_argument("--no-persist-clarification", action="store_true")
    p.add_argument("--no-future-noise", action="store_true")
    p.add_argument("--dev-people", type=int, default=20)
    p.add_argument("--test-people", type=int, default=80)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.limit < 0 or args.max_questions < 1 or args.timeout <= 0 or args.dev_people < 1 or args.test_people < 1:
        raise SystemExit("Parámetros fuera de rango")
    names = args.policies.split(",")
    if not set(names) <= {"base", "guided", "always_ask", "never_ask", "recency_only", "simple_conflict"} or len(names) != len(set(names)):
        raise SystemExit("Políticas inválidas/duplicadas")
    out = args.output.resolve()
    if args.verify_services:
        out.mkdir(parents=True, exist_ok=False)
        report = verify_services(args)
        atomic_json(out / "services.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.dataset:
        public = load_jsonl(args.dataset / "casos_agent.jsonl")
        gold = load_jsonl(args.dataset / "private" / "casos_gold.jsonl")
    else:
        public, gold = make_dataset(args.seed, args.dev_people, args.test_people)
    counts = validate_dataset(public, gold)
    selected = balanced_select(public, gold, args.split, args.limit, args.seed)
    if args.technical_pilot and (args.split != "dev" or not 5 <= args.limit <= 10):
        raise SystemExit("El piloto técnico admite solo 5–10 casos de desarrollo")
    live = not args.dry_run and (bool(set(names) & {"base", "guided"}) or args.memory_mode == "extracted" or args.retrieval == "bge" or bool(args.judge_url))
    if args.memory_mode == "oracle_full_history" and args.retrieval != "full":
        raise SystemExit("oracle_full_history exige recuperación completa")
    if live:
        # SPEC exige validación humana antes de cualquier inferencia experimental.
        # El piloto técnico acota el tamaño, pero no permite eludir ese gate.
        reasons = check_gates(selected, gold, require_human=True)
        # Ejecutar tests de nuevo, sin confiar en un marcador antiguo.
        completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, capture_output=True, text=True)
        if completed.returncode or "Ran 0 tests" in completed.stderr:
            reasons.append("Las pruebas locales no pasaron o no se descubrieron")
        if (set(names) & {"base", "guided"} or args.memory_mode == "extracted") and not args.url:
            reasons.append("Servicio de inferencia no configurado")
        if args.retrieval == "bge" and not args.ollama_url:
            reasons.append("Servicio BGE no configurado")
        if reasons:
            # No se crea una ejecución ni se llama a inferencia si un gate falla.
            print(json.dumps({"pilot_blocked": True, "reasons": reasons}, ensure_ascii=False, indent=2))
            return 2
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in {"url", "judge_url", "ollama_url", "token_env", "judge_token_env", "output", "resume", "dataset"}}
    # Identidades de servicio en hash; nunca URL/credenciales en manifiestos.
    config["service_identities"] = {key: digest(getattr(args, key)) if getattr(args, key) else None for key in ("url", "judge_url", "ollama_url")}
    config.update(public_hash=digest(public), gold_hash=digest(gold), source_hash=source_hash())
    identity = digest(config)
    if out.exists():
        if not args.resume or not (out / "manifest.json").exists():
            raise SystemExit("El directorio existe; usa otro destino o --resume con la misma configuración")
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["run_identity"] != identity:
            raise SystemExit("Reanudación rechazada: configuración/datos/código distintos")
    else:
        out.mkdir(parents=True)
        (out / "private").mkdir(mode=0o700)
        snapshot_sources(out)
        write_dataset(out / "dataset", public, gold)
        manifest = {"run_identity": identity, "configuration": config, "python": platform.python_version(),
            "created_at": datetime.now(timezone.utc).isoformat(), "people": counts, "selected_cases": len(selected),
            "selected_histories": len({c["person_id"] for c, _ in selected}), "prompt_hashes": {
                "neutral": digest(NEUTRAL), "guided": digest(GUIDED), "extraction": digest(EXTRACTION), "judge": digest(JUDGE_PROTOCOL)},
            "model_version": "pending service metadata; model ID alone is not weight provenance",
            "source_snapshot": "source_snapshot/",
            "automatic_evaluation_only": True, "live": live, "training_performed": False,
            "scientific_evaluation_eligible": live and not args.technical_pilot,
            "human_gold_validated_selected": sum(g["annotation"]["validated"] for _, g in selected),
            "technical_pilot_caveat": "Piloto acotado; no sustituye evaluación retenida de viabilidad" if args.technical_pilot else None}
        atomic_json(out / "manifest.json", manifest)
    audit_rows = [{"case_id": c["case_id"], "person_id": c["person_id"], "history": json.dumps(c["history"], ensure_ascii=False),
        "query": c["query"], "historical_conflict": "", "unresolved_conflict": "", "kind": "", "acceptable_actions": "", "reviewer": "", "validated": "", "notes": ""} for c in public]
    if not (out / "gold_review_blind.csv").exists():
        write_csv(out / "gold_review_blind.csv", audit_rows, list(audit_rows[0]))
    if args.dry_run:
        atomic_json(out / "gates.json", {"local_schema": "passed", "dataset_human_validation": "pending", "live_pilot": "blocked", "reasons": check_gates(selected, gold)})
        print(json.dumps({"dry_run": True, "cases": len(public), "people": counts, "variants": len(VARIANTS), "network_calls": 0, "output": str(out)}, ensure_ascii=False))
        return 0
    rows_path = out / "per_case.jsonl"
    rows = load_jsonl(rows_path) if rows_path.exists() else []
    done = {(r["case_id"], r["condition"]) for r in rows}
    agent_log = out / "agent_trace.jsonl"
    judge_log = out / "private" / "judge_trace.jsonl"
    current_calls = []
    current_context = {}
    def record_agent(record):
        record = {**current_context, **record}
        append_jsonl(agent_log, record)
        current_calls.append({k: v for k, v in record.items() if k not in {"messages", "content"}})
    def record_judge(record):
        record = {**current_context, **record}
        append_jsonl(judge_log, record)
        current_calls.append({k: v for k, v in record.items() if k not in {"messages", "content"}})
    client = API(args.url, args.model, args.timeout, args.max_tokens, args.seed, args.temperature, args.token_env, record_agent)
    judge = API(args.judge_url, args.judge_model, args.timeout, args.max_tokens, args.seed, args.temperature, args.judge_token_env, record_judge, "judge") if args.judge_url else None
    extraction_client = API(args.url, args.model, args.timeout, args.max_tokens, args.seed, args.temperature, args.token_env, record_agent, "extractor")
    extractor = Extractor(extraction_client) if args.memory_mode == "extracted" else None
    embedder = BGE(args.ollama_url, args.embed_model, args.timeout, out / "embedding_cache.json", record_agent) if args.retrieval == "bge" else None
    for case, annotation in selected:
        for name in names:
            condition = f"{args.memory_mode}/{args.retrieval}/k{args.topk}/{name}"
            if (case["case_id"], condition) in done:
                continue
            current_calls.clear()
            current_context.update(case_id=case["case_id"], condition=condition)
            policy = LMPolicy(client, name == "guided") if name in {"base", "guided"} else Heuristic(name)
            # Intentos incompletos se conservan; una reanudación crea una carpeta nueva.
            parent = out / "cases" / case["case_id"] / name
            attempt = 1
            while (parent / str(attempt)).exists():
                attempt += 1
            row = run_case(case, annotation, policy, condition, parent / str(attempt), args.retrieval, args.topk,
                embedder, extractor, judge, args.two_turn, args.max_questions, not args.no_persist_clarification,
                not args.no_future_noise, args.memory_mode)
            row["api_calls"] = list(current_calls)
            row["evaluation_mode"] = "technical_pilot_provisional" if args.technical_pilot else "live_provisional" if live else "offline_heuristic"
            append_jsonl(rows_path, row)
            rows.append(row)
            export_reports(out, rows, {c["case_id"]: c for c in public}, args.seed)
            print(f"{case['case_id']} {name}: {row.get('initial_action', 'NO_EVALUABLE')}", flush=True)
            if row["technical_errors"]:
                atomic_json(out / "stop.json", {"reason": "technical_error", "case_id": case["case_id"], "condition": condition})
                return 3
    print(f"Resultados guardados: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
