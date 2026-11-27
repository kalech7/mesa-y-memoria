"""Loopback laboratory for v6 decisions and the evidence stored during each visit."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from math import isfinite
import mimetypes
from pathlib import Path
import sys
from threading import Lock
from urllib.parse import unquote, urlsplit

from research.src.artifacts import digest, file_hash
from research.src.readers.symbolic import applicability
from research.tesis_timing_v6_bank import build_worlds
from research.tesis_timing_v6_demo import ROUTES, checked_adapter, demo_pair
from research.tesis_timing_v6_env import prompt_text, replay
from research.tesis_timing_v6_experiment import ARMS, PROJECT, SEEDS, check_sources, rank, read, signature
from web_demo.profiles_runtime import ProfilesError


DEFAULT_RUN = PROJECT / "reports/tesis_timing_qwen_a100_v6"
STATIC = Path(__file__).resolve().parent / "static/timing"
AGENDA = {"ask_now": "Cenas para la semana", "save": "Desayunos para la semana"}
ARM = {ARMS[0]: "Qwen SFT", ARMS[1]: "Qwen RL (GRPO)"}
ROUTE = {
    "model": "Dejar que Qwen decida",
    "ask_today": "Preguntar hoy",
    "save_for_tomorrow": "Preguntar mañana",
    "never_ask": "Continuar sin preguntar",
}
SCOPE = {"ongoing": "Preferencia habitual", "session": "Solo para esta visita"}
DEFAULTS = {"agenda": "ask_now", "arm": ARMS[1], "route": "model", "scope": "ongoing"}
MAX_REQUEST = 4096


def options(labels):
    return [{"id": key, "label": label} for key, label in labels.items()]


def validated_settings(payload):
    if not isinstance(payload, dict) or set(payload) - (DEFAULTS.keys() | {"seed", "second_scope"}):
        raise ProfilesError("Las opciones de la prueba deben ser un objeto con campos conocidos.")
    settings = {**DEFAULTS, **payload}
    settings.setdefault("second_scope", settings["scope"])
    allowed = {"agenda": AGENDA, "arm": ARM, "route": ROUTE, "scope": SCOPE, "second_scope": SCOPE}
    for name, permitted in allowed.items():
        if not isinstance(settings[name], str) or settings[name] not in permitted:
            raise ProfilesError(f"Opción desconocida para {name}.")
    if "seed" in settings and (type(settings["seed"]) is not int or settings["seed"] not in SEEDS):
        raise ProfilesError("Semilla desconocida.")
    return settings


def active_memory(records, number, date):
    return [
        {
            **record,
            "valid": applicability(
                record,
                {
                    "date": date,
                    "session_id": f"S{number}",
                    "attribute": record["attribute"],
                    "context": None,
                },
            ),
        }
        for record in records
    ]


def fit_live_context(observation, tokenizer):
    """Use the training tokenizer to preserve a recent suffix of public evidence."""
    from research.tesis_timing_v6_train import MAX_TOKENS, messages

    history, memory = observation["messages"], observation["memory"]
    if not history:
        raise ProfilesError("Falta el mensaje actual de la conversación.")

    def length(chat, records):
        text = prompt_text({**observation, "messages": chat, "memory": records})
        return len(tokenizer.apply_chat_template(messages(text), tokenize=True, add_generation_prompt=True))

    def suffix(items, fits):
        lo, hi = 0, len(items)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(items[len(items) - mid :]):
                lo = mid
            else:
                hi = mid - 1
        return items[len(items) - lo :]

    chat, records = history, memory
    size = length(chat, records)
    if size > MAX_TOKENS:
        last = history[-1:]
        if length(last, []) > MAX_TOKENS:
            raise ProfilesError(f"El mensaje y la pregunta superan el límite de {MAX_TOKENS} tokens.", 413)
        records = suffix(memory, lambda selected: length(last, selected) <= MAX_TOKENS)
        chat = suffix(history, lambda selected: length(selected, records) <= MAX_TOKENS)
        size = length(chat, records)
    return {**observation, "messages": chat, "memory": records}, {
        "messages_omitted": len(history) - len(chat),
        "memory_omitted": len(memory) - len(records),
        "tokens": size,
        "max_tokens": MAX_TOKENS,
    }


def episode(world, settings, decide=None):
    """Keep hidden scopes in the simulator; Qwen only receives replay's public observation."""
    path = []
    decisions = []
    manual = ROUTES.get(settings["route"])
    for number in (1, 2):
        node = replay(world, tuple(path), budget=1)
        margin = None
        if len(node.legal) == 1:
            action, source = node.legal[0], "forced"
        elif manual:
            action, source = manual[number - 1], "manual"
        else:
            margin = float(decide(node.observation))
            if not isfinite(margin):
                raise RuntimeError("Puntuación no finita en el checkpoint seleccionado.")
            action, source = ("ASK" if margin > 0 else "WAIT"), "qwen"
        available = node.observation["questions_remaining"]
        decisions.append(
            {
                "action": action,
                "source": source,
                "margin": margin,
                "remaining_before": available,
                "remaining_after": available - int(action == "ASK"),
            }
        )
        path.append(action)
    trace = replay(world, tuple(path), budget=1)
    remaining = decisions[-1]["remaining_after"]
    decisions.append(
        {
            "action": "READ",
            "source": "reader",
            "margin": None,
            "remaining_before": remaining,
            "remaining_after": remaining,
        }
    )

    # Operations come from the verified writer in replay. Construct snapshots
    # from its actual writes; the terminal snapshot is checked against SQLite.
    indexed = {r["memory_id"]: r for r in trace["memory"]}
    snapshots = []
    records = []
    cursor = 0
    for number, (visit, decision) in enumerate(zip(world["visits"], decisions, strict=True), start=1):
        if number <= 2:
            operations = trace["visits"][number - 1]["executed_operations"]
            records.extend(op["record"] for op in operations if "record" in op)
        else:
            operations = []
        start = cursor
        cursor += 1
        while cursor < len(trace["messages"]) and trace["messages"][cursor]["turn_id"] != f"u{number + 1}":
            cursor += 1
        snapshots.append(
            {
                "visit": number,
                "date": visit["date"],
                **decision,
                "messages": trace["messages"][start:cursor],
                "operations": operations,
                "memory": active_memory(records, number, visit["date"]),
            }
        )
    if records != trace["memory"]:
        raise RuntimeError("La traza de escrituras no coincide con el estado final de SQLite.")
    future = []
    for row in trace["future"]:
        response = row["response"]
        ids = response["evidence_ids"]
        quotes = [quote for identity in ids for quote in indexed[identity]["evidence_quotes"]]
        future.append(
            {
                "date": row["query"]["date"],
                "attribute": row["query"]["attribute"],
                "mode": response["mode"],
                "value": response["value"],
                "correct": bool(row["correct"]),
                "correct_abstention": bool(row["correct_abstention"]),
                "evidence_ids": ids,
                "quotes": quotes,
            }
        )
    return {
        "opening": world["opening"],
        "cue": world["cue"],
        "budget": 1,
        "split": "train",
        "world_id": world["world_id"],
        "root_id": world["root_id"],
        "settings": settings,
        "visits": snapshots,
        "memory": snapshots[-1]["memory"],
        "future": future,
        "metrics": trace["metrics"],
    }


class TimingLab:
    def __init__(self, run=DEFAULT_RUN, scorer_factory=None):
        self.run = Path(run).resolve()
        plan = check_sources(self.run)
        self.lock_sha = file_hash(self.run / "LOCK.json")
        self.lock = read(self.run / "LOCK.json")
        if self.lock["sha256"] != signature(self.run, self.lock["sha256"]):
            raise ValueError("La selección de modelos cambió después del sellado.")
        # Deploy one representative per arm using the same validation objective
        # as the locked checkpoint selector, before reading reserved results.
        self.deployed = {
            arm: max(SEEDS, key=lambda seed: rank(self.lock["selected"][str(seed)][arm])) for arm in ARM
        }
        reserved = read(self.run / "RESERVED_RESULTS.json")
        if reserved["lock_sha256"] != self.lock_sha:
            raise ValueError("Los resultados reservados no corresponden a estos checkpoints.")
        self.results = {k: reserved[k] for k in ("roots", "pairs", "packets", "summary", "device")}
        self.results["summary"] = {arm: reserved["summary"][arm] for arm in ARM}
        worlds = build_worlds("train")
        if digest(worlds) != plan["development_worlds_sha256"]["train"]:
            raise ValueError("El ejemplo de entrenamiento ha cambiado.")
        pair = demo_pair(worlds)
        self.opening = pair[0]["opening"]
        self.worlds = {}
        self.cues = {}
        for world in pair:
            first, second = world["visits"][:2]
            counts = Counter(q["attribute"] for q in world["future_queries"])
            high = first if world["variant"] == "ask_now" else second
            if counts[high["attribute"]] != 5:
                continue
            self.worlds[(world["variant"], first["scope"], second["scope"])] = world
            self.cues[world["variant"]] = world["cue"]
        if len(self.worlds) != len(AGENDA) * len(SCOPE) ** 2:
            raise ValueError("Faltan variantes o alcances para esta demostración.")
        self.scorer_lock = Lock()
        self.scorer_factory = scorer_factory
        self.scorer = None
        self.scorer_device = None

    def catalog(self):
        return {
            "opening": self.opening,
            "agendas": [{**item, "cue": self.cues[item["id"]]} for item in options(AGENDA)],
            "arms": options(ARM),
            "routes": options(ROUTE),
            "scopes": options(SCOPE),
            "defaults": DEFAULTS.copy(),
            "budget": 1,
            "results": self.results,
            "device": self.results["device"],
            "memory_storage": "episode_sqlite",
            "lock_sha256": self.lock_sha,
        }

    def describe_model(self, arm):
        if not isinstance(arm, str) or arm not in ARM:
            raise ProfilesError("Elige un modelo final disponible.")
        seed = self.deployed[arm]
        item = self.lock["selected"][str(seed)][arm]
        return {
            "arm": arm,
            "seed": seed,
            "round": item["round"],
            "label": ARM[arm],
            "weights_sha256": item["sha256"]["adapter_model.safetensors"],
        }

    @contextmanager
    def scoring(self, arm):
        """Hold the active adapter stable for the entire inference request."""
        seed = str(self.deployed[arm])
        with self.scorer_lock:
            try:
                adapter = checked_adapter(self.run, seed, arm, self.lock["selected"][seed][arm])
                if self.scorer is None:
                    if self.scorer_factory is None:
                        import torch

                        from research.tesis_timing_v6_train import MarginScorer

                        self.scorer = MarginScorer()
                        self.scorer_device = torch.cuda.get_device_name()
                    else:
                        self.scorer = self.scorer_factory()
                        self.scorer_device = "Prueba de integración"
                self.scorer.load(adapter)
                yield self.scorer
            except ProfilesError:
                raise
            except (ImportError, OSError, ValueError) as error:
                raise RuntimeError("No se pudo puntuar con el modelo seleccionado.") from error
            finally:
                if self.scorer is not None:
                    self.scorer.cache.clear()
                    self.scorer.encodings.clear()

    def score_public(self, arm, observation):
        model = self.describe_model(arm)
        if not observation["questions_remaining"]:
            return {"action": "WAIT", "source": "forced", "margin": None, "model": model, "device": "CPU"}
        with self.scoring(arm) as scorer:
            public, context = fit_live_context(observation, scorer.tokenizer)
            margin = float(scorer.margins([public])[0])
        if not isfinite(margin):
            raise RuntimeError("Puntuación no finita en el checkpoint seleccionado.")
        return {
            "action": "ASK" if margin > 0 else "WAIT",
            "source": "qwen",
            "margin": margin,
            "model": model,
            "device": self.scorer_device,
            "context": context,
        }

    def run_episode(self, payload):
        settings = validated_settings(payload)
        world = self.worlds[(settings["agenda"], settings["scope"], settings["second_scope"])]
        model = self.describe_model(settings["arm"])
        if "seed" in settings and settings["seed"] != model["seed"]:
            raise ProfilesError("La semilla solicitada no corresponde al modelo final.")
        settings["seed"] = model["seed"]
        used = settings["route"] == "model"
        if used:
            with self.scoring(settings["arm"]) as scorer:
                result = episode(world, settings, lambda obs: scorer.margins([obs])[0])
        else:
            result = episode(world, settings)
        result["model"] = {**model, "used": used}
        result["executed_utc"] = datetime.now(timezone.utc).isoformat()
        result["device"] = self.scorer_device if result["model"]["used"] else "CPU"
        return result


class TimingHandler(BaseHTTPRequestHandler):
    def send_bytes(self, data, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def send_data(self, result, status=200):
        self.send_bytes(
            json.dumps(result, ensure_ascii=False, allow_nan=False).encode(),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self):
        route = unquote(urlsplit(self.path).path)
        if route == "/api/timing/catalog":
            self.send_data(self.server.lab.catalog())
            return
        relative = {"/": "index.html", "/chat": "chat.html", "/chat/": "chat.html"}.get(
            route, route.lstrip("/")
        )
        asset = (STATIC / relative).resolve()
        if not asset.is_relative_to(STATIC) or not asset.is_file():
            self.send_data({"error": "Recurso desconocido."}, 404)
            return
        self.send_bytes(
            asset.read_bytes(), mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        )

    def do_POST(self):
        route = urlsplit(self.path).path
        if route not in ("/api/timing/run", "/api/food/chat"):
            self.send_data({"error": "Ruta desconocida."}, 404)
            return
        if self.headers.get_content_type() != "application/json":
            self.send_data({"error": "Envía las opciones como JSON."}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        limit = 16_384 if route == "/api/food/chat" else MAX_REQUEST
        if not 0 < length <= limit:
            self.send_data({"error": "Tamaño de solicitud no admitido."}, 413 if length > limit else 400)
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError, RecursionError):
            self.send_data({"error": "Se esperaba una solicitud JSON válida."}, 400)
            return
        try:
            if route == "/api/food/chat":
                with self.server.chat_init_lock:
                    if self.server.food_chats is None:
                        from web_demo.food_chat import FinalFoodChats

                        self.server.food_chats = FinalFoodChats(
                            self.server.lab, storage=self.server.food_storage
                        )
                result = self.server.food_chats.handle(payload)
            else:
                result = self.server.lab.run_episode(payload)
        except ProfilesError as error:
            self.send_data({"error": str(error)}, error.status)
            return
        except ValueError as error:
            if route == "/api/food/chat":
                self.send_data({"error": str(error)}, 400)
            else:
                self.log_error("No se completó el recorrido: %s", error)
                self.send_data({"error": "No se pudo reproducir el episodio."}, 503)
            return
        except (RuntimeError, OSError, ImportError) as error:
            self.log_error("No se completó el recorrido: %s", error)
            self.send_data(
                {"error": "No se pudo ejecutar el recorrido con la configuración del servidor."}, 503
            )
            return
        self.send_data(result)


def make_server(lab, port=8767, food_storage=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), TimingHandler)
    server.lab = lab
    server.food_chats = None
    server.food_storage = food_storage
    server.chat_init_lock = Lock()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    with make_server(TimingLab(args.run), args.port) as server:
        print(f"Laboratorio v6 en http://127.0.0.1:{server.server_port}", file=sys.stderr, flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
