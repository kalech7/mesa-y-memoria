"""Serve the local memory lab with only the Python standard library."""
from __future__ import annotations

import argparse
from copy import deepcopy
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit

from .cases import MODELS, ROOT, RUN, TARGETED_RUN, expected, load_cases, question_turn, show_steps
from .chat import ChatError, Chats
from .guard import load_guard_cases, load_reserved_guard
from .interactive import load_interactive, simulate
from .live import LiveInference, read_json
from .meal_models import meal_models
from .shared_budget import BudgetError, BudgetLab, DEFAULT_RUN as BUDGET_RUN
from .profiles_chat import DEFAULT_STORAGE, ProfilesChats
from .profiles_runtime import DEFAULT_RUN as PROFILES_RUN, ProfilesError

STATIC = Path(__file__).resolve().parent / "static"
CHAT_STUDY = ROOT / "research/runs/tesis_chat_memory_e2e_v1/SUMMARY.json"
MAX_REQUEST = 4096
MAX_QUESTION = 240
FINAL_GET = {"/", "/chat", "/chat/", "/timing.css", "/timing.js", "/food.css", "/food.js", "/api/timing/catalog"}
FINAL_POST = {"/api/timing/run": MAX_REQUEST, "/api/food/chat": 16_384}


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class Lab:
    def __init__(self, device, budget_run=BUDGET_RUN, profiles_run=PROFILES_RUN, profiles_storage=DEFAULT_STORAGE):
        self.live = LiveInference(device)
        self.shared_budget = BudgetLab(device, budget_run)
        self.profiles = ProfilesChats(device, profiles_run, profiles_storage)
        self.chats = Chats(self.live)
        self.cases = load_cases()
        reserved, self.study_results = load_reserved_guard()
        self.guard_cases = {**reserved, **load_guard_cases()}
        self.interactive_study, self.interactive_cases = load_interactive()
        self.selection = {
            (run, row["method"]): row for run in (RUN, TARGETED_RUN)
            for row in read_json(run / "SELECTION.json")["rows"]
        }

    def catalog(self):
        available, reason = self.live.status()
        recommendations, report = meal_models()
        models = {**MODELS, **recommendations}
        exploration_pairs = [(f"comidas-variado-exploracion-sft-{seed}",
                              f"comidas-variado-exploracion-{method}-{seed}")
                             for method in ("grpo_sampled", "grpo") for seed in (101, 202, 303)]
        trained_pairs = [pair for pair in exploration_pairs if pair[1] in models
                         and (models[pair[1]].training_epoch or 0) > 0]
        chat_pairs = (
            *trained_pairs,
            *exploration_pairs,
            ("comidas-variado-aclaracion-sft-101", "comidas-variado-aclaracion-grpo-101"),
            ("comidas-variado-qa-sft", "comidas-variado-qa-grpo"),
            ("comidas-variado-sft", "comidas-variado-grpo"),
            ("comidas-sft", "comidas-grpo"),
            ("sft", "grpo"),
        )
        sft, rl = next(pair for pair in chat_pairs if all(model in models for model in pair))
        return {
            "scenarios": [case["public"] for case in self.cases.values()],
            "guard_scenarios": [case["scenario"] for case in self.guard_cases.values()],
            "study_results": self.study_results,
            "interactive_study": self.interactive_study,
            "profiles_v5": True,
            "shared_budget": self.shared_budget.catalog(),
            "models": [model.public() for model in MODELS.values()],
            "chat_models": [model.public() for model in (*recommendations.values(), *MODELS.values())],
            "chat_choices": [{"id": sft, "label": "SFT"}, {"id": rl, "label": "RL (GRPO)"}],
            "recommendation_study": report,
            "chat_memory_study": read_json(CHAT_STUDY) if CHAT_STUDY.is_file() else None,
            "live_available": available, "live_reason": reason, "device": self.live.device,
        }

    def compare(self, body):
        if not isinstance(body, dict):
            raise RequestError("La solicitud debe ser un objeto JSON.")
        mode = body.get("mode", "replay")
        if mode not in ("replay", "live"):
            raise RequestError("Modo desconocido. Usa replay o live.")
        case_id = body.get("scenario_id")
        if not isinstance(case_id, str) or case_id not in self.cases:
            raise RequestError("Caso desconocido.")
        case = self.cases[case_id]
        scenario = dict(case["public"])
        if mode == "live" and not scenario["live_available"]:
            raise RequestError("Este caso se puede reproducir desde su traza archivada.")
        model_ids = body.get("model_ids", ["sft", "grpo"])
        if not isinstance(model_ids, list) or not model_ids or any(
            not isinstance(model_id, str) or model_id not in case["models"] for model_id in model_ids
        ):
            raise RequestError("model_ids debe incluir identificadores disponibles para este caso.")
        if len(model_ids) != len(set(model_ids)):
            raise RequestError("Cada modelo puede aparecer una sola vez.")
        question = body.get("question")
        if question is not None and (not isinstance(question, str) or not 1 <= len(question.strip()) <= MAX_QUESTION):
            raise RequestError(f"Escribe una pregunta de 1 a {MAX_QUESTION} caracteres.")
        custom = question is not None and question.strip() != scenario["question"]
        if custom and mode != "live":
            raise RequestError("Las preguntas editadas requieren inferencia local.")
        verify_writes = body.get("verify_writes", False)
        if type(verify_writes) is not bool or (verify_writes and mode != "live"):
            raise RequestError("La verificación de nuevas escrituras requiere inferencia local.")
        models = [case["models"][model_id] for model_id in model_ids]
        episode = case["episode"]
        rows = case["rows"]
        if mode == "live":
            available, reason = self.live.status()
            if not available:
                raise RequestError(reason, status=503)
            if custom:
                scenario["question"] = question.strip()
                episode = deepcopy(episode)
                question_turn(episode)["text"] = scenario["question"]
            options = {"verify_writes": True} if verify_writes else {}
            rows = self.live.compare(episode, models, **options)
        shown = []
        for model in models:
            row = rows[model.id]
            selection = self.selection[model.run, model.method]
            shown.append({
                **model.public(), "steps": show_steps(row, score=not custom),
                "metrics": row["metrics"] if not custom else None,
                "verification": row.get("guard_decision"),
                "provenance": {
                    "source": mode, "method": model.method, "step": selection["step"],
                    "training_seed": selection["training_seed"], "adapter_sha256": selection["adapter_sha256"],
                    "condition": row["condition"], "root_id": row["root_id"], "reader": row["reader_id"],
                    "latent_world_id": row["latent_world_id"], "realization_id": row["realization_id"],
                    "run": model.run.name,
                    "artifact": str(model.trajectories.relative_to(ROOT)) if mode == "replay" else None,
                    "device": self.live.device if mode == "live" else None,
                    "compute_precision": self.live.compute_precision if mode == "live" else None,
                },
            })
        return {
            "scenario": scenario, "mode": mode, "custom": custom,
            "verified_writes": verify_writes,
            "expected": expected(episode) if not custom else None, "models": shown,
        }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, data):
        self._send(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")

    def _read_json_bytes(self, limit=MAX_REQUEST):
        if self.headers.get_content_type() != "application/json":
            raise RequestError("Usa Content-Type: application/json.", status=415)
        try:
            size = int(self.headers.get("Content-Length", 0))
        except ValueError as error:
            raise RequestError("Content-Length debe ser un entero.") from error
        if not 0 < size <= limit:
            raise RequestError("El tamaño del cuerpo JSON está fuera del límite.", status=413)
        return self.rfile.read(size)

    def _proxy_final(self, path, body=None):
        headers = {"Content-Type": self.headers["Content-Type"]} if body is not None else {}
        query = urlsplit(self.path).query
        target = f"{path}?{query}" if query else path
        upstream = HTTPConnection("127.0.0.1", self.server.final_lab_port, timeout=180)
        try:
            upstream.request(self.command, target, body=body, headers=headers)
            response = upstream.getresponse()
            data = response.read()
            content_type = response.getheader("Content-Type") or "application/octet-stream"
        except TimeoutError:
            self._json(504, {"error": "El laboratorio v6 superó el tiempo de espera."})
            return
        except (HTTPException, OSError):
            self._json(502, {"error": "No se pudo conectar con el laboratorio v6."})
            return
        finally:
            upstream.close()
        self._send(response.status, data, content_type)

    def do_GET(self):
        path = urlsplit(self.path).path
        final_lab = getattr(self.server, "final_lab_port", None)
        if final_lab and path in FINAL_GET:
            self._proxy_final(path)
            return
        if path == "/api/catalog":
            self._json(200, self.server.lab.catalog())
            return
        if path == "/api/shared-budget":
            self._json(200, self.server.lab.shared_budget.catalog())
            return
        if path == "/api/profiles/catalog":
            try:
                self._json(200, self.server.lab.profiles.catalog())
            except ProfilesError as error:
                self._json(error.status, {"error": str(error)})
            return
        if path.startswith("/api/guard/"):
            case = self.server.lab.guard_cases.get(path.removeprefix("/api/guard/"))
            self._json(200 if case else 404, case or {"error": "Caso de verificación desconocido."})
            return
        if path.startswith("/api/interactive/"):
            case = self.server.lab.interactive_cases.get(path.removeprefix("/api/interactive/"))
            public = {key: value for key, value in case.items() if key != "story"} if case else None
            self._json(200 if case else 404, public or {"error": "Historia de varios turnos desconocida."})
            return
        if path == "/" or (final_lab and path == "/archivo"):
            asset = STATIC / "index.html"
        elif path.startswith("/static/"):
            asset = (STATIC / path.removeprefix("/static/")).resolve()
        else:
            asset = None
        if asset is None or not asset.is_relative_to(STATIC) or not asset.is_file():
            self._json(404, {"error": "Recurso no encontrado."})
            return
        media_type = mimetypes.guess_type(asset)[0] or "application/octet-stream"
        self._send(200, asset.read_bytes(), media_type)

    def do_POST(self):
        path = urlsplit(self.path).path
        proxy = getattr(self.server, "final_lab_port", None) and path in FINAL_POST
        if not proxy and path not in ("/api/compare", "/api/interactive", "/api/chat", "/api/shared-budget", "/api/profiles/chat"):
            self._json(404, {"error": "Recurso no encontrado."})
            return
        try:
            content = self._read_json_bytes(FINAL_POST[path] if proxy else MAX_REQUEST)
            if proxy:
                self._proxy_final(path, content)
                return
            body = json.loads(content)
            if path == "/api/compare":
                result = self.server.lab.compare(body)
            elif path == "/api/chat":
                result = self.server.lab.chats.handle(body)
            elif path == "/api/shared-budget":
                result = self.server.lab.shared_budget.play(body)
            elif path == "/api/profiles/chat":
                result = self.server.lab.profiles.handle(body)
            else:
                case_id = body.get("case_id") if isinstance(body, dict) else None
                case = self.server.lab.interactive_cases.get(case_id) if isinstance(case_id, str) else None
                actions = body.get("actions") if isinstance(body, dict) else None
                if case is None or not isinstance(actions, list) or len(actions) > 6 or not all(isinstance(action, str) for action in actions):
                    raise RequestError("Especifica una historia y una secuencia de hasta seis acciones.")
                try:
                    result = simulate(case["story"], actions)
                except ValueError as error:
                    raise RequestError(str(error)) from error
        except (RequestError, ChatError, BudgetError, ProfilesError) as error:
            self._json(error.status, {"error": str(error)})
        except (json.JSONDecodeError, UnicodeError, OverflowError):
            self._json(400, {"error": "El cuerpo no contiene JSON válido."})
        except Exception as error:
            logging.exception("Falló la ejecución del laboratorio")
            self._json(500, {"error": f"Falló la ejecución local. {error}"})
        else:
            try:
                self._json(200, result)
            except (BrokenPipeError, ConnectionResetError):
                logging.info("El navegador se desconectó antes de recibir el resultado.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--final-lab-port", type=int, help="Puerto local del laboratorio v6. Conserva v5 en /archivo.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--budget-run", type=Path, default=BUDGET_RUN,
                        help="Carpeta con LAB_MODELS.json del experimento de presupuesto compartido.")
    parser.add_argument("--profiles-run", type=Path, default=PROFILES_RUN,
                        help="Corrida v5 con checkpoints y LOCK.json.")
    parser.add_argument("--profiles-storage", type=Path, default=DEFAULT_STORAGE,
                        help="Carpeta persistente para las conversaciones y SQLite de v5.")
    args = parser.parse_args()
    if args.final_lab_port is not None and (not 1 <= args.final_lab_port <= 65535 or args.port == args.final_lab_port):
        parser.error("--final-lab-port necesita otro puerto local entre 1 y 65535.")
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.final_lab_port = args.final_lab_port
        server.lab = Lab(args.device, args.budget_run, args.profiles_run, args.profiles_storage)
        print(f"Laboratorio de memoria en http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
