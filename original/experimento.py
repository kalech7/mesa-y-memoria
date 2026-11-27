#!/usr/bin/env python3
"""Evaluación longitudinal de contradicciones DIRECTAS entre memorias personales.
Solo stdlib. API de inferencia OpenAI-compatible (vLLM) y embeddings Ollama opcionales.

No es demostración de eficacia: el juez LLM es una anotación AUTOMÁTICA PROVISIONAL.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict

from scenarios import build_cases

ACTIONS = {"ASK", "ANSWER", "CONDITIONAL"}
BASE_SYSTEM = """Eres un asistente de alimentación que conserva recuerdos de conversaciones anteriores.
Ayuda al usuario respondiendo a su consulta. Escribe SOLO un objeto JSON válido:
{"historical_conflict": false, "unresolved_conflict": false,
 "action": "ANSWER", "question": "", "answer": "", "reason": ""}
Las claves historical_conflict y unresolved_conflict deben ser booleanas.
action debe ser ASK, ANSWER o CONDITIONAL.
Si action=ASK, escribe una sola pregunta en question y deja answer vacío.
Si action=ANSWER o CONDITIONAL, escribe tu respuesta en answer y question vacío.
En reason, justifica en una frase tu decisión.
No escribas texto fuera del JSON."""
GUIDED_ADDITION = """
Al decidir, conserva el origen y el alcance de cada recuerdo. Distingue:
- Contradicción histórica: dos preferencias mutuamente excluyentes para el mismo hábito y contexto.
- Conflicto vigente: sigue sin saberse cuál preferencia prevalece AHORA.
- Si la persona confirmó una decisión en la consulta actual o corrigió explícitamente
  su preferencia anterior en una sesión posterior, sigue esa decisión sin preguntar.
- No supongas que la nota de fecha posterior anula una anterior SIN señal de rectificación;
  tampoco confundas una diferencia de contexto con contradicción directa.
- Si elegir una única recomendación requiere conocer qué preferencia está vigente, ASK
  una pregunta breve que distinga las dos preferencias, sin inferir la respuesta.
- Si el usuario pide alternativas, CONDITIONAL y expón ambas sin fingir resolver la memoria.
- Evita preguntas de detalle que no cambian la recomendación que el usuario necesita.
"""


EXTRACTION_SYSTEM = """Gestionas memoria de largo plazo de un asistente personal.
Lees UN mensaje del usuario de UNA conversación pasada, sin ver el futuro.
Extrae SOLO preferencias personales, restricciones o planes duraderos que puedan afectar
consejos futuros. No guardes acciones cotidianas efímeras ni detalles domésticos triviales.
Conserva negaciones, condiciones y alcance (desayuno/cena, casa/fuera, etc.).
NO reconcilies recuerdos anteriores: una posible contradicción debe permanecer observable.
NO inventes hechos ni deduzcas la vigencia de una preferencia.
Devuelve SOLO JSON {"memories": ["recuerdo breve en primera persona", ...]};
puedes devolver una lista vacía. Cada recuerdo debe tener sentido por separado."""


class Extractor:
    def __init__(self, agent, cache_path):
        self.agent=agent
        self.path=cache_path
        self.cache=json.loads(cache_path.read_text(encoding="utf8")) if cache_path.exists() else {}

    def extract(self, user_text, at):
        key=hashlib.sha256((self.agent.model+"|extract-v1|"+at+"|"+user_text).encode()).hexdigest()
        if key not in self.cache:
            obj,_=self.agent.get_json(EXTRACTION_SYSTEM,
                "Fecha de la sesión: "+at+"\nMensaje original del usuario: "+user_text,
                max_tokens=1536)
            memories=obj.get("memories",[])
            if not isinstance(memories,list):
                raise ValueError("Extractor: memories no es lista")
            memories=[m.strip() for m in memories if isinstance(m,str) and m.strip()]
            self.cache[key]=memories
            self.path.write_text(json.dumps(self.cache,ensure_ascii=False),encoding="utf8")
        return self.cache[key]

JUDGE_SYSTEM = """Evalúas PREGUNTAS de aclaración de un asistente con memoria.
Debes decidir si la pregunta obtiene el dato DECISIVO para resolver la contradicción,
NO si es meramente relevante a alimentación. No presupongas ninguna respuesta.
Una pregunta vaga ("¿algo más?", "¿qué tienes en casa?") o sobre otra dimensión NO sirve.
La pregunta es útil si una respuesta natural a ella permite elegir inequívocamente entre
las preferencias incompatibles, incluso si no copia textualmente sus formulaciones.
Responde SOLO JSON {"useful": true|false, "reason": "breve"}."""

ANSWER_JUDGE_SYSTEM = """Eres un verificador de FIDELIDAD a preferencias de un usuario.
Evalúa solamente si la recomendación del agente satisface la preferencia VIGENTE
indicada por el usuario en la evidencia que recibirás. No penalices estilos distintos.
Si recomienda una opción directamente incompatible: fail.
Si evita comprometerse con una opción pese a que la decisión vigente es clara: uncertain.
Si respeta claramente la preferencia: pass. Si no puedes saberlo: uncertain.
No des por correcta una respuesta solo porque repita palabras del problema.
Responde SOLO JSON {"verdict":"pass|fail|uncertain","reason":"breve"}."""


def post_json(url, payload, timeout=180, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.load(res)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read(500).decode('utf8','replace')}") from exc


def decode_json(text):
    if isinstance(text, dict):
        return text
    if not text:
        raise ValueError("El modelo no devolvió contenido final (quizás agotó razonamiento)")
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(text[pos:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("No se encontró JSON de respuesta válido")


class API:
    def __init__(self, base, model, token="", timeout=180, max_tokens=4096):
        self.url = base.rstrip("/") + "/chat/completions"
        self.model = model
        self.token = token
        self.timeout = timeout
        self.max_tokens = max_tokens

    def get_json(self, system, user, max_tokens=None):
        return self.get_messages(system, [{"role":"user", "content":user}], max_tokens)

    def get_messages(self, system, messages, max_tokens=None):
        limit = max_tokens or self.max_tokens
        for attempt in range(2):
            start = time.monotonic()
            data = post_json(self.url, {
                "model": self.model,
                "messages": [{"role":"system", "content":system}] + messages,
                "temperature": 0, "max_tokens": limit,
            }, timeout=self.timeout, token=self.token)
            elapsed = round(time.monotonic() - start, 3)
            choice = data["choices"][0]
            message = choice.get("message") or {}
            reason = choice.get("finish_reason")
            raw = message.get("content")
            try:
                return decode_json(raw), {"latency":elapsed,
                    "tokens":(data.get("usage") or {}).get("completion_tokens"),
                    "finish_reason":reason, "raw":raw}
            except ValueError:
                if attempt == 0 and (reason == "length" or not raw):
                    limit = min(limit * 2, 16384)
                    continue
                raise


class Store:
    """Memoria episódica persistida en SQLite; extracción textual por sesión, no contexto mágico."""
    def __init__(self, path):
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("""CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            at TEXT NOT NULL, session_id TEXT NOT NULL,
            channel TEXT NOT NULL, text TEXT NOT NULL,
            source TEXT NOT NULL, raw_text TEXT NOT NULL DEFAULT ''
        )""")
        self.conn.commit()

    def add(self, at, session_id, channel, text, source="historical",raw_text=None):
        self.conn.execute("INSERT INTO episodes (at,session_id,channel,text,source,raw_text) VALUES (?,?,?,?,?,?)",
                          (at,session_id,channel,text,source,raw_text if raw_text is not None else text))
        self.conn.commit()

    def all(self):
        return [{"id":r[0],"at":r[1],"session_id":r[2],"channel":r[3],
                 "text":r[4],"source":r[5],"raw_text":r[6]} for r in
                self.conn.execute("SELECT id,at,session_id,channel,text,source,raw_text FROM episodes ORDER BY at, id")]

    def close(self):
        self.conn.close()


class Embedder:
    def __init__(self, url, model, cache_path, timeout=120):
        self.url = url.rstrip("/") + "/api/embed"
        self.model = model
        self.path = cache_path
        self.timeout = timeout
        self.cache = json.loads(cache_path.read_text(encoding="utf8")) if cache_path.exists() else {}

    def vectors(self, strings):
        digests = [hashlib.sha256((self.model + "\0" + s).encode()).hexdigest() for s in strings]
        unknown = [(i,s) for i,s in enumerate(strings) if digests[i] not in self.cache]
        if unknown:
            # Ollama soporta lotes en /api/embed
            raw = post_json(self.url, {"model":self.model, "input":[s for _,s in unknown]}, self.timeout)
            result = raw.get("embeddings")
            if not result or len(result) != len(unknown):
                raise ValueError("Embeddings faltantes o inesperados de Ollama")
            for (i,_), vector in zip(unknown, result):
                self.cache[digests[i]] = vector
            self.path.write_text(json.dumps(self.cache), encoding="utf8")
        return [self.cache[k] for k in digests]


def cosine(a,b):
    ab = sum(x*y for x,y in zip(a,b))
    aa = math.sqrt(sum(x*x for x in a))
    bb = math.sqrt(sum(x*x for x in b))
    return ab/(aa*bb) if aa*bb else 0


def get_memory(store, query, retrieval, topk, embedder):
    episodes = store.all()
    if retrieval == "full":
        return episodes
    # Retrieval: el query NO contiene los términos gold ni la respuesta oculta.
    texts = [e["text"] for e in episodes]
    vectors = embedder.vectors(texts + [query])
    q = vectors[-1]
    ranked = sorted(zip(episodes,vectors[:-1]), key=lambda x: cosine(x[1],q), reverse=True)
    ids = {id(x[0]) for x in ranked[:topk]}
    return [e for e in episodes if id(e) in ids]


def rendered_memory(episodes):
    if not episodes:
        return "(no se recuperaron memorias)"
    return "\n".join(f"{e['at']} · sesión {e['session_id']} ({e['channel']}): {e['text']}" for e in episodes)


def render_user(query, records):
    return ("Recuerdos de conversaciones previas del usuario, con fecha y procedencia:\n"
            + rendered_memory(records)
            + "\n\nMENSAJE DEL USUARIO EN ESTA SESIÓN:\n" + query)


def standardize(obj):
    action = str(obj.get("action", "")).strip().upper()
    if action not in ACTIONS:
        raise ValueError("acción inválida: " + action)
    question = str(obj.get("question", "") or "").strip()
    answer = str(obj.get("answer", "") or "").strip()
    if action == "ASK" and not question:
        raise ValueError("ASK sin pregunta")
    if action != "ASK" and not answer:
        raise ValueError("ANSWER/CONDITIONAL sin respuesta")
    for field in ("historical_conflict", "unresolved_conflict"):
        if not isinstance(obj.get(field), bool):
            raise ValueError(f"{field} debe ser booleano")
    return {"historical_conflict":obj["historical_conflict"],
            "unresolved_conflict":obj["unresolved_conflict"],
            "action":action, "question":question, "answer":answer,
            "reason":str(obj.get("reason", ""))}


def judge_question(judge, case, question):
    prompt = (f"ÁMBITO: {case.scope}\n"
              f"NOTA HISTÓRICA 1: '{case.memory_statement_a}'\n"
              f"NOTA HISTÓRICA 2: '{case.memory_statement_b}'\n"
              f"PREGUNTA FORMULADA: {question}\n"
              "¿Permite esta pregunta distinguir qué decisión actual prefiere la persona "
              "en el ámbito indicado? Considera si hace falta especificar el tema; "
              "respuestas meramente sociales o sobre detalles irrelevantes no sirven.")
    obj,_ = judge.get_json(JUDGE_SYSTEM,prompt,max_tokens=1024)
    return obj.get("useful") is True, str(obj.get("reason",""))


def judge_answer(judge, case, response, expected=None):
    preferred = expected or case.truth
    obj,_ = judge.get_json(ANSWER_JUDGE_SYSTEM,(
        f"PREFERENCIA VIGENTE (no procede de una inferencia): {preferred}\n"
        f"ÁMBITO: {case.scope}\n"
        f"RESPUESTA DEL AGENTE: {response}\n"
        "¿Es la recomendación coherente con la preferencia vigente?"
    ),max_tokens=1024)
    verdict = str(obj.get("verdict", "uncertain")).lower()
    if verdict not in {"pass","fail","uncertain"}:
        verdict = "uncertain"
    return verdict, str(obj.get("reason", ""))


def run_case(case, policy, agent, judge, args, outdir, embedder, extractor=None):
    path = outdir / "db" / f"{case.id}__{policy}.sqlite"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        path.unlink()  # cada caso arranca desde historial sin contamination entre configuraciones
    store = Store(path)
    row = {"case_id":case.id,"topic_id":case.topic_id,"variant":case.variant,"policy":policy,
           "gold_conflict":case.gold_conflict_in_history,"gold_unresolved":case.gold_unresolved,
           "gold_action":case.gold_action,"expected_after_confirmation":case.truth,
           "question_useful_auto":None,"answer_after_auto":None,"followup_auto":None,
           "simulator_reply":None,"second_action":None,"followup_action":None,
           "initial_error":"", "second_error":"", "followup_error":""}
    system = BASE_SYSTEM + (GUIDED_ADDITION if policy == "guided" else "")
    try:
        for s in case.sessions:
            fragments = ([s["text"]] if extractor is None else extractor.extract(s["text"],s["at"]))
            for n,fragment in enumerate(fragments):
                store.add(s["at"],s["session_id"],s["channel"],fragment,
                          source="historical" if extractor is None else "extracted",
                          raw_text=s["text"])
        memories = get_memory(store,case.query,args.retrieval,args.topk,embedder)
        row["memories_total"] = len(store.all())
        row["memories_retrieved"] = len(memories)
        # Gold coverage is an audit on raw historical strings; not used as agent input.
        row["contradiction_pair_retrieved"] = (None if not case.gold_conflict_in_history
             else all(any(statement == e["raw_text"] for e in memories)
                      for statement in (case.memory_statement_a,case.memory_statement_b)))
        prompt = render_user(case.query,memories)
        initial, meta = agent.get_json(system,prompt)
        initial = standardize(initial)
        row.update({f"initial_{k}":v for k,v in initial.items()})
        row["initial_latency"] = meta["latency"]
        row["initial_tokens"] = meta["tokens"]
        row["initial_correct"] = initial["action"] == case.gold_action
        row["detected_conflict_correct"] = initial["historical_conflict"] == case.gold_conflict_in_history
        row["detected_unresolved_correct"] = initial["unresolved_conflict"] == case.gold_unresolved
        # Juzga directamente la respuesta solamente si existe verdad que no depende de preguntar.
        if initial["action"] == "ANSWER" and case.gold_action == "ANSWER":
            row["initial_grounded_auto"],row["initial_grounded_reason"] = judge_answer(
                judge,case,initial["answer"],case.truth)

        if args.two_turn and initial["action"] == "ASK":
            useful, why = judge_question(judge,case,initial["question"])
            row["question_useful_auto"] = useful
            row["question_judge_reason"] = why
            # El usuario solo responde la variable latente en un conflicto ABIERTO.
            # Si el agente pregunta innecesariamente ante una preferencia ya confirmada,
            # no se filtra la verdad escondida ni se suma una aclaración ficticia.
            if useful and case.gold_action == "ASK":
                reply = case.user_reply
                updated = [reply] if extractor is None else extractor.extract(reply,"2026-10-10")
                # Las preferencias nuevas solo persisten si el sistema de memoria las almacena.
                for fragment in updated:
                    store.add("2026-10-10",f"{case.id}-aclaracion", "chat", fragment,
                              source="clarification" if extractor is None else "extracted_clarification",
                              raw_text=reply)
                row["confirmation_memory_saved"] = bool(updated)
                row["clarification_recorded"] = True
            elif useful and case.gold_action != "ASK":
                reply = "Lo que dije hace un momento sigue siendo válido. ¿Puedes usarlo para responder?"
                row["clarification_recorded"] = False
            else:
                reply = "No entiendo qué dato necesitas. ¿Puedes preguntar por mi decisión concreta?"
                row["clarification_recorded"] = False
            row["simulator_reply"] = reply
            # el agente recibe realmente la pregunta y respuesta, pero NUNCA la truth interna
            try:
                # Reproducir los roles REALES de conversación; NO incrustar turnos falsos en usuario.
                second_messages = [
                    {"role":"user", "content":prompt},
                    {"role":"assistant", "content":json.dumps(initial,ensure_ascii=False)},
                    {"role":"user", "content":reply + "\n¿Puedes terminar de responder a mi consulta?"},
                ]
                second = standardize(agent.get_messages(system,second_messages)[0])
                row["second_action"] = second["action"]
                row["second_answer"] = second["answer"]
                row["second_question"] = second["question"]
                if row["clarification_recorded"] and second["action"] in {"ANSWER","CONDITIONAL"}:
                    row["answer_after_auto"],row["answer_after_reason"] = judge_answer(
                        judge,case,second["answer"])
            except Exception as exc:
                row["second_error"] = f"{type(exc).__name__}: {exc}"
            # Tercera sesión: la pregunta ya no tiene la respuesta recién dicha en contexto.
            # Únicamente el almacén SQLite conserva la aclaración.
            if row["clarification_recorded"]:
                try:
                    followmem = get_memory(store,case.followup,args.retrieval,args.topk,embedder)
                    final = standardize(agent.get_json(system,render_user(case.followup,followmem))[0])
                    row["followup_action"] = final["action"]
                    row["followup_answer"] = final["answer"]
                    row["followup_question"] = final["question"]
                    if final["action"] in {"ANSWER","CONDITIONAL"}:
                        row["followup_auto"],row["followup_reason"] = judge_answer(
                            judge,case,final["answer"])
                except Exception as exc:
                    row["followup_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        row["initial_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        store.close()
    return row


def safe_rate(rows, pred):
    return round(sum(bool(pred(r)) for r in rows)/len(rows),4) if rows else None


def summary(rows):
    out={"description":"Evaluaciones automáticas provisionales; revisar manualmente preguntas y respuestas."}
    for policy in ("base","guided"):
        sub = [r for r in rows if r["policy"]==policy]
        valid = [r for r in sub if not r["initial_error"]]
        gold_ask = [r for r in valid if r["gold_action"]=="ASK"]
        other = [r for r in valid if r["gold_action"]!="ASK"]
        queried = [r for r in valid if r.get("question_useful_auto") is not None]
        clarified = [r for r in valid if r.get("clarification_recorded") is True]
        second = [r for r in clarified if r.get("answer_after_auto")]
        followup = [r for r in clarified if r.get("followup_auto")]
        with_pair = [r for r in valid if r.get("contradiction_pair_retrieved") is not None]
        out[policy] = {
            "n":len(sub),"histories":len({r["topic_id"] for r in sub}),
            "valid":len(valid),"errors":len(sub)-len(valid),
            "decision_accuracy":safe_rate(valid,lambda r:r.get("initial_correct")),
            "historical_conflict_accuracy":safe_rate(valid,lambda r:r.get("detected_conflict_correct")),
            "unresolved_conflict_accuracy":safe_rate(valid,lambda r:r.get("detected_unresolved_correct")),
            "both_contradictory_memories_retrieved":safe_rate(with_pair,
                lambda r:r.get("contradiction_pair_retrieved") is True),
            "ask_recall":safe_rate(gold_ask,lambda r:r.get("initial_action")=="ASK"),
            "ask_false_positive_rate":safe_rate(other,lambda r:r.get("initial_action")=="ASK"),
            "auto_question_utility":safe_rate(queried,lambda r:r.get("question_useful_auto") is True),
            "clarifications_recorded":len(clarified),
            "auto_after_clarification_pass":safe_rate(second,lambda r:r["answer_after_auto"]=="pass"),
            "auto_followup_memory_pass":safe_rate(followup,lambda r:r["followup_auto"]=="pass"),
            "followup_answer_rate":safe_rate(clarified,lambda r:r.get("followup_action")=="ANSWER"),
            "mean_tokens":round(statistics.mean(r["initial_tokens"] for r in valid if isinstance(r.get("initial_tokens"),int)),1)
                           if any(isinstance(r.get("initial_tokens"),int) for r in valid) else None,
            "by_variant":{k:{"n":len(g),"accuracy":safe_rate(g,lambda r:r.get("initial_correct"))}
                          for k in sorted({r["variant"] for r in valid})
                          for g in [[r for r in valid if r["variant"]==k]]},
        }
    both = {r["case_id"]:r for r in rows if r["policy"]=="base" and not r["initial_error"]}
    guided = {r["case_id"]:r for r in rows if r["policy"]=="guided" and not r["initial_error"]}
    pairs = [(both[k],guided[k]) for k in both.keys() & guided.keys()]
    out["paired"]={"n":len(pairs),
        "guided_only_correct":sum((not a["initial_correct"]) and b["initial_correct"] for a,b in pairs),
        "base_only_correct":sum(a["initial_correct"] and (not b["initial_correct"]) for a,b in pairs),
        "both_correct":sum(a["initial_correct"] and b["initial_correct"] for a,b in pairs),
        "both_wrong":sum((not a["initial_correct"]) and (not b["initial_correct"]) for a,b in pairs),
        "caveat":"Variantes de una misma biografía están correlacionadas. No separar train/test por fila."}
    return out


def write_rows(outdir,rows):
    (outdir/"respuestas.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
    keys=sorted({k for r in rows for k in r})
    with (outdir/"respuestas.csv").open("w",newline="",encoding="utf8") as fp:
        w=csv.DictWriter(fp,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # revisión ciega: anonimizar identificador y ocultar política/variante.
    blind=[]
    for r in rows:
        review_id=hashlib.sha256((r["case_id"]+"|"+r["policy"]+"|review-v1").encode()).hexdigest()[:12]
        blind.append({"review_id":review_id,
                      "initial_action":r.get("initial_action"),
                      "question":r.get("initial_question",""),
                      "answer":r.get("initial_answer",""),
                      "second_answer":r.get("second_answer",""),
                      "followup_answer":r.get("followup_answer",""),
                      "question_useful_human":"","answer_grounded_human":"",
                      "followup_grounded_human":"","notes":""})
    random.Random(13).shuffle(blind)
    with (outdir/"anotacion_humana.csv").open("w",newline="",encoding="utf8") as fp:
        w=csv.DictWriter(fp,fieldnames=list(blind[0]) if blind else [])
        if blind:
            w.writeheader();w.writerows(blind)


def validate_cases(cases):
    ids=set()
    for c in cases:
        assert c.id not in ids, c.id
        ids.add(c.id)
        assert len(c.sessions) >= 12
        assert all(c.sessions[i]["at"]<=c.sessions[i+1]["at"] for i in range(len(c.sessions)-1))
        assert "A)" not in c.query and "B)" not in c.query
        assert c.gold_action in ACTIONS
        assert c.truth != c.other
        if c.gold_action == "ASK":
            assert c.gold_unresolved and c.gold_conflict_in_history
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",default="http://127.0.0.1:12555/v1")
    p.add_argument("--judge-url",default="http://127.0.0.1:12559/v1")
    p.add_argument("--model",default="zai-org/GLM-5.3-Flash")
    p.add_argument("--judge-model",default="zai-org/GLM-5.3-Flash")
    p.add_argument("--token-env",default="VLLM_API_KEY",help="Nombre variable entorno API key; sin compartir la clave")
    p.add_argument("--limit",type=int,default=5,help="0 para ejecutar TODOS los 100 casos")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--two-turn",action="store_true",help="Simulador + persistencia + consulta posterior")
    p.add_argument("--retrieval",choices=["full","bge"],default="full")
    p.add_argument("--memory-mode",choices=["raw","extracted"],default="raw",
                   help="raw: registro episódico; extracted: GLM decide qué memorizar")
    p.add_argument("--topk",type=int,default=8)
    p.add_argument("--ollama-url",default="http://127.0.0.1:11434")
    p.add_argument("--embed-model",default="bge-m3:latest")
    p.add_argument("--output",default="resultados_longitudinales")
    p.add_argument("--timeout",type=int,default=180)
    p.add_argument("--max-tokens",type=int,default=4096)
    p.add_argument("--dry-run",action="store_true",help="Genera dataset sin contactar la H200")
    p.add_argument("--policies",choices=["both","base","guided"],default="both")
    args=p.parse_args()
    import os
    cases=build_cases(seed=args.seed)
    validate_cases(cases)
    cases=cases[:args.limit] if args.limit>0 else cases
    outdir=pathlib.Path(args.output)
    outdir.mkdir(parents=True,exist_ok=True)
    # El dataset es un anexo privado del experimentador. Nunca se pasa directo al modelo.
    (outdir/"casos_gold.json").write_text(json.dumps([asdict(c) for c in cases],indent=2,ensure_ascii=False),encoding="utf8")
    # auditoría de procedencia y validez del dataset (retener fuera de entrenamiento)
    (outdir/"auditoria_conflictos.csv").write_text(
        "case_id,topic_id,contradiccion_directa,conflicto_vigente,accion_gold,revisor,anotacion_valida,comentario\n"
        + "".join(f"{c.id},{c.topic_id},{int(c.gold_conflict_in_history)},{int(c.gold_unresolved)},"
                  f"{c.gold_action},,,\n" for c in cases),encoding="utf8")
    if args.dry_run:
        for case in cases[:3]:
            print(f"{case.id}: {len(case.sessions)} sesiones; query='{case.query}'")
        print(f"[OK] {len(cases)} casos escritos en {outdir/'casos_gold.json'}")
        return
    token=os.environ.get(args.token_env,"")
    agent=API(args.url,args.model,token,args.timeout,args.max_tokens)
    judge=API(args.judge_url,args.judge_model,token,args.timeout,args.max_tokens)
    embedder=Embedder(args.ollama_url,args.embed_model,outdir/"embedding_cache.json") if args.retrieval=="bge" else None
    extractor=Extractor(agent,outdir/"extraction_cache.json") if args.memory_mode=="extracted" else None
    policies=["base","guided"] if args.policies=="both" else [args.policies]
    rows=[]
    for idx,case in enumerate(cases,1):
        for policy in policies:
            row=run_case(case,policy,agent,judge,args,outdir,embedder,extractor)
            rows.append(row)
            status="OK" if not row["initial_error"] else "ERROR"
            print(f"[{idx}/{len(cases)}] {case.id:35} {policy:6} {status:5} "
                  f"{row.get('initial_action','?'):11} gold={case.gold_action}",flush=True)
            # Escribe incrementalmente para recuperar resultados aunque caiga VPN.
            write_rows(outdir,rows)
            (outdir/"metricas.json").write_text(json.dumps(summary(rows),indent=2,ensure_ascii=False),encoding="utf8")
    print("\nFIN. Archivos:",outdir.resolve())
    print(json.dumps(summary(rows),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
