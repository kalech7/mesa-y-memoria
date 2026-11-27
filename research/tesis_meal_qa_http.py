"""Score held-out multi-turn dinner chats through the HTTP interface with verification off."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic
from urllib.request import Request, urlopen

from web_demo.meal_policy import recommendation_menu


ROOT = Path(__file__).resolve().parents[1]


CLARIFICATION = "Pedir una aclaración"
INGREDIENTS = {item["name"]: item["ingredients"] for item in
               recommendation_menu([], "dinner_preference", shuffle=False)}


def call(url, endpoint, timeout, payload=None):
    encoded = json.dumps(payload).encode() if payload is not None else None
    request = Request(url + endpoint, data=encoded, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def score(response, current):
    messages = response["messages"]
    assistant = messages[-1]
    detail = assistant.get("recommendation") or {}
    traces = assistant.get("trace", [])
    record = next((record for record in traces if record.get("phase") == "recommendation"), {})
    raw = json.loads(record["raw"]) if record else {}
    dish = raw.get("plato")
    menu = raw.get("menú", {})
    ingredients = INGREDIENTS.get(dish)
    excluded = set(current["excluded"])
    clashes = sorted(excluded.intersection(ingredients or ()))
    has_compatible_dish = any(values and excluded.isdisjoint(values) for values in INGREDIENTS.values())
    expects_clarification = current.get("clarify", False) or not has_compatible_dish
    uses_model_directly = (
        detail.get("source") == "model" and detail.get("raw") is True
        and assistant.get("role") == "assistant" and assistant.get("verification") is None
        and not assistant.get("fallback", False) and not assistant.get("context_limited", False)
        and dish in INGREDIENTS and dish == detail.get("model_choice")
        and menu.get(raw.get("letra")) == dish and set(menu.values()) == set(INGREDIENTS)
        and not response["pending"] and not response["memory"]
    )
    if not uses_model_directly:
        verdict = "no_raw_menu_decision"
    elif dish == CLARIFICATION:
        verdict = "correct_clarification" if expects_clarification else "unnecessary_clarification"
    elif expects_clarification:
        verdict = "missed_clarification"
    elif clashes:
        verdict = "excluded_ingredient"
    else:
        verdict = "compatible_dish"
    return {
        "passed": verdict in ("compatible_dish", "correct_clarification"),
        "verdict": verdict, "expected_clarification": expects_clarification,
        "conflicting_ingredients": clashes, "ingredients": ingredients, "raw_dish": dish,
        "raw_letter": raw.get("letra"), "first_unconstrained_token": raw.get("primera_salida_libre"),
        "menu": menu, "uses_model_directly": uses_model_directly,
        "displays_selected_dish": bool(ingredients and dish.casefold() in assistant["text"].casefold()),
        "assistant": assistant, "pending": response["pending"], "memory": response["memory"],
        "attribute": response["attribute"], "revision": response["revision"],
    }


def dump(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def gather(cases, models, url, handle, catalog, timeout):
    rows = []
    for model in models:
        for case in cases:
            print(f"{model} {case['id']}", file=sys.stderr, flush=True)
            try:
                start = call(url, "/api/chat", timeout, {"action": "start", "model": model})
                session_error = None
            except Exception as error:
                start = {}
                session_error = f"{type(error).__name__}: {error}"
            revision = start.get("revision")
            chat_id = start.get("id")
            for index, current in enumerate(case["turns"], start=1):
                payload = {"action": "send", "id": chat_id, "request_id": str(index),
                           "revision": revision, "text": current["text"], "verify": False}
                start_time = monotonic()
                if session_error is None:
                    try:
                        response = call(url, "/api/chat", timeout, payload)
                        revision = response["revision"]
                        result = score(response, current)
                        if response.get("model") != catalog[model] or response.get("id") != chat_id \
                           or response["messages"][-2]["text"] != current["text"] \
                           or revision != payload["revision"] + 1:
                            raise ValueError("La respuesta HTTP no coincide con el turno solicitado.")
                    except Exception as error:
                        session_error = f"{type(error).__name__}: {error}"
                if session_error is not None:
                    result = {"passed": False, "verdict": "http_or_response_error", "error": session_error}
                row = {"type": "turn", "case_id": case["id"], "family": case["family"],
                       "turn": index, "model": model, "gold": current, "request": payload,
                       "seconds": round(monotonic() - start_time, 3), **result}
                rows.append(row)
                dump(handle, row)
    return rows


def totals(rows):
    by_dialog = defaultdict(list)
    by_family = defaultdict(list)
    for row in rows:
        by_dialog[row["case_id"]].append(row["passed"])
        by_family[row["family"]].append(row["passed"])
    return {
        "turns": len(rows), "turns_passed": sum(row["passed"] for row in rows),
        "dialogs": len(by_dialog), "dialogs_passed": sum(all(turns) for turns in by_dialog.values()),
        "raw_decisions": sum(row.get("uses_model_directly", False) for row in rows),
        "displayed_dishes": sum(row.get("displays_selected_dish", False) for row in rows),
        "turns_with_conflicting_ingredient": sum(bool(row.get("conflicting_ingredients")) for row in rows),
        "verdicts": dict(Counter(row["verdict"] for row in rows)),
        "by_family": {family: {"turns": len(turns), "passed": sum(turns)}
                      for family, turns in by_family.items()},
    }


def summarize(rows, models):
    panels = {model: totals([row for row in rows if row["model"] == model]) for model in models}
    paired = defaultdict(list)
    for row in rows:
        paired[(row["case_id"], row["turn"])].append(row)
    joint = {}
    if len(models) > 1:
        paired_outcomes = Counter()
        mismatch, missing_menu = [], []
        for key, candidates in paired.items():
            if any(not row.get("menu") for row in candidates):
                missing_menu.append(key)
            elif any(row["menu"] != candidates[0]["menu"] for row in candidates[1:]):
                mismatch.append(key)
            winners = tuple(row["model"] for row in candidates if row["passed"])
            paired_outcomes[winners] += 1
        dialog_results = defaultdict(lambda: defaultdict(list))
        for row in rows:
            dialog_results[row["case_id"]][row["model"]].append(row["passed"])
        dialog_outcomes = Counter(tuple(model for model in models if all(results[model]))
                                  for results in dialog_results.values())
        joint = {
            "different_menu_order": mismatch, "missing_menu": missing_menu,
            "outcomes": [{"models_passing": winners, "turns": turn_count}
                         for winners, turn_count in paired_outcomes.items()],
            "dialog_outcomes": [{"models_passing": winners, "dialogs": count}
                                for winners, count in dialog_outcomes.items()],
        }
    return {"type": "summary", "panels": panels, "paired": joint}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--bank", type=Path, default=ROOT / "research/data/meal_qa_reserved.json")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if len(set(args.model)) != len(args.model):
        parser.error("Cada identificador de modelo se puede indicar una sola vez.")
    bank_path = args.bank.resolve()
    bank_bytes = bank_path.read_bytes()
    bank_label = bank_path.relative_to(ROOT) if bank_path.is_relative_to(ROOT) else bank_path
    cases = json.loads(bank_bytes)
    url = args.url.rstrip("/")
    server = call(url, "/api/catalog", args.timeout)
    catalog = {model["id"]: model for model in server["chat_models"]}
    if not server["live_available"] or set(args.model) - catalog.keys():
        parser.error(f"Comprueba el estado y los modelos en {url}/api/catalog.")
    with args.output.open(mode="x", encoding="utf-8") as handle:
        dump(handle, {"type": "metadata", "bank": str(bank_label), "bank_sha256": sha256(bank_bytes).hexdigest(),
                      "url": url, "models": [catalog[model] for model in args.model], "verify": False,
                      "device": server["device"], "started_at": datetime.now(timezone.utc).isoformat(),
                      "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "ingredients": INGREDIENTS})
        rows = gather(cases, args.model, url, handle, catalog, args.timeout)
        summary = summarize(rows, args.model)
        dump(handle, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
