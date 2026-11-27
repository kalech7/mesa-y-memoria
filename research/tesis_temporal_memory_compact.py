"""CPU ablation that scores public memory actions without an option-letter pointer."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import re
import unicodedata
from zlib import crc32

import torch
from torch import nn

from research.src.artifacts import digest, file_hash, write_json
from research.tesis_temporal_memory_data import build_split
from research.tesis_temporal_memory_reward import score_action
from research.tesis_temporal_memory_train import QA, SEEDS, preparation
from web_demo.cases import ROOT


RUN = ROOT / "research/runs/tesis_temporal_memory_compact_v1"
WIDTH = 1024
BATCH = 16
WARMUP = 6
EPOCHS = 12
PASSES = 2
DRAWS = 16
LR = .004
EXPLORATION = .25
KL = .06
CLIP = .20
DAY_COUNTS = {"un": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7}
LEXICAL_SIGNALS = {
    "day": r"\b(?:solo (?:por |para )?hoy|esta noche|medianoche|este dia|por el dia|horas que quedan|de hoy)\b",
    "session": r"\b(?:est[ae] (?:chat|hilo|sesion|conversacion|charla)|encerrado|aca|otro chat vuelve)\b",
    "tomorrow": r"\bmanana\b",
    "until": r"\b(?:hasta|final|ultimo dia|termin[ae]|inclui\w*|inclusive)\b",
    "end_inclusive": r"\b(?:inclusive|inclui\w*|contando ese dia)\b",
    "ongoing": r"\b(?:habitual\w*|permanente|siempre|(?:en|de aqui en) adelante|desde ahora|costumbre)\b",
    "context": r"\b(?:oficina|casa|trabajo|cuando este|visitas)\b",
    "ambiguous": r"\b(?:ultimamente|estos dias|estas veces|estos tiempos|por ahora|ando|antoja|apetece\w*"
                 r"|estado prefiriendo|esta gustando|todavia|inclinand\w*|ha dado por|he notado|viendo)\b",
    "nonresponse": r"\b(?:no (?:puedo precisar|se|he decidido)|no estoy segur\w*|aun no)\b",
    "outside_person": r"\b(?:mi (?:vecin\w*|ti[oa]|pareja|colega|sobrin\w*|herman\w*|amig\w*)"
                       r"|otra persona|la invitada|el mensajero|su gusto|de el|de ella)\b",
    "no_write": r"\b(?:no guardes|no anotes|no lo recuerdes|no quiero registrar|sin guardar|no hace falta guardar"
                r"|manten el gusto|deja como esta|sugerencia|fue una opcion)\b",
}


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def offset(d, current):
    return (date.fromisoformat(d) - current).days if d else None


def seen_date(text, raw):
    return bool(raw and raw in text)


def signal_features(observation):
    users = [message for message in observation["user_messages"] if message.get("role", "user") == "user"]
    text = " ".join(normalize(message["text"]) for message in users)
    last = normalize(users[-1]["text"]) if users else ""
    all_signals = set()
    for name, expression in LEXICAL_SIGNALS.items():
        if re.search(expression, text):
            all_signals.add(name)
        if re.search(expression, last):
            all_signals.add("last_" + name)
    counts = re.search(r"\b(\d+|un|uno|dos|tres|cuatro|cinco|seis|siete)\s+dias?\b", last)
    duration = (int(counts[1]) if counts[1].isdigit() else DAY_COUNTS[counts[1]]) if counts else None
    if duration is not None:
        all_signals.add("duration")
    has_date = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    if has_date.search(last):
        all_signals.add("absolute")
    if any(message.get("in_reply_to") for message in users) or len(users) >= 2:
        all_signals.add("reply")
    if any(mark in text for mark in ("«", "»", "“", "”", '"')):
        all_signals.add("quote")
    return all_signals, last, text, duration, {message["turn_id"]: message["text"] for message in users}


def put_flags(out, flags):
    out.update({name: float(value) for name, value in flags.items()})


def kind_of(scope, query):
    kind = scope["kind"]
    if kind not in ("interval", "contextual") or not scope["valid_until_exclusive"]:
        return kind
    current = date.fromisoformat(query["date"])
    start, end = (offset(scope[key], current) for key in ("valid_from", "valid_until_exclusive"))
    duration = end - start if start is not None else None
    if duration != 1:
        return "period"
    return {0: "day", 1: "tomorrow"}.get(start, "other_day")


def write_features(observation, action, signals):
    cues, last, text, spoken_duration, evidence = signals
    query = observation["query"]
    out = {}
    if "slot" in action:
        role = "ask_" + action["slot"]
    elif action["operations"][0]["op"] == "NOOP":
        role = "noop"
    else:
        op = action["operations"][0]
        candidate = op["record"]
        scope = candidate["scope"]
        role = kind_of(scope, query)
        value = normalize(candidate["value"])
        sources = candidate["source_turn_ids"]
        current = date.fromisoformat(query["date"])
        start, end = (offset(scope[key], current) for key in ("valid_from", "valid_until_exclusive"))
        target_ids = set(op.get("target_ids", []))
        prior = {r["memory_id"] for r in observation["memory"] if r["attribute"] == candidate["attribute"]
                 and r["scope"]["kind"] == scope["kind"] and r["scope"]["context"] == scope["context"]
                 and not r["scope"]["valid_until_exclusive"]}
        factual = all(q["text"] == evidence.get(q["turn_id"]) for q in candidate["evidence_quotes"])
        duration = end - start if start is not None and end is not None else None
        write_limits = len(action["operations"])
        put_flags(out, {
            "w:value_visible": value in text,
            "w:value_recent": value in last,
            "w:attribute": candidate["attribute"] == query["attribute"],
            "w:evidence_visible": factual,
            "w:evidence_contains": any(value in normalize(evidence.get(ref, "")) for ref in sources),
            "w:evidence_both": len(sources) > 1,
            "w:has_target": bool(target_ids),
            "w:same_target": target_ids == prior,
            "w:location_supported": isinstance(scope["context"], str) and normalize(scope["context"]) in text,
            "w:session_matches": scope["session_id"] == query["session_id"],
            "w:date_start_mentioned": seen_date(last, scope["valid_from"]),
            "w:date_end_mentioned": seen_date(last, scope["valid_until_exclusive"]),
            "w:previous_day_mentioned": end is not None and any(
                (date.fromisoformat(mark) - current).days + 1 == end for mark in re.findall(r"\d{4}-\d{2}-\d{2}", last)
                if _valid_iso(mark)),
            "w:duration_matches": spoken_duration is not None and duration == spoken_duration,
            "w:non_today_start": start is not None and start > 0,
            "w:both_claims": write_limits > 1,
        })
        for flag in ("date_start_mentioned", "date_end_mentioned", "previous_day_mentioned", "duration_matches",
                     "location_supported", "same_target"):
            out[f"w:{role}:{flag}"] = out["w:" + flag]
    out["w:" + role] = 1.
    for cue in cues:
        out[f"w:{role}:{cue}"] = 1.
    return out


def _valid_iso(value):
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def memory_flags(record, query):
    scope = record["scope"]
    current = date.fromisoformat(query["date"])
    start = offset(scope["valid_from"], current)
    end = offset(scope["valid_until_exclusive"], current)
    return {
        "attribute": record["attribute"] == query["attribute"],
        "expired": end is not None and end <= 0,
        "scheduled": start is not None and start > 0,
        "session_miss": scope["kind"] == "session" and scope["session_id"] != query["session_id"],
        "context_miss": scope["context"] is not None and query["context"] is not None
                        and scope["context"] != query["context"],
        "context_missing": scope["context"] is not None and query["context"] is None,
        "unresolved": scope["kind"] == "unknown" or record["status"] == "unresolved",
        "period": end is not None,
        "session": scope["kind"] == "session",
        "context": scope["context"] is not None,
        "updates": bool(record.get("supersedes")),
    }


def recall_features(observation, action):
    query = observation["query"]
    memory = observation["memory"]
    by_id = {r["memory_id"]: r for r in memory}
    facts = {identity: memory_flags(record, query) for identity, record in by_id.items()}
    available = set()
    uncertain = set()
    for identity, flags in facts.items():
        if flags["attribute"] and not any(flags[k] for k in ("expired", "scheduled", "session_miss", "context_miss")):
            if flags["context_missing"] or flags["unresolved"]:
                uncertain.add(identity)
            else:
                available.add(identity)
    role = action["mode"]
    references = set(action["evidence_ids"])
    records = [by_id[mid] for mid in references if mid in by_id]
    all_valid = available | uncertain
    replaces = {older for newer in available for older in by_id[newer].get("supersedes", [])}
    covers_uncertainty = bool(uncertain and uncertain <= references)
    covered = len(references & all_valid)
    out = {f"r:{role}": 1.}
    put_flags(out, {
        f"r:{role}:memory_empty": not by_id,
        f"r:{role}:nothing_available": not all_valid,
        f"r:{role}:some_available": bool(available),
        f"r:{role}:conflict_possible": bool(uncertain),
        f"r:{role}:covers_uncertainty": covers_uncertainty,
        f"r:{role}:has_valid": covered > 0,
        f"r:{role}:missing_ref": len(records) != len(references),
        f"r:{role}:uncovered": bool(references - all_valid),
        f"r:{role}:superseded": bool(references & replaces),
        f"r:{role}:uses_record_value": action.get("value") in {r["value"] for r in records},
        f"r:{role}:correct_branch_values": bool(action["branches"]) and
            {r["value"] for r in records} == {b["value"] for b in action["branches"]},
    })
    for identity, flags in facts.items():
        if identity in references:
            for kind, value in flags.items():
                out[f"r:{role}:ref_{kind}"] = max(out.get(f"r:{role}:ref_{kind}", 0.), float(value))
        elif identity in available:
            for kind in ("period", "session", "context", "updates"):
                out[f"r:{role}:other_{kind}"] = max(out.get(f"r:{role}:other_{kind}", 0.), float(flags[kind]))
    return out


def features(phase, observation, options):
    signals = signal_features(observation) if phase == "write" else None
    tensor = torch.zeros((len(options), WIDTH), dtype=torch.float32)
    for index, candidate in enumerate(options):
        if phase == "write":
            values = write_features(observation, candidate, signals)
        else:
            values = recall_features(observation, candidate)
        for name, value in values.items():
            tensor[index, crc32(name.encode("utf-8")) % WIDTH] += value
    return tensor


class CompactActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(WIDTH, 24), nn.Tanh(), nn.Linear(24, 1))
        self.linear = nn.Linear(WIDTH, 1, bias=False)

    def forward(self, values):
        return (self.score(values) + self.linear(values)).squeeze(-1)

    @classmethod
    def load(cls, checkpoint):
        actor = cls()
        actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        return actor.eval()

    @torch.inference_mode()
    def choose(self, phase, observation, options, key):
        scores = self(features(phase, observation, options))
        choice = int(scores.argmax())
        action = options[choice]
        trace = {"phase": phase, "action": action, "raw": json.dumps(
            {"policy": "compact", "option": choice, "scores": scores.tolist(), "key": key}, ensure_ascii=False)}
        return action, trace


def pack(rows):
    data = []
    for row, options, rewards, best in preparation(rows):
        x = features(row["phase"], row["observation"], options)
        r = torch.tensor(rewards)
        data.append((x, r, torch.isclose(r, torch.tensor(best)), row["phase"]))
    return data


def pad(data, indices):
    items = [data[i] for i in indices]
    width = max(len(item[0]) for item in items)
    x = torch.zeros((len(items), width, WIDTH))
    rewards = torch.zeros((len(items), width))
    legal = torch.zeros((len(items), width), dtype=torch.bool)
    targets = torch.zeros_like(legal)
    for index, (features_, scores, correct, _) in enumerate(items):
        n = len(features_)
        x[index, :n] = features_
        rewards[index, :n] = scores
        legal[index, :n] = True
        targets[index, :n] = correct
    return x, rewards, legal, targets


def policy(actor, features_, legal):
    return actor(features_).masked_fill(~legal, -1e4).log_softmax(-1)


def behavior(logp, legal):
    probs = (1. - EXPLORATION) * logp.exp() + EXPLORATION * legal / legal.sum(-1, keepdim=True)
    return probs.clamp_min(1e-30).log()


def learn(actor, opt, data, schedule, method, rng, initial=None, *, passes=PASSES):
    actor.train()
    varied = 0
    for batch in schedule:
        x, rewards, legal, gold = pad(data, batch)
        if method == "grpo":
            with torch.no_grad():
                reference = policy(initial, x, legal)
                old = behavior(policy(actor, x, legal), legal)
                chosen = torch.multinomial(old.exp(), DRAWS, replacement=True, generator=rng)
                returns = rewards.gather(1, chosen)
                centered = returns - returns.mean(-1, keepdim=True)
                advantage = centered / returns.std(-1, keepdim=True, unbiased=False).clamp_min(.4)
                varied += int((returns.amax(-1) > returns.amin(-1)).sum())
        for _ in range(passes):
            opt.zero_grad(set_to_none=True)
            logp = policy(actor, x, legal)
            if method == "sft":
                loss = -logp.masked_fill(~gold, -1e4).logsumexp(-1).mean()
            else:
                ratio = (behavior(logp, legal).gather(1, chosen) - old.gather(1, chosen)).exp()
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage)
                loss = -surrogate.mean() + KL * (logp.exp() * (logp - reference)).sum(-1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.)
            opt.step()
    return varied


@torch.inference_mode()
def measure(actor, data):
    counts = Counter()
    actor.eval()
    for x, r, good, phase in data:
        selected = int(actor(x).argmax())
        counts[phase] += 1
        counts[f"{phase}_hits"] += bool(good[selected])
        counts["reward"] += float(r[selected])
    return dict(counts)


def rank(stats):
    rates = [stats[f"{phase}_hits"] / stats[phase] for phase in ("write", "recall")]
    return (sum(rates) / 2, min(rates), stats["reward"])


def schedule(size, rng):
    order = torch.randperm(size, generator=rng).tolist()
    return [order[start:start + BATCH] for start in range(0, size, BATCH)]


def train(output):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    splits = {name: build_split(name) for name in ("train", "dev")}
    train_data, dev = (pack(splits[name]) for name in ("train", "dev"))
    plan = {
        "method": "feature candidate CPU ablation", "heldout_run": False, "heldout_sha256": file_hash(QA),
        "inputs_sha256": {name: digest(rows) for name, rows in splits.items()},
        "source_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in (
            "tesis_temporal_memory_compact.py", "tesis_temporal_memory_data.py",
            "tesis_temporal_memory_train.py", "tesis_temporal_memory_reward.py", "tesis_temporal_memory_policy.py")},
        "parameters": {"warmup_epochs": WARMUP, "epochs": EPOCHS, "batch": BATCH, "passes": PASSES,
                       "draws": DRAWS, "lr": LR, "exploration": EXPLORATION, "kl": KL, "clip": CLIP},
        "seeds": SEEDS,
    }
    write_json(output / "PLAN.json", plan)
    result = {}
    for seed in SEEDS:
        folder = output / str(seed)
        folder.mkdir()
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        common = CompactActor()
        opt = torch.optim.AdamW(common.parameters(), lr=LR, weight_decay=0.)
        for _ in range(WARMUP):
            learn(common, opt, train_data, schedule(len(train_data), rng), "sft", rng, passes=1)
        baseline = measure(common, dev)
        schedules = [schedule(len(train_data), rng) for _ in range(EPOCHS)]
        print(seed, "common", baseline, flush=True)
        methods = {}
        for method in ("sft", "grpo"):
            actor = deepcopy(common)
            opt = torch.optim.AdamW(actor.parameters(), lr=LR, weight_decay=0.)
            history = [baseline]
            best, weights, diversity = 0, deepcopy(common.state_dict()), 0
            for epoch, groups in enumerate(schedules, 1):
                diversity += learn(actor, opt, train_data, groups, method, rng, initial=common)
                stats = measure(actor, dev)
                history.append(stats)
                if rank(stats) > rank(history[best]):
                    best, weights = epoch, deepcopy(actor.state_dict())
            checkpoint = folder / f"{method}.pt"
            torch.save(weights, checkpoint)
            methods[method] = {"selected_epoch": best,
                               "validation": {str(epoch): stats for epoch, stats in enumerate(history)},
                               "groups_seen": len(train_data) * EPOCHS, "groups_with_signal": diversity,
                               "updates": PASSES * sum(map(len, schedules)), "adapter_sha256": file_hash(checkpoint)}
            print(seed, method, best, history[best], flush=True)
        write_json(folder / "SELECTION.json", methods)
        result[str(seed)] = methods
    write_json(output / "SELECTION.json", result)
    write_json(output / "FREEZE.json", {"plan_sha256": file_hash(output / "PLAN.json"), "selections": {
        str(seed): file_hash(output / str(seed) / "SELECTION.json") for seed in SEEDS}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    train(args.output)


if __name__ == "__main__":
    main()
