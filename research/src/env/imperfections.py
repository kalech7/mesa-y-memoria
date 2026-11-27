"""Opt-in, reproducible observation faults, independent of policy and gold.

These are controlled synthetic stressors, not measured rates of human errors.
Memory omission is a delivery fault during S2; SQLite and S3 remain intact.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re


VERSION = "environment_imperfections_v1"


@dataclass(frozen=True)
class ImperfectionConfig:
    seed: int = 7331
    memory_omission_rate: float = 0.0
    partial_reply_rate: float = 0.0
    unavailable_reply_rate: float = 0.0

    def __post_init__(self):
        if type(self.seed) is not int:
            raise ValueError("IMPERFECTION_SEED_INTEGER_REQUIRED")
        for field in ("memory_omission_rate", "partial_reply_rate", "unavailable_reply_rate"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("INVALID_IMPERFECTION_RATE:" + field)
        if self.partial_reply_rate + self.unavailable_reply_rate > 1:
            raise ValueError("REPLY_IMPERFECTION_RATES_EXCEED_ONE")

    @property
    def active(self):
        return bool(self.memory_omission_rate or self.partial_reply_rate or self.unavailable_reply_rate)

    def to_dict(self):
        return asdict(self)


def normalize_config(value=None):
    if value is None:
        return ImperfectionConfig()
    if isinstance(value, ImperfectionConfig):
        return value
    if isinstance(value, dict):
        try:
            return ImperfectionConfig(**value)
        except TypeError as exc:
            raise ValueError("UNKNOWN_IMPERFECTION_CONFIG_FIELD") from exc
    raise ValueError("IMPERFECTION_CONFIG_OBJECT_REQUIRED")


def uniform(config, root_id, channel, key):
    """Stable assignment: no method, phase, branch, model seed or truth input."""
    payload = json.dumps([VERSION, config.seed, root_id, channel, key], ensure_ascii=False,
                         separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def omit_memory(records, config, root_id):
    visible, omitted = [], []
    for record in records:
        mid = record["memory_id"]
        if uniform(config, root_id, "s2_memory", mid) < config.memory_omission_rate:
            omitted.append(mid)
        else:
            visible.append(deepcopy(record))
    return visible, {"input_ids": [r["memory_id"] for r in records],
                     "visible_ids": [r["memory_id"] for r in visible], "omitted_ids": omitted,
                     "scope": "S2_observation_only"}


def supported_context_fragment(replies, requested_slot, full_reply):
    """Return a literal independent context fact, withholding requested scope.

Only the existing simulator table is consulted. We deliberately do not cut a
preference sentence: dropping 'only today' could turn it into a false habit.
A standalone current-location sentence is independently meaningful. If the
requested response is already just that fact, no partial reduction is possible.
Unsupported wording fails closed to nonresponse rather than paraphrasing gold.
"""
    # Runtime import avoids a module-import cycle with UserSimulator. A neutral
    # or explicitly unavailable source must not become a definite location just
    # because a later sentence happens to match the supported grammar.
    from .user_simulator import response_availability

    full_text = full_reply["text"].strip()
    candidates = [(requested_slot, full_reply)]
    if requested_slot != "CURRENT_CONTEXT" and "CURRENT_CONTEXT" in replies:
        candidates.append(("CURRENT_CONTEXT", replies["CURRENT_CONTEXT"]))
    for source_slot, source in candidates:
        source = {"text": source} if isinstance(source, str) else source
        if response_availability(source) != "available":
            continue
        text = source.get("text", "")
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            if re.fullmatch(r"(?:Hoy )?[Ee]stoy en (?:la )?(?:casa|oficina)[.]?", sentence):
                if sentence.casefold().rstrip(".") == full_text.casefold().rstrip("."):
                    continue
                return sentence, source_slot
    return None, None
