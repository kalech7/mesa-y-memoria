import json
import os
import time
import urllib.error
import urllib.request


class TechnicalError(RuntimeError):
    def __init__(self, kind, attempts=None, status=None):
        super().__init__(kind)
        self.kind, self.attempts, self.status = kind, attempts or [], status


def http_json(url, payload=None, timeout=20, token=""):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # No copiar cuerpo HTTP, URL ni credenciales a los resultados.
        raise TechnicalError("http_error", status=exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise TechnicalError("connection_or_timeout") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise TechnicalError("invalid_http_json") from None


class API:
    def __init__(self, base, model, timeout=60, max_tokens=4096, seed=42, temperature=0,
                 token_env="VLLM_API_KEY", recorder=None, name="agent"):
        self.base, self.model = base.rstrip("/"), model
        self.timeout, self.max_tokens, self.seed, self.temperature = timeout, max_tokens, seed, temperature
        self.token = os.environ.get(token_env, "")
        self.recorder, self.name = recorder, name

    def complete(self, messages, max_tokens=None):
        return self._complete(messages, max_tokens, structured=True)

    def complete_text(self, messages, max_tokens=None):
        """Respuesta natural: sin esquema JSON ni clasificación solicitada al modelo."""
        return self._complete(messages, max_tokens, structured=False)

    def _complete(self, messages, max_tokens=None, structured=True):
        attempts, limit = [], max_tokens or self.max_tokens
        for number in range(2):
            record = {"client": self.name, "model": self.model, "messages": messages,
                      "temperature": self.temperature, "seed": self.seed, "max_tokens": limit,
                      "attempt": number + 1}
            payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                       "seed": self.seed, "max_tokens": limit}
            if structured:
                record["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "evaluation_object", "schema": {"type": "object", "additionalProperties": True}}}
                payload["response_format"] = record["response_format"]
            start = time.monotonic()
            try:
                data = http_json(self.base + "/chat/completions", payload, self.timeout, self.token)
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise TechnicalError("missing_choices")
                choice = choices[0]
                content = (choice.get("message") or {}).get("content")
                usage = data.get("usage") or {}
                record.update(finish_reason=choice.get("finish_reason"), content=content, usage=usage,
                    reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    model_returned=data.get("model"), system_fingerprint=data.get("system_fingerprint"))
                if not isinstance(content, str) or not content.strip():
                    raise TechnicalError("empty_content")
                if choice.get("finish_reason") == "length":
                    raise TechnicalError("token_limit")
                obj = content
                if structured:
                    try:
                        obj = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
                    except json.JSONDecodeError:
                        raise TechnicalError("parse_error") from None
                    if not isinstance(obj, dict):
                        raise TechnicalError("parse_error")
                record["latency_seconds"] = time.monotonic() - start
                attempts.append(record)
                if self.recorder:
                    self.recorder(record)
                return obj, attempts
            except TechnicalError as exc:
                record.update(error=exc.kind, http_status=exc.status, latency_seconds=time.monotonic() - start)
                attempts.append(record)
                if self.recorder:
                    self.recorder(record)
                if number == 0 and exc.kind in {"empty_content", "token_limit"}:
                    limit = min(limit * 2, 16384)
                    continue
                raise TechnicalError(exc.kind, attempts, exc.status) from None
            except (KeyError, TypeError, AttributeError, IndexError):
                record.update(error="invalid_api_schema", latency_seconds=time.monotonic() - start)
                attempts.append(record)
                if self.recorder:
                    self.recorder(record)
                raise TechnicalError("invalid_api_schema", attempts) from None

