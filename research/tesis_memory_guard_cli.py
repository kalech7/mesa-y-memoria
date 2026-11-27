"""Preview the effect of the WRITE guard on a public observation and proposal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.tesis_memory_guard import MODES, plan_write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON object with 'observation' and 'proposed' (null if malformed)")
    parser.add_argument("--mode", choices=MODES, default="repair")
    parser.add_argument("--output", type=Path, help="Fresh output path. Defaults to standard output.")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"observation", "proposed"}:
            raise ValueError("Input must contain exactly observation and proposed")
        decision = plan_write(payload["observation"], payload["proposed"], mode=args.mode)
        result = json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is None:
            print(result, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
