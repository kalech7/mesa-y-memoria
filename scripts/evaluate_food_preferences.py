#!/usr/bin/env python3
"""Local convenience entry point; use python -m evaluation.food_preferences too."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.food_preferences import main

if __name__ == "__main__":
    raise SystemExit(main())
