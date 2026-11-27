"""Local model used by integration tests that assert Qwen's exact token IDs."""
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def qwen_05b_dir():
    path = Path(__file__).resolve().parents[2] / "models" / "Qwen2.5-0.5B-Instruct"
    if not path.exists():
        pytest.skip(f"Optional Qwen2.5-0.5B-Instruct model bundle absent: {path}")
    return path
