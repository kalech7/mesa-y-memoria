import json
from zipfile import ZipFile

import pytest

from research.review_packets import BANK, FORMS, export_packets
from research.score_ai_reviews import agreement


def test_export_blinds_and_preserves_existing_files(tmp_path):
    export_packets(BANK, tmp_path)
    export_packets(BANK, tmp_path)
    ids = []
    for side in ("reviewer_a", "reviewer_b"):
        with ZipFile(tmp_path / f"{side}.zip") as bundle:
            assert tuple(bundle.namelist()) == FORMS
            for name in FORMS:
                assert bundle.read(name) == (BANK / "review" / side / name).read_bytes()
            ids.append([case["audit_id"] for case in json.loads(bundle.read(FORMS[0]))["cases"]])
    assert ids[0] != ids[1] and len(set(ids[0]) & set(ids[1])) == 100
    existing = tmp_path / "reviewer_a.zip"
    existing.write_bytes(b"mis datos")
    with pytest.raises(FileExistsError):
        export_packets(BANK, tmp_path)
    assert existing.read_bytes() == b"mis datos"


def test_kappa_counts_chance_and_constant_categories():
    assert agreement([(1, 1), (1, 0), (0, 1), (0, 0)]) == {"agree": 2, "n": 4, "kappa": 0}
    assert agreement([(1, 1)] * 5) == {"agree": 5, "n": 5, "kappa": None}
