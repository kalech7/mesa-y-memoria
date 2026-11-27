"""Distribute and reconcile the two blind packets of the frozen thesis bank v1."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from research.src.artifacts import file_hash
from research.tesis_bank_review import reconcile_review


BANK = Path(__file__).resolve().parent / "data/tesis_bank_v1"
MANIFEST_SHA256 = "74468df63da63bc2e9b8654e37248387cf9529dbbb2e0b14aee2f1f51a722acf"
FORMS = ("01_casos.json", "02_replicas.json", "03_respuestas_en_blanco.json")


def verify_materials(bank):
    manifest_path = bank / "manifest.json"
    if file_hash(manifest_path) != MANIFEST_SHA256:
        raise ValueError("El manifiesto difiere del banco v1 congelado.")
    manifest = json.loads(manifest_path.read_bytes())
    for item in (*manifest["partitions"].values(), *manifest["teachers"].values(), manifest["audit_contracts"]):
        if file_hash(bank / item["path"]) != item["sha256"]:
            raise ValueError(f"El archivo difiere del manifiesto: {item['path']}")
    for item in manifest["review"]["artifacts"].values():
        if file_hash(bank / "review" / item["path"]) != item["sha256"]:
            raise ValueError(f"El material de revisión difiere del manifiesto: {item['path']}")
    return manifest


def export_packets(bank, output):
    verify_materials(bank)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as staging:
        for side in ("reviewer_a", "reviewer_b"):
            temp = Path(staging) / f"{side}.zip"
            with ZipFile(temp, "w") as archive:
                for name in FORMS:
                    data = (bank / "review" / side / name).read_bytes()
                    archive.writestr(ZipInfo(name), data, compress_type=ZIP_DEFLATED)
            destination = output / temp.name
            if destination.exists():
                if file_hash(destination) != file_hash(temp):
                    raise FileExistsError(f"Ya existe otro archivo en {destination}")
            else:
                destination.hardlink_to(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output", required=True, type=Path)
    review = sub.add_parser("reconcile-human")
    review.add_argument("--review-a", required=True, type=Path)
    review.add_argument("--review-b", required=True, type=Path)
    review.add_argument("--adjudications", type=Path)
    review.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "export":
        export_packets(BANK, args.output)
        return 0
    result = reconcile_review(BANK, verify_materials(BANK), args.review_a, args.review_b,
                              args.output, adjudications=args.adjudications)
    print(json.dumps({k: result[k] for k in ("status", "roots", "expected_roots", "unresolved_audit_ids")}))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
