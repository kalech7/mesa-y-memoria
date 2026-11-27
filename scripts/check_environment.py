"""Check the CPU test environment or the v6 inference bundle without opening a server."""
from __future__ import annotations

import argparse
from importlib import import_module
from importlib.metadata import version
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-v6", action="store_true", help="verificar pesos y CUDA para el chat v6")
    args = parser.parse_args()

    packages = {"torch": "torch", "transformers": "transformers", "peft": "peft", "safetensors": "safetensors"}
    if not args.demo_v6:
        packages.update({
            "pytest": "pytest", "yaml": "PyYAML", "numpy": "numpy",
            "tensorboard": "tensorboard", "jsonschema": "jsonschema", "playwright.sync_api": "playwright",
        })
    errors = []
    for name, distribution in packages.items():
        try:
            import_module(name)
            print(f"{distribution} {version(distribution)}")
        except (ImportError, OSError, RuntimeError) as error:
            errors.append(f"{distribution}: {error}")

    if args.demo_v6:
        from web_demo.timing_service import TimingLab, checked_adapter

        try:
            lab = TimingLab()
            for arm, seed in lab.deployed.items():
                checked_adapter(lab.run, str(seed), arm, lab.lock["selected"][str(seed)][arm])
            print("Modelo Qwen2.5-1.5B-Instruct, adaptadores y selección v6 verificados.")
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"Archivos de la demo v6: {error}")
        torch = sys.modules.get("torch")
        if torch is not None and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            errors.append("La demo v6 requiere una GPU CUDA con soporte bfloat16.")

    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print("El entorno está preparado.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
