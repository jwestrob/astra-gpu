#!/usr/bin/env python3
"""Create or validate a plan7-gpu pressed-profile manifest."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "python" / "plan7_gpu"


def _load_manifest_module():
    # Loading under a private package keeps this offline verifier independent of
    # the plan7_gpu facade, which imports the optional CUDA extension.
    package_name = "_plan7_gpu_manifest_cli"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[package_name] = package
    name = f"{package_name}.pressed_manifest"
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE_DIR / "pressed_manifest.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pressed-manifest implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser(
        "create", help="fully audit a database and write a manifest"
    )
    create.add_argument("database", type=Path)
    create.add_argument("output", type=Path)
    create.add_argument("--force", action="store_true")

    validate = commands.add_parser(
        "validate", help="authenticate a database against a manifest"
    )
    validate.add_argument("database", type=Path)
    validate.add_argument("manifest", type=Path)
    validate.add_argument(
        "--require-stat-token",
        action="store_true",
        help="reject a copied database instead of checking its SHA256 hashes",
    )

    args = parser.parse_args()
    module = _load_manifest_module()
    try:
        if args.command == "create":
            manifest = module.create_pressed_manifest(
                args.database, args.output, overwrite=args.force
            )
            print(
                f"audited {manifest['database']['model_count']} models; "
                f"wrote {args.output}"
            )
        else:
            result = module.validate_pressed_manifest(
                args.database,
                args.manifest,
                allow_hash_fallback=not args.require_stat_token,
            )
            method = "SHA256" if result.used_content_hashes else "stat token"
            print(f"validated {result.model_count} models by {method}")
    except (module.PressedManifestError, OSError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
