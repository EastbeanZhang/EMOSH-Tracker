"""Copy required files from existing downloads without changing their source directories."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehm_tracker.assets import REQUIRED_FILES, check_assets, validate_model_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        required=True,
        help="Extracted checkpoint or existing tracker asset directories",
    )
    parser.add_argument("--destination", type=Path, default=Path("assets"))
    args = parser.parse_args()
    errors = []
    for relative in [p for files in REQUIRED_FILES.values() for p in files]:
        destination = args.destination / relative
        if destination.exists():
            continue
        exact = [root / relative for root in args.source if (root / relative).is_file()]
        candidates = exact or sorted(
            set(
                path
                for root in args.source
                for path in root.rglob(Path(relative).name)
                if path.is_file()
            )
        )
        if not candidates:
            errors.append(f"Not found: {relative}")
            continue
        if len(candidates) != 1:
            errors.append(f"Ambiguous: {relative}; place the correct file manually")
            continue
        try:
            validate_model_file(candidates[0])
        except ValueError as error:
            errors.append(str(error))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], destination)
        print(f"Imported {relative}")
    try:
        check_assets(args.destination)
    except (FileNotFoundError, ValueError) as error:
        errors.append(str(error))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
