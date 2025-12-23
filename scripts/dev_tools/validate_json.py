from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

from jsonschema import Draft202012Validator, exceptions

from scripts.dev_tools.json_config import iter_governed_files


class ValidateResult:
    def __init__(self) -> None:
        self.failed = False
        self.messages: list[str] = []


def _cache_path(cache_dir: Path, uri: str) -> Path:
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _load_schema(
    uri: str, cache_dir: Path, base_path: Path | None = None
) -> dict[str, Any]:
    parsed = urlparse(uri)

    if not parsed.scheme:
        if base_path is None:
            raise ValueError("Unsupported schema URI scheme: missing")

        local_path = (base_path.parent / uri).resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"Schema file not found: {local_path}")

        return json.loads(local_path.read_text())

    if parsed.scheme == "file":
        local_path = Path(parsed.path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Schema file not found: {local_path}")

        return json.loads(local_path.read_text())

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported schema URI scheme: {parsed.scheme or 'missing'}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, uri)
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    resp = urllib.request.urlopen(uri)  # noqa: S310 - fetching trusted schema URL
    with resp:
        content = resp.read().decode("utf-8")
    cache_file.write_text(content)
    return json.loads(content)


def validate_file(path: Path, cache_dir: Path) -> tuple[bool, str]:
    try:
        data_raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"{path}: invalid JSON ({exc})"

    if not isinstance(data_raw, dict):
        return False, f"{path}: JSON root must be an object for validation"

    data = cast(dict[str, Any], data_raw)
    schema_value = data.get("$schema")
    if not isinstance(schema_value, str):
        return False, f"{path}: missing $schema"
    schema_uri: str = schema_value

    try:
        schema = _load_schema(schema_uri, cache_dir, path)
        validator = Draft202012Validator(schema)
        raw_errors = cast(Any, validator).iter_errors(data)
        errors_iter = cast("Iterable[exceptions.ValidationError]", raw_errors)
        errors = list(sorted(errors_iter, key=lambda e: e.path))
        if errors:
            messages = [f"{list(err.path)}: {err.message}" for err in errors]
            return False, f"{path}: schema validation failed: {'; '.join(messages)}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{path}: validation error ({exc})"

    return True, f"{path}: ok"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate governed JSON files against their $schema"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional specific files/dirs; defaults to governed globs",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-file status")
    parser.add_argument(
        "--cache-dir", default=".cache/schemas", help="Schema cache directory"
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def collect_targets(root: Path, paths: Iterable[str]) -> list[Path]:
    if paths:
        targets: list[Path] = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                targets.extend(path.rglob("*.json"))
            else:
                targets.append(path)
        return targets
    return list(iter_governed_files(root))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / args.cache_dir
    targets = collect_targets(root, args.paths)

    result = ValidateResult()
    for path in targets:
        ok, msg = validate_file(path, cache_dir)
        if args.verbose or not ok:
            result.messages.append(msg)
        if not ok:
            result.failed = True

    for msg in result.messages:
        print(msg)

    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
