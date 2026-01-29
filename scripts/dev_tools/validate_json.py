from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

from scripts.dev_tools.json_config import iter_governed_files

_jsonschema_spec = importlib.util.find_spec("jsonschema")
if _jsonschema_spec is None:
    _jsonschema_module = None
else:
    _jsonschema_module = importlib.import_module("jsonschema")


class ValidateResult:
    """
    Aggregate validation outcomes for the validate_json CLI.

    Purpose:
        Track whether any validation failed and collect per-file messages for
        user-facing reporting.

    Usage:
        Instantiate once per run, update `failed` and append to `messages`
        as files are processed.

    Flow:
        1. Initialize with defaults.
        2. Mark `failed` when a validation error occurs.
        3. Collect messages and print them at the end of execution.

    Invariants / Constraints:
        `messages` always contains strings that are safe to print directly.

    Side Effects:
        None. This class only stores state for the caller.

    Attributes:
        failed (bool): Indicates whether any validation failed.
        messages (list[str]): Collected status or error messages.
    """

    def __init__(self) -> None:
        """
        Initialize an empty validation result container.

        Purpose:
            Provide default state for a new validation run.

        Args:
            None.

        Returns:
            None: The instance is initialized in-place.

        Raises:
            None.

        Side Effects:
            Sets `failed` to False and initializes `messages` to an empty list.
        """
        self.failed = False
        self.messages: list[str] = []


def _cache_path(cache_dir: Path, uri: str) -> Path:
    """
    Build a deterministic cache path for a schema URI.

    Purpose:
        Convert schema URLs into stable filenames so cached schemas can be reused.

    Args:
        cache_dir (Path): Directory where cached schema files live.
        uri (str): Schema URI used to generate the cache key.

    Returns:
        Path: Full path for the cached schema file.

    Raises:
        None.

    Side Effects:
        None.
    """
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _collect_schema_errors(
    schema: Mapping[str, Any], data: Mapping[str, Any]
) -> list[str]:
    """
    Validate data against a minimal subset of JSON Schema keywords.

    Purpose:
        Provide a lightweight validator when jsonschema is unavailable,
        supporting the schema features used in tests.

    Args:
        schema (Mapping[str, Any]): JSON schema dictionary to validate against.
        data (Mapping[str, Any]): Parsed JSON object to validate.

    Returns:
        list: Human-readable error strings describing validation failures.

    Raises:
        ValueError: When the schema expects a non-object root but data is not a dict.

    Side Effects:
        None.
    """
    errors: list[str] = []
    schema_type = schema.get("type")
    # Reject non-object roots when schema expects an object.
    if schema_type == "object" and not isinstance(data, dict):
        raise ValueError("Schema expects an object at the root.")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Track missing required properties for clear diagnostic output.
    for key in required:
        if key not in data:
            errors.append(f"['{key}']: is a required property")

    # Validate types for schema-defined properties present in the data.
    for key, descriptor in properties.items():
        if key not in data:
            continue
        expected = descriptor.get("type")
        if expected == "number" and not isinstance(data[key], int | float):
            errors.append(f"['{key}']: expected number")

    return errors


def _load_schema(
    uri: str, cache_dir: Path, base_path: Path | None = None
) -> dict[str, Any]:
    """
    Resolve a JSON schema from local paths or trusted remote endpoints.

    Purpose:
        Load a JSON schema from a relative path, file URI, or a remote HTTPS
        endpoint while caching downloads for repeatable validation runs.

    Args:
        uri (str): Schema URI or relative path to resolve.
        cache_dir (Path): Directory used to store cached schema downloads.
        base_path (Path | None): Optional base file path for relative schemas.

    Returns:
        dict[str, Any]: Parsed JSON schema payload.

    Raises:
        ValueError: When the schema URI cannot be resolved or uses an unsupported
            scheme.
        FileNotFoundError: When a local schema file cannot be found.

    Side Effects:
        May create cache directories and write cached schema files to disk.
    """
    parsed = urlparse(uri)

    # Treat missing schemes as relative paths scoped to the referring file.
    if not parsed.scheme:
        if base_path is None:
            raise ValueError("Unsupported schema URI scheme: missing")

        local_path = (base_path.parent / uri).resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"Schema file not found: {local_path}")

        return json.loads(local_path.read_text())

    # Resolve explicit file:// URIs directly from disk.
    if parsed.scheme == "file":
        local_path = Path(parsed.path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Schema file not found: {local_path}")

        return json.loads(local_path.read_text())

    # Only allow HTTP/S for remote schema lookups.
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported schema URI scheme: {parsed.scheme or 'missing'}")

    # Normalize the json-schema.org endpoint to HTTPS for strict transport safety.
    if parsed.scheme == "http" and parsed.netloc == "json-schema.org":
        uri = parsed._replace(scheme="https", fragment="").geturl()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, uri)
    # Reuse cached schema payloads to avoid repeat network calls.
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    request = (
        urllib.request.Request(  # noqa: S310 - trusted HTTPS endpoint: json-schema.org
            uri, headers={"User-Agent": "lexile-corpus-tuner-json-validator"}
        )
    )
    resp = (
        urllib.request.urlopen(  # noqa: S310 - trusted HTTPS endpoint: json-schema.org
            request, timeout=30
        )
    )
    with resp:
        content = resp.read().decode("utf-8")
    cache_file.write_text(content)
    return json.loads(content)


def validate_file(path: Path, cache_dir: Path) -> tuple[bool, str]:
    """
    Validate a JSON file against its declared $schema.

    Purpose:
        Load a JSON document, resolve its schema, and report validation
        success or a descriptive error message.

    Args:
        path (Path): Path to the JSON file being validated.
        cache_dir (Path): Directory used to cache fetched schemas.

    Returns:
        tuple[bool, str]: A tuple of success flag and a human-readable message.

    Raises:
        None: All validation errors are returned as part of the result tuple.

    Side Effects:
        Reads the JSON file, may write cached schemas to disk.
    """
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
        # Branch by whether the optional jsonschema dependency is available.
        if _jsonschema_module is None:
            errors = _collect_schema_errors(schema, data)
            if errors:
                return (
                    False,
                    f"{path}: schema validation failed: {'; '.join(errors)}",
                )
        else:
            validator = _jsonschema_module.Draft202012Validator(schema)
            errors_iter = validator.iter_errors(data)
            errors = list(sorted(errors_iter, key=lambda e: e.path))
            if errors:
                messages = [f"{list(err.path)}: {err.message}" for err in errors]
                return (
                    False,
                    f"{path}: schema validation failed: {'; '.join(messages)}",
                )
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
