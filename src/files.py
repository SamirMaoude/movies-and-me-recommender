"""Read and write JSON files, and fingerprint files with SHA-256."""

import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    """Write indented UTF-8 JSON with LF line endings, creating the parent folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
