#!/usr/bin/env python3
"""Deterministic public-tree credential and machine-path scanner.

The scanner prints only file names and counts. It never dumps matching text.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

FORBIDDEN_NAMES = {".env", "wallets.txt", "private_keys.txt", "addresses.txt", "proxies.txt", "inventory.json", "transfer-results.json"}
# A bare 64-hex public SHA vector is not a credential. Require the
# conventional 0x private-key spelling or a key-named quoted assignment.
PRIVATE_KEY = re.compile(r"0x[0-9a-f]{64}\b|(?:private[_-]?key|secret|seed|mnemonic)\s*[=:]\s*['\"](?:0x)?[0-9a-f]{64}['\"]", re.I)
KEY_URL = re.compile(r"https?://[^\s\"']+/(?:v2|key|token|auth)/[A-Za-z0-9_-]{12,}", re.I)
SECRET_ASSIGNMENT = re.compile(r"(?:password|private[_-]?key|api[_-]?key|secret|access[_-]?token)\s*[=:]\s*['\"][^'\"]{8,}['\"]", re.I)
ABSOLUTE_WINDOWS = re.compile(r"\b[A-Za-z]:[\\/][^\s\"']+")
PRIVATE_HOST = re.compile(r"\b(?:root@)?(?:\d{1,3}\.){3}\d{1,3}:\d+\b")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    hits: list[tuple[str, str]] = []
    files = 0
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(args.root).as_posix()
        if path.name in FORBIDDEN_NAMES or path.name.startswith(".env") and path.name != ".env.example":
            hits.append((rel, "forbidden-name"))
            continue
        try:
            data = path.read_bytes()
        except OSError:
            hits.append((rel, "unreadable"))
            continue
        files += 1
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        for name, pattern in (("private-key-shape", PRIVATE_KEY), ("key-url", KEY_URL), ("secret-assignment", SECRET_ASSIGNMENT), ("windows-path", ABSOLUTE_WINDOWS), ("private-host", PRIVATE_HOST)):
            if pattern.search(text):
                hits.append((rel, name))
    print(f"files_scanned={files}")
    print(f"findings={len(hits)}")
    for rel, kind in hits:
        print(f"finding={kind} file={rel}")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
