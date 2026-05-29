#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def candidates_for(source: Path, raw: str) -> list[Path]:
    target = raw.split("#", 1)[0].split("?", 1)[0].strip()
    target = unquote(target)
    if target.startswith("/"):
        path = DOCS / target.lstrip("/")
    else:
        path = source.parent / target

    candidates = [path]
    if path.suffix == "":
        candidates.extend([path / "README.md", path.with_suffix(".md")])
    return candidates


def is_external(raw: str) -> bool:
    return raw.startswith(("http://", "https://", "mailto:", "#"))


def main() -> int:
    missing: list[tuple[Path, int, str]] = []

    for path in DOCS.rglob("*.md"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        in_code_block = False
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line_no, line in enumerate(lines, 1):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            for match in LINK.finditer(line):
                raw = match.group(1).strip()
                if not raw or is_external(raw):
                    continue
                if not any(candidate.exists() for candidate in candidates_for(path, raw)):
                    missing.append((path, line_no, raw))

    for path, line_no, raw in missing:
        rel = path.relative_to(ROOT)
        print(f"{rel}:{line_no}: missing internal link: {raw}")

    if missing:
        print(f"\n{len(missing)} missing internal links")
        return 1

    print("No missing internal links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
