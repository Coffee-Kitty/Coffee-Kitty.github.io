#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK = re.compile(r"(!?\[[^\]]*\]\()([^)]+)(\))")
HTML_ATTR = re.compile(r"\b(src|href)=(['\"])(.*?)\2")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#", "data:")
LEGACY_PICTURE_MARKER = "picture.asset/"
LEGACY_PICTURE_DIR = DOCS / "assets" / "legacy-picture"

SPECIAL_TARGETS = {
    ROOT / "picture.asset" / "compass.png": DOCS / "assets" / "research" / "misc" / "compass.png",
    DOCS / "life" / "assets" / "空间" / "image.png": DOCS / "assets" / "research" / "llm-agent" / "representation-space" / "image.png",
    DOCS / "life" / "assets" / "资料搜索" / "image.png": DOCS / "assets" / "tech" / "dev-tools" / "search" / "image.png",
    DOCS / "code" / "QQ_1741180290375.png": DOCS / "assets" / "tech" / "dev-tools" / "vscode" / "QQ_1741180290375.png",
    DOCS / "code" / "image.png": DOCS / "assets" / "tech" / "dev-tools" / "vscode" / "image.png",
}


def norm(path: Path) -> Path:
    return Path(os.path.normpath(str(path)))


def split_target(raw: str) -> tuple[str, str]:
    for sep in ("#", "?"):
        idx = raw.find(sep)
        if idx != -1:
            return raw[:idx], raw[idx:]
    return raw, ""


def docs_url(path: Path) -> str:
    rel = path.relative_to(DOCS).as_posix()
    return "/" + quote(rel, safe="/._-%")


def resolve(source: Path, target: str) -> Path:
    target = unquote(target.strip())
    if target.startswith("/"):
        return norm(DOCS / target.lstrip("/"))
    return norm(source.parent / target)


def replacement_url(source: Path, raw: str) -> str | None:
    target, suffix = split_target(raw.strip())
    if not target or raw.startswith(EXTERNAL_PREFIXES):
        return None

    if LEGACY_PICTURE_MARKER in target:
        legacy_name = target.split(LEGACY_PICTURE_MARKER, 1)[1]
        new = norm(LEGACY_PICTURE_DIR / unquote(legacy_name))
        if new.exists():
            return docs_url(new) + suffix
        return None

    old = resolve(source, target)
    new = SPECIAL_TARGETS.get(old, old)

    if DOCS in new.parents and new.exists():
        return docs_url(new) + suffix

    return None


def main() -> None:
    changed = 0
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")

        def replace(match: re.Match[str]) -> str:
            prefix, raw, suffix = match.groups()
            new_url = replacement_url(path, raw)
            if not new_url or new_url == raw.strip():
                return match.group(0)
            return f"{prefix}{new_url}{suffix}"

        def replace_html_attr(match: re.Match[str]) -> str:
            attr, quote_char, raw = match.groups()
            new_url = replacement_url(path, raw)
            if not new_url or new_url == raw.strip():
                return match.group(0)
            return f"{attr}={quote_char}{new_url}{quote_char}"

        updated = LINK.sub(replace, text)
        updated = HTML_ATTR.sub(replace_html_attr, updated)
        if updated == text:
            continue
        path.write_text(updated, encoding="utf-8")
        changed += 1
        print(f"normalized {path.relative_to(ROOT)}")
    print(f"normalized_files={changed}")


if __name__ == "__main__":
    main()
