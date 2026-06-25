#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LEGACY = DOCS / "assets" / "legacy-picture"
PRIVATE_UNUSED = ROOT / "private" / "raw-assets" / "legacy-picture-unused"
LEGACY_REF = re.compile(r"/assets/legacy-picture/([^\)\"\s>]+)")


def referenced_legacy_files() -> set[str]:
    refs: set[str] = set()
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        refs.update(unquote(match) for match in LEGACY_REF.findall(text))
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description="Move unreferenced legacy public images into private raw assets.")
    parser.add_argument("--move", action="store_true", help="Move files instead of only reporting.")
    args = parser.parse_args()

    refs = referenced_legacy_files()
    files = {path.name: path for path in LEGACY.glob("*") if path.is_file()}
    unused = sorted(set(files) - refs)
    missing = sorted(refs - set(files))

    unused_bytes = sum(files[name].stat().st_size for name in unused)
    print(f"legacy_files={len(files)}")
    print(f"referenced={len(refs)}")
    print(f"unused={len(unused)}")
    print(f"unused_mb={unused_bytes / 1024 / 1024:.1f}")
    print(f"missing={len(missing)}")

    if missing:
        for name in missing[:50]:
            print(f"missing: {name}")
        return 1

    if not args.move:
        for name in unused[:80]:
            print(f"unused: {name}")
        return 0

    PRIVATE_UNUSED.mkdir(parents=True, exist_ok=True)
    for name in unused:
        src = files[name]
        dst = PRIVATE_UNUSED / name
        if dst.exists():
            raise FileExistsError(dst)
        shutil.move(str(src), str(dst))
    print(f"moved={len(unused)}")
    print(f"destination={PRIVATE_UNUSED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
