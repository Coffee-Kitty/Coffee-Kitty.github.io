#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DIARY_DIR = DOCS / "life" / "diary"
SHANGHAI = dt.timezone(dt.timedelta(hours=8))
PRIVATE_PREFIXES = (
    "life/diary/",
    "drawio画图/日记/",
    "简历/",
)

DATE_IN_NAME = re.compile(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})")
H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
FRONT_DATE = re.compile(r"^date:\s*[\"']?(\d{4})[-/](\d{1,2})[-/](\d{1,2})", re.MULTILINE)
LAST_EDIT = re.compile(r"@LastEditTime:\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
CREATE_DATE = re.compile(r"@Date:\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


def as_date(match: re.Match[str] | None) -> dt.date | None:
    if not match:
        return None
    year, month, day = (int(match.group(i)) for i in range(1, 4))
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def title_from_file(path: Path, text: str) -> str:
    match = H1.search(text)
    if match:
        return clean_title(match.group(1).strip())

    stem = path.stem
    date_match = DATE_IN_NAME.search(stem)
    if date_match:
        suffix = stem[date_match.end() :].strip("-_ .")
        date_text = "-".join(date_match.groups())
        return clean_title(f"{date_text} {suffix}".strip())

    return clean_title(stem.replace("_", " ").strip())


def clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) <= 72:
        return title
    return title[:69].rstrip() + "..."


def date_from_file(path: Path, text: str) -> dt.date | None:
    for pattern in (FRONT_DATE, LAST_EDIT, CREATE_DATE):
        found = as_date(pattern.search(text))
        if found:
            return found
    return as_date(DATE_IN_NAME.search(path.stem))


def docs_link(path: Path) -> str:
    rel = path.relative_to(DOCS).as_posix()
    return "/" + quote(rel, safe="/._-%")


def is_private_path(path: Path) -> bool:
    rel = path.relative_to(DOCS).as_posix()
    return any(rel.startswith(prefix) for prefix in PRIVATE_PREFIXES)


def month_title(date: dt.date) -> str:
    return f"{date.year} 年 {date.month:02d} 月"


def collect_dated_pages() -> list[tuple[dt.date, str, Path]]:
    pages: list[tuple[dt.date, str, Path]] = []
    for path in DOCS.rglob("*.md"):
        if path.name.startswith("_"):
            continue
        if path.name in {"timeline.md"}:
            continue
        if is_private_path(path):
            continue
        text = read_text(path)
        date = date_from_file(path, text)
        if not date:
            continue
        pages.append((date, title_from_file(path, text), path))
    return sorted(pages, key=lambda item: (item[0], item[2].as_posix()), reverse=True)


def collect_diary_pages() -> list[tuple[dt.date, str, Path]]:
    pages: list[tuple[dt.date, str, Path]] = []
    for path in DIARY_DIR.glob("*.md"):
        if path.name == "README.md":
            continue
        text = read_text(path)
        date = as_date(DATE_IN_NAME.search(path.stem))
        if not date:
            continue
        pages.append((date, title_from_file(path, text), path))
    return sorted(pages, key=lambda item: (item[0], item[2].name), reverse=True)


def render_grouped_pages(
    title: str,
    description: str,
    pages: list[tuple[dt.date, str, Path]],
) -> str:
    today = dt.datetime.now(SHANGHAI).date().isoformat()
    lines = [
        f"# {title}",
        "",
        f"> {description}",
        "",
        f"> 自动生成于 {today}。更新命令：`python3 tools/build_blog_index.py`。",
        "",
        f"共 {len(pages)} 篇。",
        "",
    ]

    current_month = ""
    for date, page_title, path in pages:
        month = month_title(date)
        if month != current_month:
            current_month = month
            if lines and lines[-1] != "":
                lines.append("")
            lines.extend([f"## {month}", ""])
        lines.append(f"* {date.isoformat()} - [{page_title}]({docs_link(path)})")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    timeline = render_grouped_pages(
        "公开学习时间线",
        "按文件名或文章元信息中的日期自动汇总。默认不展示日记、简历等个人内容。",
        collect_dated_pages(),
    )
    (DOCS / "timeline.md").write_text(timeline, encoding="utf-8")

    diary_lines = [
        "# 简记",
        "",
        "> 这部分是个人记录，不再作为公开博客入口展示。",
        "",
        "日记文件仍保留在仓库中，但不会出现在侧边栏、首页和公开时间线里。",
        "如果需要真正不公开，应把 `docs/life/diary/` 移出公开仓库或迁移到私有仓库。",
        "",
    ]
    (DIARY_DIR / "README.md").write_text("\n".join(diary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
