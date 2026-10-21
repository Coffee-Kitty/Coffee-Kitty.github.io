#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SHANGHAI = dt.timezone(dt.timedelta(hours=8))
PUBLIC_PREFIXES = (
    "research/",
    "tech/",
)

DATE_IN_NAME = re.compile(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})")
H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
FRONT_DATE = re.compile(r"^date:\s*[\"']?(\d{4})[-/](\d{1,2})[-/](\d{1,2})", re.MULTILINE)
LAST_EDIT = re.compile(r"@LastEditTime:\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
CREATE_DATE = re.compile(r"@Date:\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


def as_date(match: re.Match[str] | None) -> dt.date | None:
    if not match:
        return None
    try:
        year, month, day = (int(match.group(i)) for i in (1, 2, 3))
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


def is_public_page(path: Path) -> bool:
    rel = path.relative_to(DOCS).as_posix()
    return any(rel.startswith(prefix) for prefix in PUBLIC_PREFIXES)


def month_title(date: dt.date) -> str:
    return f"{date.year} 年 {date.month:02d} 月"


def collect_dated_pages() -> list[tuple[dt.date, str, Path]]:
    pages: list[tuple[dt.date, str, Path]] = []
    for path in DOCS.rglob("*.md"):
        if path.name.startswith("_"):
            continue
        if path.name in {"timeline.md"}:
            continue
        if not is_public_page(path):
            continue
        text = read_text(path)
        date = date_from_file(path, text)
        if not date:
            continue
        pages.append((date, title_from_file(path, text), path))
    return sorted(pages, key=lambda item: (item[0], item[2].as_posix()), reverse=True)


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


CATEGORY_META = [
    ("research/llm-agent", "LLM / Agent", "Research", "#c2603d"),
    ("research/code-agent", "Code Agent", "Research", "#2f6f68"),
    ("research/se-llm4se", "SE / LLM4SE", "Research", "#5b7db1"),
    ("tech/ai-engineering", "AI Engineering", "Tech", "#d9a441"),
    ("tech/programming", "Programming", "Tech", "#7d8f4e"),
    ("tech/dev-tools", "Dev Tools", "Tech", "#8a6fa8"),
    ("tech/algorithms", "Algorithms", "Tech", "#c25b7a"),
    ("tech/cs-courses", "CS Courses", "Tech", "#8f6b4a"),
]


def collect_category_posts() -> list[dict[str, Any]]:
    categories: list[dict[str, Any]] = []
    for prefix, name, group, color in CATEGORY_META:
        root = DOCS / prefix
        posts: list[dict[str, Any]] = []
        if root.is_dir():
            for path in sorted(root.rglob("*.md")):
                if path.name == "README.md":
                    continue
                text = read_text(path)
                rel_dir = path.parent.relative_to(root).as_posix()
                date = date_from_file(path, text)
                posts.append(
                    {
                        "title": title_from_file(path, text),
                        "url": docs_link(path),
                        "date": date.isoformat() if date else None,
                        "sub": "" if rel_dir == "." else rel_dir,
                    }
                )
        posts.sort(
            key=lambda item: (item["date"] is not None, item["date"] or "", item["title"]),
            reverse=True,
        )
        categories.append(
            {
                "id": prefix,
                "name": name,
                "group": group,
                "color": color,
                "url": "/" + prefix + "/README.md",
                "posts": posts,
            }
        )
    return categories


def write_site_graph(categories: list[dict[str, Any]]) -> None:
    payload = {
        "generated": dt.datetime.now(SHANGHAI).date().isoformat(),
        "totalPosts": sum(len(cat["posts"]) for cat in categories),
        "categories": categories,
    }
    target = DOCS / "assets" / "site-graph.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    timeline = render_grouped_pages(
        "公开学习时间线",
        "按文件名或文章元信息中的日期自动汇总。只展示 Research 和 Tech 下的公开内容。",
        collect_dated_pages(),
    )
    _ = (DOCS / "timeline.md").write_text(timeline, encoding="utf-8")
    write_site_graph(collect_category_posts())


if __name__ == "__main__":
    main()
