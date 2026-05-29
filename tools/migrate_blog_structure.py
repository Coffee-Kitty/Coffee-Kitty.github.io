#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from urllib.parse import quote, unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PRIVATE = ROOT / "private"

MARKDOWN_LINK = re.compile(r"(!?\[[^\]]*\]\()([^)]+)(\))")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#", "data:")


def repo(path: str) -> Path:
    return ROOT / path


def add(mapping: dict[Path, Path], src: str, dst: str) -> None:
    mapping[repo(src)] = repo(dst)


def add_dir(mapping: dict[Path, Path], src: str, dst: str) -> None:
    src_path = repo(src)
    dst_path = repo(dst)
    if not src_path.exists():
        return
    for child in src_path.rglob("*"):
        if child.is_file():
            rel = child.relative_to(src_path)
            mapping[child] = dst_path / rel


def build_mapping() -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}

    # Whole-topic migrations.
    add_dir(mapping, "docs/传统算法学习", "docs/tech/algorithms")
    add_dir(mapping, "docs/深度学习", "docs/tech/ai-engineering/deep-learning")
    add(mapping, "docs/深度学习/test_rnn.py", "private/raw-assets/tech/deep-learning/test_rnn.py")
    add_dir(mapping, "docs/cpython", "docs/tech/programming/cpython")
    add_dir(mapping, "docs/分布式系统", "docs/tech/cs-courses/distributed-systems")
    add_dir(
        mapping,
        "docs/nju静态程序分析",
        "docs/tech/cs-courses/program-analysis/nju-static-analysis",
    )
    add_dir(
        mapping,
        "docs/北大程序分析",
        "docs/tech/cs-courses/program-analysis/pku-program-analysis",
    )

    # Research: LLM / Agent.
    for name in [
        "LLMBOOK.md",
        "PIE 数据集.md",
        "Qwen2代码学习.md",
        "Qwen2论文.md",
        "RAG.md",
        "Sentence-Bert.md",
        "evalplus.md",
        "game_to_moe.md",
        "llama.md",
        "minimind学习.md",
        "qwen.md",
        "rome.md",
        "tiny-llm.md",
        "大模型微调.md",
        "幻觉综述.md",
        "ARE.md",
        "Casper.md",
        "DINM.md",
        "LLIFT.md",
    ]:
        add(mapping, f"docs/paper/{name}", f"docs/research/llm-agent/models-data-evals/{name}")

    for name in [
        "PentestGPT.md",
        "应用大模型.md",
    ]:
        add(mapping, f"docs/paper/{name}", f"docs/research/llm-agent/agents/{name}")

    for name in [
        "agent-pro.md",
        "agent.md",
        "langchain.md",
        "openmanus.md",
        "pentestAssistant.md",
    ]:
        add(mapping, f"docs/llm/{name}", f"docs/research/llm-agent/agents/{name}")

    add(mapping, "docs/llm-lr.md", "docs/research/llm-agent/models-data-evals/tiny-universe.md")
    add(mapping, "docs/llm/sft.md", "docs/research/llm-agent/models-data-evals/sft.md")

    # Research: SE / LLM4SE.
    for name in [
        "PredEx.md",
        "buglen.md",
        "empirical.md",
        "iris.md",
        "knighter.md",
        "llm4PFA.md",
        "llm4sa.md",
        "llmdfa.md",
    ]:
        add(mapping, f"docs/paper/{name}", f"docs/research/se-llm4se/code-intelligence/{name}")

    # Tech: AI engineering.
    for name in [
        "ollama.md",
        "vllm.md",
        "显存估计.md",
    ]:
        add(mapping, f"docs/llm/{name}", f"docs/tech/ai-engineering/{name}")

    add(mapping, "docs/paper/vllm.md", "docs/tech/ai-engineering/vllm-paper.md")

    # Tech: programming.
    for name in [
        "PVZ.md",
        "python.md",
        "python爬虫.md",
        "设计模式.md",
    ]:
        add(mapping, f"docs/code/{name}", f"docs/tech/programming/{name}")

    for name in [
        "pandas学习.ipynb",
        "python爬虫.ipynb",
        "奇奇怪怪的python.ipynb",
    ]:
        add(mapping, f"docs/code/{name}", f"private/raw-assets/tech/programming/notebooks/{name}")

    for name in ["analyze_cpp.py", "analyze_java.py", "tree-sitter.py"]:
        add(mapping, f"docs/code/{name}", f"private/raw-assets/tech/programming/scripts/{name}")

    add(mapping, "docs/life/编程语言.md", "docs/tech/programming/编程语言.md")

    # Tech: dev tools.
    for name in [
        "Typora.md",
        "Typora.pdf",
        "linux代理配置.md",
        "playwright.md",
        "vscode技巧.md",
        "vscode技巧.pdf",
        "zsh.md",
    ]:
        add(mapping, f"docs/code/{name}", f"docs/tech/dev-tools/{name}")

    for name in ["network_test.ipynb", "tmp.ipynb"]:
        add(mapping, f"docs/code/{name}", f"private/raw-assets/tech/dev-tools/notebooks/{name}")

    for name in [
        "docker配置dvwa.md",
        "docker配置ssh与code-server.md",
        "dokcer配置jupter.md",
    ]:
        add(mapping, f"docs/llm/{name}", f"docs/tech/dev-tools/docker/{name}")

    add(mapping, "docs/dosify升级之路.md", "docs/tech/dev-tools/docsify-upgrade.md")
    add(mapping, "docs/timeline-plugin.js", "private/drafts/timeline-plugin.js")

    # Tech: courses and public technical notes.
    add(mapping, "docs/life/计算机导论.md", "docs/tech/cs-courses/computer-intro.md")
    add(mapping, "docs/life/资料搜索.md", "docs/tech/dev-tools/search.md")
    add(mapping, "docs/life/空间.md", "docs/research/llm-agent/models-data-evals/representation-space.md")

    for name in [
        "22551260-薛世超-金融软件分析.doc",
        "22551260-薛世超-金融软件分析.pptx",
        "A1.zip",
    ]:
        add(mapping, f"docs/nju静态程序分析/{name}", f"private/raw-assets/courses/program-analysis/nju/{name}")

    add(mapping, "docs/北大程序分析/程序分析-PKU.zip", "private/raw-assets/courses/program-analysis/pku/程序分析-PKU.zip")
    for name in ["course_project_1.zip", "course_project_2.zip"]:
        add(
            mapping,
            f"docs/北大程序分析/程序分析-PKU/绋嬪簭鍒嗘瀽-PKU/{name}",
            f"private/raw-assets/courses/program-analysis/pku/{name}",
        )

    # Public assets that belong with moved public content.
    add_dir(mapping, "docs/paper/assets", "docs/assets/research/paper")
    add_dir(mapping, "docs/llm/assets", "docs/assets/research/llm-agent")
    add_dir(mapping, "docs/llm/image", "docs/assets/research/llm-agent/image")
    add_dir(
        mapping,
        "docs/llm/langchain Simple LLM application_files",
        "docs/assets/research/llm-agent/langchain-files",
    )
    add_dir(mapping, "docs/code/assets", "docs/assets/tech/code-assets")
    add_dir(mapping, "docs/code/image", "docs/assets/tech/code-image")

    # Private content.
    add_dir(mapping, "docs/life/diary", "private/diary")
    add_dir(mapping, "docs/life/assets", "private/raw-assets/life-assets")
    add_dir(mapping, "docs/life/graduate_stu", "private/applications/graduate-stu")
    add_dir(mapping, "docs/life/实习", "private/reviews/internship")
    add_dir(mapping, "docs/drawio画图", "private/raw-assets/drawio/remaining")
    add_dir(mapping, "docs/简历", "private/resume")
    add_dir(mapping, "docs/2025", "private/applications/2025")
    add_dir(mapping, "docs/my", "private/drafts/my")

    for name in [
        "2024年末回顾.md",
        "2025年末回顾.md",
        "README.md",
        "diary.md",
        "找实习.md",
        "找实习项目.md",
        "英语学习.md",
        "那年那up那些事.md",
    ]:
        add(mapping, f"docs/life/{name}", f"private/reviews/life/{name}")

    for name in ["日记.md", "研一.md", "研究plan.md"]:
        add(mapping, f"docs/paper/{name}", f"private/research-logs/{name}")

    add(mapping, "docs/llm/graduate-design.md", "private/research-logs/graduate-design.md")
    add(mapping, "docs/memo.md", "private/drafts/memo.md")
    add(mapping, "docs/导师申请表.docx", "private/applications/导师申请表.docx")
    add(mapping, "docs/导师申请表.pdf", "private/applications/导师申请表.pdf")
    add(mapping, "docs/中国计算机学会推荐国际学术会议和期刊目录-2022（拟定）.pdf", "private/raw-assets/ccf/中国计算机学会推荐国际学术会议和期刊目录-2022（拟定）.pdf")

    # Root-level private/raw files.
    add(mapping, "MEGA-恢复密钥.txt", "private/secrets/MEGA-恢复密钥.txt")
    add(mapping, "Python源码剖析.pdf", "private/raw-assets/books/Python源码剖析.pdf")
    add(mapping, "Python高性能编程.chs.(z-lib.org).pdf", "private/raw-assets/books/Python高性能编程.chs.(z-lib.org).pdf")
    for name in [
        "case_understand.ipynb",
        "dask.ipynb",
        "rich-test.ipynb",
        "swefficiency代码理解.ipynb",
        "奇奇怪怪的python.ipynb",
    ]:
        add(mapping, name, f"private/raw-assets/notebooks/{name}")
    add_dir(mapping, "picture.asset", "docs/assets/legacy-picture")
    add_dir(mapping, "docs/picture.asset", "docs/assets/legacy-picture")

    # Research slides and notebooks that should not be first-class public pages.
    for name in [
        ".~proper展示ppt.pptx",
        "22551260-薛世超-金融软件分析.pptx",
        "Proper.pptx",
        "learn.ipynb",
        "llm-as-optimizer.pptx",
        "minimnd.ipynb",
        "numpy学习.ipynb",
        "proper展示ppt.pptx",
        "sentence-transformer.py",
        "tiny-llm.ipynb",
        "实用科研.pptx",
        "手撕atten.ipynb",
    ]:
        add(mapping, f"docs/paper/{name}", f"private/raw-assets/research/paper/{name}")

    add(mapping, "docs/llm/test_sft.html", "docs/assets/research/llm-agent/models-data-evals/test_sft.html")

    for name in [
        "langchain Simple LLM application.ipynb",
        "test_sft.ipynb",
    ]:
        add(mapping, f"docs/llm/{name}", f"private/raw-assets/research/llm/{name}")

    for name in [
        "QQ_1741180290375.png",
        "image.png",
        "优设标题黑.ttf",
    ]:
        add(mapping, f"docs/code/{name}", f"private/raw-assets/code/{name}")

    add_dir(mapping, "docs/code/test", "private/raw-assets/code/test")
    add_dir(mapping, "docs/code/.ipynb_checkpoints", "private/raw-assets/code/ipynb-checkpoints")

    # Keep generated macOS archive metadata out of public content.
    for child in DOCS.rglob(".DS_Store"):
        rel = child.relative_to(DOCS)
        mapping[child] = PRIVATE / "raw-assets" / "junk" / "ds-store" / rel
    for macos_dir in DOCS.rglob("__MACOSX"):
        add_dir(mapping, str(macos_dir.relative_to(ROOT)), f"private/raw-assets/junk/{macos_dir.relative_to(DOCS)}")

    return mapping


def is_external(raw: str) -> bool:
    raw = raw.strip()
    return raw.startswith(EXTERNAL_PREFIXES)


def split_target(raw: str) -> tuple[str, str]:
    for sep in ("#", "?"):
        idx = raw.find(sep)
        if idx != -1:
            return raw[:idx], raw[idx:]
    return raw, ""


def docs_url(path: Path) -> str:
    rel = path.relative_to(DOCS).as_posix()
    return "/" + quote(rel, safe="/._-%")


def resolve_link(source: Path, raw_target: str) -> Path:
    target, _suffix = split_target(raw_target)
    target = unquote(target.strip())
    if target.startswith("/"):
        return DOCS / target.lstrip("/")
    return source.parent / target


def moved_target(path: Path, mapping: dict[Path, Path]) -> Path | None:
    if path in mapping:
        return mapping[path]
    if path.suffix == "":
        readme = path / "README.md"
        if readme in mapping:
            return mapping[readme]
        markdown = path.with_suffix(".md")
        if markdown in mapping:
            return mapping[markdown]
    return None


def rewrite_markdown_links(text: str, old_source: Path, mapping: dict[Path, Path]) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix, raw, suffix = match.groups()
        raw = raw.strip()
        if is_external(raw):
            return match.group(0)

        target_part, anchor = split_target(raw)
        if not target_part:
            return match.group(0)

        old_target = resolve_link(old_source, target_part)
        new_target = moved_target(old_target, mapping)

        if new_target is None and old_target.exists() and DOCS in old_target.parents:
            new_target = old_target

        if new_target is None:
            return match.group(0)

        if new_target == old_target and target_part.startswith("/"):
            return match.group(0)

        if DOCS in new_target.parents:
            return f"{prefix}{docs_url(new_target)}{anchor}{suffix}"

        return match.group(0)

    return MARKDOWN_LINK.sub(replace, text)


def move_files(mapping: dict[Path, Path], dry_run: bool) -> dict[Path, Path]:
    moved: dict[Path, Path] = {}
    for src, dst in sorted(mapping.items(), key=lambda item: len(item[0].parts)):
        if not src.exists():
            continue
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        moved[src] = dst
        print(f"MOVE {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        if dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return moved


def rewrite_docs(mapping: dict[Path, Path], moved: dict[Path, Path], dry_run: bool) -> None:
    origin_by_dest = {dst: src for src, dst in moved.items()}
    markdown_files = list(DOCS.rglob("*.md"))
    for path in markdown_files:
        old_source = origin_by_dest.get(path, path)
        original = path.read_text(encoding="utf-8", errors="ignore")
        updated = rewrite_markdown_links(original, old_source, mapping)
        if updated == original:
            continue
        print(f"REWRITE {path.relative_to(ROOT)}")
        if not dry_run:
            path.write_text(updated, encoding="utf-8")


def prune_empty_dirs(dry_run: bool) -> None:
    for path in sorted(DOCS.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_dir():
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            print(f"RMDIR {path.relative_to(ROOT)}")
            if not dry_run:
                path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mapping = build_mapping()
    moved = move_files(mapping, args.dry_run)
    if not args.dry_run:
        rewrite_docs(mapping, moved, args.dry_run)
    prune_empty_dirs(args.dry_run)


if __name__ == "__main__":
    main()
