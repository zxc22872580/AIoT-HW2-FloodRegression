# -*- coding: utf-8 -*-
"""
Rename every 7114056XXX placeholder to your real student ID and build the
submission zip.

Usage
    python package_for_submission.py 7114056001
    python package_for_submission.py 7114056001 --dry-run

What it does
    1. Replaces the placeholder inside .py / .md files (text only).
    2. Renames files whose NAME contains the placeholder.
    3. Zips the deliverables into <ID>_hw2.zip.

It never deletes anything, and --dry-run shows the plan without touching disk.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

PLACEHOLDER = "7114056XXX"
ROOT = Path(__file__).resolve().parent

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json"}
# Directories copied wholesale into the zip
INCLUDE_DIRS = ["outputs", "data"]
# Files that must not be shipped (raw Kaggle CSVs are large and redistributable
# only under the competition rules)
EXCLUDE_NAMES = {"train.csv", "test.csv", "sample_submission.csv"}
# The raw .gz downloads are duplicated in decompressed form under data/, so
# only the decompressed copies ship. data/original.csv IS kept despite its
# size: section 3.4 of the report proves the generative rule from it, and a
# claim that cannot be re-run from the submitted archive is not evidence.
EXCLUDE_DIRS = {"kaggle_mirror"}
EXCLUDE_LARGE: set[str] = set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("student_id", help="e.g. 7114056001")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sid = args.student_id.strip()
    if not re.fullmatch(r"\w{6,20}", sid):
        print(f"ERROR: '{sid}' does not look like a student ID.")
        return 1
    if sid == PLACEHOLDER:
        print("ERROR: pass your REAL student ID, not the placeholder.")
        return 1

    act = "[dry-run] " if args.dry_run else ""

    # ---- 1. rewrite text content ---------------------------------------- #
    print(f"{act}Replacing '{PLACEHOLDER}' -> '{sid}' inside text files")
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if p.name == Path(__file__).name:
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if PLACEHOLDER not in txt:
            continue
        n = txt.count(PLACEHOLDER)
        print(f"    {p.relative_to(ROOT)}  ({n} occurrence(s))")
        if not args.dry_run:
            p.write_text(txt.replace(PLACEHOLDER, sid), encoding="utf-8")

    # ---- 2. rename files ------------------------------------------------- #
    print(f"\n{act}Renaming files containing the placeholder")
    for p in sorted(ROOT.rglob(f"*{PLACEHOLDER}*")):
        if not p.is_file():
            continue
        tgt = p.with_name(p.name.replace(PLACEHOLDER, sid))
        print(f"    {p.name}  ->  {tgt.name}")
        if not args.dry_run:
            if tgt.exists():
                print(f"      SKIPPED: {tgt.name} already exists")
                continue
            p.rename(tgt)

    # ---- 3. build the zip ------------------------------------------------ #
    zip_path = ROOT / f"{sid}_hw2.zip"
    print(f"\n{act}Building {zip_path.name}")

    members: list[Path] = []
    for pat in (f"{sid}_hw2.py", "REPORT.md", "README.md",
                "package_for_submission.py", "make_report_pdf.py", "*.pdf"):
        members.extend(sorted(ROOT.glob(pat)))
    for d in INCLUDE_DIRS:
        dp = ROOT / d
        if dp.is_dir():
            members.extend(sorted(q for q in dp.rglob("*") if q.is_file()))

    kept, skipped = [], []
    for m in members:
        rel = m.relative_to(ROOT)
        drop = (m.name in EXCLUDE_NAMES
                or m.name in EXCLUDE_LARGE
                or any(part in EXCLUDE_DIRS for part in rel.parts))
        (skipped if drop else kept).append(m)

    if not args.dry_run:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for m in kept:
                zf.write(m, m.relative_to(ROOT))

    total = sum(m.stat().st_size for m in kept)
    print(f"    {len(kept)} file(s), {total/1e6:.2f} MB uncompressed")
    for m in kept:
        print(f"      + {m.relative_to(ROOT)}")
    for m in skipped:
        print(f"      - {m.relative_to(ROOT)}  (excluded: raw competition data)")

    if args.dry_run:
        print("\nDry run only - nothing was changed.")
    else:
        print(f"\nDone -> {zip_path.name}")

        todo = []
        if not any(m.suffix.lower() == ".pdf" and "report" in m.name.lower()
                   for m in kept):
            todo.append("報告 PDF          執行 python make_report_pdf.py")
        if not any("gpt" in m.name.lower() and m.suffix.lower() == ".pdf"
                   for m in kept):
            todo.append("GPT_conversation.pdf  以 pdfCrowd 匯出 ChatGPT 對話")
        if "NotebookLM 研究摘要" in (ROOT / "REPORT.md").read_text(
                encoding="utf-8") and "摘要草稿" in (ROOT / "REPORT.md").read_text(
                encoding="utf-8"):
            todo.append("NotebookLM 摘要   取代 REPORT.md 第 11 節的草稿")
        if not (ROOT / "data" / "train.csv").exists():
            todo.append(
                "官方完整資料   （選用）放入 data/train.csv（1,117,957 列）後重跑。"
                "目前使用 5,000 列公開鏡像抽樣，分析有效但精度較低")

        if todo:
            print("\n繳交前仍待處理：")
            for t in todo:
                print(f"  * {t}")
        else:
            print("\n所有項目皆已齊備。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
