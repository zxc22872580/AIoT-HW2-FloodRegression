# -*- coding: utf-8 -*-
"""
將 REPORT.md 轉成適合繳交的中文 PDF。

路線：Markdown -> HTML（內嵌列印用 CSS、指定微軟正黑體）-> Edge/Chrome 無頭列印

為什麼走這條路
    Windows 上 pandoc / wkhtmltopdf / weasyprint 都需要額外安裝龐大的相依套件，
    而 Edge 是 Windows 11 內建，且能正確處理中文字型、emoji 與 CSS 分頁控制。

用法
    python make_report_pdf.py                 # REPORT.md -> <學號>_hw2_report.pdf
    python make_report_pdf.py --md README.md  # 轉其他 Markdown 檔
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent

# Chrome 排在前面：實測 Edge 在某些版本／系統政策下，無頭列印會以 returncode 0
# 正常結束卻完全不產生檔案，且不輸出任何錯誤訊息。兩者都會嘗試。
BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
}
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
    font-family: "Microsoft JhengHei", "Microsoft YaHei", "Segoe UI",
                 "PingFang TC", sans-serif;
    font-size: 10.5pt;
    line-height: 1.75;
    color: #1a1a1a;
    margin: 0;
}
h1 {
    font-size: 20pt; font-weight: 700; color: #0f3d6e;
    border-bottom: 2.5px solid #0f3d6e;
    padding-bottom: 6px; margin: 0 0 4px 0;
    break-after: avoid; page-break-after: avoid;
}
h2 {
    font-size: 14.5pt; font-weight: 700; color: #0f3d6e;
    border-left: 5px solid #0f3d6e; padding-left: 10px;
    margin: 22px 0 10px 0;
    break-after: avoid; page-break-after: avoid;
}
h3 {
    font-size: 12pt; font-weight: 700; color: #22527a;
    margin: 16px 0 8px 0;
    break-after: avoid; page-break-after: avoid;
}
h4 { font-size: 11pt; font-weight: 700; margin: 12px 0 6px 0;
     break-after: avoid; page-break-after: avoid; }
p { margin: 7px 0; text-align: justify; }
ul, ol { margin: 7px 0; padding-left: 22px; }
li { margin: 3px 0; }
strong { color: #0b2d4f; }

table {
    border-collapse: collapse; width: 100%;
    margin: 12px 0; font-size: 9.5pt;
    break-inside: avoid; page-break-inside: avoid;
}
th {
    background: #0f3d6e; color: #fff; font-weight: 700;
    border: 1px solid #0f3d6e; padding: 6px 8px; text-align: left;
}
td { border: 1px solid #c3cfdb; padding: 5px 8px; vertical-align: top; }
/* 第一欄多為短標籤，keep-all 可避免中文詞被硬拆成兩行（如「可稽核/性」） */
th:first-child, td:first-child { word-break: keep-all; }
tr:nth-child(even) td { background: #f4f7fa; }

img {
    max-width: 100%; height: auto; display: block;
    margin: 12px auto; border: 1px solid #d7dee5; border-radius: 3px;
    break-inside: avoid; page-break-inside: avoid;
}

blockquote {
    background: #fff8e6; border-left: 5px solid #e0a800;
    margin: 12px 0; padding: 10px 14px; border-radius: 3px;
    break-inside: avoid; page-break-inside: avoid;
}
blockquote p { margin: 5px 0; }
blockquote h3 { margin-top: 2px; color: #8a6100; }

code {
    font-family: Consolas, "Courier New", monospace;
    background: #eef2f6; padding: 1.5px 5px; border-radius: 3px;
    font-size: 9pt; color: #b03030;
}
pre {
    background: #1e2a36; color: #e6edf3; padding: 11px 13px;
    border-radius: 4px; overflow-x: auto; font-size: 8.5pt;
    line-height: 1.5;
    break-inside: avoid; page-break-inside: avoid;
}
pre code { background: transparent; color: inherit; padding: 0; font-size: 8.5pt; }

hr { border: none; border-top: 1px solid #ccd5de; margin: 20px 0; }
a { color: #11598f; text-decoration: none; word-break: break-all; }

/* 讓每個主要章節從新頁開始，排版較整齊 */
h2.newpage { break-before: page; page-break-before: always; }
"""

HTML_TMPL = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def fix_cjk_emphasis(text: str) -> str:
    """
    修正 Python-Markdown 在中文環境下的粗體失效問題。

    Python-Markdown 不解析「詞內」粗體：當 **粗體** 的結尾星號後面緊接著一個
    word character 時，它會視為 intraword emphasis 而原樣輸出星號。中日韓文字
    都算 word character，所以「改用**某某**執行」這種寫法會印出字面上的星號，
    而「……**某某**，」（後接標點）則正常。

    這裡在轉換前先把 **...** 直接換成 <strong>...</strong>，並跳過程式碼區塊與
    行內程式碼，以免動到內容。
    """
    blocks = re.split(r"(```.*?```)", text, flags=re.S)
    for i, block in enumerate(blocks):
        if i % 2 == 1:                      # 圍欄程式碼區塊，原封不動
            continue
        spans = re.split(r"(`[^`\n]*`)", block)
        for j, span in enumerate(spans):
            if j % 2 == 1:                  # 行內程式碼，原封不動
                continue
            spans[j] = re.sub(r"\*\*(?=\S)([^\n]+?)(?<=\S)\*\*",
                              r"<strong>\1</strong>", span)
        blocks[i] = "".join(spans)
    return "".join(blocks)


def find_browsers() -> list[str]:
    """回傳所有可用的瀏覽器路徑，依偏好排序。全部都會被嘗試。"""
    found = [c for c in BROWSER_CANDIDATES if Path(c).exists()]
    for name in ("chrome", "msedge"):
        w = shutil.which(name)
        if w and w not in found:
            found.append(w)
    if not found:
        sys.exit("找不到 Chrome 或 Edge，無法轉檔。")
    return found


def build(md_path: Path, pdf_path: Path, keep_html: bool) -> int:
    if not md_path.exists():
        sys.exit(f"找不到 {md_path.name}")

    text = md_path.read_text(encoding="utf-8")
    n_before = len(re.findall(r"\*\*", text))
    text = fix_cjk_emphasis(text)
    n_after = len(re.findall(r"\*\*", text))
    print(f"  粗體修正 : 轉換 {(n_before - n_after) // 2} 組中文粗體標記")

    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "nl2br"],
        output_format="html5",
    )

    # 讓編號章節（## 1. ～ ## 13.）各自從新頁開始，但第一章不用
    def newpage(m: re.Match) -> str:
        num = int(m.group(1))
        cls = "" if num == 1 else ' class="newpage"'
        return f"<h2{cls}>{num}."
    html_body = re.sub(r"<h2>(\d+)\.", newpage, html_body)

    html = HTML_TMPL.format(title=md_path.stem, css=CSS, body=html_body)
    html_path = ROOT / f"_{md_path.stem}_print.html"
    html_path.write_text(html, encoding="utf-8")

    browsers = find_browsers()
    print(f"  可用瀏覽器 : {', '.join(Path(b).name for b in browsers)}")
    print(f"  來源   : {md_path.name}  ({len(text):,} 字元)")

    # 先輸出到暫存檔，成功後才取代既有 PDF。
    # 若直接寫入目標路徑並事先刪除，一旦轉檔失敗就會連舊檔都失去。
    tmp_pdf = ROOT / f"_{pdf_path.stem}_tmp.pdf"
    if tmp_pdf.exists():
        tmp_pdf.unlink()

    # 獨立的使用者設定檔目錄是必要的：若使用者已開著 Edge，無頭程序會接上
    # 同一個設定檔然後立刻結束，不留任何錯誤訊息，導致轉檔靜默失敗。
    profile = Path(tempfile.mkdtemp(prefix="mdpdf_"))

    def attempts_for(browser: str) -> list[list[str]]:
        base = [
            browser,
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            f"--user-data-dir={profile}",
            "--virtual-time-budget=20000",
        ]
        tail = [f"--print-to-pdf={tmp_pdf}", html_path.as_uri()]
        return [
            base + ["--headless=new", "--no-pdf-header-footer"] + tail,
            base + ["--headless", "--print-to-pdf-no-header"] + tail,
        ]

    def ok() -> bool:
        return tmp_pdf.exists() and tmp_pdf.stat().st_size > 0

    res = None
    for browser in browsers:
        name = Path(browser).name
        for i, cmd in enumerate(attempts_for(browser), 1):
            try:
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=180)
            except subprocess.TimeoutExpired:
                print(f"  {name} 模式 {i}: 逾時")
                continue
            if ok():
                print(f"  成功 : {name}（模式 {i}）")
                break
            print(f"  {name} 模式 {i} 失敗 "
                  f"(returncode={res.returncode}，未產生檔案)")
        if ok():
            break

    shutil.rmtree(profile, ignore_errors=True)

    if not ok():
        if res is not None:
            print((res.stdout or "")[-1200:])
            print((res.stderr or "")[-1200:])
        print(f"\n提示：HTML 中間檔保留於 {html_path.name}，"
              "可直接用瀏覽器開啟後 Ctrl+P 另存為 PDF。")
        sys.exit("轉檔失敗，既有 PDF 未被更動。")

    tmp_pdf.replace(pdf_path)

    if not keep_html:
        html_path.unlink()

    size = pdf_path.stat().st_size
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(str(pdf_path)).pages)
        print(f"  完成   : {pdf_path.name}  {pages} 頁, {size/1e6:.2f} MB")
    except Exception:
        print(f"  完成   : {pdf_path.name}  {size/1e6:.2f} MB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="REPORT.md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    md_path = ROOT / args.md
    if args.out:
        pdf_path = ROOT / args.out
    else:
        main_py = sorted(ROOT.glob("*_hw2.py"))
        sid = main_py[0].name.split("_")[0] if main_py else "report"
        pdf_path = ROOT / f"{sid}_hw2_report.pdf"

    return build(md_path, pdf_path, args.keep_html)


if __name__ == "__main__":
    sys.exit(main())
