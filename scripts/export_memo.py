#!/usr/bin/env python3
"""
scripts/export_memo.py

Exports docs/paper/redesign_review_memo.md to a single, self-contained HTML file
and a publication-grade PDF using headless Google Chrome.
Inlines all referenced PNG figures as Base64 data URIs.
"""

import os
import sys
import re
import base64
import subprocess
import shutil
from pathlib import Path
import markdown

REPO_ROOT = Path(__file__).resolve().parent.parent
MEMO_MD = REPO_ROOT / "docs" / "paper" / "redesign_review_memo.md"
OUT_HTML = REPO_ROOT / "docs" / "paper" / "redesign_review_memo.html"
OUT_PDF = REPO_ROOT / "docs" / "paper" / "redesign_review_memo.pdf"
ART_DIR = Path("/home/yoavh/.gemini/antigravity/brain/fb95950a-187d-465f-ac10-b4b09aebc503")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Technical Review Memo: Shared-Target Reconstruction</title>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']],
    processEscapes: true
  },
  svg: { fontCache: 'global' },
  startup: {
    pageReady: () => {
      return MathJax.startup.defaultPageReady().then(() => {
        document.body.classList.add('mathjax-ready');
      });
    }
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({ startOnLoad: true, theme: 'neutral' });
</script>
<style>
@page {
    size: letter;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-right {
        content: counter(page);
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #24292f;
    line-height: 1.55;
    font-size: 13.5px;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    background: #fff;
}

h1 {
    font-size: 24px;
    border-bottom: 2px solid #0969da;
    padding-bottom: 8px;
    margin-top: 10px;
    color: #1f2328;
}

h2 {
    font-size: 18px;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 5px;
    margin-top: 28px;
    color: #0969da;
    page-break-after: avoid;
}

h3 {
    font-size: 15px;
    margin-top: 20px;
    color: #24292f;
    page-break-after: avoid;
}

p {
    margin: 0.8em 0;
}

hr {
    border: 0;
    border-top: 1px solid #d0d7de;
    margin: 20px 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    font-size: 12px;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #d0d7de;
    padding: 7px 10px;
    text-align: left;
}

th {
    background-color: #f6f8fa;
    font-weight: 600;
}

tr:nth-child(even) {
    background-color: #fcfcfc;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 15px auto;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    page-break-inside: avoid;
}

blockquote {
    border-left: 4px solid #0969da;
    padding: 6px 16px;
    color: #57606a;
    background: #f6f8fa;
    margin: 12px 0;
}

code {
    background: #eff1f3;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 88%;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}

pre {
    background: #f6f8fa;
    padding: 12px;
    border-radius: 5px;
    border: 1px solid #d0d7de;
    overflow-x: auto;
}

.figure-caption {
    font-size: 11.5px;
    color: #57606a;
    text-align: center;
    margin-top: -8px;
    margin-bottom: 16px;
    font-style: italic;
}
</style>
</head>
<body>
{CONTENT}
</body>
</html>
"""


def image_to_base64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def build_self_contained_html():
    with open(MEMO_MD, "r") as f:
        md_text = f.read()

    # Replace markdown image links with base64 data URIs
    # e.g., ![Caption](figures/fig_wild4_crossover_curve.png) or ![](/path/to/fig.png)
    def repl_img(match):
        caption = match.group(1)
        rel_path = match.group(2)
        
        # Check locations
        possible_paths = [
            REPO_ROOT / "docs" / "paper" / rel_path,
            REPO_ROOT / rel_path,
            ART_DIR / Path(rel_path).name,
            Path(rel_path)
        ]
        found = None
        for p in possible_paths:
            if p.exists() and p.is_file():
                found = p
                break
                
        if found:
            b64 = image_to_base64(found)
            return f'<div class="figure-container"><img src="{b64}" alt="{caption}"><div class="figure-caption">{caption}</div></div>'
        else:
            return match.group(0)

    # Regex for ![caption](path)
    md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl_img, md_text)

    # Convert mermaid block to a live SVG diagram container
    def repl_mermaid(match):
        code = match.group(1).strip()
        return f'<div class="mermaid" style="text-align:center; margin: 15px 0;">{code}</div>'

    md_text = re.sub(r'```mermaid\s*([\s\S]*?)\s*```', repl_mermaid, md_text)

    # Convert markdown to html using python-markdown with tables extension
    html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    
    full_html = HTML_TEMPLATE.replace("{CONTENT}", html_content)

    with open(OUT_HTML, "w") as f:
        f.write(full_html)
    print(f"Generated self-contained HTML: {OUT_HTML} ({len(full_html) / 1024:.1f} KB)")


def convert_html_to_pdf():
    print(f"Converting HTML to PDF via headless Google Chrome...")
    cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=6000",
        f"--print-to-pdf={OUT_PDF}",
        str(OUT_HTML)
    ]
    subprocess.check_call(cmd)
    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"Generated standalone PDF: {OUT_PDF} ({size_kb:.1f} KB)")

    if ART_DIR.exists():
        shutil.copy2(OUT_PDF, ART_DIR / "redesign_review_memo.pdf")
        shutil.copy2(OUT_HTML, ART_DIR / "redesign_review_memo.html")
        print(f"Copied PDF & HTML to artifacts directory: {ART_DIR}")


if __name__ == "__main__":
    build_self_contained_html()
    convert_html_to_pdf()
