"""Render a Markdown report (e.g. docs/report/status_report_*.md) to an A4 PDF.

Markdown -> HTML with python-markdown (tables, sane_lists), then headless Chrome
prints the HTML to PDF. Image paths resolve relative to the .md file, so
`![..](figures/x.png)` works as written. The intermediate HTML goes to a temp
dir unless --html is given.

Usage (from the repo root):
    .venv/bin/python scripts/md_to_pdf.py docs/report/status_report_2026-10-14.md \
        --img-width fig_recognition=100 --img-width fig1_lag=85 \
        --img-width fig2_topic_timing=55 --img-width fig4_grounding=55

The PDF defaults to the .md path with a .pdf suffix. Requires google-chrome
(or --chrome PATH) and `pip install markdown`.
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import tempfile

import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'DejaVu Sans', 'Helvetica', sans-serif; font-size: 10pt; line-height: 1.45; color: #1a1a1a; background: #fff; }
h1 { font-size: 19pt; margin: 0 0 6pt; line-height: 1.2; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
blockquote { margin: 8pt 0 10pt; padding: 8pt 12pt; background: #eef4fc; border-left: 4px solid #2a78d6; border-radius: 3px; break-inside: avoid; }
blockquote p { margin: 0 0 4pt; }
blockquote ul, blockquote ol { margin: 2pt 0 0; padding-left: 16pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 8.6pt; width: 100%; break-inside: avoid; }
th, td { border: 1px solid #d4d4d4; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f3f3f2; }
tr:nth-child(even) td { background: #fbfbfa; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.4pt; background: #f3f3f2; padding: 0 2px; border-radius: 2px; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; break-inside: avoid; }
hr { border: none; border-top: 1px solid #e0e0e0; margin: 14pt 0; }
em { color: #444; }
"""

LIST_ITEM = re.compile(r'^(> ?)?( *)([-*]|\d+\.) ')


def prep(text: str) -> str:
    """Adapt GitHub-style Markdown to python-markdown's stricter list rules.

    python-markdown needs 4-space nesting and a blank line before a list that
    follows a paragraph (inside or outside a blockquote); GitHub accepts 2-3
    spaces and no blank line.
    """
    out, prev = [], ''
    for line in text.split('\n'):
        m = LIST_ITEM.match(line)
        if m:
            q, ind = m.group(1) or '', m.group(2)
            if len(ind) in (2, 3):
                line = q + '    ' + line[len(q) + len(ind):]
            if not LIST_ITEM.match(prev) and prev.strip() not in ('', '>') and prev.startswith('>') == bool(q):
                out.append('>' if q else '')
        elif re.match(r'^ {2,3}\S', line) and out and re.match(r'^ *([-*]|\d+\.) ', prev):
            line = '    ' + line.lstrip()
        out.append(line)
        prev = line
    return '\n'.join(out)


def to_html(src: pathlib.Path, img_widths: dict[str, int]) -> str:
    body = markdown.markdown(prep(src.read_text()), extensions=['tables', 'sane_lists'])
    for name, w in img_widths.items():
        body = body.replace(f'src="figures/{name}.png"', f'src="figures/{name}.png" style="width:{w}%"')
    h1 = re.search(r'^# (.+)$', src.read_text(), re.M)
    title = h1.group(1) if h1 else src.stem
    return (f"<!doctype html><html><head><meta charset='utf-8'><base href='{src.parent.as_uri()}/'>"
            f"<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('md', type=pathlib.Path)
    ap.add_argument('-o', '--out', type=pathlib.Path, help='PDF path (default: <md>.pdf)')
    ap.add_argument('--html', type=pathlib.Path, help='also keep the intermediate HTML here')
    ap.add_argument('--img-width', action='append', default=[], metavar='NAME=PCT',
                    help='width in %% of page for figures/NAME.png (repeatable; default: natural size, capped at 100%%)')
    ap.add_argument('--chrome', default=shutil.which('google-chrome') or 'google-chrome')
    args = ap.parse_args()

    src = args.md.resolve()
    pdf = (args.out or src.with_suffix('.pdf')).resolve()
    widths = {k: int(v) for k, v in (s.split('=') for s in args.img_width)}
    html = to_html(src, widths)

    with tempfile.TemporaryDirectory() as tmp:
        html_path = args.html.resolve() if args.html else pathlib.Path(tmp) / 'report.html'
        html_path.write_text(html)
        subprocess.run([args.chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                        f'--print-to-pdf={pdf}', html_path.as_uri()], check=True)
    print(f'wrote {pdf}')


if __name__ == '__main__':
    main()
