"""Fetch an arXiv paper's full text as plain text (HTML version, else the PDF through pdftotext).

Usage: python -I scripts/lit_search/fetch_fulltext.py ARXIV_ID OUTDIR
Writes OUTDIR/ARXIV_ID.txt. Downloads are untrusted data: give each paper its own new OUTDIR
outside the repo (e.g. under the session scratchpad), and run with python -I.
"""
import html
import os
import re
import subprocess
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 lit-review"}


def main() -> None:
    aid, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    txt_path = os.path.join(outdir, aid + ".txt")
    try:
        raw = urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/html/{aid}", headers=UA),
                                     timeout=60).read().decode("utf-8", "replace")
        raw = re.sub(r"(?s)<(script|style|math)[^>]*>.*?</\1>", " ", raw)
        raw = re.sub(r"(?i)<(br|p|div|h[1-6]|li|tr|table|section|figcaption)[^>]*>", "\n", raw)
        raw = re.sub(r"(?i)</t[dh]>", " | ", raw)
        text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        text = re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", text))
        if len(text) < 5000:
            raise ValueError("HTML version too short")
    except Exception as e:  # noqa: BLE001
        print(f"HTML failed ({e}); trying the PDF", file=sys.stderr)
        pdf = os.path.join(outdir, aid + ".pdf")
        with open(pdf, "wb") as f:
            f.write(urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/pdf/{aid}", headers=UA),
                                           timeout=120).read())
        text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
    with open(txt_path, "w") as f:
        f.write(text)
    print(txt_path, len(text))


if __name__ == "__main__":
    main()
