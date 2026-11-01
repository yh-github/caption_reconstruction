#!/usr/bin/env python3
"""Check that a prose edit of the paper kept its numbers.

Compares the numbers in a LaTeX file against a git revision (or a snapshot) of
the same file, section by section. Fails if a number disappears from the whole
paper or occurs more often than before (an edit that only rewords removes
repeats; an extra occurrence means a number was added or substituted). Prints
per-section count changes for review (a number may move between sections, e.g.
out of the abstract into the results).

    .venv/bin/python scripts/paper_numbers.py docs/paper/acl2027/caption_index.tex [--rev HEAD]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# LaTeX that holds numbers which are not results: labels, refs, cites, sizes, layout.
STRIP = [
    r'(?<!\\)%.*',                                   # comments
    r'\\(?:label|ref|eqref|cite[pt]?|citet|citep|includegraphics|input|bibliography'
    r'|graphicspath|usepackage|documentclass|begin|end|resizebox|setlength|hspace|vspace'
    r'|DeclareUnicodeCharacter|newcommand|lstset|S)\b(?:\[[^\]]*\])*(?:\{[^{}]*\})*',
    r'\\begin\{tabular\}\{[^}]*\}',
    r'\\(?:columnwidth|textwidth|linewidth)',
    r'\$10\^\{\d\}\$',
]
NUM = re.compile(r'(?<![\w.])[+\-−]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')


def norm(tok: str) -> str:
    tok = tok.replace('−', '-').rstrip(',')
    if re.fullmatch(r'[+\-]?\d{1,3}(?:,\d{3})+', tok):   # 1,025 -> 1025
        tok = tok.replace(',', '')
    if re.fullmatch(r'[+\-]?\d+\.\d*0', tok):             # 0.20 -> 0.2 (same value)
        tok = tok.rstrip('0').rstrip('.')
    return tok.lstrip('+')


def sections(text: str) -> list[tuple[str, str]]:
    parts = re.split(r'(\\(?:sub)?section\*?\{[^}]*\}|\\begin\{abstract\}|\\appendix)', text)
    out, name = [], 'preamble'
    for p in parts:
        if re.match(r'\\(?:sub)?section|\\begin\{abstract\}|\\appendix', p):
            name = re.sub(r'\\(?:sub)?section\*?\{|\}|\\begin\{|\\', '', p)
        else:
            out.append((name, p))
    return out


def numbers(chunk: str) -> Counter:
    for pat in STRIP:
        chunk = re.sub(pat, ' ', chunk)
    chunk = re.sub(r'\$-\$', '-', chunk)
    chunk = re.sub(r'(?<=[\d%])\s*--\s*', ' to ', chunk)   # 1/23--1/26 is a range, not a minus
    return Counter(norm(m.group()) for m in NUM.finditer(chunk))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tex', type=Path)
    ap.add_argument('--rev', default='HEAD', help='git revision to compare against')
    ap.add_argument('--old', type=Path, help='compare against this file instead of a git revision')
    ap.add_argument('--allow', nargs='*', default=[], help='new numbers that are expected (e.g. a new section count)')
    a = ap.parse_args()

    if a.old:
        old_text = a.old.read_text()
    else:
        root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True,
                              check=True).stdout.strip()
        rel = a.tex.resolve().relative_to(root)
        old_text = subprocess.run(['git', 'show', f'{a.rev}:{rel}'], capture_output=True, text=True,
                                  check=True, cwd=root).stdout
    new_text = a.tex.read_text()

    old_sec, new_sec = sections(old_text), sections(new_text)
    old_all = sum((numbers(c) for _, c in old_sec), Counter())
    new_all = sum((numbers(c) for _, c in new_sec), Counter())

    allow = {norm(x) for x in a.allow}
    lost = sorted(set(old_all) - set(new_all))
    # A prose edit removes repeats; a number that occurs more often than before was added or substituted.
    added = sorted(f'{k}(+{new_all[k] - old_all[k]})' for k in new_all
                   if new_all[k] > old_all[k] and k not in allow)
    print(f'numbers: {sum(old_all.values())} -> {sum(new_all.values())} tokens, '
          f'{len(old_all)} -> {len(new_all)} distinct')

    old_by = {n: numbers(c) for n, c in old_sec}
    new_by = {n: numbers(c) for n, c in new_sec}
    for name in dict.fromkeys([n for n, _ in old_sec] + [n for n, _ in new_sec]):
        o, n = old_by.get(name, Counter()), new_by.get(name, Counter())
        if o != n:
            minus = ' '.join(f'{k}×{v}' if v > 1 else k for k, v in sorted((o - n).items()))
            plus = ' '.join(f'{k}×{v}' if v > 1 else k for k, v in sorted((n - o).items()))
            print(f'  [{name}]  -{{{minus}}}  +{{{plus}}}')

    if lost:
        print(f'LOST from the whole paper ({len(lost)}): {" ".join(lost)}')
    if added:
        print(f'MORE occurrences than before ({len(added)}): {" ".join(added)}')
    return 1 if lost or added else 0


if __name__ == '__main__':
    sys.exit(main())
