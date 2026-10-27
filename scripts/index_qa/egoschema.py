#!/usr/bin/env python3
"""EgoSchema questions and videos for the caption-index length test (docs/new_direction/nextqa_plan.md §5).

Data: lmms-eval/egoschema on Hugging Face (MIT, ungated). Its "Subset" split has the 500 questions with public
answers, one per 180 s egocentric video, 5 options each. The videos sit in five zips of about 21 GB; fetch reads
each zip's central directory with HTTP range requests and extracts only the videos a question set needs, into
local/egoschema/videos/ (log D6: about 3 s per video). Windows, frames, captions and reads then go through
tempcompass.py, which routes UUID video IDs here (36 windows of 5 s, 180 frames per video).

Sets: "ego_dev" (SIZES, sets the predictions) and "ego_confirm" (the confirmatory questions), disjoint, drawn in
one seeded shuffle of the subset without EXCLUDED.

    .venv/bin/python scripts/index_qa/egoschema.py sample --set ego_dev      # writes the set's items file
    .venv/bin/python scripts/index_qa/egoschema.py fetch --set ego_dev       # the set's videos
    .venv/bin/python scripts/index_qa/tempcompass.py frames --set ego_dev    # the set's frames
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from index_qa import tempcompass as tc  # noqa: E402

HF = "https://huggingface.co/datasets/lmms-eval/egoschema/resolve/main"
SUBSET_URL = f"{HF}/Subset/test-00000-of-00001.parquet"
CHUNKS = [f"{HF}/videos_chunked_0{k}.zip" for k in range(1, 6)]
SUBSET = tc.EGO / "subset.parquet"
ZIP_INDEX = tc.EGO / "zip_index.json"
VIDEOS = tc.EGO / "videos"
# Videos whose result was seen before any sample was drawn (log D6)
EXCLUDED = {"8345d177-0ed4-4965-8455-bd17d2b42399": "D6 pilot: one Haiku call on all 180 frames, answer seen"}
SIZES = {"ego_dev": 40, "ego_confirm": 160}
PREFIX = {"ego_dev": "egd", "ego_confirm": "egc"}
SEED = 0
LETTERS = "ABCDE"


class HttpRange(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, so zipfile can read members of a remote zip."""

    def __init__(self, url: str):
        super().__init__()
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.url  # the redirect target (the CDN), resolved once
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        """Bytes start..end inclusive."""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for wait in (5, 30, 0):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except (urllib.error.URLError, TimeoutError):
                if not wait:
                    raise
                time.sleep(wait)

    def readinto(self, b) -> int:
        if self.pos >= self.size or not len(b):
            return 0
        data = self._get(self.pos, min(self.pos + len(b), self.size) - 1)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRange(url), buffer_size=1 << 20))


def parse_row(question: str, options, answer) -> dict:
    """One subset row as a question: options "A. text" in letter order, answer an index into them."""
    opts = {}
    for k, o in enumerate(options):
        m = tc.OPTION.match(o)
        if not m or m.group(1) != LETTERS[k]:
            raise ValueError(f"option {k} is not {LETTERS[k]}: {o!r}")
        opts[m.group(1)] = m.group(2)
    return dict(stem=question.strip(), options=opts, answer=LETTERS[int(answer)])


def load_questions() -> list[dict]:
    import pandas as pd
    if not SUBSET.exists():
        SUBSET.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SUBSET_URL, SUBSET)
    d = pd.read_parquet(SUBSET)
    return [dict(video_id=r.video_idx, dim="egoschema", **parse_row(r.question, r.option, r.answer))
            for r in d.itertuples()]


def sample(qs: list[dict], set_: str, seed: int = SEED) -> list[dict]:
    """The set's questions: one seeded shuffle of the subset without EXCLUDED, ego_dev first, then ego_confirm, so
    the sets never share a video."""
    pool = sorted((q for q in qs if q["video_id"] not in EXCLUDED), key=lambda q: q["video_id"])
    random.Random(seed).shuffle(pool)
    start = 0 if set_ == "ego_dev" else SIZES["ego_dev"]
    out = [dict(q) for q in pool[start: start + SIZES[set_]]]
    for i, q in enumerate(out):
        q["item"] = f"{PREFIX[set_]}{i:03d}|{q['video_id']}"
    return out


def zip_index() -> dict[str, list[str]]:
    """Per chunk URL, the mp4 member names (cached: each central directory is a few range requests)."""
    if ZIP_INDEX.exists():
        return json.loads(ZIP_INDEX.read_text())
    index = {url: [i.filename for i in open_zip(url).infolist() if i.filename.endswith(".mp4")] for url in CHUNKS}
    ZIP_INDEX.parent.mkdir(parents=True, exist_ok=True)
    ZIP_INDEX.write_text(json.dumps(index))
    return index


def fetch(video_ids: list[str]) -> int:
    """Extract the videos not yet on disk from their chunks; returns how many were written."""
    todo = {v for v in video_ids if not tc.video_path(v).exists()}
    where = {Path(name).stem: (url, name) for url, names in zip_index().items() for name in names}
    missing = sorted(todo - set(where))
    if missing:
        sys.exit(f"{len(missing)} videos are in no chunk, e.g. {missing[0]}")
    VIDEOS.mkdir(parents=True, exist_ok=True)
    made = 0
    for url in CHUNKS:
        names = sorted(where[v][1] for v in todo if where[v][0] == url)
        if not names:
            continue
        z = open_zip(url)
        for name in names:
            dest = tc.video_path(Path(name).stem)
            tmp = dest.with_suffix(".tmp")
            t0 = time.time()
            tmp.write_bytes(z.read(name))
            tmp.replace(dest)
            made += 1
            print(f"  {dest.name}: {dest.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.1f} s", flush=True)
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "fetch"])
    ap.add_argument("--set", dest="set_", choices=sorted(tc.EGO_SETS), default="ego_dev")
    args = ap.parse_args()
    if args.cmd == "sample":
        path = tc.SETS[args.set_]
        if path.exists():
            sys.exit(f"{path} exists; a sample is fixed once drawn")
        qs = sample(load_questions(), args.set_)
        tc.OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(qs, indent=1))
        print(f"{len(qs)} items on {len({q['video_id'] for q in qs})} videos -> {path.relative_to(tc.ROOT)}")
    else:
        print(f"{fetch([q['video_id'] for q in tc.items(args.set_)])} videos written to {VIDEOS.relative_to(tc.ROOT)}")


if __name__ == "__main__":
    main()
