#!/usr/bin/env python3
"""C12: export the open-weights reader's prompts for Kaggle, and import its outputs (research log, C12).

bundle  writes local/kaggle_c12/ and local/kaggle_c12.zip for qwen_kaggle.py:
        - captions.jsonl: one caption job per confirmatory window, with the frames and the prompt the self-captions
          use (caption.frames_prompt), images before the prompt, as self_caption.py sends them;
        - reads.jsonl: one reader prompt per confirmatory question and C12 arm, made by reader.build with each frame
          as its path. A qwencaptions prompt is built with <<CAPTION p>> in place of Qwen's caption of window p,
          which qwen_kaggle.py fills in;
        - the frames those prompts use, at their repo paths; qwen_kaggle.py; manifest.json.
        Kaggle allows 1,000 files in an upload it unzips, so local/kaggle_c12_upload.zip holds only kaggle_c12.zip
        (Kaggle then unzipped both levels; qwen_kaggle.py also extracts an archive itself if needed).
import  reads qwen_kaggle.py's outputs (its out folder, or a .tar.gz / .zip of it): writes Qwen's captions to
        results/index_qa/captions/<CAPTIONERS["qwencaptions"]>/, and the reads, as reader rows, to
        results/index_qa/open/qwen.jsonl (reader.rows_path), replacing that file. Every window must have a caption,
        and every reader prompt Qwen saw must equal reader.build's for the same item once the captions are in;
        otherwise no reads are written. Letters are parsed with reader.parse.

    .venv/bin/python scripts/index_qa/open_reader.py bundle
    .venv/bin/python scripts/index_qa/open_reader.py import ~/Downloads/c12_out.tar.gz
    .venv/bin/python scripts/index_qa/reader.py c12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from index_qa import reader as R  # noqa: E402
from index_qa import tempcompass as tc  # noqa: E402
from index_qa.caption import caption_path, caption_windows, frames_prompt  # noqa: E402

MODEL = "qwen"
ARMS = [a.split(":")[1] for a in R.C12_ARMS]
BUNDLE = tc.ROOT / "local/kaggle_c12"
SLOT = "<<CAPTION {p}>>"


def _path_block(f: Path) -> dict:
    return {"type": "image", "path": str(f.relative_to(tc.ROOT))}


def slot_captions(video_id: str, model: str) -> list[str]:
    return [SLOT.format(p=p) for p in range(len(caption_windows(video_id, model)))]


def reader_prompts() -> list[dict]:
    """reader.build's prompt for every confirmatory question and C12 arm, with frames as paths; qwencaptions with
    caption slots."""
    out, real = [], R.captions
    try:
        R.captions = lambda v, m: (slot_captions(v, m) if m == R.CAPTIONERS["qwencaptions"] else real(v, m))
        for it in R.item_set(None, "confirm"):
            if it["arm"] in ARMS:
                blocks, _ = R.build(it, image=_path_block)
                out.append(dict(item=it["item"], question_item=it["question_item"], video_id=it["video_id"],
                                arm=it["arm"], answer=it["answer"], blocks=blocks))
    finally:
        R.captions = real
    return out


def caption_jobs() -> list[dict]:
    jobs = []
    for q in tc.items("confirm"):
        v = q["video_id"]
        for p, ((s, e), frames) in enumerate(zip(tc.windows(tc.duration_s(v)), tc.window_frames(v))):
            jobs.append(dict(video_id=v, window=p, start_s=s, end_s=e,
                             frames=[str(f.relative_to(tc.ROOT)) for f in frames], prompt=frames_prompt(len(frames), e - s)))
    return jobs


def cmd_bundle(args):
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    jobs, reads = caption_jobs(), reader_prompts()
    frames = sorted({f for j in jobs for f in j["frames"]}
                    | {b["path"] for r in reads for b in r["blocks"] if b["type"] == "image"})
    for f in frames:
        dest = BUNDLE / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tc.ROOT / f, dest)
    for name, rows in (("captions.jsonl", jobs), ("reads.jsonl", reads)):
        (BUNDLE / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    shutil.copy2(Path(__file__).with_name("qwen_kaggle.py"), BUNDLE / "qwen_kaggle.py")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=tc.ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=tc.ROOT).stdout)
    (BUNDLE / "manifest.json").write_text(json.dumps(dict(
        experiment="C12", model=R.OPEN_MODELS[MODEL], arms=ARMS, n_caption_jobs=len(jobs), n_reads=len(reads),
        n_frames=len(frames), git_commit=commit, git_dirty=dirty), indent=1))
    zp = BUNDLE.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(BUNDLE.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(BUNDLE))
    # Kaggle unzips an uploaded dataset and allows at most 1,000 files, so the upload is a zip holding only zp.
    up = BUNDLE.with_name(BUNDLE.name + "_upload.zip")
    with zipfile.ZipFile(up, "w", zipfile.ZIP_STORED) as z:
        z.write(zp, zp.name)
    print(f"{len(jobs)} caption jobs, {len(reads)} reader prompts ({len(reads) // len(ARMS)} questions x {len(ARMS)} "
          f"arms), {len(frames)} frames -> {BUNDLE}/ and {zp} ({zp.stat().st_size / 1e6:.1f} MB), uploaded to Kaggle "
          f"as {up}; commit {commit[:7]}" + (" (DIRTY tree: commit before the run)" if dirty else ""))


def _rows(path: Path) -> list[dict]:
    """A JSONL file's rows; a partial line from a stopped run is skipped (qwen_kaggle.py reruns its item)."""
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _outputs(src: Path, tmp: Path) -> Path:
    """The folder holding reads.jsonl: src itself, or inside the archive src."""
    if src.is_file():
        if src.suffix == ".zip":
            zipfile.ZipFile(src).extractall(tmp)
        else:
            tarfile.open(src).extractall(tmp, filter="data")
        src = tmp
    hits = [p.parent for p in src.rglob("reads.jsonl") if p.parent.name != "smoke"]
    if len(hits) != 1:
        sys.exit(f"expected one reads.jsonl (outside smoke/) under {src}, found {hits}")
    return hits[0]


def cmd_import(args):
    with tempfile.TemporaryDirectory() as tmp:
        out = _outputs(args.src.expanduser(), Path(tmp))
        envd = json.loads((out / "env.json").read_text())
        caps = {(r["video_id"], r["window"]): r for r in _rows(out / "captions.jsonl")}
        reads = {r["item"]: r for r in _rows(out / "reads.jsonl")}
    jobs = {(j["video_id"], j["window"]): j for j in caption_jobs()}
    if missing := sorted(set(jobs) - set(caps)):
        sys.exit(f"{len(missing)} windows have no Qwen caption, e.g. {missing[:3]}: finish the caption stage first")
    if extra := sorted(set(caps) - set(jobs)):
        sys.exit(f"{len(extra)} Qwen captions are for windows the bundle does not have, e.g. {extra[:3]}")
    name, model_id = R.CAPTIONERS["qwencaptions"], R.OPEN_MODELS[MODEL]
    for (v, p), r in caps.items():
        j = jobs[(v, p)]
        dest = caption_path(name, v, p)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(dict(
            video_id=v, window=p, start_s=j["start_s"], end_s=j["end_s"], caption=r["caption"], captioner=model_id,
            decoding="greedy", frames=j["frames"], prompt_sha256=hashlib.sha256(j["prompt"].encode()).hexdigest()[:16],
            usage=dict(input_tokens=r["input_tokens"], visual_tokens=r["visual_tokens"],
                       output_tokens=r["output_tokens"]), hit_limit=r["hit_limit"], seconds=r["seconds"],
            time=r["time"], env=envd), indent=1))
    print(f"{len(caps)} Qwen captions -> {caption_path(name, 'x', 0).parents[1]}")
    items = {it["item"]: it for it in R.item_set(None, "confirm") if it["arm"] in ARMS}
    bad = [i for i, r in reads.items() if i not in items or R.build(items[i], image=_path_block)[1] != r["prompt"]]
    if bad:
        sys.exit(f"{len(bad)} reader prompts differ from reader.build's, e.g. {bad[:3]}; no reads written")
    if unread := len(set(items) - set(reads)):
        print(f"warning: {unread} of {len(items)} reader prompts have no Qwen answer yet")
    rows = []
    for i, r in reads.items():
        it = items[i]
        letter = R.parse(r["raw"], list(it["options"]))
        rows.append(dict(item=i, question_item=it["question_item"], video_id=it["video_id"], dim=it["dim"],
                         arm=it["arm"], n_options=len(it["options"]), answer=it["answer"], letter=letter,
                         correct=letter == it["answer"], model=MODEL, model_id=model_id, decoding="greedy",
                         status="ok" if letter else "invalid", raw=r["raw"],
                         usage=dict(input_tokens=r["input_tokens"], visual_tokens=r["visual_tokens"],
                                    output_tokens=r["output_tokens"]), hit_limit=r["hit_limit"],
                         seconds=r["seconds"], time=r["time"], env=envd))
    path = R.rows_path(MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} reads of {len(items)} prompts (all prompts equal reader.build's) -> {path}; unparsed: "
          f"{sum(r['status'] != 'ok' for r in rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bundle")
    im = sub.add_parser("import")
    im.add_argument("src", type=Path, help="qwen_kaggle.py's out folder, or a .tar.gz / .zip of it")
    args = ap.parse_args()
    {"bundle": cmd_bundle, "import": cmd_import}[args.cmd](args)


if __name__ == "__main__":
    main()
