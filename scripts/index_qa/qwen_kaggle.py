#!/usr/bin/env python3
"""C12 on Kaggle: Qwen2.5-VL-7B-Instruct captions the confirmatory windows and answers the reader prompts that
open_reader.py exported (research log, C12). Self-contained: needs only torch, transformers, accelerate and Pillow.

The bundle (open_reader.py bundle) holds captions.jsonl (one caption job per window: its frames and the
self-captions' prompt), reads.jsonl (one reader prompt per question and arm, built by reader.build, with frames as
paths; a qwencaptions prompt has <<CAPTION p>> where Qwen's caption of window p goes) and the frames. Stages:
    smoke    3 captions and 12 reads into <out>/smoke/, printed, to check the model loads and answers sanely
    caption  every window not yet captioned, appended to <out>/captions.jsonl
    read     every reader prompt not yet answered, appended to <out>/reads.jsonl (qwencaptions prompts need all of
             their video's captions)
    all      caption, then read
Each stage resumes where a stopped run left off. Decoding is greedy (the run is deterministic up to GPU numerics),
with the model's shipped generation config otherwise; there is no retry, since a greedy retry repeats itself, so an
unparsed answer scores as wrong at import. One user message per call, with the model's default system prompt; a
text block is followed by a newline. The frames go through the processor as shipped (224x126 -> 224x112, 32
visual tokens, where Claude's rule gives 40; over all C12 frames Qwen gets 0.82 of Claude's visual tokens). The
model is split over the GPUs (device_map="auto"); in float16 it needs both of Kaggle's T4s.

    python qwen_kaggle.py --stage smoke
    python qwen_kaggle.py --stage all
Outputs: <out>/captions.jsonl, <out>/reads.jsonl, <out>/env.json; open_reader.py import reads them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
import zipfile
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CAPTION_TOKENS = 200  # the prompt asks for at most 45 words
READ_TOKENS = 128  # the prompt asks for JSON only
SLOT = re.compile(r"<<CAPTION (\d+)>>")


def find_bundle(root: Path, work: Path) -> Path:
    """The folder holding reads.jsonl under root; a bundle uploaded as an archive is extracted into work first."""
    hits = sorted(root.rglob("reads.jsonl"))
    if hits:
        return hits[0].parent
    for a in sorted(root.rglob("*.zip")) + sorted(root.rglob("*.tar*")):
        dest = work / "bundle"
        if not (dest / "reads.jsonl").exists():
            print(f"extracting {a} to {dest}")
            if a.suffix == ".zip":
                zipfile.ZipFile(a).extractall(dest)
            else:
                tarfile.open(a).extractall(dest)
        hits = sorted(dest.rglob("reads.jsonl"))
        if hits:
            return hits[0].parent
    sys.exit(f"no reads.jsonl (or archive holding one) under {root}")


def jsonl(path: Path) -> list[dict]:
    """The rows of a JSONL file; a partial last line (from a stopped run) is skipped."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def end_line(path: Path) -> None:
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("a") as f:
            f.write("\n")


class Qwen:
    def __init__(self, model_path: str, dtype: str, fp32_vision: bool):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=getattr(torch, dtype), device_map="auto", attn_implementation="sdpa").eval()
        if fp32_vision:
            (getattr(self.model, "visual", None) or self.model.model.visual).float()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def generate(self, content: list[dict], images: list, max_new_tokens: int) -> dict:
        """content: Qwen chat parts ({"type": "text"} or {"type": "image"}), images: the PIL images in order."""
        msgs = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1]:]
        return dict(text=self.processor.decode(new, skip_special_tokens=True).strip(),
                    input_tokens=int(inputs["input_ids"].shape[1]),
                    visual_tokens=int((inputs["input_ids"] == self.pad_id).sum()),
                    output_tokens=int(len(new)), hit_limit=len(new) >= max_new_tokens,
                    seconds=round(time.time() - t0, 2))


def load_image(bundle: Path, rel: str):
    from PIL import Image
    return Image.open(bundle / rel).convert("RGB")


def caption_call(q: Qwen, bundle: Path, job: dict) -> dict:
    """Images first, then the prompt, as self_caption.py sends them to Claude."""
    imgs = [load_image(bundle, f) for f in job["frames"]]
    r = q.generate([*({"type": "image"} for _ in imgs), {"type": "text", "text": job["prompt"]}], imgs,
                   CAPTION_TOKENS)
    return dict(video_id=job["video_id"], window=job["window"], caption=r.pop("text"), **r)


def render(blocks: list[dict], caps: dict[int, str] | None) -> tuple[list[dict], list[str], str]:
    """Qwen chat parts, frame paths, and the prompt as reader.build logs it (texts and "[image <path>]" joined by
    newlines), with a qwencaptions prompt's slots filled from caps."""
    parts, frames, log = [], [], []
    for i, b in enumerate(blocks):
        if b["type"] == "image":
            parts.append({"type": "image"})
            frames.append(b["path"])
            log.append(f"[image {b['path']}]")
            continue
        t = SLOT.sub(lambda m: caps[int(m.group(1))], b["text"]) if caps is not None else b["text"]
        log.append(t)
        parts.append({"type": "text", "text": t + ("\n" if i < len(blocks) - 1 else "")})
    return parts, frames, "\n".join(log)


def read_call(q: Qwen, bundle: Path, item: dict, caps: dict[int, str] | None) -> dict:
    parts, frames, log = render(item["blocks"], caps)
    r = q.generate(parts, [load_image(bundle, f) for f in frames], READ_TOKENS)
    m = re.findall(r'"answer"\s*:\s*"([A-F])', r["text"])
    return dict(item=item["item"], arm=item["arm"], raw=r.pop("text"), prompt=log, quick_letter=m[-1] if m else None,
                answer=item["answer"], **r)


def video_caps(rows: list[dict]) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for r in rows:
        out.setdefault(r["video_id"], {})[r["window"]] = r["caption"]
    return out


def run_captions(q, bundle, jobs, path: Path, limit=None):
    done = {(r["video_id"], r["window"]) for r in jsonl(path)}
    todo = [j for j in jobs if (j["video_id"], j["window"]) not in done][:limit]
    print(f"captions: {len(jobs)} windows, {len(done)} done, running {len(todo)}", flush=True)
    end_line(path)
    with path.open("a") as f:
        for n, j in enumerate(todo):
            r = dict(caption_call(q, bundle, j), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{n + 1}/{len(todo)}] {j['video_id']} w{j['window']}: {len(r['caption'].split())} words, "
                  f"in={r['input_tokens']} vis={r['visual_tokens']} out={r['output_tokens']} {r['seconds']}s"
                  + (f"\n    {r['caption']}" if n < 3 else ""), flush=True)


def run_reads(q, bundle, items, path: Path, cap_rows: list[dict], limit=None, verbose=0):
    caps, n_win = video_caps(cap_rows), {}
    for j in jsonl(bundle / "captions.jsonl"):
        n_win[j["video_id"]] = n_win.get(j["video_id"], 0) + 1
    done = {r["item"] for r in jsonl(path)}
    todo, waiting = [], 0
    for it in items:
        if it["item"] in done:
            continue
        if it["arm"] == "qwencaptions" and len(caps.get(it["video_id"], {})) < n_win[it["video_id"]]:
            waiting += 1
            continue
        todo.append(it)
    todo = todo[:limit]
    print(f"reads: {len(items)} prompts, {len(done)} done, {waiting} waiting for captions, running {len(todo)}",
          flush=True)
    end_line(path)
    right = 0
    with path.open("a") as f:
        for n, it in enumerate(todo):
            c = caps[it["video_id"]] if it["arm"] == "qwencaptions" else None
            r = dict(read_call(q, bundle, it, c), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            right += r["quick_letter"] == r["answer"]
            print(f"  [{n + 1}/{len(todo)}] {it['item']}: {r['quick_letter']}/{r['answer']} in={r['input_tokens']} "
                  f"vis={r['visual_tokens']} out={r['output_tokens']} {r['seconds']}s (running acc "
                  f"{right / (n + 1):.2f})" + (f"\n    raw: {r['raw'][:300]!r}" if n < verbose else ""), flush=True)


def env(args, q: Qwen) -> dict:
    import torch
    import transformers
    gc = q.model.generation_config.to_dict()
    return dict(model=args.model_path, model_commit=getattr(q.model.config, "_commit_hash", None),
                dtype=args.dtype, fp32_vision=args.fp32_vision, torch=torch.__version__,
                transformers=transformers.__version__, cuda=torch.version.cuda,
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                device_map={k: str(v) for k, v in (getattr(q.model, "hf_device_map", None) or {}).items()},
                decoding=dict(do_sample=False, caption_max_new_tokens=CAPTION_TOKENS,
                              read_max_new_tokens=READ_TOKENS,
                              repetition_penalty=gc.get("repetition_penalty")),
                time=time.strftime("%Y-%m-%dT%H:%M:%S"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["smoke", "caption", "read", "all"], required=True)
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"), help="where to look for the bundle")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/c12"))
    ap.add_argument("--model-path", default=MODEL_ID, help="a Hugging Face ID or a local copy (e.g. a Kaggle model)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="float32 only for a CPU test with a tiny model")
    ap.add_argument("--fp32-vision", action="store_true", help="the vision tower in float32 (if float16 overflows)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bundle = find_bundle(args.input, args.out.parent)
    manifest = json.loads((bundle / "manifest.json").read_text())
    jobs, items = jsonl(bundle / "captions.jsonl"), jsonl(bundle / "reads.jsonl")
    print(f"bundle {bundle}: {len(jobs)} caption jobs, {len(items)} reader prompts; exported from commit "
          f"{manifest['git_commit']}", flush=True)
    q = Qwen(args.model_path, args.dtype, args.fp32_vision)
    e = env(args, q)
    print(json.dumps(e, indent=1), flush=True)
    if args.stage == "smoke":
        out = args.out / "smoke"
        out.mkdir(exist_ok=True)
        (out / "env.json").write_text(json.dumps(e, indent=1))
        run_captions(q, bundle, jobs, out / "captions.jsonl", limit=3)
        qs = list(dict.fromkeys(it["question_item"] for it in items))[:6]
        smoke = [it for it in items if it["question_item"] in qs and it["arm"] in ("blind", "frames5")]
        run_reads(q, bundle, smoke, out / "reads.jsonl", [], verbose=len(smoke))
        return
    (args.out / "env.json").write_text(json.dumps(e, indent=1))
    if args.stage in ("caption", "all"):
        run_captions(q, bundle, jobs, args.out / "captions.jsonl", args.limit)
    if args.stage in ("read", "all"):
        run_reads(q, bundle, items, args.out / "reads.jsonl", jsonl(args.out / "captions.jsonl"), args.limit)
    print("done")


if __name__ == "__main__":
    main()
