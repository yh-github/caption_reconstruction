"""Pilot videos, caption sets, embeddings and text tags for the unshuffling study."""
from __future__ import annotations

import csv
import functools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from unshuffle.core import N_PIECES, PIECE_S  # noqa: E402
from unshuffle.cues import remove_cues  # noqa: E402

CAPTIONER = "gemini-3.8-flash"
PC = ROOT / "results/piece_captions"
VIS_DIR = ROOT / "local/wild_videos_embs_siglip"
OUT = ROOT / "results/unshuffle"
TEXT_TAGS = PC / "check/text_tags.csv"
CAPTION_SETS = ["isolated", "in_context", "in_context_nocue", "original", "original_nocue"]


def pilot_videos(n: int = 30) -> list[str]:
    """The pilot, in the seeded pilot order (caption_pieces.select_pilot, seed 0): one video per channel."""
    from caption_pieces import select_pilot
    return select_pilot(n, 0)


def file_id(video: str) -> str:
    return video.replace("'", "_")


@functools.lru_cache(maxsize=None)
def captions(video: str) -> dict[str, dict[int, str]]:
    """Caption set -> {piece: caption} for the sets in CAPTION_SETS that exist for this video."""
    from piece_caption_check import load_sets
    sets = {k: v for k, v in load_sets(video, CAPTIONER).items() if k in ("isolated", "in_context", "original")}
    for k in ("in_context", "original"):
        if k in sets:
            sets[f"{k}_nocue"] = {p: remove_cues(t) for p, t in sets[k].items()}
    return sets


@functools.lru_cache(maxsize=None)
def frame_rows(video: str) -> np.ndarray:
    """Unit-normalised SigLIP 2 frame embeddings, row t = the frame at t s (first 60 s)."""
    V = np.load(VIS_DIR / f"{file_id(video)}.npy")[: N_PIECES * PIECE_S].astype(np.float32)
    return unit(V)


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def piece_means(video: str) -> np.ndarray:
    V = frame_rows(video)
    return unit(np.stack([V[p * PIECE_S:(p + 1) * PIECE_S].mean(0) for p in range(N_PIECES)]))


def piece_edges(video: str) -> tuple[np.ndarray, np.ndarray]:
    """(first frame, last frame) of each piece."""
    V = frame_rows(video)
    return V[[p * PIECE_S for p in range(N_PIECES)]], V[[p * PIECE_S + PIECE_S - 1 for p in range(N_PIECES)]]


@functools.lru_cache(maxsize=None)
def _embedders():
    from llm.local_embedder import LocalEmbedder, SiglipTextEmbedder
    # The stored frames are SigLIP 2 (timm v2_webli), so captions must use the SigLIP 2 text tower.
    return {"siglip2_text": SiglipTextEmbedder("google/siglip2-base-patch16-224"),
            "mpnet": LocalEmbedder("all-mpnet-base-v2")}


def caption_embeddings(video: str, caption_set: str, encoder: str) -> np.ndarray:
    """Unit-normalised embeddings of the 12 captions of one set, row p = piece p (cached by text)."""
    caps = captions(video)[caption_set]
    texts = [caps[p] for p in range(N_PIECES)]
    return unit(_embedders()[encoder].get_embeddings(f"{video}_pieces_{caption_set}", texts))


@functools.lru_cache(maxsize=None)
def text_tagged() -> dict[str, set[int]]:
    """video -> isolated pieces whose caption quotes or names on-screen text (piece_caption_check.py)."""
    out: dict[str, set[int]] = {}
    with open(TEXT_TAGS) as f:
        for r in csv.DictReader(f):
            if r["set"] == "isolated" and r["quotes_text"] == "True":
                out.setdefault(r["video_id"], set()).add(int(r["piece"]))
    return out
