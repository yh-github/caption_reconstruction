"""
src/shared_target/pools.py

Candidate pool generator for stratified retrieval evaluation:
- same_video_near (within 10s of gap edge, excluding boundary and gap, <= 18)
- same_video_far (remaining same-video seconds, 30 sampled)
- other_video_other_channel (same split & domain, different channel & stem, 30 sampled)
- other_video_same_channel (same split & channel, different stem, 30 sampled)
- boundary_diag (same_video_near + boundary frames, <= 20)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class Candidate:
    video_id: str
    second: int


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    dataset: str
    category: str
    channel: str
    stem: str
    total_seconds: int = 60


def parse_video_id(video_id: str) -> tuple[str, str]:
    """
    Parses video_id into (channel, stem):
    e.g. 'AiirSource-Military_1-clip-0' -> ('AiirSource-Military', 'AiirSource-Military_1')
         'How-Farms-Work_9-manual'      -> ('How-Farms-Work', 'How-Farms-Work_9')
    """
    channel = video_id.split("_")[0]
    stem = video_id.split("-clip-")[0].split("-manual")[0]
    return channel, stem


def get_gap_boundaries(gap_seconds: Sequence[int], total_seconds: int = 60) -> tuple[int | None, int | None]:
    """
    Returns (left_boundary, right_boundary) for a gap.
    Returns None if boundary falls outside [0, total_seconds - 1].
    """
    t_start = min(gap_seconds)
    t_end = max(gap_seconds)
    left_b = t_start - 1 if t_start > 0 else None
    right_b = t_end + 1 if t_end < total_seconds - 1 else None
    return left_b, right_b


def build_same_video_near_candidates(
    video_id: str,
    gap_seconds: Sequence[int],
    window_sec: int = 10,
    total_seconds: int = 60
) -> list[Candidate]:
    """
    Seconds within window_sec of gap edges, EXCLUDING gap seconds and boundary seconds.
    """
    t_start = min(gap_seconds)
    t_end = max(gap_seconds)
    left_b, right_b = get_gap_boundaries(gap_seconds, total_seconds)
    
    gap_set = set(gap_seconds)
    bound_set = {b for b in (left_b, right_b) if b is not None}
    excluded = gap_set | bound_set
    
    near_seconds = set()
    # Left near window: [t_start - window_sec, t_start - 1]
    for s in range(max(0, t_start - window_sec), t_start):
        if s not in excluded:
            near_seconds.add(s)
            
    # Right near window: [t_end + 1, min(total_seconds - 1, t_end + window_sec)]
    for s in range(t_end + 1, min(total_seconds, t_end + window_sec + 1)):
        if s not in excluded:
            near_seconds.add(s)
            
    return [Candidate(video_id, s) for s in sorted(near_seconds)]


def build_boundary_diag_candidates(
    video_id: str,
    gap_seconds: Sequence[int],
    window_sec: int = 10,
    total_seconds: int = 60
) -> list[Candidate]:
    """
    same_video_near candidates PLUS available boundary frames.
    """
    near_cands = build_same_video_near_candidates(video_id, gap_seconds, window_sec, total_seconds)
    left_b, right_b = get_gap_boundaries(gap_seconds, total_seconds)
    
    res = list(near_cands)
    if left_b is not None:
        res.append(Candidate(video_id, left_b))
    if right_b is not None:
        res.append(Candidate(video_id, right_b))
        
    # Deduplicate and sort by second
    res_dict = {c.second: c for c in res}
    return [res_dict[s] for s in sorted(res_dict.keys())]


def build_same_video_far_candidates(
    video_id: str,
    gap_seconds: Sequence[int],
    sample_size: int = 30,
    window_sec: int = 10,
    total_seconds: int = 60,
    rng: np.random.Generator | None = None
) -> list[Candidate]:
    """
    Remaining same-video seconds: [0, total_seconds - 1] excluding near, gap, and boundary.
    Sample sample_size distractors (without replacement if available >= sample_size).
    """
    t_start = min(gap_seconds)
    t_end = max(gap_seconds)
    left_b, right_b = get_gap_boundaries(gap_seconds, total_seconds)
    
    gap_set = set(gap_seconds)
    bound_set = {b for b in (left_b, right_b) if b is not None}
    near_cands = build_same_video_near_candidates(video_id, gap_seconds, window_sec, total_seconds)
    near_set = {c.second for c in near_cands}
    
    excluded = gap_set | bound_set | near_set
    available = [s for s in range(total_seconds) if s not in excluded]
    
    if rng is None:
        rng = np.random.default_rng(42)
        
    n_sample = min(len(available), sample_size)
    if n_sample == 0:
        return []
        
    sampled_sec = rng.choice(available, size=n_sample, replace=False)
    return [Candidate(video_id, int(s)) for s in sorted(sampled_sec)]


def build_other_video_candidates(
    target_meta: VideoMeta,
    all_videos: Sequence[VideoMeta],
    same_channel: bool = False,
    sample_size: int = 30,
    rng: np.random.Generator | None = None
) -> list[Candidate]:
    """
    Samples distractors from other videos in the same dataset split.
    If same_channel=False (other_video_other_channel):
        Same split, same category, different channel, different stem.
    If same_channel=True (other_video_same_channel):
        Same split, same channel, different stem.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    eligible_videos = []
    for vm in all_videos:
        if vm.dataset != target_meta.dataset:
            continue
        if vm.stem == target_meta.stem:
            continue
            
        if same_channel:
            if vm.channel == target_meta.channel:
                eligible_videos.append(vm)
        else:
            if vm.category == target_meta.category and vm.channel != target_meta.channel:
                eligible_videos.append(vm)
                
    if not eligible_videos:
        return []
        
    # Build candidate universe of (video_id, second) across eligible videos
    pool = []
    for vm in eligible_videos:
        for s in range(vm.total_seconds):
            pool.append(Candidate(vm.video_id, s))
            
    n_sample = min(len(pool), sample_size)
    sampled_idx = rng.choice(len(pool), size=n_sample, replace=False)
    return [pool[i] for i in sampled_idx]
