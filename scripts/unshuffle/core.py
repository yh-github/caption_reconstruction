"""Puzzles, metrics, exact ordering DPs and the channel-cluster bootstrap. Pure functions: no data, no models.

Pieces are numbered by time (piece p = seconds [5p, 5p + 5)), so a puzzle's true order is its pieces sorted.
An order is a list of piece numbers, earliest first.
"""
from __future__ import annotations

import random
import re
from itertools import combinations

import numpy as np

N_PIECES, PIECE_S = 12, 5
SEED = 2026

# Puzzle sets (pilot pre-registration §2). K4: 4 pieces with spacing d; the starts spread over the clip.
SETS: dict[str, list[list[int]]] = {
    "P12": [list(range(12))],
    "C6": [list(range(0, 6)), list(range(6, 12))],
    "G6": [list(range(0, 12, 2)), list(range(1, 12, 2))],
    "K4d1": [[s + k for k in range(4)] for s in (0, 4, 8)],
    "K4d2": [[s + 2 * k for k in range(4)] for s in (0, 3, 5)],
    "K4d3": [[s + 3 * k for k in range(4)] for s in (0, 1, 2)],
}


def gap_s(pieces: list[int]) -> int | None:
    """Seconds not shown between consecutive pieces if they are evenly spaced, else None."""
    steps = {b - a for a, b in zip(pieces, pieces[1:])}
    return (steps.pop() - 1) * PIECE_S if len(steps) == 1 else None


def channel(video_id: str) -> str:
    """The YouTube channel (movie_id): the video id without its _<n>-clip-<k> / _<n>-manual suffix."""
    return re.sub(r"_\d+(-clip-\d+|-manual)?$", "", video_id).replace("Olly_s", "Olly's")


def make_puzzles(videos: list[str], sets: dict[str, list[list[int]]] = SETS, seed: int = SEED) -> list[dict]:
    """One dict per (video, set, index). Shuffle 1 is a seeded uniform permutation; shuffle 2 is its reverse, so
    a reader that echoes the presentation order scores tau = 0 exactly over the pair."""
    out = []
    for v in videos:
        for name, groups in sets.items():
            for k, pieces in enumerate(groups):
                pid = f"{v}|{name}|{k}"
                s1 = random.Random(f"{seed}|{pid}").sample(pieces, len(pieces))
                out.append(dict(puzzle=pid, video=v, set=name, k=k, pieces=list(pieces), gap_s=gap_s(pieces),
                                shuffles=[s1, s1[::-1]]))
    return out


def reduce_puzzle(pieces: list[int], drop: set[int]) -> list[int]:
    return [p for p in pieces if p not in drop]


def restrict(order: list[int], keep: list[int]) -> list[int]:
    """The induced order of `keep` within `order`."""
    ks = set(keep)
    return [p for p in order if p in ks]


# ----------------------------------------------------------------------------- metrics
def kendall_tau(order: list[int]) -> float:
    n = len(order)
    if n < 2:
        return float("nan")
    s = sum(1 if a < b else -1 for a, b in combinations(order, 2))
    return s / (n * (n - 1) / 2)


def score(order: list[int]) -> dict:
    """Metrics of one predicted order against the truth (its pieces sorted). Pre-registration §4."""
    truth = sorted(order)
    n = len(order)
    true_adj = list(zip(truth, truth[1:]))
    pred_adj = set(zip(order, order[1:]))
    directed = sum((a, b) in pred_adj for a, b in true_adj)
    undirected = sum((a, b) in pred_adj or (b, a) in pred_adj for a, b in true_adj)
    return dict(n=n, tau=kendall_tau(order), adj_und=undirected / (n - 1), adj_dir=directed / (n - 1),
                local_dir=directed / undirected if undirected else float("nan"),
                exact=float(np.mean([a == b for a, b in zip(order, truth)])))


def chance(n: int) -> dict:
    """Expected metrics of a uniform random order (an invalid answer scores this). local_dir is 0.5 by symmetry."""
    return dict(n=n, tau=0.0, adj_und=2 / n, adj_dir=1 / n, local_dir=0.5, exact=1 / n)


def position_bias(order: list[int], presented: list[int]) -> float:
    """Kendall tau between the predicted order and the presentation order (1 = echoed the presentation)."""
    pos = {p: i for i, p in enumerate(presented)}
    return kendall_tau([pos[p] for p in order])


# ----------------------------------------------------------------------------- exact ordering
def best_path(W: np.ndarray) -> list[int]:
    """Maximum-weight Hamiltonian path, W[i, j] = weight of i immediately followed by j (exact DP over subsets).
    Returns indices 0..n-1. For a symmetric W the path's direction is arbitrary: callers choose it."""
    n = W.shape[0]
    if n == 1:
        return [0]
    full = 1 << n
    dp = np.full((full, n), -np.inf)
    back = np.full((full, n), -1, dtype=np.int64)
    for j in range(n):
        dp[1 << j, j] = 0.0
    js_all = np.arange(n)
    for mask in range(1, full):
        js = js_all[(mask >> js_all) & 1 == 1]
        if len(js) < 2:
            continue
        prev = mask ^ (1 << js)                      # mask without j, one per j
        cand = dp[prev] + W[:, js].T                 # cand[a, i] = dp[mask - j_a, i] + W[i, j_a]
        best = cand.argmax(axis=1)
        dp[mask, js] = cand[np.arange(len(js)), best]
        back[mask, js] = best
    mask, j = full - 1, int(dp[full - 1].argmax())
    path = []
    while j >= 0:
        path.append(j)
        mask, j = mask ^ (1 << j), int(back[mask, j])
    return path[::-1]


def best_path_batch(W: np.ndarray) -> np.ndarray:
    """best_path for a batch of weight matrices W[b] (B, n, n), vectorised over the batch (for Monte Carlo).
    Returns (B, n) paths."""
    B, n, _ = W.shape
    if n == 1:
        return np.zeros((B, 1), dtype=np.int64)
    full = 1 << n
    dp = np.full((full, B, n), -np.inf)
    back = np.full((full, B, n), -1, dtype=np.int8)
    for j in range(n):
        dp[1 << j, :, j] = 0.0
    js_all = np.arange(n)
    Wt = np.transpose(W, (2, 0, 1))  # Wt[j, b, i] = W[b, i, j]
    for mask in range(1, full):
        js = js_all[(mask >> js_all) & 1 == 1]
        if len(js) < 2:
            continue
        cand = dp[mask ^ (1 << js)] + Wt[js]          # (len(js), B, n): cand[a, b, i] = dp[mask - j_a, b, i] + W[b, i, j_a]
        best = cand.argmax(axis=2)
        dp[mask][:, js] = np.take_along_axis(cand, best[..., None], axis=2)[..., 0].T
        back[mask][:, js] = best.T
    rows = np.arange(B)
    j = dp[full - 1].argmax(axis=1)
    mask = np.full(B, full - 1)
    out = np.empty((B, n), dtype=np.int64)
    for pos in range(n - 1, -1, -1):
        out[:, pos] = j
        prev = back[mask, rows, j]
        mask = mask ^ (1 << j)
        j = prev.astype(np.int64)
    return out


def path_weight(W: np.ndarray, path: list[int]) -> float:
    return float(sum(W[a, b] for a, b in zip(path, path[1:])))


def best_linear_order(P: np.ndarray) -> list[int]:
    """Order maximising sum over pairs placed a-before-b of P[a, b] (the linear ordering problem; Kemeny-style
    when P holds pairwise log-odds). Exact DP over the set of pieces already placed. Returns indices 0..n-1."""
    n = P.shape[0]
    full = 1 << n
    dp = np.full(full, -np.inf)
    back = np.full(full, -1, dtype=np.int64)
    dp[0] = 0.0
    bits = np.array([[(m >> k) & 1 for k in range(n)] for m in range(full)], dtype=bool)
    for mask in range(full - 1):
        if dp[mask] == -np.inf:
            continue
        rest = ~bits[mask]
        for j in np.flatnonzero(rest):
            # placing j next puts it before every other piece not yet placed
            others = rest.copy()
            others[j] = False
            val = dp[mask] + P[j, others].sum()
            nm = mask | (1 << j)
            if val > dp[nm]:
                dp[nm], back[nm] = val, j
    order, mask = [], full - 1
    while mask:
        j = int(back[mask])
        order.append(j)
        mask ^= 1 << j
    return order[::-1]


def linear_order_value(P: np.ndarray, order: list[int]) -> float:
    return float(sum(P[a, b] for a, b in combinations(order, 2)))


# ----------------------------------------------------------------------------- inference
def cluster_boot(values: dict[str, float], n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean over videos and a 95% percentile CI, resampling channels (pre-registration §4).
    values: video id -> per-video mean (NaNs dropped)."""
    by: dict[str, list[float]] = {}
    for v, x in values.items():
        if x == x:
            by.setdefault(channel(v), []).append(x)
    groups = [np.asarray(g) for g in by.values()]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sums = np.array([g.sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(sums.sum() / cnts.sum()), float(lo), float(hi)
