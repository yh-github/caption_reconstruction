"""No-reader and learned ordering arms (pilot pre-registration §3). Each arm maps a puzzle's pieces to an order.

    chain (symmetric)  best Hamiltonian path under cosine similarity of piece embeddings; the direction is a
                       seeded coin flip, so the arm carries no direction by construction (Prop. 2)
    edge (directed)    best path under similarity of piece i's last frame to piece j's first frame
    probe              ridge regression embedding -> relative position, leave-one-channel-out; sort by prediction
    pairwise           logistic regression on embedding differences (P(i before j)), same folds; on a linear
                       model the exact linear-ordering optimum is the sort by the learned score (tested)
"""
from __future__ import annotations

import random

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

from unshuffle.core import N_PIECES, SEED, best_path, channel

ALPHAS = [0.1, 1, 10, 100, 1000]


def coin(*key) -> bool:
    return random.Random("|".join(map(str, (SEED, "coin", *key)))).random() < 0.5


# Every arm gets the pieces in the puzzle's presentation order (shuffle 1), never in true order: the DPs break
# exact ties by index (a symmetric path and its reverse always tie), and sorting keeps the input order of ties,
# so a true-order input would leak the answer into those choices.
def chain(E: np.ndarray, presented: list[int], key: tuple) -> list[int]:
    """Symmetric chaining: E holds one unit row per piece of the clip (row p = piece p)."""
    X = E[presented]
    path = [presented[i] for i in best_path(X @ X.T)]
    return path[::-1] if coin(*key) else path


def edge_chain(first: np.ndarray, last: np.ndarray, presented: list[int]) -> list[int]:
    W = last[presented] @ first[presented].T  # W[i, j]: piece i's last frame -> piece j's first frame
    return [presented[i] for i in best_path(W)]


def order_by_score(score: np.ndarray, presented: list[int]) -> list[int]:
    """Earliest first: higher score = earlier."""
    return sorted(presented, key=lambda p: -score[p])


# ----------------------------------------------------------------------------- learned arms
def _xy_position(E: dict[str, np.ndarray], vids: list[str]):
    X = np.concatenate([E[v] for v in vids])
    y = np.tile(np.arange(N_PIECES) / (N_PIECES - 1), len(vids))
    return X, y


def _xy_pairs(E: dict[str, np.ndarray], vids: list[str]):
    i, j = np.where(~np.eye(N_PIECES, dtype=bool))
    X = np.concatenate([E[v][i] - E[v][j] for v in vids])
    y = np.tile((i < j).astype(int), len(vids))
    return X, y


def _fit(kind: str, alpha: float, X, y):
    if kind == "probe":
        return Ridge(alpha=alpha).fit(X, y)
    return LogisticRegression(C=1 / alpha, fit_intercept=False, max_iter=2000).fit(X, y)


def _loss(kind: str, model, X, y) -> float:
    if kind == "probe":
        return float(np.mean((model.predict(X) - y) ** 2))
    p = np.clip(model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _score(kind: str, model, E_v: np.ndarray) -> np.ndarray:
    """Higher = earlier."""
    if kind == "probe":
        return -model.predict(E_v)
    return E_v @ model.coef_[0]  # P(i before j) = sigmoid(w . (e_i - e_j))


def _choose_alpha(kind: str, E: dict[str, np.ndarray], vids: list[str]) -> float:
    xy = _xy_position if kind == "probe" else _xy_pairs
    groups = [channel(v) for v in vids]
    folds = GroupKFold(n_splits=min(5, len(set(groups))))
    losses = {a: [] for a in ALPHAS}
    for tr, te in folds.split(vids, groups=groups):
        Xtr, ytr = xy(E, [vids[k] for k in tr])
        Xte, yte = xy(E, [vids[k] for k in te])
        for a in ALPHAS:
            losses[a].append(_loss(kind, _fit(kind, a, Xtr, ytr), Xte, yte))
    return min(ALPHAS, key=lambda a: np.mean(losses[a]))


def _one_fold(kind: str, E: dict[str, np.ndarray], held: list[str], train: list[str]):
    alpha = _choose_alpha(kind, E, train)
    xy = _xy_position if kind == "probe" else _xy_pairs
    model = _fit(kind, alpha, *xy(E, train))
    return {v: (_score(kind, model, E[v]), alpha) for v in held}


def loco_scores(kind: str, E: dict[str, np.ndarray], n_jobs: int = 8) -> dict[str, tuple[np.ndarray, float]]:
    """Leave-one-channel-out: video -> (per-piece score, higher = earlier; the alpha chosen in its fold)."""
    vids = sorted(E)
    by_ch: dict[str, list[str]] = {}
    for v in vids:
        by_ch.setdefault(channel(v), []).append(v)
    jobs = [(held, [v for v in vids if channel(v) != c]) for c, held in by_ch.items()]
    out: dict[str, tuple[np.ndarray, float]] = {}
    for part in Parallel(n_jobs=n_jobs)(delayed(_one_fold)(kind, E, h, t) for h, t in jobs):
        out.update(part)
    return out
