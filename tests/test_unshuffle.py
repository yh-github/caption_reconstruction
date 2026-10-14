"""Unit tests for scripts/unshuffle (core metrics, exact ordering DPs, puzzles, cue removal, chaining)."""
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from unshuffle import core  # noqa: E402
from unshuffle.cues import gerund, remove_cues, third  # noqa: E402


# ----------------------------------------------------------------------------- metrics
def test_score_perfect_and_reversed():
    s = core.score(list(range(12)))
    assert s["tau"] == 1 and s["adj_und"] == 1 and s["adj_dir"] == 1 and s["exact"] == 1 and s["local_dir"] == 1
    r = core.score(list(range(12))[::-1])
    assert r["tau"] == -1 and r["adj_und"] == 1 and r["adj_dir"] == 0 and r["local_dir"] == 0


def test_score_swapped_halves():
    # kickoff plan 3.4: halves right but swapped -> tau = -0.09, directed adjacency 10/11
    s = core.score(list(range(6, 12)) + list(range(6)))
    assert s["tau"] == pytest.approx(-6 / 66)
    assert s["adj_dir"] == pytest.approx(10 / 11) and s["adj_und"] == pytest.approx(10 / 11)
    assert s["exact"] == 0


def test_score_gapped_pieces_use_puzzle_neighbours():
    s = core.score([0, 2, 4, 6, 8, 10])
    assert s["tau"] == 1 and s["adj_dir"] == 1
    s = core.score([2, 0, 4])
    assert s["adj_und"] == pytest.approx(0.5) and s["adj_dir"] == 0  # (0,2) adjacent reversed, (2,4) not


def test_chance_matches_enumeration():
    n = 5
    rows = [core.score(list(p)) for p in permutations(range(n))]
    ch = core.chance(n)
    for k in ["tau", "adj_und", "adj_dir", "exact"]:
        assert np.mean([r[k] for r in rows]) == pytest.approx(ch[k])


def test_position_bias():
    pres = [3, 1, 2, 0]
    assert core.position_bias(pres, pres) == 1
    assert core.position_bias(pres[::-1], pres) == -1


def test_echo_reader_scores_zero_over_the_shuffle_pair():
    for pz in core.make_puzzles(["v_1-clip-0"]):
        s1, s2 = pz["shuffles"]
        assert s2 == s1[::-1] and sorted(s1) == pz["pieces"]
        if len(s1) > 1:
            assert core.kendall_tau(s1) + core.kendall_tau(s2) == pytest.approx(0)


# ----------------------------------------------------------------------------- puzzles
def test_sets_and_gaps():
    assert core.SETS["K4d2"] == [[0, 2, 4, 6], [3, 5, 7, 9], [5, 7, 9, 11]]
    assert core.SETS["K4d3"] == [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
    gaps = {name: core.gap_s(g[0]) for name, g in core.SETS.items()}
    assert gaps == {"P12": 0, "C6": 0, "G6": 5, "K4d1": 0, "K4d2": 5, "K4d3": 10}
    assert all(max(p) < core.N_PIECES for g in core.SETS.values() for p in g)


def test_puzzles_deterministic_and_video_specific():
    a = core.make_puzzles(["x_1-clip-0", "y_2-clip-1"])
    b = core.make_puzzles(["x_1-clip-0", "y_2-clip-1"])
    assert a == b
    p12 = {p["video"]: p["shuffles"][0] for p in a if p["set"] == "P12"}
    assert p12["x_1-clip-0"] != p12["y_2-clip-1"]
    assert len(a) == 2 * sum(len(g) for g in core.SETS.values())


def test_channel():
    assert core.channel("Olly_s-Farm_11-clip-1") == "Olly's-Farm"
    assert core.channel("USA-Military-Channel_6-manual") == "USA-Military-Channel"
    assert core.channel("Joe-Robinet_9-clip-2") == "Joe-Robinet"


def test_restrict_and_reduce():
    assert core.reduce_puzzle([0, 1, 2, 3], {1}) == [0, 2, 3]
    assert core.restrict([3, 1, 0, 2], [0, 2, 3]) == [3, 0, 2]


# ----------------------------------------------------------------------------- exact DPs
@pytest.mark.parametrize("n,seed", [(2, 0), (3, 1), (5, 2), (6, 3), (7, 4)])
def test_best_path_is_optimal(n, seed):
    W = np.random.default_rng(seed).normal(size=(n, n))
    best = max(core.path_weight(W, list(p)) for p in permutations(range(n)))
    path = core.best_path(W)
    assert sorted(path) == list(range(n))
    assert core.path_weight(W, path) == pytest.approx(best)


@pytest.mark.parametrize("n,seed", [(2, 0), (3, 1), (5, 2), (6, 3), (7, 4)])
def test_best_linear_order_is_optimal(n, seed):
    A = np.random.default_rng(seed).normal(size=(n, n))
    P = A - A.T  # antisymmetric log-odds
    best = max(core.linear_order_value(P, list(p)) for p in permutations(range(n)))
    order = core.best_linear_order(P)
    assert sorted(order) == list(range(n))
    assert core.linear_order_value(P, order) == pytest.approx(best)


def test_best_linear_order_on_a_score_is_sorting():
    s = np.random.default_rng(5).normal(size=8)
    P = s[None, :] - s[:, None]  # P[a, b] > 0 when a has the lower score
    assert core.best_linear_order(P) == list(np.argsort(s))


def test_best_path_recovers_a_chain():
    # similarity decays with distance: the best path is the true chain (either direction)
    x = np.arange(9)
    W = -np.abs(x[:, None] - x[None, :]).astype(float)
    path = core.best_path(W)
    assert path in (list(x), list(x[::-1]))


# ----------------------------------------------------------------------------- bootstrap
def test_cluster_boot_mean_and_ci():
    vals = {f"ch{c}_{i}-clip-0": float(c) for c in range(10) for i in range(2)}
    m, lo, hi = core.cluster_boot(vals, n_boot=2000)
    assert m == pytest.approx(4.5) and lo < 4.5 < hi
    m, lo, hi = core.cluster_boot({"a_1-clip-0": 1.0, "b_1-clip-0": float("nan")})
    assert m == 1.0


# ----------------------------------------------------------------------------- cue removal
def test_inflection():
    assert [third(v) for v in ["cut", "carry", "push", "go", "be", "play"]] == \
        ["cuts", "carries", "pushes", "goes", "is", "plays"]
    assert [gerund(v) for v in ["make", "cut", "see", "tie", "walk"]] == \
        ["making", "cutting", "seeing", "tying", "walking"]


# 20 captions from the pilot's in-context and original sets (2026-10-14), with the rewrite read and accepted.
CUE_CASES = [
    ("The man continues to hammer bamboo poles into the wall frame.",
     "The man hammers bamboo poles into the wall frame."),
    ("The river continues rushing over smooth river stones from a low vantage point.",
     "The river is rushing over smooth river stones from a low vantage point."),
    ("The hikers continue wading through the water and exit to the right side of the frame.",
     "The hikers are wading through the water and exit to the right side of the frame."),
    ("The two hikers remain seated on the rocky shore against the shadow of the canyon cliffs.",
     "The two hikers are seated on the rocky shore against the shadow of the canyon cliffs."),
    ("The clear stream keeps flowing between the massive canyon cliffs.",
     "The clear stream is flowing between the massive canyon cliffs."),
    ("The person strikes the fire rod again, successfully igniting the wood shavings into a bright flame.",
     "The person strikes the fire rod, successfully igniting the wood shavings into a bright flame."),
    ("Still under the tractor, the man reaches into the undercarriage opening.",
     "Under the tractor, the man reaches into the undercarriage opening."),
    ("The man stands still for a moment, enjoying the forest breeze.",
     "The man stands still for a moment, enjoying the forest breeze."),
    ("The aircraft continues its journey across the vast and empty desert environment.",
     "The aircraft makes its journey across the vast and empty desert environment."),
    ("The camera continues down the trail as the forest opens slightly.",
     "The camera moves down the trail as the forest opens slightly."),
    ("The storm continues with high-speed winds and torrential rain in the dark.",
     "The storm shows high-speed winds and torrential rain in the dark."),
    ("The shoveling continues as the man clears the access point.",
     "The shoveling is visible as the man clears the access point."),
    ("He makes another cut on one of the few remaining ridges.",
     "He makes a cut on one of the few ridges."),
    ("The video begins with a dark and blurry view from inside the car.",
     "The video shows a dark and blurry view from inside the car."),
    ("The hovercraft is now halfway into the ship's docking area.",
     "The hovercraft is halfway into the ship's docking area."),
    ("The tornado is still quite large and powerful as it moves away from the road.",
     "The tornado is quite large and powerful as it moves away from the road."),
    ("The man positions his knife near the top of the cactus stem and begins cutting into the flesh.",
     "The man positions his knife near the top of the cactus stem and is cutting into the flesh."),
    ("The crew members begin to move away from their signaling positions.",
     "The crew members move away from their signaling positions."),
    ("Continuing along the ridge with rolling hills and water.",
     "Moving along the ridge with rolling hills and water."),
    ("A man walks past a red barn next to another tractor before the camera cuts away.",
     "A man walks past a red barn next to a tractor before the camera cuts away."),
]


@pytest.mark.parametrize("src,want", CUE_CASES)
def test_remove_cues(src, want):
    assert remove_cues(src) == want


def test_remove_cues_leaves_cue_free_text_alone():
    s = 'A man in a blue jacket walks toward the "27 NOVEMBER" title card, then turns left before a tree.'
    assert remove_cues(s) == s
