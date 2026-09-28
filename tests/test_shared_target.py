"""
tests/test_shared_target.py

Pre-registered automated unit tests for shared-target evaluation (Phase 1):
1. Metric identity & tie-handling: c in [-1, 1], mid-ranks, exact formula matches
2. Shuffle null: permuted scores yield mean c within bootstrap tolerance of 0.0
3. Pool integrity: no boundary contamination, no gap inclusion, no stem collisions, deterministic seeds
4. Arm generation & alignment parser checks
5. Oracle identity checks (V_Oracle in frame-home = 1.0, T_Oracle in caption-home = 1.0)
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from shared_target.metrics import (
    compute_mid_rank,
    compute_calibrated_score,
    compute_mrr,
    compute_top_k,
    compute_headline_score
)
from shared_target.pools import (
    Candidate,
    VideoMeta,
    parse_video_id,
    get_gap_boundaries,
    build_same_video_near_candidates,
    build_same_video_far_candidates,
    build_boundary_diag_candidates,
    build_other_video_candidates
)
from shared_target.arms import (
    normalize_vector,
    compute_lerp_alpha,
    get_nearest_boundary_idx,
    ArmGenerator
)
from shared_target.stats import (
    classify_outcome,
    stratified_cluster_bootstrap,
    holm_bonferroni_correction,
    PairedTestResult
)


class TestMetrics:
    def test_metric_identity_extreme_ranks(self):
        """Tests c formula at rank 1, rank N, and chance (N+1)/2 across varying N."""
        for N in [5, 10, 50, 100, 200]:
            # Perfect rank r=1 -> c = +1.0
            assert compute_calibrated_score(1.0, N) == pytest.approx(1.0)
            # Worst rank r=N -> c = -1.0
            assert compute_calibrated_score(float(N), N) == pytest.approx(-1.0)
            # Chance rank r=(N+1)/2 -> c = 0.0
            chance_r = (N + 1.0) / 2.0
            assert compute_calibrated_score(chance_r, N) == pytest.approx(0.0)

    def test_mid_rank_and_ties(self):
        """Verifies mid-rank computation with strictly greater, strictly smaller, and ties."""
        # Target = 0.8, Distractors = [0.9, 0.7, 0.6] -> 1 strictly greater -> rank 2
        r, has_tie = compute_mid_rank(0.8, [0.9, 0.7, 0.6])
        assert r == 2.0
        assert not has_tie

        # Target = 0.8, Distractors = [0.8, 0.7, 0.6] -> 1 equal -> mid-rank 1 + 0.5 = 1.5
        r, has_tie = compute_mid_rank(0.8, [0.8, 0.7, 0.6])
        assert r == 1.5
        assert has_tie

        # Target = 0.8, Distractors = [0.8, 0.8, 0.8] -> 3 equal -> mid-rank 1 + 1.5 = 2.5
        r, has_tie = compute_mid_rank(0.8, [0.8, 0.8, 0.8])
        assert r == 2.5
        assert has_tie

    def test_mrr_and_top_k(self):
        assert compute_mrr(1.0) == 1.0
        assert compute_mrr(2.0) == 0.5
        assert compute_top_k(1.0, k=1) == 1.0
        assert compute_top_k(1.5, k=1) == 0.0
        assert compute_top_k(1.5, k=5) == 1.0
        assert compute_top_k(6.0, k=5) == 0.0

    def test_shuffle_null(self):
        """Permutes scores within pools; mean c must lie within CI of 0.0."""
        rng = np.random.default_rng(12345)
        n_draws = 2000
        pool_size = 31  # 1 target + 30 distractors
        
        calibrated_scores = []
        for _ in range(n_draws):
            scores = rng.standard_normal(pool_size)
            target_score = scores[0]
            distractor_scores = scores[1:]
            r, _ = compute_mid_rank(target_score, distractor_scores)
            c = compute_calibrated_score(r, pool_size)
            calibrated_scores.append(c)
            
        mean_c = np.mean(calibrated_scores)
        se_c = np.std(calibrated_scores, ddof=1) / np.sqrt(n_draws)
        # Verify 99% tolerance around 0.0
        assert abs(mean_c) < 2.58 * se_c, f"Shuffle null biased! mean={mean_c:.5f}, se={se_c:.5f}"


class TestPoolIntegrity:
    def test_video_id_parsing(self):
        ch, st = parse_video_id("AiirSource-Military_1-clip-0")
        assert ch == "AiirSource-Military"
        assert st == "AiirSource-Military_1"
        
        ch2, st2 = parse_video_id("How-Farms-Work_9-manual")
        assert ch2 == "How-Farms-Work"
        assert st2 == "How-Farms-Work_9"

    def test_same_video_near_exclusions(self):
        """Verifies near pool excludes gap seconds and boundary seconds."""
        vid = "test_vid_1"
        gap = [27, 28, 29, 30, 31, 32]  # W=6, mid gap
        near = build_same_video_near_candidates(vid, gap, window_sec=10, total_seconds=60)
        near_secs = {c.second for c in near}
        
        # Excluded: gap [27..32], left boundary 26, right boundary 33
        for s in range(26, 34):
            assert s not in near_secs, f"Second {s} contaminated the near pool!"
            
        # Left window: [27 - 10, 25] = [17..25] (9 seconds)
        # Right window: [34, 32 + 10] = [34..42] (9 seconds)
        assert near_secs == set(range(17, 26)) | set(range(34, 43))
        assert len(near) == 18

    def test_boundary_diag_includes_boundaries(self):
        vid = "test_vid_1"
        gap = [27, 28, 29, 30, 31, 32]
        diag = build_boundary_diag_candidates(vid, gap, window_sec=10, total_seconds=60)
        diag_secs = {c.second for c in diag}
        
        # Boundaries 26 and 33 MUST be present
        assert 26 in diag_secs
        assert 33 in diag_secs
        # Gap itself must NOT be present
        for s in gap:
            assert s not in diag_secs
        assert len(diag) == 20

    def test_same_video_far_exclusions_and_reproducibility(self):
        vid = "test_vid_1"
        gap = [27, 28, 29, 30, 31, 32]
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        far1 = build_same_video_far_candidates(vid, gap, sample_size=30, rng=rng1)
        far2 = build_same_video_far_candidates(vid, gap, sample_size=30, rng=rng2)
        
        # Reproducibility under same seed
        assert [c.second for c in far1] == [c.second for c in far2]
        assert len(far1) == 30
        
        # Exclusions: gap [27..32], boundaries [26, 33], near [17..25, 34..42]
        excluded = set(range(17, 43))
        for c in far1:
            assert c.second not in excluded, f"Far candidate {c.second} in excluded set!"

    def test_other_video_channel_and_split_isolation(self):
        target = VideoMeta("AiirSource-Military_1-clip-0", "Wild5", "Military", "AiirSource-Military", "AiirSource-Military_1")
        same_stem = VideoMeta("AiirSource-Military_1-clip-1", "Wild5", "Military", "AiirSource-Military", "AiirSource-Military_1")
        same_ch = VideoMeta("AiirSource-Military_2-clip-0", "Wild5", "Military", "AiirSource-Military", "AiirSource-Military_2")
        diff_ch_same_domain = VideoMeta("WarLeaks-Military-Blog_3-clip-0", "Wild5", "Military", "WarLeaks-Military-Blog", "WarLeaks-Military-Blog_3")
        diff_split = VideoMeta("WarLeaks-Military-Blog_4-clip-0", "Wild4", "Military", "WarLeaks-Military-Blog", "WarLeaks-Military-Blog_4")
        
        all_videos = [target, same_stem, same_ch, diff_ch_same_domain, diff_split]
        
        # other_video_other_channel
        cands = build_other_video_candidates(target, all_videos, same_channel=False, sample_size=10, rng=np.random.default_rng(42))
        for c in cands:
            assert c.video_id == "WarLeaks-Military-Blog_3-clip-0"
            assert c.video_id != target.video_id
            assert c.video_id != same_stem.video_id
            assert c.video_id != same_ch.video_id
            assert c.video_id != diff_split.video_id

        # other_video_same_channel
        cands_same_ch = build_other_video_candidates(target, all_videos, same_channel=True, sample_size=10, rng=np.random.default_rng(42))
        for c in cands_same_ch:
            assert c.video_id == "AiirSource-Military_2-clip-0"


class TestArmsAndOracles:
    def test_oracle_identities(self):
        """V_Oracle in frame-home must give c=1.0. T_Oracle in caption-home must give c=1.0."""
        rng = np.random.default_rng(999)
        dim = 768
        
        # Mock frame and caption embeddings for 60 seconds
        gt_frames = rng.standard_normal((60, dim))
        gt_frames = gt_frames / np.linalg.norm(gt_frames, axis=1, keepdims=True)
        
        gt_texts = [f"caption {s}" for s in range(60)]
        text_embs = rng.standard_normal((60, dim))
        text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
        
        text_cache = {gt_texts[s]: text_embs[s] for s in range(60)}

        def mock_embed_fn(texts):
            res = []
            for txt in texts:
                if txt in text_cache:
                    res.append(text_cache[txt])
                else:
                    v = rng.standard_normal(dim)
                    res.append(v / np.linalg.norm(v))
            return res
        
        arm_gen = ArmGenerator(embed_text_fn=mock_embed_fn)
        gap = [27, 28, 29, 30, 31, 32]
        t = 29
        
        preds = arm_gen.generate_arms(
            t=t,
            gap_seconds=gap,
            gt_captions=gt_texts,
            gt_frames=gt_frames,
            llama_captions={29: "predicted caption"}
        )
        
        # Frame-home oracle check
        v_oracle_pred = preds["V_Oracle"].vector
        target_frame = gt_frames[t]
        distractor_frames = [gt_frames[s] for s in [10, 15, 20, 45, 50]]
        
        target_sim = float(np.dot(v_oracle_pred, target_frame))
        dist_sims = [float(np.dot(v_oracle_pred, df)) for df in distractor_frames]
        r_frame, has_tie = compute_mid_rank(target_sim, dist_sims)
        c_frame = compute_calibrated_score(r_frame, 1 + len(distractor_frames))
        
        assert target_sim == pytest.approx(1.0)
        assert r_frame == 1.0
        assert c_frame == 1.0
        assert not has_tie

        # Caption-home oracle check
        t_oracle_pred = preds["T_Oracle"].vector
        target_caption_emb = text_embs[t]
        distractor_caption_embs = [text_embs[s] for s in [10, 15, 20, 45, 50]]
        
        target_cap_sim = float(np.dot(t_oracle_pred, target_caption_emb))
        dist_cap_sims = [float(np.dot(t_oracle_pred, dc)) for dc in distractor_caption_embs]
        r_cap, has_tie_cap = compute_mid_rank(target_cap_sim, dist_cap_sims)
        c_cap = compute_calibrated_score(r_cap, 1 + len(distractor_caption_embs))
        
        assert target_cap_sim == pytest.approx(1.0)
        assert r_cap == 1.0
        assert c_cap == 1.0
        assert not has_tie_cap


class TestStatisticalInference:
    def test_outcome_classification(self):
        assert classify_outcome(0.01, 0.08, 0.05) == "superior"
        assert classify_outcome(-0.08, -0.01, 0.05) == "inferior"
        assert classify_outcome(-0.03, 0.04, 0.05) == "equivalent"
        assert classify_outcome(-0.06, 0.04, 0.05) == "inconclusive"

    def test_holm_bonferroni(self):
        tests = [
            PairedTestResult("C1", 0.05, 0.01, 0.09, 0.02, 0.01),
            PairedTestResult("C2", 0.02, -0.01, 0.05, 0.02, 0.04),
            PairedTestResult("C3", 0.00, -0.03, 0.03, 0.02, 0.80),
            PairedTestResult("C4", -0.01, -0.04, 0.02, 0.02, 0.50),
        ]
        corrected = holm_bonferroni_correction(tests)
        # C1 has lowest p (0.01) * 4 = 0.04
        assert corrected[0].p_adjusted == pytest.approx(0.04)
        # C2 has 2nd lowest p (0.04) * 3 = 0.12
        assert corrected[1].p_adjusted == pytest.approx(0.12)
