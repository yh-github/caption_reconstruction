import numpy as np
import pytest
from data.video_surprisal import VideoSurprisalScorer, VideoSurprisalResult
from analysis.categorize_videos import get_extended_video_complexity, get_video_complexity


class TestVideoSurprisalScorer:
    def setup_method(self):
        self.scorer = VideoSurprisalScorer()

    def test_empty_or_single_frame(self):
        # When less than 2 frames, returns default 0 values with unit rank and tortuosity
        res_empty = self.scorer.calculate_surprisal(np.empty((0, 768)))
        assert res_empty.avg_cosine_distance == 0.0
        assert res_empty.max_cosine_distance == 0.0
        assert res_empty.effective_rank == 1.0
        assert res_empty.tortuosity == 1.0

        res_single = self.scorer.calculate_surprisal(np.ones((1, 768)))
        assert res_single.avg_cosine_distance == 0.0
        assert res_single.max_cosine_distance == 0.0
        assert res_single.effective_rank == 1.0
        assert res_single.tortuosity == 1.0

    def test_identical_embeddings(self):
        # Static video: all frames are identical unit vectors
        v = np.random.randn(768)
        v = v / np.linalg.norm(v)
        embeddings = np.tile(v, (10, 1))

        res = self.scorer.calculate_surprisal(embeddings)
        assert np.isclose(res.avg_cosine_distance, 0.0, atol=1e-5)
        assert np.isclose(res.max_cosine_distance, 0.0, atol=1e-5)
        assert np.isclose(res.p95_cosine_distance, 0.0, atol=1e-5)
        assert np.isclose(res.variance_cosine_distance, 0.0, atol=1e-5)
        # For rank 1 matrix, effective rank should be approx 1
        assert np.isclose(res.effective_rank, 1.0, atol=1e-1)

    def test_orthogonal_embeddings(self):
        # 3 mutually orthogonal unit vectors
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        e3 = np.array([0.0, 0.0, 1.0])
        embeddings = np.stack([e1, e2, e3])

        res = self.scorer.calculate_surprisal(embeddings)
        # Cosine distance between e1 and e2 is 1 - 0 = 1.0; between e2 and e3 is 1 - 0 = 1.0
        assert np.isclose(res.avg_cosine_distance, 1.0, atol=1e-5)
        assert np.isclose(res.max_cosine_distance, 1.0, atol=1e-5)
        assert np.isclose(res.p95_cosine_distance, 1.0, atol=1e-5)
        # Effective rank for 3 equal singular values: exp(ln(3)) = 3.0
        assert np.isclose(res.effective_rank, 3.0, atol=1e-2)
        # Path length = ||e2 - e1|| + ||e3 - e2|| = sqrt(2) + sqrt(2) = 2*sqrt(2)
        # Displacement = ||e3 - e1|| = sqrt(2)
        # Tortuosity = 2*sqrt(2) / sqrt(2) = 2.0
        assert np.isclose(res.tortuosity, 2.0, atol=1e-4)

    def test_p95_metric(self):
        # 20 frames: first 18 have distance 0.1, last 1 has distance 0.9
        # Check that p95 captures the high percentile
        T = 20
        dists_expected = [0.1] * 18 + [0.9]
        # Construct embeddings in 2D with these angular steps
        angles = [0.0]
        for d in dists_expected:
            # cos(theta) = 1 - d
            theta = np.arccos(np.clip(1.0 - d, -1.0, 1.0))
            angles.append(angles[-1] + theta)
        embeddings = np.column_stack([np.cos(angles), np.sin(angles)])

        res = self.scorer.calculate_surprisal(embeddings)
        assert res.max_cosine_distance >= 0.89
        assert res.p95_cosine_distance > res.avg_cosine_distance


class TestExtendedVideoComplexity:
    def test_get_extended_video_complexity(self):
        embeddings = np.random.randn(15, 64)
        result_dict = get_extended_video_complexity(embeddings)

        expected_keys = {
            "mean_surprisal",
            "max_surprisal",
            "variance_surprisal",
            "p95_surprisal",
            "effective_rank",
            "tortuosity",
        }
        assert set(result_dict.keys()) == expected_keys
        assert result_dict["mean_surprisal"] >= 0.0
        assert result_dict["max_surprisal"] >= result_dict["mean_surprisal"]
        assert result_dict["effective_rank"] >= 1.0
        assert result_dict["tortuosity"] >= 1.0
