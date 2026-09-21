from __future__ import annotations
import numpy as np
import pytest
from evaluations.evidence_retrieval import (
    calculate_evidence_retrieval_metrics,
    build_evidence_retrieval_index,
)


def test_perfect_retrieval():
    # 5 frames, dim=4
    # Query is [1, 0, 0, 0]
    # Frame 2 is [1, 0, 0, 0] (exact match)
    # Other frames are orthogonal or opposing
    query = np.array([1.0, 0.0, 0.0, 0.0])
    index = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],  # target (index 2)
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
    ])
    metrics = calculate_evidence_retrieval_metrics(query, index, target_indices={2})
    assert metrics["rank"] == 1
    assert metrics["mrr"] == 1.0
    assert metrics["recall_at_1"] == 1.0
    assert metrics["recall_at_5"] == 1.0
    assert pytest.approx(metrics["best_target_sim"], 0.01) == 1.0
    assert metrics["contrast_margin"] > 0


def test_worst_retrieval():
    # Target frame has the lowest similarity
    query = np.array([1.0, 0.0, 0.0, 0.0])
    index = np.array([
        [0.9, 0.1, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],  # target (index 3, orthogonal)
    ])
    metrics = calculate_evidence_retrieval_metrics(query, index, target_indices={3})
    assert metrics["rank"] == 4
    assert pytest.approx(metrics["mrr"], 0.01) == 0.25
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 1.0


def test_index_builders():
    T, D = 10, 4
    orig = np.random.randn(T, D).astype(np.float32)

    # Oracle
    oracle_idx = build_evidence_retrieval_index(orig, target_indices={3, 4}, condition="oracle")
    assert np.allclose(oracle_idx, orig)

    # Masked
    masked_idx = build_evidence_retrieval_index(orig, target_indices={3, 4}, condition="masked")
    assert np.allclose(masked_idx[3], 0.0)
    assert np.allclose(masked_idx[4], 0.0)
    assert np.allclose(masked_idx[0:3], orig[0:3])
    assert np.allclose(masked_idx[5:], orig[5:])

    # Baseline Repeat
    repeat_idx = build_evidence_retrieval_index(orig, target_indices={3, 4}, condition="baseline_repeat")
    assert np.allclose(repeat_idx[3], orig[2])
    assert np.allclose(repeat_idx[4], orig[2])

    # Reconstructed
    recon_vecs = np.ones((2, D), dtype=np.float32)
    recon_idx = build_evidence_retrieval_index(
        orig, target_indices={3, 4}, condition="reconstructed", reconstructed_embeddings=recon_vecs
    )
    assert np.allclose(recon_idx[3], 1.0)
    assert np.allclose(recon_idx[4], 1.0)
    assert np.allclose(recon_idx[0:3], orig[0:3])
