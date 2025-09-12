import numpy as np
import pytest
from evaluations.evaluation import MetricsMetadata, MetricsRecordRaw, MetricsRecord
from evaluations.eval_vectors import VectorStats


@pytest.fixture
def sample_metadata() -> MetricsMetadata:
    """Provides a sample MetricsMetadata object for tests."""
    return MetricsMetadata(
        data_type="test",
        recon_strategy="MeanClosest",
        video_id="vid123",
        size=10,
        masked=[0, 5, 9]
    )

@pytest.fixture
def sample_raw_record(sample_metadata: MetricsMetadata) -> MetricsRecordRaw:
    """Provides a sample MetricsRecordRaw object for tests."""
    return MetricsRecordRaw(
        metadata=sample_metadata,
        raw_metrics={
            "similarity": np.array([0.8, 0.9, 0.7]),
            "distance": np.array([10.0, 20.0, 30.0]),
        }
    )


def test_vector_stats_from_vector():
    """Tests the VectorStats.from_vector class method."""
    vector = [1, 2, 3, 4, 5]
    stats = VectorStats.from_vector(vector)
    assert stats.mean == 3.0
    np.testing.assert_allclose(stats.std, 1.41421356)
    assert stats.min == 1.0
    assert stats.max == 5.0


def test_vector_stats_from_empty_vector():
    """Tests the edge case of an empty vector."""
    stats = VectorStats.from_vector([])
    assert stats.mean == 0.0
    assert stats.std == 0.0
    assert stats.min == 0.0
    assert stats.max == 0.0


def test_metrics_record_to_flat_dict(sample_metadata: MetricsMetadata):
    """Tests the flattening logic of the MetricsRecord."""
    record = MetricsRecord(
        metadata=sample_metadata,
        metrics={
            "similarity": VectorStats(mean=0.8, std=0.1, min=0.7, max=0.9),
            "distance": VectorStats(mean=20, std=8.16, min=10, max=30)
        }
    )
    flat_dict = record.to_flat_dict()

    # Check some metadata fields
    assert flat_dict["video_id"] == "vid123"
    assert flat_dict["recon_strategy"] == "MeanClosest"

    # Check flattened metrics fields
    assert flat_dict["similarity_mean"] == 0.8
    assert flat_dict["distance_std"] == 8.16
    assert flat_dict["similarity_max"] == 0.9


def test_metrics_record_raw_stats(sample_raw_record: MetricsRecordRaw):
    """Tests the stats() method for converting raw metrics to VectorStats."""
    stats_record = sample_raw_record.stats()

    # Check that the metadata is preserved
    assert stats_record.metadata == sample_raw_record.metadata

    # Check the calculations for one of the metrics
    similarity_stats = stats_record.metrics["similarity"]
    np.testing.assert_allclose(similarity_stats.mean, 0.8)
    np.testing.assert_allclose(similarity_stats.std, 0.081649658)
    assert similarity_stats.min == 0.7
    assert similarity_stats.max == 0.9


def test_metrics_record_raw_stats_z_score(sample_raw_record: MetricsRecordRaw):
    """Tests the stats_z_score() method for normalization."""
    z_score_record = sample_raw_record.stats_z_score({
        "similarity": VectorStats(mean=0.5, std=0.2, max=1., min=0.0),
        "distance": VectorStats(mean=0.5, std=0.2, max=1., min=0.0)
    })

    # The raw "similarity" vector was [0.8, 0.9, 0.7]
    # The z-scored vector should be ([0.8, 0.9, 0.7] - 0.5) / 0.2
    # which is [1.5, 2.0, 1.0]
    expected_z_vector = np.array([1.5, 2.0, 1.0])

    # Calculate the expected stats from the z-scored vector
    expected_stats = VectorStats.from_vector(expected_z_vector)

    # Get the actual stats from the result
    actual_stats = z_score_record.metrics["similarity"]

    assert actual_stats.mean == pytest.approx(expected_stats.mean)
    assert actual_stats.std == pytest.approx(expected_stats.std)
    assert actual_stats.min == pytest.approx(expected_stats.min)
    assert actual_stats.max == pytest.approx(expected_stats.max)
