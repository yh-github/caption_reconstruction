import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# Import all the classes and functions we need to test or use
from evaluation import (
    ReconstructionEvaluator,
    ReconstructionEvaluator_BertScore,
    ReconstructionEvaluator_EmbSimilarity,
    VectorReconstructionEvaluator,
    MetricsRecordRaw, MetricsMetadata
)
from data_models.captions_only import CaptionedVideo, CaptionedClip, TimestampRange
from reconstruction_strategies import Reconstructed
from vectors.eval_vectors import VectorStats


@pytest.fixture
def sample_metrics_records():
    """Provides a list of sample MetricsRecordRaw objects for testing aggregation."""
    # Create real, concrete instances of the metadata object
    metadata1 = MetricsMetadata(
        data_type="video_captions",
        recon_strategy="test_strategy",
        video_id="vid1",
        size=10,
        masked=[2, 3]
    )
    metadata2 = MetricsMetadata(
        data_type="video_captions",
        recon_strategy="test_strategy",
        video_id="vid2",
        size=8,
        masked=[4, 5]
    )

    return [
        MetricsRecordRaw(metadata=metadata1, raw_metrics={"score": np.array([0.5, 0.6])}),
        MetricsRecordRaw(metadata=metadata2, raw_metrics={"score": np.array([0.7, 0.8])}),
    ]
@pytest.fixture
def sample_data():
    """Provides sample original and reconstructed data for tests using the new models."""
    original_video = CaptionedVideo(
        video_id="test_vid",
        clips=[
            CaptionedClip(index=0, timestamp=TimestampRange(start=0.0, duration=1.0),caption="clip one original"),
            CaptionedClip(index=1, timestamp=TimestampRange(start=1.0, duration=1.0), caption="clip two original"),
            CaptionedClip(index=2, timestamp=TimestampRange(start=2.0, duration=1.0), caption="clip three original")
        ]
    )
    # Use a mock for Reconstructed that has a predictable align method
    reconstructed_data = MagicMock(spec=Reconstructed)
    reconstructed_data.align.return_value = (["clip one recon"], ["clip one original"])
    return original_video, reconstructed_data


# --- Tests for ReconstructionEvaluator (Static Methods) ---

def test_agg_metrics(sample_metrics_records):
    """Tests that the aggregation function correctly calculates the mean of means."""
    agg = ReconstructionEvaluator.agg_metrics(sample_metrics_records, "mean")
    # Mean of video 1 is 0.55, mean of video 2 is 0.75.
    # The mean of these two means is (0.55 + 0.75) / 2 = 0.65
    assert agg['mean_score_mean'] == pytest.approx(0.65)
    assert agg['num_of_instances'] == 2


def test_global_stats(sample_metrics_records):
    """Tests that global stats correctly concatenates vectors."""
    stats:dict[str, VectorStats] = ReconstructionEvaluator.global_stats(sample_metrics_records)
    actual_stats = stats['score']
    expected_stats = VectorStats(mean=0.65, std=0.1118033988749895, min=0.5, max=0.8)
    print(actual_stats)
    assert actual_stats.mean == expected_stats.mean
    assert actual_stats.std == expected_stats.std
    assert actual_stats.min == expected_stats.min
    assert actual_stats.max == expected_stats.max


# --- Test for VectorReconstructionEvaluator ---

def test_vector_reconstruction_evaluator_with_real_data():
    """
    Tests the vector evaluator's full logic using real, predictable vectors.
    """
    # Arrange
    evaluator = VectorReconstructionEvaluator()

    # Create simple vectors where the cosine similarity is easy to calculate
    # Vector 1: Identical vectors (cosine similarity = 1.0)
    # Vector 2: Orthogonal vectors (cosine similarity = 0.0)
    # Vector 3: Opposite vectors (cosine similarity = -1.0)
    pred_vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    true_vecs = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])

    # Act
    metrics = evaluator.evaluate(pred_vecs, true_vecs)

    # Assert
    # Check that the calculated cosine similarities are correct
    expected_similarities = np.array([1.0, 0.0, -1.0])
    assert "cos_sim" in metrics
    assert np.allclose(metrics['cos_sim'], expected_similarities)

# --- Test for ReconstructionEvaluator_BertScore ---

@patch('evaluation.BERTScorer')
def test_bert_score_evaluator(MockBERTScorer, sample_text_data):
    """
    Tests the BERTScore evaluator. We mock BERTScorer to avoid loading the model.
    """
    # Arrange
    # Configure the mock instance that will be created
    mock_scorer_instance = MockBERTScorer.return_value
    mock_scorer_instance.score.return_value = (
        torch.tensor([0.9]), torch.tensor([0.8]), torch.tensor([0.85])
    )

    original_video, reconstructed_data = sample_text_data
    evaluator = ReconstructionEvaluator_BertScore(model_type="mock-model")

    # Act
    metrics = evaluator.evaluate(reconstructed_data, original_video)

    # Assert
    # 1. Check that align was called to get the sentences
    reconstructed_data.align.assert_called_once_with(original_video.clips)

    # 2. Check that the score method of our mock was called
    mock_scorer_instance.score.assert_called_once()

    # 3. Check that the metrics are correctly converted to numpy arrays
    assert "bs_f1" in metrics
    assert isinstance(metrics["bs_f1"], np.ndarray)
    assert metrics["bs_f1"][0] == pytest.approx(0.85)


# --- Test for ReconstructionEvaluator_EmbSimilarity ---

@patch('evaluation.Embedder')
def test_emb_similarity_evaluator(MockEmbedder, sample_text_data):
    """
    Tests the embedding similarity evaluator. We mock the Embedder.
    """
    # Arrange
    mock_embedder_instance = MockEmbedder.return_value
    mock_embedder_instance.get_embeddings.side_effect = [
        np.array([[1.0, 0.0]]),  # Embedding for candidate
        np.array([[0.0, 1.0]])  # Embedding for reference
    ]

    original_video, reconstructed_data = sample_text_data
    evaluator = ReconstructionEvaluator_EmbSimilarity(embedder=mock_embedder_instance)

    # Act
    metrics = evaluator.evaluate(reconstructed_data, original_video)

    # Assert
    # 1. Check that get_embeddings was called twice (for cands and refs)
    assert mock_embedder_instance.get_embeddings.call_count == 2

    # 2. Check that the cosine similarity is calculated correctly (should be 0 for orthogonal vectors)
    assert "cos_sim" in metrics
    assert metrics["cos_sim"][0] == pytest.approx(0.0)