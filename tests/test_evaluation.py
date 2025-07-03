# tests/test_evaluation.py

import pytest

import torch

from evaluation import evaluate_reconstruction

from data_models import TranscriptClip, NarrativeOnlyPayload

from constants import DATA_MISSING



def test_evaluate_reconstruction_with_bert_score(mocker):

    """

    Tests that the evaluation function correctly calls bert_score

    with the aligned candidate and reference sentences.

    """

    # Arrange

    # Mock the bert_score.score function

    mock_bert_scorer = mocker.patch('evaluation.bert_score')

    # Make it return predictable tensor values

    mock_bert_scorer.return_value = (

        torch.tensor([0.9, 0.95]), # Precision

        torch.tensor([0.8, 0.85]), # Recall

        torch.tensor([0.85, 0.9])  # F1

    )



    # Create a ground truth transcript

    ground_truth = [

        TranscriptClip(timestamp=1.0, data=NarrativeOnlyPayload(description="original one")),

        TranscriptClip(timestamp=2.0, data=NarrativeOnlyPayload(description="original two")),

        TranscriptClip(timestamp=3.0, data=NarrativeOnlyPayload(description="original three")),

    ]

    

    # Create a reconstructed transcript where the middle clip was filled in

    reconstruction = [

        TranscriptClip(timestamp=1.0, data=DATA_MISSING),

        TranscriptClip(timestamp=2.0, data=NarrativeOnlyPayload(description="reconstructed two")),

        TranscriptClip(timestamp=3.0, data=DATA_MISSING),

    ]



    # Act

    metrics = evaluate_reconstruction(reconstruction, ground_truth)



    # Assert

    # 1. Check that bert_score was called

    mock_bert_scorer.assert_called_once()

    

    # 2. Check that it was called with the correct sentences

    args, kwargs = mock_bert_scorer.call_args

    assert args[0] == ["reconstructed two"] # Candidates

    assert args[1] == ["original two"]      # References

    

    # 3. Check that the metrics were calculated correctly (mean of the tensors)

    assert "bert_score_f1" in metrics

    assert metrics["bert_score_f1"] == pytest.approx(0.875) # (0.85 + 0.9) / 2
