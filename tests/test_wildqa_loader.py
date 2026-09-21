from __future__ import annotations
import json
import pytest
from pathlib import Path
from data.wildqa_loader import WildQAPair, load_wildqa_dataset


def test_wildqa_pair_properties():
    item = {
        "video_id": "test_vid_1",
        "domain": "Agriculture",
        "question_type": ["Reasoning"],
        "question_base": ["scene"],
        "question": "What is the farmer doing?",
        "answer": "Plowing the field.",
        "alter_answers": ["He is plowing.", "Plowing ground."],
        "evidences": [{"0": [12.5, 17.0]}],
        "duration": 60.0
    }
    qa = WildQAPair.model_validate(item)
    assert qa.is_scene_grounded is True
    assert qa.is_single_evidence is True
    assert qa.primary_evidence == (12.5, 17.0)
    assert qa.evidence_intervals == [(12.5, 17.0)]
    assert qa.all_reference_answers == ["Plowing the field.", "He is plowing.", "Plowing ground."]
    indices = qa.evidence_indices(max_duration=60)
    assert indices == {12, 13, 14, 15, 16}


def test_wildqa_pair_audio_only():
    item = {
        "video_id": "test_vid_2",
        "domain": "Military",
        "question_type": ["Entity"],
        "question_base": ["audio"],
        "question": "What sound is heard?",
        "answer": "Helicopter rotor.",
        "evidences": [{"0": [5.0, 10.0]}],
    }
    qa = WildQAPair.model_validate(item)
    assert qa.is_scene_grounded is False
    assert qa.evidence_intervals == [(5.0, 10.0)]


def test_wildqa_multi_evidence():
    item = {
        "video_id": "test_vid_3",
        "domain": "Human Survival",
        "question_type": ["Motion"],
        "question_base": ["scene", "audio"],
        "question": "Where does he walk?",
        "answer": "Through the woods.",
        "evidences": [{"0": [1.0, 5.0]}, {"1": [20.0, 25.0]}],
    }
    qa = WildQAPair.model_validate(item)
    assert qa.is_scene_grounded is True
    assert qa.is_single_evidence is False
    assert len(qa.evidence_intervals) == 2
    assert qa.evidence_indices(max_duration=30) == {1, 2, 3, 4, 20, 21, 22, 23, 24}


def test_load_wildqa_dev_file():
    dev_path = Path("datasets/wildQA/dev.json")
    if not dev_path.exists():
        pytest.skip("datasets/wildQA/dev.json does not exist")

    all_scene = load_wildqa_dataset(dev_path, filter_scene_only=True, max_duration=60.0)
    assert len(all_scene) > 0

    single_clean = load_wildqa_dataset(
        dev_path,
        filter_scene_only=True,
        max_duration=60.0,
        single_evidence_only=True,
        max_evidence_duration=15.0
    )
    assert len(single_clean) > 0
    assert len(single_clean) <= len(all_scene)
    for q in single_clean:
        assert q.is_single_evidence
        assert q.is_scene_grounded
        start, end = q.primary_evidence
        assert end <= 60.0
        assert end - start <= 15.0
