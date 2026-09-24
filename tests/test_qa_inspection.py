from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Ensure both src and scripts are discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data.wildqa_loader import WildQAPair
from prepare_qa_inspection_dataset import (
    build_inspection_item,
    check_lexical_leakage,
    export_html_report,
    export_markdown_summary,
    extract_top_k_candidates,
)


def test_check_lexical_leakage():
    answers = ["The fire is used for heat.", "heat and warmth"]
    # Positive case: exact substring or significant word 'heat' appears
    context_with_leak = [
        "The fire crackles in the cold night.",
        "The flames give off substantial heat in the camp.",
    ]
    assert check_lexical_leakage(context_with_leak, answers) is True

    # Negative case: unrelated context
    context_no_leak = [
        "A bird flies across the blue sky.",
        "The camera tilts up to the trees.",
    ]
    assert check_lexical_leakage(context_no_leak, answers) is False

    # Stopwords only should not trigger false positives
    assert check_lexical_leakage(["and with the from that this"], ["with that"]) is False


def test_extract_top_k_candidates():
    # 5 frames, dim=3
    # Frame 1 is closest to query
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    index = np.array([
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],  # Frame 1: highest sim
        [0.5, 0.5, 0.0],  # Frame 2: second highest
        [0.0, 0.0, 1.0],
        [0.1, 0.9, 0.0],
    ], dtype=np.float32)

    captions = [f"Caption at {i}" for i in range(5)]
    target_indices = {1}  # Target is frame 1

    top_3 = extract_top_k_candidates(index, query, captions, target_indices, top_k=3)

    assert len(top_3) == 3
    assert top_3[0]["rank"] == 1
    assert top_3[0]["second"] == 1
    assert top_3[0]["is_in_evidence"] is True
    assert top_3[1]["rank"] == 2
    assert top_3[1]["second"] == 2
    assert top_3[1]["is_in_evidence"] is False


def test_build_inspection_item():
    qa = WildQAPair(
        assignment_id="test_qa_1",
        video_id="video_test_1",
        domain="Agriculture",
        question_type=["Reasoning"],
        question="What tool is the farmer using?",
        answer="A shovel",
        alter_answers=["A small spade"],
        evidences=[{"0": [10.0, 12.5]}],  # indices 10, 11, 12
    )

    # 60 dummy captions
    captions = [f"Second {i} caption" for i in range(60)]
    captions[9] = "Before gap: farmer walks to garden."
    captions[13] = "After gap: farmer plants seeds."

    recon_dict = {
        10: "Recon second 10",
        11: "Recon second 11",
        12: "Recon second 12",
    }

    item = build_inspection_item(
        qa=qa,
        split="dev",
        original_captions=captions,
        recon_captions=recon_dict,
        embedder=None,
        boundary_window=2,
    )

    assert item["video_id"] == "video_test_1"
    assert item["evidence_span"] == "10-12"
    assert item["evidence_duration_sec"] == 3
    assert item["recon_available"] is True
    assert len(item["boundary_context"]["pre_gap"]) == 2  # indices 8, 9
    assert len(item["boundary_context"]["post_gap"]) == 2  # indices 13, 14
    assert len(item["gap_conditions"]["oracle"]) == 3
    assert len(item["gap_conditions"]["reconstructed"]) == 3
    assert len(item["gap_conditions"]["baseline_repeat"]) == 3
    # Baseline repeat should use left boundary index 9
    assert item["gap_conditions"]["baseline_repeat"][0]["caption"] == captions[9]


def test_export_html_and_markdown():
    qa = WildQAPair(
        assignment_id="test_qa_1",
        video_id="video_test_1",
        domain="Agriculture",
        question_type=["Reasoning"],
        question="What tool is the farmer using?",
        answer="A shovel",
        alter_answers=[],
        evidences=[{"0": [10.0, 11.0]}],
    )
    captions = [f"Caption {i}" for i in range(60)]
    item = build_inspection_item(
        qa=qa,
        split="dev",
        original_captions=captions,
        recon_captions=None,
        embedder=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        html_out = tmp_path / "test.html"
        md_out = tmp_path / "test.md"

        export_html_report([item], html_out, embedder_name="none")
        export_markdown_summary([item], md_out)

        assert html_out.exists()
        assert md_out.exists()
        assert "What tool is the farmer using?" in html_out.read_text(encoding="utf-8")
        assert "What tool is the farmer using?" in md_out.read_text(encoding="utf-8")
