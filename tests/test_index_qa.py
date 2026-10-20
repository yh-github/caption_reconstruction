"""Unit tests for scripts/index_qa (TempCompass parsing, windows, answer parsing). No data or API needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from index_qa import tempcompass as tc  # noqa: E402
from index_qa.caption import GEMINI_LEAD, frame_clip, frames_prompt, prompt_for  # noqa: E402
from index_qa.reader import (ARMS, CAPTIONERS, CLAUDE_CAPTIONER, HIGH, INDEX_SOURCE, LOW, QSTAR_PAIRS,  # noqa: E402
                             contrast_boot, list_usd, paired_boot, parse, qstar_boot, retention_boot, spearman,
                             thinking_tokens)
from index_qa.self_caption import captioner  # noqa: E402


def test_parse_question_plain():
    q = tc.parse_question("What is the man doing in the video?\nA. dunking a basketball\nB. dribbling a basketball\n"
                          "C. passing a basketball", "A. dunking a basketball")
    assert q["stem"] == "What is the man doing in the video?"
    assert q["options"] == {"A": "dunking a basketball", "B": "dribbling a basketball", "C": "passing a basketball"}
    assert q["answer"] == "A"


def test_parse_question_malformed_options():
    # the dataset's three malformed questions: a missing space after the letter, a leading space
    q = tc.parse_question("Which event occurs first?\nA. dancing\nB. sending an air kiss\n"
                          "C.Both events happen at the same time", "A. dancing")
    assert q["options"]["C"] == "Both events happen at the same time"
    q = tc.parse_question("Spin?\nA. towards the screen\nB. anticlockwise\nC. clockwise\n D. away from the screen",
                          "B. anticlockwise")
    assert set(q["options"]) == set("ABCD") and q["answer"] == "B"


def test_parse_question_rejects_unknown_answer():
    with pytest.raises(ValueError):
        tc.parse_question("Q?\nA. x\nB. y", "C. z")


def test_windows_and_frame_times():
    assert tc.windows(10.0) == [(0, 5), (5, 10)]
    assert tc.windows(10.02) == [(0, 5), (5, 10)]  # a last window under 0.5 s is dropped
    assert tc.windows(10.6) == [(0, 5), (5, 10), (10, 10.6)]
    assert tc.windows(7.5) == [(0, 5), (5, 7.5)]
    assert tc.windows(1.5) == [(0, 1.5)]
    assert tc.frame_times(0, 5) == [0, 1, 2, 3, 4]
    assert tc.frame_times(5, 7.5) == [5, 6, 7]
    assert tc.frame_times(0, 1.5) == [0, 1]


def test_middle_frame():
    assert tc.middle(list("01234")) == "2"
    assert tc.middle(list("567")) == "6"
    assert tc.middle(list("01")) == "1"


def test_first_per_dim():
    qs = [dict(dim=d, i=i) for i, d in enumerate("aabab" "bba")]
    assert [q["i"] for q in tc.first_per_dim(qs, 2)] == [0, 1, 2, 4]
    assert tc.first_per_dim(qs, None) == qs


def test_prompt_for_short_window():
    base = "Describe what happens in the attached 5-second video clip."
    assert prompt_for(base, 5.0) == base
    assert prompt_for(base, 2.5) == "Describe what happens in the attached 2-second video clip."  # round half to even
    assert prompt_for(base, 1.2) == "Describe what happens in the attached 1-second video clip."


def test_parse_answer():
    assert parse('{"answer": "B"}', ["A", "B", "C"]) == "B"
    assert parse('thinking... {"answer": "a"} then {"answer": "C."}', ["A", "B", "C"]) == "C"
    assert parse('{"answer": "D"}', ["A", "B", "C"]) is None
    assert parse("B", ["A", "B", "C"]) is None


def test_paired_boot():
    est, lo, hi = paired_boot([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert est == pytest.approx(0.2) and lo <= est <= hi


def test_paired_boot_level():
    d = [1, 0, -1, 1, 0, 0, 1, 0, 0, -1] * 3
    _, lo95, hi95 = paired_boot(d)
    _, lo98, hi98 = paired_boot(d, level=1 - 0.05 / 3)
    assert lo98 <= lo95 and hi98 >= hi95


def test_contrast_boot():
    est, lo, hi = contrast_boot([-1, 0, -1, 0], [0, 0, 1, 0])
    assert est == pytest.approx(-0.75) and lo <= est <= hi


def test_retention_boot():
    est, lo, hi = retention_boot([1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0])
    assert est == pytest.approx(2 / 3) and lo <= est <= hi


def test_aspect_groups_partition_the_aspects():
    assert LOW | HIGH == set(tc.DIMS) and not LOW & HIGH


def test_claude_captioner_folders_match_the_reader_arms():
    assert captioner("haiku") == "claude-haiku-5-5-self"  # K2's folder name, kept
    assert captioner("sonnet") == "claude-sonnet-5-5-frames"
    for arm, model in CLAUDE_CAPTIONER.items():
        assert arm in ARMS and CAPTIONERS[arm] == captioner(model)


def test_thinking_tokens():
    assert thinking_tokens({"usage": {"output_tokens_details": {"thinking_tokens": 42}}}) == 42
    assert thinking_tokens({"usage": {"output_tokens": 11}}) == 0
    assert thinking_tokens({"usage": None}) == 0


def test_list_usd_matches_a_logged_cli_cost():
    # a logged Haiku question-only call: 1032 input and 11 output tokens, costUSD 0.0001087
    assert list_usd("haiku", {"input_tokens": 1032, "output_tokens": 11}) == pytest.approx(0.0001087)
    assert list_usd("sonnet", None) == 0


def test_qstar_boot():
    est, lo, hi, never = qstar_boot([1.0] * 4, [0.0] * 4, [0.5] * 4)
    assert (est, lo, hi, never) == pytest.approx((2.0, 2.0, 2.0, 0.0))
    est, _, hi, never = qstar_boot([1.0] * 4, [0.5] * 4, [0.5] * 4)  # no saving per question: never pays
    assert est == float("inf") and hi == float("inf") and never == 1


def test_every_qstar_caption_arm_has_an_index_cost():
    assert all(a.split(":")[1] in INDEX_SOURCE and b.split(":")[1] not in INDEX_SOURCE for a, b in QSTAR_PAIRS)


def test_spearman():
    import pandas as pd
    assert spearman(pd.Series([1, 2, 3, 4]), pd.Series([10, 20, 30, 1000])) == pytest.approx(1.0)
    assert spearman(pd.Series([1, 2, 3]), pd.Series([3, 2, 1])) == pytest.approx(-1.0)


def test_frames_prompt_changes_only_the_lead():
    p = frames_prompt(3, 2.6)
    assert GEMINI_LEAD not in p and p.startswith("The attached images are 3 frames from a 3-second video clip")
    assert p.split("Describe what happens in the clip,", 1)[1] == \
        prompt_for(tc.ROOT.joinpath("prompts/yt_video_processing/piece_isolated.txt").read_text(), 5).split(
            GEMINI_LEAD, 1)[1]


def test_frame_clip_plays_each_frame_for_one_second(tmp_path):
    import cv2
    import numpy as np
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    frames = []
    for i, c in enumerate(colors):
        f = tmp_path / f"{i}.jpg"
        cv2.imwrite(str(f), np.full((126, 224, 3), c, np.uint8))
        frames.append(f)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(frame_clip(frames))
    cap = cv2.VideoCapture(str(clip))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(1.0)
    decoded = []
    while (got := cap.read())[0]:
        decoded.append(got[1])
    assert len(decoded) == 3 and decoded[0].shape == (126, 224, 3)
    for d, c in zip(decoded, colors):
        assert np.abs(d.mean((0, 1)) - c).max() < 12
