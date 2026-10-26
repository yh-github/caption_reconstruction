"""Unit tests for scripts/index_qa (TempCompass parsing, windows, answer parsing). No data or API needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from index_qa import tempcompass as tc  # noqa: E402
from index_qa.caption import GEMINI_LEAD, frame_clip, frames_prompt, prompt_for  # noqa: E402
from index_qa import reader  # noqa: E402
from index_qa.reader import (ARMS, CAPTIONER_LONG_EDGE, CAPTIONERS, CLAUDE_CAPTIONER, FRAMES_ALL, HIGH, INDEX_SOURCE, LOW,  # noqa: E402
                             PROMPTS, QSTAR_PAIRS, clip_clusters, cluster_boot, contrast_boot, list_usd, paired_boot,
                             parse, qstar_boot, retention_boot, spearman, thinking_tokens)
from index_qa.caption import (ASPECT_BULLET, LAST_BULLET, LENGTH_CLAUSE, end_line, frames_prompt,  # noqa: E402
                              whole_prompt, with_aspects)
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


def test_retention_boot_over_clusters():
    cap, frames, blind = [1, 1, 0, 0, 1, 0] * 5, [1, 1, 1, 0, 1, 1] * 5, [0, 1, 0, 0, 0, 0] * 5
    # one question per cluster (labels that sort in order): the same draws as resampling questions
    assert retention_boot(cap, frames, blind, clusters=[f"{i:02d}" for i in range(30)]) == pytest.approx(
        retention_boot(cap, frames, blind))
    est, lo, hi = retention_boot(cap, frames, blind, clusters=[i // 6 for i in range(30)])
    assert est == pytest.approx(2 / 4) and lo <= est <= hi


def test_cluster_boot():
    est, lo, hi = cluster_boot([1, 1, 0, 0, -1, 0], ["a", "a", "b", "b", "c", "c"])
    assert est == pytest.approx(1 / 6) and lo <= est <= hi
    # one question per cluster: the same percentile CI as resampling questions
    d = [1, 0, -1, 1, 0, 0, 1, 0, 0, -1] * 3
    assert cluster_boot(d, list(range(len(d)))) == pytest.approx(paired_boot(d))


def test_cluster_boot_wider_when_clusters_are_alike():
    d = [1] * 10 + [0] * 10
    _, lo, hi = paired_boot(d)
    _, clo, chi = cluster_boot(d, [0] * 10 + [1] * 10)  # two clusters, each all alike
    assert chi - clo > hi - lo


def test_video_type_and_source_clips():
    assert [tc.video_type(v) for v in ["1034419625", "1034419625_reverse", "1059718130_concat_0",
                                       "1059072773_1059718130_0"]] == ["plain", "reversed", "stacked", "two_clip"]
    assert set(tc.VIDEO_TYPES) == {"plain", "reversed", "stacked", "two_clip"}
    assert tc.source_clips("1034419625_reverse") == ["1034419625"]
    assert tc.source_clips("1059718130_concat_2") == ["1059718130"]
    assert tc.source_clips("1059072773_1059718130_0") == ["1059072773", "1059718130"]
    with pytest.raises(ValueError):
        tc.video_type("abc_reverse")


def test_clip_clusters_join_videos_through_shared_clips():
    v = ["1", "1_reverse", "2_1_0", "2_concat_1", "3", "4_5_1"]
    c = clip_clusters(v)
    assert c[0] == c[1] == c[2] == c[3]  # 1 and 2 are linked by the two-clip video 2_1_0
    assert len({c[0], c[4], c[5]}) == 3


def test_aspect_groups_partition_the_aspects():
    assert LOW | HIGH == set(tc.DIMS) and not LOW & HIGH


def test_claude_captioner_folders_match_the_reader_arms():
    assert captioner("haiku") == "claude-haiku-5-5-self"  # K2's folder name, kept
    assert captioner("sonnet") == "claude-sonnet-5-5-frames"
    assert captioner("sonnet", 448) == "claude-sonnet-5-5-frames448"
    assert captioner("sonnet", whole=True) == "claude-sonnet-5-5-frames-whole"
    assert captioner("sonnet", whole=True, aspects=True) == "claude-sonnet-5-5-frames-aspects-whole"
    for arm, model in CLAUDE_CAPTIONER.items():
        assert arm in ARMS and CAPTIONERS[arm] == captioner(model, CAPTIONER_LONG_EDGE.get(arm, tc.LONG_EDGE),
                                                            arm in ("sonnetwhole", "sonnetaspects"),
                                                            arm == "sonnetaspects")
        assert arm in INDEX_SOURCE


def test_with_aspects_adds_only_the_bullet():
    base = whole_prompt(12, 12.0, 3)
    p = with_aspects(base)
    assert ASPECT_BULLET in p and p.replace(ASPECT_BULLET + "\n", "") == base
    assert p.index(ASPECT_BULLET) < p.index(LAST_BULLET)
    assert "revers" not in ASPECT_BULLET.lower() and "stack" not in ASPECT_BULLET.lower()  # no benchmark tricks


def test_latest_skips_a_partial_row(tmp_path, monkeypatch):
    monkeypatch.setattr(reader, "OUT", tmp_path)
    (tmp_path / "haiku.jsonl").write_text('{"item": "a", "status": "ok"}\n{"item": "b", "sta')
    assert list(reader.latest("haiku")) == ["a"]
    end_line(tmp_path / "haiku.jsonl")
    assert (tmp_path / "haiku.jsonl").read_text().endswith('"sta\n')


def test_whole_prompt_scales_only_the_length_limit():
    assert whole_prompt(5, 5.0, 1) == frames_prompt(5, 5.0)
    p, base = whole_prompt(12, 12.0, 3), frames_prompt(12, 12.0)
    assert "in at most 6 sentences of at most 135 words in total" in p
    assert p.replace("in at most 6 sentences of at most 135 words in total", LENGTH_CLAUSE) == base


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


def test_siblings():
    ids = ["1", "1_reverse", "2", "1_concat_0", "1_concat_1", "3_concat_0", "1_2_0", "2_1_1", "1_3_0"]
    assert tc.siblings("1", ids) == ["1_reverse"]
    assert tc.siblings("1_reverse", ids) == ["1"]
    assert tc.siblings("2", ids) == []
    assert tc.siblings("1_concat_0", ids) == ["1_concat_1"]
    assert tc.siblings("1_2_0", ids) == ["2_1_1"]


def test_sample_pairs_keeps_the_confirmatory_wording(monkeypatch):
    conf = dict(item="tcc000|1_reverse|direction", video_id="1_reverse", dim="direction", stem="Which way?",
                options={"A": "left", "B": "right"}, answer="A")
    sib_same = dict(video_id="1", dim="direction", stem="Which way?", options={"A": "left", "B": "right"}, answer="A")
    sib = dict(video_id="1", dim="direction", stem="Which way?", options={"A": "right", "B": "left"}, answer="A")
    other = dict(video_id="1", dim="speed", stem="How fast?", options={"A": "fast", "B": "slow"}, answer="B")
    monkeypatch.setattr(tc, "load_questions", lambda: [conf, sib_same, sib, other])
    monkeypatch.setattr(tc, "items", lambda set_: [conf])
    out = tc.sample_pairs(lambda v: True)
    assert len(out) == 1
    p = out[0]
    assert p["video_id"] == "1" and p["pair_of"] == conf["item"] and p["options"] == conf["options"]
    assert p["answer"] == "B"  # "right", the sibling's correct option, in the confirmatory option order
    assert p["item"] == "tcp000|1|direction"
    assert tc.sample_pairs(lambda v: False) == []  # siblings without Gemini captions are left out


def test_frame_size_arms():
    assert tc.frame_path("v", 1, 2) == tc.FRAMES / "v/01_2.jpg"
    assert tc.frame_path("v", 1, 2, 448) == tc.TC / "frames_448/v/01_2.jpg"
    assert FRAMES_ALL["frames5"][0] == tc.LONG_EDGE and FRAMES_ALL["frames5_448"][0] == 448
    for arm, (_, template) in FRAMES_ALL.items():
        assert arm in ARMS and (PROMPTS / template).exists()
    base, desc = ((PROMPTS / t).read_text().split("Question:")[0] for t in ("mcq.txt", "mcq_describe.txt"))
    assert base == desc  # describe5 changes only the answer instruction


def test_parse_describe_then_answer():
    text = '{"windows": ["a man walks left", "he stops"], "answer": "B"}'
    assert parse(text, ["A", "B", "C"]) == "B"


def test_pair_frame(monkeypatch):
    pairs = [dict(item="tcp000|1|direction", pair_of="tcc000|1_reverse|direction", pair_video="1_reverse",
                  options={"A": "l", "B": "r"}),
             dict(item="tcp001|2_concat_1|speed", pair_of="tcc001|2_concat_0|speed", pair_video="2_concat_0",
                  options={"A": "x", "B": "y", "C": "z"})]
    monkeypatch.setattr(tc, "items", lambda set_: pairs)

    def row(q, arm, letter, correct):
        return dict(question_item=q, arm=arm, model="haiku", letter=letter, correct=correct)
    rows = {"haiku": {i: r for i, r in enumerate([
        row("tcc000|1_reverse|direction", "frames5", "A", True), row("tcp000|1|direction", "frames5", "A", False),
        row("tcc001|2_concat_0|speed", "frames5", "B", True), row("tcp001|2_concat_1|speed", "frames5", "C", True)])},
        "sonnet": {}}
    monkeypatch.setattr(reader, "latest", lambda m: rows[m])
    both, same, orig, sib, group, clus, chance = reader.pair_frame()
    assert both["haiku:frames5"].tolist() == [0.0, 1.0]
    assert same["haiku:frames5"].tolist() == [1.0, 0.0]
    assert group.tolist() == ["reversal", "stacked"]
    assert chance.tolist() == [0.25, 1 / 9]


def test_audit_parse_label_and_kappa():
    from index_qa.audit import cohen_kappa, parse_label
    assert parse_label('{"label": "Silent", "evidence": ""}') == ("silent", "")
    assert parse_label('x {"label": "maybe"} {"label": "wrong", "evidence": "pans right"}') == ("wrong", "pans right")
    assert parse_label("no json") == (None, "")
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-12


def test_qwen_render_logs_like_build_and_fills_caption_slots():
    from index_qa.open_reader import SLOT
    from index_qa.qwen_kaggle import render
    blocks = [{"type": "text", "text": "head"}, {"type": "text", "text": "0-5 s:"},
              {"type": "image", "path": "local/f/00_0.jpg"}, {"type": "text", "text": "tail"}]
    parts, frames, log = render(blocks, None)
    assert log == "head\n0-5 s:\n[image local/f/00_0.jpg]\ntail"  # reader.build's log format
    assert frames == ["local/f/00_0.jpg"]
    assert [p.get("text") for p in parts] == ["head\n", "0-5 s:\n", None, "tail"]  # a newline after each text block
    text = f"intro\n\n0-5 s: {SLOT.format(p=0)}\n5-10 s: {SLOT.format(p=1)}\n\nQ"
    parts, frames, log = render([{"type": "text", "text": text}], {0: "a dog runs", 1: "it stops"})
    assert log == "intro\n\n0-5 s: a dog runs\n5-10 s: it stops\n\nQ" and parts[0]["text"] == log and not frames


def test_open_reader_is_c12_and_reads_own_folder():
    assert "qwencaptions" in ARMS and "qwencaptions" in CAPTIONERS and "qwencaptions" not in CLAUDE_CAPTIONER
    assert "qwencaptions" not in INDEX_SOURCE  # captioned on a free GPU: no list price
    assert reader.rows_path("qwen").parent.name == "open" and reader.rows_path("haiku").parent.name == "claude"
    assert all(a.split(":")[0] == "qwen" for a in reader.C12_ARMS)
    assert {t[1] for t in reader.C12_TERMS} | {t[2] for t in reader.C12_TERMS} <= set(reader.C12_ARMS) | {
        "haiku:sonnetcaptions", "haiku:blind"}
