from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Iterator
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WildQAPair(BaseModel):
    """
    Represents a single Question-Answer pair from the WildQA benchmark (Castro et al., COLING 2022).
    Includes ground-truth evidence interval annotations, domain tags, and multi-reference answers.
    """
    assignment_id: str | None = None
    video_id: str
    domain: str
    question_type: list[str] = Field(default_factory=list)
    question_base: list[str] = Field(default_factory=list)
    question: str
    answer: str
    alter_answers: list[str] = Field(default_factory=list)
    evidences: list[dict[str, list[float]]] = Field(default_factory=list)
    alter_evidences: list[list[list[float]]] = Field(default_factory=list)
    duration: float | None = None
    confidence: str | None = None
    objective: str | None = None

    @property
    def all_reference_answers(self) -> list[str]:
        """Returns the primary answer plus all human alternate answers."""
        refs = [self.answer]
        for alt in self.alter_answers:
            if alt and alt.strip() and alt not in refs:
                refs.append(alt)
        return refs

    @property
    def evidence_intervals(self) -> list[tuple[float, float]]:
        """
        Extracts all primary evidence intervals as a list of (start_sec, end_sec) tuples.
        """
        intervals: list[tuple[float, float]] = []
        for ev_dict in self.evidences:
            for k, v in ev_dict.items():
                if len(v) >= 2:
                    intervals.append((float(v[0]), float(v[1])))
        return intervals

    @property
    def primary_evidence(self) -> tuple[float, float] | None:
        """Returns the first primary evidence interval, or None if no evidence is annotated."""
        intervals = self.evidence_intervals
        return intervals[0] if intervals else None

    @property
    def is_single_evidence(self) -> bool:
        """True if the question has exactly one evidence interval."""
        return len(self.evidence_intervals) == 1

    @property
    def is_scene_grounded(self) -> bool:
        """True if the question is based on visual/scene evidence (not audio-only)."""
        return "scene" in self.question_base

    def evidence_indices(self, max_duration: int = 60) -> set[int]:
        """
        Converts primary evidence interval(s) into second-level integer clip indices
        [start_idx, end_idx) bounded by max_duration.
        """
        indices: set[int] = set()
        for start, end in self.evidence_intervals:
            s_idx = max(0, int(start))
            e_idx = min(max_duration, int(end) + 1 if end > int(end) else int(end))
            for i in range(s_idx, e_idx):
                if 0 <= i < max_duration:
                    indices.add(i)
        return indices


def load_wildqa_dataset(
    path: Path | str,
    filter_scene_only: bool = True,
    max_duration: float | None = 60.0,
    single_evidence_only: bool = False,
    max_evidence_duration: float | None = None
) -> list[WildQAPair]:
    """
    Loads and filters WildQA Question-Answer pairs from dev.json or test.json.

    Args:
        path: Path to the WildQA JSON file (dev.json or test.json).
        filter_scene_only: If True, excludes questions that do not have 'scene' in question_base.
        max_duration: If set, ensures all evidence intervals finish within this duration (seconds).
        single_evidence_only: If True, only returns questions with exactly one evidence interval.
        max_evidence_duration: If set, excludes evidence intervals longer than this duration (seconds).

    Returns:
        List of filtered WildQAPair objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WildQA dataset file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: list[WildQAPair] = []
    for item in data:
        qa = WildQAPair.model_validate(item)

        if filter_scene_only and not qa.is_scene_grounded:
            continue

        intervals = qa.evidence_intervals
        if not intervals:
            continue

        if single_evidence_only and len(intervals) != 1:
            continue

        if max_duration is not None:
            if any(end > max_duration for _, end in intervals):
                continue

        if max_evidence_duration is not None:
            if any((end - start) > max_evidence_duration for start, end in intervals):
                continue

        results.append(qa)

    logger.info(
        f"Loaded {len(results)}/{len(data)} WildQA pairs from {path.name} "
        f"(scene_only={filter_scene_only}, max_dur={max_duration}, single_only={single_evidence_only})"
    )
    return results
