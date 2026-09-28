"""
src/shared_target/arms.py

Defines prediction pathways (arms) in SigLIP embedding space:
- Text arms: T_LLM, T_RepeatClosest, T_MeanClosest, T_Oracle, T_RandWithinDomain, T_RandCorpus, T_LLM_gapmean
- Visual arms: V_MeanClosest, V_RepeatClosest, V_Oracle, V_RandFrame
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np


@dataclass
class ArmPrediction:
    vector: np.ndarray  # shape: (D,), L2-normalized
    arm_name: str
    arm_space: str      # 'text' or 'visual'
    parse_ok: bool = True
    misaligned: bool = False
    truncated: bool = False


def normalize_vector(v: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Normalizes vector to unit L2 norm."""
    norm = np.linalg.norm(v)
    if norm <= tol:
        return v
    return v / norm


def get_nearest_boundary_idx(
    t: int,
    t_start: int,
    t_end: int,
    total_seconds: int = 60
) -> int:
    """
    Finds index of nearest boundary: (t_start - 1) or (t_end + 1).
    Handles edge gaps where only one boundary is valid.
    """
    has_left = (t_start > 0)
    has_right = (t_end < total_seconds - 1)
    
    if has_left and has_right:
        left_dist = abs(t - (t_start - 1))
        right_dist = abs(t - (t_end + 1))
        return (t_start - 1) if left_dist <= right_dist else (t_end + 1)
    elif has_left:
        return t_start - 1
    elif has_right:
        return t_end + 1
    else:
        raise ValueError("Both boundaries are invalid!")


def compute_lerp_alpha(t: int, t_start: int, width: int) -> float:
    """
    Interpolation parameter alpha_t:
        alpha_t = (t - t_start + 1) / (width + 1)
    """
    return float(t - t_start + 1) / float(width + 1)


class ArmGenerator:
    """
    Generates unit-norm prediction vectors for all 10 arms at a given target second.
    """
    def __init__(self, embed_text_fn: Callable[[list[str]], list[np.ndarray]]):
        self.embed_text_fn = embed_text_fn
        
    def generate_arms(
        self,
        t: int,
        gap_seconds: Sequence[int],
        gt_captions: list[str],
        gt_frames: np.ndarray,
        llama_captions: dict[int, str] | None = None,
        rand_text_within_domain: str | None = None,
        rand_text_corpus: str | None = None,
        rand_frame_within_domain: np.ndarray | None = None,
    ) -> dict[str, ArmPrediction]:
        """
        Generates predictions for all arms for a specific target second t in gap_seconds.
        """
        t_start = min(gap_seconds)
        t_end = max(gap_seconds)
        width = len(gap_seconds)
        total_sec = len(gt_captions)
        
        has_left = (t_start > 0)
        has_right = (t_end < total_sec - 1)
        is_edge = (not has_left) or (not has_right)
        
        near_b_idx = get_nearest_boundary_idx(t, t_start, t_end, total_sec)
        
        # -------------------------------------------------------------
        # 1. TEXT ARMS
        # -------------------------------------------------------------
        # Gather text to embed in batch
        texts_to_embed = []
        text_keys = []
        
        # Oracle text
        texts_to_embed.append(gt_captions[t])
        text_keys.append("oracle")
        
        # Repeat closest text
        texts_to_embed.append(gt_captions[near_b_idx])
        text_keys.append("repeat")
        
        # Boundary texts for LERP
        if has_left:
            texts_to_embed.append(gt_captions[t_start - 1])
            text_keys.append("left")
        if has_right:
            texts_to_embed.append(gt_captions[t_end + 1])
            text_keys.append("right")
            
        # Llama caption for t
        llama_t_text = None
        if llama_captions and t in llama_captions:
            llama_t_text = llama_captions[t].strip()
            if llama_t_text:
                texts_to_embed.append(llama_t_text)
                text_keys.append("llama")
                
        # Llama all gap texts for gapmean
        llama_gap_texts = []
        if llama_captions:
            for s in gap_seconds:
                if s in llama_captions and llama_captions[s].strip():
                    llama_gap_texts.append(llama_captions[s].strip())
            for idx_g, gt in enumerate(llama_gap_texts):
                texts_to_embed.append(gt)
                text_keys.append(f"llama_gap_{idx_g}")
                
        # Random controls
        if rand_text_within_domain:
            texts_to_embed.append(rand_text_within_domain)
            text_keys.append("rand_domain")
        if rand_text_corpus:
            texts_to_embed.append(rand_text_corpus)
            text_keys.append("rand_corpus")
            
        # Embed all requested texts
        embs = self.embed_text_fn(texts_to_embed)
        emb_map = {k: normalize_vector(v) for k, v in zip(text_keys, embs)}
        
        preds: dict[str, ArmPrediction] = {}
        
        # T_Oracle
        preds["T_Oracle"] = ArmPrediction(
            vector=emb_map["oracle"],
            arm_name="T_Oracle",
            arm_space="text"
        )
        
        # T_RepeatClosest
        preds["T_RepeatClosest"] = ArmPrediction(
            vector=emb_map["repeat"],
            arm_name="T_RepeatClosest",
            arm_space="text"
        )
        
        # T_MeanClosest (LERP)
        if not is_edge and has_left and has_right:
            alpha = compute_lerp_alpha(t, t_start, width)
            lerp_t = (1.0 - alpha) * emb_map["left"] + alpha * emb_map["right"]
            lerp_t = normalize_vector(lerp_t)
        else:
            lerp_t = emb_map["repeat"]
        preds["T_MeanClosest"] = ArmPrediction(
            vector=lerp_t,
            arm_name="T_MeanClosest",
            arm_space="text"
        )
        
        # T_LLM
        if "llama" in emb_map:
            preds["T_LLM"] = ArmPrediction(
                vector=emb_map["llama"],
                arm_name="T_LLM",
                arm_space="text",
                parse_ok=True
            )
        else:
            preds["T_LLM"] = ArmPrediction(
                vector=np.zeros_like(emb_map["oracle"]),
                arm_name="T_LLM",
                arm_space="text",
                parse_ok=False
            )
            
        # T_LLM_gapmean
        gap_vectors = [emb_map[f"llama_gap_{idx_g}"] for idx_g in range(len(llama_gap_texts))]
        if gap_vectors:
            mean_gap = np.mean(gap_vectors, axis=0)
            preds["T_LLM_gapmean"] = ArmPrediction(
                vector=normalize_vector(mean_gap),
                arm_name="T_LLM_gapmean",
                arm_space="text",
                parse_ok=True
            )
        else:
            preds["T_LLM_gapmean"] = ArmPrediction(
                vector=np.zeros_like(emb_map["oracle"]),
                arm_name="T_LLM_gapmean",
                arm_space="text",
                parse_ok=False
            )
            
        # T_RandWithinDomain
        if "rand_domain" in emb_map:
            preds["T_RandWithinDomain"] = ArmPrediction(
                vector=emb_map["rand_domain"],
                arm_name="T_RandWithinDomain",
                arm_space="text"
            )
            
        # T_RandCorpus
        if "rand_corpus" in emb_map:
            preds["T_RandCorpus"] = ArmPrediction(
                vector=emb_map["rand_corpus"],
                arm_name="T_RandCorpus",
                arm_space="text"
            )
            
        # -------------------------------------------------------------
        # 2. VISUAL ARMS
        # -------------------------------------------------------------
        # V_Oracle
        preds["V_Oracle"] = ArmPrediction(
            vector=normalize_vector(gt_frames[t]),
            arm_name="V_Oracle",
            arm_space="visual"
        )
        
        # V_RepeatClosest
        v_rep = normalize_vector(gt_frames[near_b_idx])
        preds["V_RepeatClosest"] = ArmPrediction(
            vector=v_rep,
            arm_name="V_RepeatClosest",
            arm_space="visual"
        )
        
        # V_MeanClosest (LERP)
        if not is_edge and has_left and has_right:
            alpha = compute_lerp_alpha(t, t_start, width)
            v_left = normalize_vector(gt_frames[t_start - 1])
            v_right = normalize_vector(gt_frames[t_end + 1])
            v_lerp = (1.0 - alpha) * v_left + alpha * v_right
            v_lerp = normalize_vector(v_lerp)
        else:
            v_lerp = v_rep
        preds["V_MeanClosest"] = ArmPrediction(
            vector=v_lerp,
            arm_name="V_MeanClosest",
            arm_space="visual"
        )
        
        # V_RandFrame
        if rand_frame_within_domain is not None:
            preds["V_RandFrame"] = ArmPrediction(
                vector=normalize_vector(rand_frame_within_domain),
                arm_name="V_RandFrame",
                arm_space="visual"
            )
            
        return preds
