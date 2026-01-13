# Generative Reconstruction of Dense Video Captions: Beyond Vector Interpolation

## Abstract
Dense video captioning requires understanding the temporal evolution of events. We investigate the task of **caption reconstruction**: recovering a missing caption in a dense sequence given its surrounding context. We compare a generative Large Language Model (Phi-3) against a strong vector-based interpolation baseline. Our experiments on the Wild-Dev dataset reveal that while vector methods provide a robust temporal baseline, generative models significantly outperform them in reconstructing short-to-medium duration gaps (3-9s), particularly in specialized semantic domains (e.g., Military, Survival). Furthermore, we introduce **temporal retrieval metrics** (Windowed Recall and Temporal NDCG), demonstrating that even when generative models fail exact reconstruction, they maintain high temporal coherence, hallucinating plausible events that align with the narrative flow.

## 1. Introduction
Video understanding models often rely on dense captions to index and retrieve content. However, these captions can be noisy, sparse, or missing. The ability to *reconstruct* a missing caption from its temporal context is a proxy for a model's understanding of narrative causal structure. 

Existing approaches largely rely on embedding space interpolation—assuming that the "meaning" of a missing segment is the average of its neighbors. While effective for slow-moving semantic drifts, this fails to capture distinct, discrete events (e.g., a specific action like "lighting a fire" between "gathering wood" and "cooking"). 

In this work, we propose a generative approach using **Phi-3**, a lightweight LLM, to Hallucinate the missing caption based strictly on textual context. We conduct a rigorous comparative analysis against a vector-space baseline across varying gap sizes (mask widths). We find that the generative approach offers superior precision for distinct events and maintains high semantic fidelity, whereas vector interpolation smooths over critical details.

## 2. Methodology

### 2.1 Task Definition
Given a sequence of dense captions $C = \{c_1, c_2, ..., c_T\}$ ordered by time, we mask a contiguous subsequence of width $W$ starting at index $i$: $M = \{c_i, ..., c_{i+W-1}\}$. The task is to reconstruct each $c_j \in M$ given the visible context $C \setminus M$.

### 2.2 Models
*   **Generative Approach (Phi-3)**: We prompt Phi-3 with the *pre-mask* ($c_{i-K}...c_{i-1}$) and *post-mask* ($c_{i+W}...c_{i+W+K}$) context. The model generates the missing text directly. We retrieve the closest ground-truth caption from the video's pool using embedding similarity to the generated text, enabling standard retrieval metrics.
*   **Vector Baseline (MeanClosest)**: A strong non-generative baseline. For any missing index $j$, we compute the mean of the nearest available past and future embeddings: $v_j = \text{mean}(v_{known\_prev}, v_{known\_next})$. This represents the "smooth transition" hypothesis.

### 2.3 Metrics
We evaluate precision and temporal coherence:
*   **Exact Retrieval**: MRR (Mean Reciprocal Rank) and R@1 (Recall at 1) against the exact ground truth index.
*   **Temporal Metrics**:
    *   **Windowed R@1 (W=k)**: Success if the retrieved caption is within $k$ steps of the true index. This accounts for semantic redundancy in dense video benchmarks.
    *   **Temporal NDCG**: A distance-weighted metric where relevance decays as $1/(1 + |i_{pred} - i_{true}|)$, rewarding retrieval of temporally adjacent events.

## 3. Experiments & Results

We evaluate on the **Wild-Dev** dataset, comprising diverse "in-the-wild" video sequences. We iterate mask widths $W \in \{3, ..., 30\}$ frames across varying start positions.

### 3.1 Generative Precision vs. Gap Size
As shown in **Figure 1** (see `mrr_vs_width`), the generative model (Phi-3) achieves significantly higher MRR for small gaps ($W \le 6$), peaking at >0.61 MRR compared to the baseline's best case.
*   **Short Gaps**: The LLM effectively infers discrete missing actions (e.g., "loading the gun") from immediate context.
*   **Degradation**: Performance degrades near-linearly as $W$ increases. By $W=12$, the generative advantage narrows as the hallucination search space becomes too large.

### 3.2 Semantic Wins
Category-wise analysis reveals that Phi-3 dominates in specialized domains such as **Military** and **Survival** (+0.35 MRR Delta). In these domains, structured procedural knowledge (e.g., steps to build a shelter) allows the LLM to predict missing steps accurately, whereas vector interpolation merely blurs the distinct actions.

### 3.3 Temporal Coherence
A key finding is the robustness of **Temporal NDCG**. Even when exact R@1 drops at large widths ($W>15$), the Temporal NDCG for Phi-3 remains high (~0.80), comparable to the interpolation baseline.
*   **Interpretation**: When the LLM fails to guess the *exact* caption, it typically hallucinates an event that is semantically compatible and temporally adjacent (a "near miss").
*   **Baseline Competitiveness**: The vector baseline performs surprisingly well on temporal metrics because averaging naturally lands in the "middle" of the semantic space, ensuring retrieval of mid-segment captions. However, it lacks the precise event resolution of the generative model for short gaps.

### 3.4 Positional Bias
We observe a **Start Bias**: reconstruction accuracy is consistently higher when masking starts at the beginning of the context window ($i=0$) rather than the middle or end. This suggests that "forward prediction" from a known start is more robust than "in-filling" or "backward, inference".

## 4. Conclusion
We demonstrate that generative LLMs are strong candidates for dense video caption imputation, offering superior fine-grained event reconstruction compared to vector interpolation. Future work will explore larger context windows and multimodal integration.
