> [!CAUTION]
> **Historical Artifact (Superseded)**: This document represents an exploratory early draft (evaluating Phi-3 Mini against 384-dimensional video embeddings and raw retrieval MRR). It has been superseded by the unified SigLIP benchmark, Llama 3.1 8B evaluation, and the Question → Hypotheses → Tests → Answers manuscript in [`docs/paper/draft.md`](draft.md).

# Generative Reconstruction of Dense Video Captions: Beyond Vector Interpolation (Legacy Draft)

## Abstract
Dense video captioning requires understanding the temporal evolution of events. We investigate the task of **caption reconstruction**: recovering a missing caption in a dense sequence given its surrounding context. We compare a generative Large Language Model (Phi-3) against a video embedding vector-based interpolation baseline. Our experiments on the densely captioned Wild-Dev dataset reveal that while vector methods provide a robust temporal baseline, generative models significantly outperform them in reconstructing short-to-medium duration gaps (3-9s), particularly in specialized semantic domains (e.g., Military, Survival). Furthermore, we introduce **temporal retrieval metrics** (Windowed Recall and Temporal NDCG), demonstrating that even when generative models fail exact reconstruction, they maintain high temporal coherence, hallucinating plausible events that align with the narrative flow.

## 1. Introduction
Video understanding models often rely on dense captions to index and retrieve content. However, these captions can be noisy, sparse, or missing. The ability to *reconstruct* a missing caption from its temporal context is a proxy for a model's understanding of narrative causal structure. 

Existing approaches largely rely on embedding space interpolation—assuming that the "meaning" of a missing segment is the average of its neighbors. While effective for slow-moving semantic drifts, this fails to capture distinct, discrete events (e.g., a specific action like "lighting a fire" between "gathering wood" and "cooking"). 

In this work, we propose a generative approach using **Phi-3**, a lightweight LLM, to hallucinate the missing caption based strictly on textual context. We conduct a comparative analysis against a vector-space baseline across varying gap sizes (mask widths). We find that the generative approach offers superior precision for distinct events and maintains high semantic fidelity, whereas vector interpolation smooths over critical details.

## 2. Methodology

### 2.1 Task Definition
Given a sequence of dense captions \(C = \{c_1, c_2, \dots, c_T\}\) ordered by time, we mask a contiguous subsequence of width \(W\) starting at index \(i\): \(M = \{c_i, \dots, c_{i+W-1}\}\). The task is to reconstruct each \(c_j \in M\) given the visible context \(C \setminus M\).

### 2.2 Models
* **Generative Approach (Phi-3)**: We prompt Phi-3 with the *pre-mask* (\(c_{i-K} \dots c_{i-1}\)) and *post-mask* (\(c_{i+W} \dots c_{i+W+K}\)) context. The model generates the missing text directly. We retrieve the closest ground-truth caption from the video's pool using embedding similarity to the generated text, enabling standard retrieval metrics.
* **Vector Baseline (MeanClosest)**: A non-generative baseline. For any missing index \(j\), we compute the mean of the nearest available past and future embeddings: \(v_j = \text{mean}(v_{\text{known\_prev}}, v_{\text{known\_next}})\). This represents the "smooth transition" hypothesis.

### 2.3 Metrics
We evaluate precision and temporal coherence:
* **Exact Retrieval**: MRR (Mean Reciprocal Rank) and R@1 (Recall at 1) against the exact ground truth index.
* **Temporal Metrics**:
  * **Windowed R@1 (\(W=k\))**: Success if the retrieved caption is within \(k\) steps of the true index.
  * **Temporal NDCG**: A distance-weighted metric where relevance decays as \(1/(1 + |i_{\text{pred}} - i_{\text{true}}|)\), rewarding retrieval of temporally adjacent events.

## 3. Experiments & Results (Historical)

We evaluate on the **Wild-Dev** dataset, comprising diverse "in-the-wild" video sequences. We iterate mask widths \(W \in \{3, \dots, 30\}\) frames across varying start positions.

### 3.1 Generative Precision vs. Gap Size
As observed in early runs, the generative model (Phi-3) achieves higher MRR for small gaps (\(W \le 6\)), peaking at \(>0.61\) MRR compared to the baseline's best case.
* **Short Gaps**: The LLM infers discrete missing actions (e.g., "loading the gun") from immediate context.
* **Degradation**: Performance degrades near-linearly as \(W\) increases. By \(W=12\), the generative advantage narrows as the hallucination search space becomes too large.

### 3.2 Semantic Wins
Category-wise analysis reveals that Phi-3 performs well in specialized domains such as **Military** and **Survival** (+0.35 MRR Delta). In these domains, structured procedural knowledge (e.g., steps to build a shelter) allows the LLM to predict missing steps, whereas vector interpolation blurs the distinct actions.

### 3.3 Temporal Coherence
A key finding is the robustness of **Temporal NDCG**. Even when exact R@1 drops at large widths (\(W>15\)), the Temporal NDCG for Phi-3 remains high (\(\approx 0.80\)), comparable to the interpolation baseline.
* **Interpretation**: When the LLM fails to guess the exact caption, it typically hallucinates an event that is semantically compatible and temporally adjacent (a "near miss").
* **Baseline Competitiveness**: The vector baseline performs surprisingly well on temporal metrics because averaging naturally lands in the "middle" of the semantic space, ensuring retrieval of mid-segment captions.

### 3.4 Positional Bias
We observe a **Start Bias**: reconstruction accuracy is consistently higher when masking starts at the beginning of the context window (\(i=0\)) rather than the middle or end.

## 4. Conclusion
We demonstrate that generative LLMs are candidates for dense video caption imputation. *(Note: see [`docs/paper/draft.md`](draft.md) for the final standardized SigLIP benchmark with Llama 3.1 8B, channel clustering, and hypothesis testing).*
