# A Framework for Comparative Analysis of Video Caption In-filling

## 1. Abstract
- Video understanding often deals with partial or noisy data.
- We introduce a **comparative framework** to evaluate different approaches for reconstructing missing dense video captions.
- Specifically, we contrast **Structure-based (Text/LLM)** reconstruction against **Content-based (Video)** reconstruction.
- We investigate the question: *When* can pure text models (LLMs), relying only on temporal context and world knowledge, outperform models with access to the actual video content?
- Our findings reveal distinct performance regimes based on video dynamics and semantic predictability.

## 2. Introduction
- The complementary nature of Video (pixel-rich, high entropy) and Text (semantic-rich, logical structure).
- The task: In-filling missing dense captions in a sequence.
- Motivation: Understanding the limits of LLM world models vs. visual perception.
- Contribution:
    1. A rigorous framework for comparing multimodal reconstruction methods.
    2. Empirical analysis identifying scenarios where "hallucinating" plausible futures (LLM) is more accurate than imperfect retrieval (Video).

## 3. Related Work
- Video In-filling (Visual).
- Video Captioning & Dense Captioning.
- Multimodal Alignment (Video-Text embeddings).
- LLMs as World Models / Temporal Reasoners.

## 4. Methodology: The Comparative Framework
### 4.1 Problem Formulation
- Video as a sequence of semantic units (captions) and visual units (embeddings).
- Masking capability as a stress test for context understanding.

### 4.2 Reconstruction Approaches
- **Text-Only (The "Planner")**: Uses LLM (Gemini/GPT) to predict missing actions based on causal logic and context.
- **Video-Only (The "Observer")**: Uses visual similarity to interpolate missing content from surrounding frames.
- **Text-Embedding (The "Smoother")**: Latent space interpolation of semantic vectors.

### 4.3 Evaluation Metrics
- **Semantic Similarity**: Cosine similarity in a shared embedding space.
- **Normalized Performance (Z-Score)**: Accounting for the varying "difficulty" (entropy) of different videos.
- **Ranking Robustness**: Ability to distinguish the correct reconstruction from distractors.

## 5. Experiments
### 5.1 Dataset and Categorization
- **Source**: 100 videos from the **WildQA** dataset ("dev" set) [MichiganNLP/In-the-wild-QA].
- **Characteristics**: "Wild" videos from diverse domains (Survival, Manufacturing, Military, Nature).
- **Categorization**: Grouping videos by dynamic intensity and semantic structure (e.g., "Procedural/Scripted" vs. "Stochastic/Nature") to analyze *when* LLMs succeed.

### 5.2 Comparative Results
- **Overall Performance**: Aggregate comparison of LLM vs. Video baselines.
- **Conditional Analysis ("The When")**:
    - Performance delta ($Score_{LLM} - Score_{Video}$) across categories.
    - Hypothesis: LLMs excel in *Procedural* tasks (e.g., cooking, building) where A implies B. Video excels in *Stochastic* tasks (e.g., weather, animals) where specific visual detauls matter.
- **Ablation**: Impact of context window and model size.

### 5.3 Qualitative Case Studies
- "Hallucination vs. Perception": Examples where specific visual details are lost but semantic truth is preserved (or vice versa).

## 6. Discussion
- The tradeoff between *Semantic Plausibility* (LLM) and *Visual Fidelity* (Video).
- Implications for efficient video transmission and understanding (transmitting only keyframes + captions).

## 7. Conclusion
- Summary of the "When" findings.
- Future directions for hybrid models.
