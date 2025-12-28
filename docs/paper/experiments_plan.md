# Experiments Plan

## 1. Video Categorization (The "When")
**Objective**: Group videos to allow conditional analysis of performance.
**Action**:
- **Tagging**: Categorize the 100 videos based on filename/content (e.g., "Survival", "Military", "Nature", "Farms").
- **Meta-Tagging**: Group into higher-level structural categories:
    - **Procedural**: Logical steps, human-driven (Survival, Farming, Military).
    - **Stochastic/Ambient**: Nature, Weather, Scenery.

## 2. Conditional Performance Analysis
**Objective**: Comparison of LLM vs. Baseline performance across valid categories.
**Hypothesis**: LLMs excel in *Procedural* tasks where A implies B, while Video embeddings excel in *Stochastic* tasks where visual details matter.
**Analysis**:
- Compute $\Delta = Score_{LLM} - Score_{Video}$ for each video.
- Plot: Bar chart of average $\Delta$ per category.

## 3. Main Comparative Study

**Objective**: Demonstrate that LLM-based in-filling outperforms naive embedding-based interpolation.

**Baselines**:
1. **Video Embedding Baseline**: Reconstruct missing segment using the mean or closest vector of surrounding *video* embeddings.
2. **Text Embedding Baseline**: Reconstruct missing segment using the mean or closest vector of surrounding *text* embeddings (of observed captions).

**Our Method**:
- **LLM In-filling**: Use LLM to generate text for missing segments given context.

**Metrics**:
- Cosine Similarity (Mean).
- Z-Score Normalized Cosine Similarity.

## 2. Masking Strategies
Run experiments under different data loss scenarios:
- **Random Masking**: Randomly dropping % of frames (e.g., 10%, 25%, 50%).
- **Block Masking**: Dropping contiguous blocks (e.g., 5s, 10s gaps).

## 3. Stability and Ranking Analysis
- Explore how well the reconstructed captions distinguish the correct video from distractors.
- Plot Rank vs Masking Ratio.

## 4. Ablation Studies
- **Prompt Engineering**: Zero-shot vs One-shot vs Few-shot.
- **Temperature**: Effect of sampling temperature (e.g., 0.0, 0.7, 1.0) on diversity and accuracy.

## 5. Qualitative Examples
- Select 3-5 diverse videos.
- Show "Ground Truth" vs "Baseline Reconstruction" vs "LLM Reconstruction".
- Highlight cases where LLM infers actions not explicitly stated in context but logically implied.

## Action Items
- [ ] **Categorize videos** (New script/step).
- [ ] Run `wild_dev_sim_one_shot_t=1` (Done)
- [ ] Run `wild_dev_sim_vec` (Done)
- [ ] Run `wild_dev_sim_vec_vid` (Done)
- [ ] (Optional) Run Zero-shot variations if not already cached.
- [ ] Generate tables comparing Mean scores.
- [ ] Generate plots for Ranking analysis.
- [ ] Extract qualitative examples.
