> [!NOTE]
> **Historical Metric Specification**: This document details the preliminary exploratory metric formulation (using `vit_small_patch16_224` 384-dim and `gemini-embedding-001` 512-dim). It was subsequently superseded by the shared **SigLIP** (`google/siglip-base-patch16-224`, 768-dim) framework and rank-based evaluation detailed in [`docs/theory/cross_modal_evaluation_metrics.md`](../theory/cross_modal_evaluation_metrics.md).

# Data

The data is 100 videos, each 60 seconds long.  
The score is cosine similarity (which is \(1 - \text{distance}\)), between \(-1\) and \(1\).  
Segment length is 1 second.  
Videos are sampled at 1 FPS.  

The score is the mean of scores of all videos.  
The score of a video is the minimum of its cosine similarity over all reconstructed vectors:
\[
\text{Score}(\text{method}, \text{masking}) = \text{MEAN}(\text{MIN}(\text{CosineSimilarity}(v_{\text{original}}, v_{\text{reconstructed}})))
\]

The Z-Score normalized version is:
\[
\text{ScoreZ}(\text{method}, \text{masking}) = \text{MEAN}(\text{MIN}(Z, (\text{CosineSimilarity}(v_{\text{original}}, v_{\text{reconstructed}}))))
\]

The global mean and standard deviation are calculated over all maskings per method.

# Methods

### MethodA: video\_embeddings

**Input:** All VEVs beside VEV of \[i…i+w\] segment  
**Output:** (challenging) VEV\_P for \[i…i+w\] segment (prediction)  
Reconstruction of vectors using the **mean of the closest vectors**, or copying **(repeating) the *closest vector*.**

### MethodB: text\_embeddings

**Input:** All CEVs beside CEV of \[i…i+w\] segment  
**Output:** CEV\_P for \[i…i+w\] segment  
Reconstruction of vectors using the **mean of the closest vectors**, or copying **(repeating) the *closest vector*.**

### MethodC: LLM completion \-\> text\_embeddings

**Input:** All CEVs beside CEV of \[i…i+w\] segment  
**Output:** CEV\_P for \[i…i+w\] segment  
Reconstruction using an LLM (**CaptionedVideo\_\_pro\_d\_zero\_shot\_v1.1\_\_t=0.7**)

* Video vectors generated using **vit\_small\_patch16\_224** (output\_dimensionality: 384\)  
* Text vectors generated using **gemini-embedding-001 (**output\_dimensionality: 512,  task\_type: "SEMANTIC\_SIMILARITY")  
* Text compilation generated using **gemini-2.5-pro** (thought\_budget: auto)
