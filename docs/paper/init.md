# **Masked video reconstruction**

Raw Input: 100 videos clips, 60 seconds each, various types  
Masking: for each video clip, generate masked versions (continuous masking in a random spot in the video)  
algorithm: reconstruct the mask section using video embeddings or first use an LLM to caption the video and reconstruct on the text  
Hypothesis: the video method and text method will differ in the types of videos they are best at (evaluated via vector similarity to the unmasked initial representation)

Venue: CVPR/ICCP Workshop or aACL or EMNLP Workshop. Archival in any case. Short format (4 pages).

### **Possible abstract \#1:**

We propose a simple and interpretable framework for analyzing **multimodal redundancy in video understanding**. Videos are decomposed into fine-grained temporal units, from which subsets of segments are deliberately removed. Two independent pathways—**visual reconstruction** and **semantic (text-only) reconstruction**—are tasked with predicting the missing content using the remaining context. Reconstruction quality is evaluated via embedding similarity to the removed segments.

This setup exposes a clear boundary between **semantic predictability** and **visual necessity**. We show that in structurally constrained or procedural videos, semantic models can accurately infer missing segments without visual input, while in visually driven or stochastic content, visual inference dominates. Based on this behavior, videos naturally cluster into **semantic-oriented**, **visual-oriented**, and **hybrid** categories.

Beyond analysis, the framework enables **temporal semantic compression** of long videos by retaining only the minimal subset of segments required for accurate reconstruction. Our method is model-agnostic, extendable, and provides a diagnostic tool for understanding when multimodal processing is essential—and when it is redundant.

### **Possible abstract \#2:**

Recent Video-LLMs typically treat video understanding as a continuous stream of visual encoding. However, real-world events often follow structured, semantic scripts that pre-trained language models can predict without immediate visual evidence. In this work, we investigate the boundary between **semantic inference** (what *must* happen) and **visual perception** (what *actually* happened) through a novel *Caption Reconstruction* task.

We propose a comparative framework that analyzes video content across two distinct regimes: **Procedural** (logically deterministic contexts, such as instructional videos) and **Stochastic** (unpredictable environments, such as vlogs). By comparing a text-only LLM against a visual-interpolation baseline on the WildQA dataset, we demonstrate that LLMs can effectively "hallucinate" accurate intermediate captions in procedural regimes, rendering visual processing redundant for significant temporal segments. Conversely, we identify the specific stochastic conditions where visual grounding remains indispensable. This framework offers a diagnostic tool for measuring multimodal redundancy and suggests semantic compression strategies for efficient video understanding.
