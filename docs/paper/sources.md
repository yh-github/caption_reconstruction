# Main Sources

### 

### **1\. Text-Enhanced Zero-Shot Action Recognition: A training-free approach**

* #### **Publication**

  * **Venue:** ICPR (International Conference on Pattern Recognition)  
  * **Date:** 2024  
  * **Authors & Affiliations:**  
    * Massimo Bosetti (University of Trento, Italy)  
    * Shibingfeng Zhang (University of Trento, Italy)  
    * Benedetta Liberatori (University of Trento, Italy)  
    * Giacomo Zara (University of Trento, Italy)  
    * Elisa Ricci (University of Trento, Italy)  
    * Paolo Rota (University of Trento, Italy)

* #### **Key points**

  1. **TEAR (Text-Enhanced Action Recognition):** A novel, training-free method for AI to recognize actions in videos.  
  2. **Problem Addressed:** Difficulty of Vision-Language Models (VLMs) in recognizing dynamic actions in videos compared to object identification in static images.  
  3. **Limitations of Existing ZS-VAR (Zero-Shot Video Action Recognition) Methods:** Often resource-intensive and prone to bias due to extensive training.  
  4. **TEAR's Solution:** Leverages detailed action descriptors and contextual information in text format.  
  5. **Benefits:** Eliminates the need for dedicated training data or significant computational resources.

### 

### **2\. VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control**

* #### **Publication** 

  * **Venue:** SIGGRAPH (Special Interest Group on Computer Graphics and Interactive Techniques Conference)  
  * **Date:** 2025  
  * **Authors & Affiliations:**  
    * Yuxuan Bian (ARC Lab, Tencent PCG, China; The Chinese University of Hong Kong, Hong Kong, China)  
    * Zhaoyang Zhang (The Chinese University of Hong Kong, Hong Kong, China)  
    * Xuan Ju (The Chinese University of Hong Kong, Hong Kong, China)  
    * Mingdeng Cao (The University of Tokyo, Japan)  
    * Liangbin Xie (University of Macau, Macau, China)  
    * Ying Shan (ARC Lab, Tencent PCG, China)  
    * Qiang Xu (The Chinese University of Hong Kong, Hong Kong, China)

* #### **Key points**

  * **VideoPainter:** An AI tool for seamless video inpainting and editing of any length.  
  * **Addresses Limitations:** Overcomes challenges of existing methods that struggle with fully masked objects, long videos, and background consistency.  
  * **Novel Dual-Stream System:** Features a Context Encoder for background clues and target region ID resampling for any-length capability.  
  * **Superior Performance:** Demonstrates better results in both inpainting and editing tasks.  
  * **New Dataset and Benchmark:** Authors contributed VPData and VPBench to support further research.

### **3\. Lost in Embeddings: Information Loss in Vision-Language Models**

* #### **Publication**

  * **Venue:** EMNLP (Findings of the Conference on Empirical Methods in Natural Language Processing)  
  * **Date:** 2025  
  * **Authors & Affiliations:**  
    * Wenyan Li (University of Copenhagen, Denmark)  
    * Raphael Tang (Microsoft, USA)  
    * Chengzu Li (University of Cambridge, UK)  
    * Caiqi Zhang (University of Cambridge, UK)  
    * Ivan Vulić (University of Cambridge, UK)  
    * Anders Søgaard (University of Copenhagen, Denmark)

* #### **Key points**

  * **Problem:** Advanced Vision-Language Models (VLMs) sometimes make errors because they lose important visual information during the conversion of images into a language-comprehensible format.  
  * **Cause:** The "connector" component, which projects visual representations into the language model's embedding space, acts like a flawed compression algorithm.  
  * **Consequences:**  
    1. **Semantic Distortion:** Visual information is distorted, making similar images appear less related (neighbors in visual space diverge by 40–60%).  
    2. **Performance Impact:** This information loss negatively affects the model's ability to retrieve or recall information.  
    3. **Prediction of Failure:** Researchers can predict when the model will struggle with visually grounded question answering by identifying areas of significant information loss.  
  * **Conclusion:** VLMs underperform because the critical "connector" component discards vital details when bridging visual and language understanding.

---

### **4. FastV: An Image is Worth 1/2 Tokens After Layer 2**

* #### **Publication**
  * **Venue:** ECCV (European Conference on Computer Vision)  
  * **Date:** 2024 (arXiv:2403.06764)  
  * **Authors & Affiliations:**  
    * Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, Baobao Chang (Peking University, Alibaba Group)

* #### **Key points**
  * **Visual Attention Inefficiency:** Investigates spatial token redundancy in Large Vision-Language Models (LVLMs). Early transformer layers attend to dense visual tokens, but deep layers focus primarily on text and summary tokens.
  * **Plug-and-Play Pruning:** Proposes FastV, a training-free inference acceleration method that prunes up to 50% of visual tokens after layer 2 based on attention ranks.
  * **Relevance to Caption Reconstruction:** FastV addresses spatial token sparsification during forward inference. Our caption reconstruction framework addresses temporal sparsification across video spans: determining when entire multi-second video intervals contain redundant visual information that can be skipped or reconstructed from temporal context.

---

### **5. DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification**

* #### **Publication**
  * **Venue:** NeurIPS (Advances in Neural Information Processing Systems)  
  * **Date:** 2021  
  * **Authors & Affiliations:**  
    * Yongming Rao, Wenxun Zhao, Zheng Zhu, Jiwen Lu, Jie Zhou (Tsinghua University)

* #### **Key points**
  * **Dynamic Pruning:** Introduces lightweight prediction modules to dynamically drop non-informative visual tokens hierarchically across vision transformer layers.
  * **Relevance:** Foundational framework for adaptive visual computation, motivating our temporal counterpart (evaluating non-parametric continuity vs. parametric language generation across time).