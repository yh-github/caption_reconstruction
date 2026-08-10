# Dataset Audit Report: `wild3`

**Date**: 2026-08-10  
**Target Directory**: `datasets/wildQA/captions__wild3`  
**Total Clips Audited**: `71`  

## Executive Summary

| Metric | Result | Health Status |
| :--- | :--- | :--- |
| **Suspected Duplicate Clusters** | `1` | 🔴 Issue Found |
| **Ad / Commercial Boilerplate Hits** | `2` | 🟡 Warning |
| **Repetitive / Loop Captions** | `1` | 🟡 Warning |
| **Domain Keyword Alignment** | `54/71 (76.1%)` | 🟢 Strong (>70%) |

---

## 1. Duplication & Similarity Analysis

Found **38** pairs of videos with Cosine Similarity > 0.70:

| Video 1 | Video 2 | Cosine Sim | 3-Gram Jaccard | Status |
| :--- | :--- | :--- | :--- | :--- |
| `Bertram-Craft_3-clip-8` | `Bertram-Craft_7-clip-6` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_3-clip-8` | `Bertram-Craft_8-clip-5` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_3-clip-8` | `Climate-Change_1-clip-10` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_3-clip-8` | `Climate-Change_10-clip-14` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_3-clip-8` | `Climate-Change_12-clip-16` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_3-clip-8` | `Climate-Change_8-clip-9` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_3-clip-8` | `Disaster-Compilations_4-clip-2` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_3-clip-8` | `Gung-Ho-Vids_3-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_7-clip-6` | `Bertram-Craft_8-clip-5` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_7-clip-6` | `Climate-Change_1-clip-10` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_7-clip-6` | `Climate-Change_10-clip-14` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_7-clip-6` | `Climate-Change_12-clip-16` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_7-clip-6` | `Climate-Change_8-clip-9` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_7-clip-6` | `Disaster-Compilations_4-clip-2` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_7-clip-6` | `Gung-Ho-Vids_3-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_8-clip-5` | `Climate-Change_1-clip-10` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_8-clip-5` | `Climate-Change_10-clip-14` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_8-clip-5` | `Climate-Change_12-clip-16` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_8-clip-5` | `Climate-Change_8-clip-9` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_8-clip-5` | `Disaster-Compilations_4-clip-2` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Bertram-Craft_8-clip-5` | `Gung-Ho-Vids_3-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_1-clip-10` | `Climate-Change_10-clip-14` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_1-clip-10` | `Climate-Change_12-clip-16` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_1-clip-10` | `Climate-Change_8-clip-9` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_1-clip-10` | `Disaster-Compilations_4-clip-2` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_1-clip-10` | `Gung-Ho-Vids_3-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_10-clip-14` | `Climate-Change_12-clip-16` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_10-clip-14` | `Climate-Change_8-clip-9` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_10-clip-14` | `Disaster-Compilations_4-clip-2` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_10-clip-14` | `Gung-Ho-Vids_3-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_12-clip-16` | `Climate-Change_8-clip-9` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_12-clip-16` | `Disaster-Compilations_4-clip-2` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_12-clip-16` | `Gung-Ho-Vids_3-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_8-clip-9` | `Disaster-Compilations_4-clip-2` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_8-clip-9` | `Gung-Ho-Vids_3-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Disaster-Compilations_4-clip-2` | `Gung-Ho-Vids_3-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Dan-Robinson_0-clip-2` | `Dan-Robinson_7-clip-1` | `0.747` | `0.075` | 🟡 Similar Content |
| `Dan-Robinson_10-clip-0` | `Dan-Robinson_9-clip-0` | `0.715` | `0.060` | 🟡 Similar Content |


## 2. Ad & Commercial Boilerplate Screening

Found **2** files containing potential ad/boilerplate keywords:

* **`AiirSource-Military_0-clip-2`**: Matched Known sponsor brand (1x)
  * *Snippet*: "A tan military tank is positioned on a metal transport platform. Multiple tan armored vehicles are lined up on a concrete lot next to a train. A soldi..."

* **`How-Farms-Work_1-clip-8`**: Matched Channel subscription request (1x)
  * *Snippet*: "A man operates a red post driver attached to a tractor. The post driver strikes a wooden post into the green field. The red post driver continues hamm..."

## 3. Domain Alignment Analysis

Comparison of keyword profile predictions against ground-truth dataset domains:

### Mismatched Domain Outliers (17 files)

| Video ID | Ground Truth | Predicted Domain | GT Domain Score | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| `4k-Relaxation_4-clip-4` | `Geography` | `Human Survival` | `29` | "A view of a shallow river flowing between high canyon walls. The river flows ove..." |
| `AiirSource-Military_10-clip-0` | `Military` | `Human Survival` | `4` | "The AIIRSOURCE.COM logo is displayed on a white background. The AIIRSOURCE.COM l..." |
| `AiirSource-Military_2-clip-0` | `Military` | `Human Survival` | `0` | "An LCAC approaches the well deck, creating a large mist of water spray. The LCAC..." |
| `Army-military-2018_11-clip-4` | `Military` | `Human Survival` | `17` | "Camera moves rapidly, capturing blurry textures of military gear. Continued rapi..." |
| `BC-Bushcraft_1-clip-8` | `Human Survival` | `Agriculture` | `3` | "A man in a green shirt pulls a large branch through a dense forest. The man pick..." |
| `BC-Bushcraft_6-clip-0` | `Human Survival` | `Geography` | `5` | "A black screen is shown at the start. A logo for 'B C Bushcraft' with a wolf sil..." |
| `Chad-Zuber_1-clip-10` | `Human Survival` | `Agriculture` | `0` | "A man's hands mix water into a heap of soil inside a curved bark container. The ..." |
| `Chad-Zuber_2-clip-4` | `Human Survival` | `Geography` | `16` | "A man with long hair sits thoughtfully in a rocky canyon wearing a crude cloak. ..." |
| `Climate-Change_1-clip-10` | `Natural Disaster` | `Human Survival` | `0` | "A person in athletic gear is standing on top of a brick wall. They take a deep b..." |
| `Climate-Change_10-clip-14` | `Natural Disaster` | `Human Survival` | `0` | "A person in athletic gear is standing on top of a brick wall. They take a deep b..." |
| `Climate-Change_12-clip-16` | `Natural Disaster` | `Human Survival` | `0` | "A person in athletic gear is standing on top of a brick wall. They take a deep b..." |
| `Climate-Change_8-clip-9` | `Natural Disaster` | `Human Survival` | `0` | "A person in athletic gear is standing on top of a brick wall. They take a deep b..." |
| `Disaster-Compilations_4-clip-2` | `Natural Disaster` | `Human Survival` | `0` | "A person in athletic gear is standing on top of a brick wall. They take a deep b..." |
| `Gung-Ho-Vids_3-clip-0` | `Military` | `Human Survival` | `0` | "A person in athletic gear is standing on top of a brick wall. They take a deep b..." |
| `Hamiltonville-Farm_10-clip-8` | `Agriculture` | `Human Survival` | `0` | "A person in gloves uses a speed square to mark a wooden plank. The person uses a..." |
| `Hamiltonville-Farm_7-clip-15` | `Agriculture` | `Geography` | `2` | "A gloved hand points to a pinecone lodged in a small, thin tree. The gloved hand..." |
| `Hamiltonville-Farm_9-manual` | `Agriculture` | `Human Survival` | `1` | "A person's hand reaches for a brown cap on a dark side table. The person's hand ..." |

