# Dataset Audit Report: `wild2`

**Date**: 2026-08-10  
**Target Directory**: `datasets/wildQA/captions__wild2`  
**Total Clips Audited**: `95`  

## Executive Summary

| Metric | Result | Health Status |
| :--- | :--- | :--- |
| **Suspected Duplicate Clusters** | `4` | 🔴 Issue Found |
| **Ad / Commercial Boilerplate Hits** | `18` | 🟡 Warning |
| **Repetitive / Loop Captions** | `3` | 🟡 Warning |

---

## 1. Duplication & Similarity Analysis

Found **43** pairs of videos with Cosine Similarity > 0.70:

| Video 1 | Video 2 | Cosine Sim | 3-Gram Jaccard | Status |
| :--- | :--- | :--- | :--- | :--- |
| `Hamiltonville-Farm_6-clip-18` | `Survival-Instinct_9-clip-2` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Hamiltonville-Farm_6-clip-18` | `Weathershot_3-clip-1` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Survival-Instinct_9-clip-2` | `Weathershot_3-clip-1` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_0-clip-4` | `Nick-Gaillard_1-clip-2` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_0-clip-4` | `Survival-Instinct_2-clip-1` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_0-clip-4` | `Survival-Instinct_7-clip-8` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_7-clip-3` | `Gung-Ho-Vids_5-clip-2` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_7-clip-3` | `Survival-Skills-Primitive_3-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Gung-Ho-Vids_5-clip-2` | `Survival-Skills-Primitive_3-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Nick-Gaillard_1-clip-2` | `Survival-Instinct_2-clip-1` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Nick-Gaillard_1-clip-2` | `Survival-Instinct_7-clip-8` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Survival-Instinct_2-clip-1` | `Survival-Instinct_7-clip-8` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_2-clip-4` | `Climate-Change_6-clip-7` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_2-clip-4` | `Gung-Ho-Vids_8-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_2-clip-4` | `King-Kong-Amazon_5-clip-14` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_2-clip-4` | `Primal-Earth-Sounds_0-clip-44` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_2-clip-4` | `Survival-Instinct_11-clip-10` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_2-clip-4` | `Survival-Instinct_8-clip-4` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_2-clip-4` | `Weathershot_7-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_6-clip-7` | `Gung-Ho-Vids_8-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_6-clip-7` | `King-Kong-Amazon_5-clip-14` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_6-clip-7` | `Primal-Earth-Sounds_0-clip-44` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_6-clip-7` | `Survival-Instinct_11-clip-10` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_6-clip-7` | `Survival-Instinct_8-clip-4` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Climate-Change_6-clip-7` | `Weathershot_7-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Gung-Ho-Vids_8-clip-0` | `King-Kong-Amazon_5-clip-14` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Gung-Ho-Vids_8-clip-0` | `Primal-Earth-Sounds_0-clip-44` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Gung-Ho-Vids_8-clip-0` | `Survival-Instinct_11-clip-10` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Gung-Ho-Vids_8-clip-0` | `Survival-Instinct_8-clip-4` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Gung-Ho-Vids_8-clip-0` | `Weathershot_7-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `King-Kong-Amazon_5-clip-14` | `Primal-Earth-Sounds_0-clip-44` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `King-Kong-Amazon_5-clip-14` | `Survival-Instinct_11-clip-10` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `King-Kong-Amazon_5-clip-14` | `Survival-Instinct_8-clip-4` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `King-Kong-Amazon_5-clip-14` | `Weathershot_7-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Primal-Earth-Sounds_0-clip-44` | `Survival-Instinct_11-clip-10` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Primal-Earth-Sounds_0-clip-44` | `Survival-Instinct_8-clip-4` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Primal-Earth-Sounds_0-clip-44` | `Weathershot_7-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Survival-Instinct_11-clip-10` | `Survival-Instinct_8-clip-4` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Survival-Instinct_11-clip-10` | `Weathershot_7-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Survival-Instinct_8-clip-4` | `Weathershot_7-clip-0` | `1.000` | `1.000` | 🔴 Suspected Duplicate |
| `Tornado-Trackers_10-clip-4` | `Ultimate-Chase_9-clip-0` | `0.773` | `0.050` | 🟡 Similar Content |
| `Dan-Robinson_3-clip-3` | `Tornado-Trackers_6-clip-2` | `0.733` | `0.074` | 🟡 Similar Content |
| `Tornado-Trackers_1-clip-0` | `Ultimate-Chase_9-clip-0` | `0.705` | `0.041` | 🟡 Similar Content |


## 2. Ad & Commercial Boilerplate Screening

Found **18** files containing potential ad/boilerplate keywords:

* **`Climate-Change_0-clip-4`**: Matched Suspect fabric ad content (5x)
  * *Snippet*: "A person's hands are shown holding a piece of black fabric with a white grid pattern. Hands smooth out the black gridded fabric on a wooden cutting ma..."

* **`Climate-Change_2-clip-4`**: Matched Suspect VR headset ad content (17x)
  * *Snippet*: "A close-up shot of a person's hands operating a black gaming controller, with their thumbs on the joysticks. The camera tilts up to reveal a person we..."

* **`Climate-Change_6-clip-7`**: Matched Suspect VR headset ad content (17x)
  * *Snippet*: "A close-up shot of a person's hands operating a black gaming controller, with their thumbs on the joysticks. The camera tilts up to reveal a person we..."

* **`Climate-Change_7-clip-3`**: Matched Suspect bread ad content (1x)
  * *Snippet*: "A person's hands hold up a large, round loaf of freshly baked bread against a dark background. A close-up shot focuses on the crusty texture of the go..."

* **`Gung-Ho-Vids_5-clip-2`**: Matched Suspect bread ad content (1x)
  * *Snippet*: "A person's hands hold up a large, round loaf of freshly baked bread against a dark background. A close-up shot focuses on the crusty texture of the go..."

* **`Gung-Ho-Vids_8-clip-0`**: Matched Suspect VR headset ad content (17x)
  * *Snippet*: "A close-up shot of a person's hands operating a black gaming controller, with their thumbs on the joysticks. The camera tilts up to reveal a person we..."

* **`Hamiltonville-Farm_8-clip-3`**: Matched Channel subscription request (2x)
  * *Snippet*: "A man in a brown jacket and camo hat stands in front of a pole barn, speaking to the camera. The man continues to speak, looking slightly off-camera. ..."

* **`King-Kong-Amazon_5-clip-14`**: Matched Suspect VR headset ad content (17x)
  * *Snippet*: "A close-up shot of a person's hands operating a black gaming controller, with their thumbs on the joysticks. The camera tilts up to reveal a person we..."

* **`Nick-Gaillard_1-clip-2`**: Matched Suspect fabric ad content (5x)
  * *Snippet*: "A person's hands are shown holding a piece of black fabric with a white grid pattern. Hands smooth out the black gridded fabric on a wooden cutting ma..."

* **`Primal-Earth-Sounds_0-clip-44`**: Matched Suspect VR headset ad content (17x)
  * *Snippet*: "A close-up shot of a person's hands operating a black gaming controller, with their thumbs on the joysticks. The camera tilts up to reveal a person we..."

* **`Survival-Instinct_11-clip-10`**: Matched Suspect VR headset ad content (17x)
  * *Snippet*: "A close-up shot of a person's hands operating a black gaming controller, with their thumbs on the joysticks. The camera tilts up to reveal a person we..."

* **`Survival-Instinct_2-clip-1`**: Matched Suspect fabric ad content (5x)
  * *Snippet*: "A person's hands are shown holding a piece of black fabric with a white grid pattern. Hands smooth out the black gridded fabric on a wooden cutting ma..."

* **`Survival-Instinct_7-clip-8`**: Matched Suspect fabric ad content (5x)
  * *Snippet*: "A person's hands are shown holding a piece of black fabric with a white grid pattern. Hands smooth out the black gridded fabric on a wooden cutting ma..."

* **`Survival-Instinct_8-clip-4`**: Matched Suspect VR headset ad content (17x)
  * *Snippet*: "A close-up shot of a person's hands operating a black gaming controller, with their thumbs on the joysticks. The camera tilts up to reveal a person we..."

* **`Survival-Skills-Primitive_3-clip-0`**: Matched Suspect bread ad content (1x)
  * *Snippet*: "A person's hands hold up a large, round loaf of freshly baked bread against a dark background. A close-up shot focuses on the crusty texture of the go..."

* **`TreadmillTV_3-clip-12`**: Matched Known sponsor brand (1x)
  * *Snippet*: "A first-person view moves forward on a dirt path in a sunlit forest, with bright lens flare from the sun. The camera proceeds down the sun-dappled for..."

* **`Ultimate-Chase_9-clip-0`**: Matched Copyright boilerplate (1x)
  * *Snippet*: "A title card reads "Close Tornado Encounter, South Dakota - May 24th, 2010". A title card reads "Close Tornado Encounter, South Dakota - May 24th, 201..."

* **`Weathershot_7-clip-0`**: Matched Suspect VR headset ad content (17x)
  * *Snippet*: "A close-up shot of a person's hands operating a black gaming controller, with their thumbs on the joysticks. The camera tilts up to reveal a person we..."

## 3. Domain Alignment Analysis

Comparison of keyword profile predictions against ground-truth dataset domains:

🟢 **All evaluated videos align with their ground-truth domains.**
