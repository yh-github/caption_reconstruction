# Dataset Audit Report: `wild4`

**Date**: 2026-08-10  
**Target Directory**: `datasets/wildQA/captions__wild4`  
**Total Clips Audited**: `100`  

## Executive Summary

| Metric | Result | Health Status |
| :--- | :--- | :--- |
| **Suspected Duplicate Clusters** | `0` | 🟢 Clean (0 duplicates) |
| **Ad / Commercial Boilerplate Hits** | `9` | 🟡 Warning |
| **Repetitive / Loop Captions** | `1` | 🟡 Warning |
| **Domain Keyword Alignment** | `79/100 (79.0%)` | 🟢 Strong (>70%) |

---

## 1. Duplication & Similarity Analysis

Found **7** pairs of videos with Cosine Similarity > 0.70:

| Video 1 | Video 2 | Cosine Sim | 3-Gram Jaccard | Status |
| :--- | :--- | :--- | :--- | :--- |
| `Tornado-Trackers_1-clip-0` | `Ultimate-Chase_9-clip-0` | `0.814` | `0.043` | 🟡 Similar Content |
| `Dan-Robinson_3-clip-3` | `Ultimate-Chase_9-clip-0` | `0.799` | `0.060` | 🟡 Similar Content |
| `Dan-Robinson_3-clip-3` | `Tornado-Trackers_1-clip-0` | `0.786` | `0.045` | 🟡 Similar Content |
| `Tornado-Trackers_10-clip-4` | `Tornado-Trackers_6-clip-2` | `0.760` | `0.040` | 🟡 Similar Content |
| `Dan-Robinson_3-clip-3` | `Tornado-Trackers_6-clip-2` | `0.755` | `0.069` | 🟡 Similar Content |
| `Dan-Robinson_3-clip-3` | `Tornado-Trackers_10-clip-4` | `0.711` | `0.029` | 🟡 Similar Content |
| `Tornado-Trackers_1-clip-0` | `Tornado-Trackers_6-clip-2` | `0.707` | `0.055` | 🟡 Similar Content |


## 2. Ad & Commercial Boilerplate Screening

Found **9** files containing potential ad/boilerplate keywords:

* **`Army-military-2018_8-clip-73`**: Matched Known sponsor brand (1x)
  * *Snippet*: "Aerial view of a military vehicle on a dirt road in a field. The vehicle remains stationary as the camera observes from above. A massive fireball erup..."

* **`Gung-Ho-Vids_12-clip-1`**: Matched Known sponsor brand (1x)
  * *Snippet*: "An F-35 fighter jet is parked on the airfield with a ground crew member signaling nearby. The F-35's engine is running as the ground crew member conti..."

* **`Hamiltonville-Farm_8-clip-3`**: Matched Channel subscription request (2x)
  * *Snippet*: "A man in a brown jacket and camo hat speaks to the camera outdoors. The man continues his introduction while standing in front of a metal shed. He men..."

* **`King-Kong-Amazon_11-clip-7`**: Matched Known sponsor brand (1x)
  * *Snippet*: "A wooden structure with a thatched roof sits amidst lush green jungle foliage. A small hut with a palm leaf roof is nestled in a dense tropical forest..."

* **`MilitaryNotes_11-clip-1`**: Matched Suspect VR headset ad content (2x)
  * *Snippet*: "A Chinook helicopter flies through a thick, dusty haze. The helicopter continues its flight in the overcast, dusty sky. A Chinook helicopter descends ..."

* **`Sandboxx_0-clip-4`**: Matched Suspect VR headset ad content (1x)
  * *Snippet*: "A B-2 Spirit bomber is parked on a wet tarmac with ground crew working around it. A fuel truck is parked next to the B-2 Spirit bomber on the wet tarm..."

* **`TreadmillTV_6-clip-2`**: Matched Known sponsor brand (1x)
  * *Snippet*: "A person walks across the white salt flat with mountains in the distance. A man in a black shirt and shorts walks towards the camera on the salt crust..."

* **`Ultimate-Chase_9-clip-0`**: Matched Copyright boilerplate (1x)
  * *Snippet*: "A title screen displays "Close Tornado Encounter South Dakota - May 24th, 2010". The title screen remains with information about the video copyright a..."

* **`Weathershot_3-clip-1`**: Matched Known sponsor brand (1x)
  * *Snippet*: "Heavy rain and wind lash against trees and a house in a residential area. A white car is visible on a flooded street as rain pours down intensely. The..."

## 3. Domain Alignment Analysis

Comparison of keyword profile predictions against ground-truth dataset domains:

### Mismatched Domain Outliers (21 files)

| Video ID | Ground Truth | Predicted Domain | GT Domain Score | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| `AiirSource-Military_1-clip-0` | `Military` | `Human Survival` | `7` | "A firefighter in a silver proximity suit adjusts their gas mask. The firefighter..." |
| `BC-Bushcraft_10-clip-8` | `Human Survival` | `Geography` | `5` | "The camera pans across a mossy forest floor with scattered leaves. A close-up vi..." |
| `BC-Bushcraft_2-clip-2` | `Human Survival` | `Geography` | `8` | "A person in a blue raincoat walks through a mossy forest carrying a blue bow saw..." |
| `Climate-Change_0-clip-4` | `Natural Disaster` | `Human Survival` | `8` | "A car sits in a thick orange haze caused by smoke in Yakutia. Smoke from wildfir..." |
| `Disaster-Compilations_10-clip-0` | `Natural Disaster` | `Human Survival` | `5` | "Large waves crash against a wooden staircase on a pier. White foam from the wave..." |
| `Disaster-Compilations_5-clip-0` | `Natural Disaster` | `Human Survival` | `0` | "A yellow bulldozer is parked on a dirt road next to a burning forest. A person i..." |
| `Gung-Ho-Vids_0-clip-2` | `Military` | `Geography` | `10` | "A person on a dirt bike rides across a flat, sandy desert landscape. The dirt bi..." |
| `Hamiltonville-Farm_8-clip-3` | `Agriculture` | `Human Survival` | `0` | "A man in a brown jacket and camo hat speaks to the camera outdoors. The man cont..." |
| `How-Farms-Work_5-clip-0` | `Agriculture` | `Human Survival` | `1` | "A person in a black jacket and jeans works on a wire fence. The person manipulat..." |
| `How-Farms-Work_9-manual` | `Agriculture` | `Human Survival` | `1` | "A person's hand touches a damaged red metal component on a piece of farm machine..." |
| `John-Suscovich_0-clip-1` | `Agriculture` | `Human Survival` | `1` | "A group of yellow chicks are huddled together on wood shavings. The chicks move ..." |
| `King-Kong-Amazon_11-clip-7` | `Human Survival` | `Geography` | `14` | "A wooden structure with a thatched roof sits amidst lush green jungle foliage. A..." |
| `Primal-Earth-Sounds_9-clip-26` | `Geography` | `Human Survival` | `8` | "A butterfly with orange, black, and white patterns on its wings rests on a clust..." |
| `Primitive-Technology_6-clip-3` | `Human Survival` | `Agriculture` | `0` | "A view of a small patch of land with some young green plants growing out of the ..." |
| `Sandboxx_7-clip-0` | `Military` | `Geography` | `5` | "Title card displays "TRAINING WEEK 9 BWT & LAND NAVIGATION" against a black back..." |
| `Sandboxx_9-clip-4` | `Military` | `Geography` | `2` | "An F-35 jet flies over salt flats with circular patterns. Two F-35 jets fly in f..." |
| `Survival-Skills-Primitive_3-clip-0` | `Human Survival` | `Agriculture` | `0` | "Two shirtless men walk through a muddy, shrubby area. The men continue walking t..." |
| `TK-Hinshaw_7-clip-0` | `Geography` | `Human Survival` | `9` | "A view from inside an airplane looking out at a scenic coastline. The camera foc..." |
| `TreadmillTV_3-clip-12` | `Geography` | `Human Survival` | `13` | "A person runs along a dirt path through a lush green forest. Sunlight filters th..." |
| `USA-Military-Channel_0-clip-6` | `Military` | `Human Survival` | `0` | "A black inflatable boat with several people on board speeds across the water. Tw..." |
| `Welker-Farms-Inc_3-clip-4` | `Agriculture` | `Human Survival` | `0` | "A man in a black jacket and grey beanie holds up a large fish he caught while ic..." |

