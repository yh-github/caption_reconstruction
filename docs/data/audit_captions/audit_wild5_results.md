# Dataset Audit Report: `wild5`

**Date**: 2026-08-10  
**Target Directory**: `datasets/wildQA/captions__wild5`  
**Total Clips Audited**: `109`  

## Executive Summary

| Metric | Result | Health Status |
| :--- | :--- | :--- |
| **Suspected Duplicate Clusters** | `0` | 🟢 Clean (0 duplicates) |
| **Ad / Commercial Boilerplate Hits** | `6` | 🟡 Warning |
| **Repetitive / Loop Captions** | `1` | 🟡 Warning |
| **Domain Keyword Alignment** | `84/109 (77.1%)` | 🟢 Strong (>70%) |

---

## 1. Duplication & Similarity Analysis

Found **7** pairs of videos with Cosine Similarity > 0.70:

| Video 1 | Video 2 | Cosine Sim | 3-Gram Jaccard | Status |
| :--- | :--- | :--- | :--- | :--- |
| `Dan-Robinson_0-clip-2` | `Dan-Robinson_10-clip-0` | `0.807` | `0.052` | 🟡 Similar Content |
| `Army-military-2018_7-clip-2` | `Military-Archive_11-clip-0` | `0.792` | `0.064` | 🟡 Similar Content |
| `Dan-Robinson_10-clip-0` | `Disaster-Compilations_4-clip-2` | `0.785` | `0.031` | 🟡 Similar Content |
| `Dan-Robinson_0-clip-2` | `Disaster-Compilations_4-clip-2` | `0.776` | `0.044` | 🟡 Similar Content |
| `Dan-Robinson_10-clip-0` | `Dan-Robinson_7-clip-1` | `0.752` | `0.050` | 🟡 Similar Content |
| `Dan-Robinson_7-clip-1` | `Disaster-Compilations_4-clip-2` | `0.727` | `0.037` | 🟡 Similar Content |
| `Dan-Robinson_0-clip-2` | `Dan-Robinson_7-clip-1` | `0.721` | `0.044` | 🟡 Similar Content |


## 2. Ad & Commercial Boilerplate Screening

Found **6** files containing potential ad/boilerplate keywords:

* **`Army-military-2018_10-clip-0`**: Matched Known sponsor brand (1x)
  * *Snippet*: "Bright lights and flares are visible in the distance over a dark desert horizon. Multiple projectiles leave glowing trails across the night sky as the..."

* **`Army-military-2018_6-clip-1`**: Matched Suspect VR headset ad content (1x)
  * *Snippet*: "A soldier sits inside a helicopter, looking out the open door at the landscape. The soldier continues to observe the terrain as the helicopter flies. ..."

* **`Disaster-Compilations_6-clip-2`**: Matched Known sponsor brand (1x)
  * *Snippet*: "A group of people are huddled together inside a building, appearing distressed. Several individuals are seen sitting closely on the floor of an indoor..."

* **`John-Suscovich_4-clip-0`**: Matched Known sponsor brand (1x)
  * *Snippet*: "Three mobile chicken coops are positioned in a row in a grassy field. Chickens are visible through the wire mesh of the coops, which have white covers..."

* **`Military-Archive_6-clip-0`**: Matched Suspect VR headset ad content (1x)
  * *Snippet*: "Two pilots in flight suits walk away from the B-2 Spirit bomber. The pilots carry their flight bags across the airfield tarmac. One pilot adjusts his ..."

* **`Millennial-Farmer_10-clip-0`**: Matched Sponsor disclaimer (2x)
  * *Snippet*: "A man in a tan jacket speaks to the camera in front of a farm building. The man continues speaking as he stands in the farmyard. He points toward a la..."

## 3. Domain Alignment Analysis

Comparison of keyword profile predictions against ground-truth dataset domains:

### Mismatched Domain Outliers (25 files)

| Video ID | Ground Truth | Predicted Domain | GT Domain Score | Snippet |
| :--- | :--- | :--- | :--- | :--- |
| `AiirSource-Military_10-clip-0` | `Military` | `Human Survival` | `0` | "A title card displays the website "AiirSOURCE.com" on a white background. The "A..." |
| `AiirSource-Military_2-clip-0` | `Military` | `Human Survival` | `0` | "A crew member's silhouette watches a hovercraft approach a ship through a misty ..." |
| `BC-Bushcraft_1-clip-8` | `Human Survival` | `Agriculture` | `4` | "A man enters a forest clearing carrying a long wooden log on his shoulder. He wa..." |
| `BC-Bushcraft_5-clip-10` | `Human Survival` | `Geography` | `0` | "A man in a flannel shirt is in a forest, leaning over a fallen tree. The man use..." |
| `BC-Bushcraft_6-clip-0` | `Human Survival` | `Geography` | `7` | "A circular logo with a wolf silhouette and the letters B C appears on a black ba..." |
| `Chad-Zuber_1-clip-10` | `Human Survival` | `UNCLASSIFIED` | `0` | "A person mixes mud with their hands in a container. The person continues kneadin..." |
| `Chad-Zuber_2-clip-4` | `Human Survival` | `Geography` | `5` | "A man with long hair sits in a rocky outdoor setting. The man looks into the dis..." |
| `Disaster-Compilations_2-clip-2` | `Natural Disaster` | `Human Survival` | `6` | "The video begins with a black screen. A car drives past a building with a massiv..." |
| `Disaster-Compilations_3-clip-2` | `Natural Disaster` | `Human Survival` | `17` | "A black screen quickly transitions into a view of an ash-covered street from ins..." |
| `Gung-Ho-Vids_9-clip-0` | `Military` | `Geography` | `5` | "The video starts with a black screen. The screen remains black. A gray C-130 mil..." |
| `Hamiltonville-Farm_0-clip-4` | `Agriculture` | `Natural Disaster` | `13` | "A man in a blue t-shirt speaks to the camera outdoors. The camera pans down to s..." |
| `Hamiltonville-Farm_10-clip-8` | `Agriculture` | `Human Survival` | `0` | "A person wearing gloves holds a speed square against a wooden plank. The person ..." |
| `Hamiltonville-Farm_7-clip-15` | `Agriculture` | `Geography` | `1` | "A person's gloved hand points towards a tree in a forest filled with fallen bran..." |
| `Hamiltonville-Farm_9-manual` | `Agriculture` | `Human Survival` | `1` | "A hand reaches for a brown cap and sunglasses on a side table. The hand picks up..." |
| `Joe-Robinet_11-clip-2` | `Human Survival` | `Agriculture` | `10` | "A hand reaches out towards dried plant heads in a grassy field. The hand picks a..." |
| `John-Suscovich_1-clip-0` | `Agriculture` | `Human Survival` | `4` | "A man in a plaid shirt kneels in a chicken brooder with many small chicks. A man..." |
| `John-Suscovich_6-clip-1` | `Agriculture` | `Human Survival` | `0` | "A man in a white t-shirt and cap is kneeling in a chicken coop, taking yellow ch..." |
| `John-Suscovich_9-manual` | `Agriculture` | `Human Survival` | `0` | "A white freezer door features a sign labeled "Pork in Freezer" with a pig drawin..." |
| `King-Kong-Amazon_0-clip-5` | `Human Survival` | `UNCLASSIFIED` | `0` | "The person picks up a piece of dark meat from the green leaves. The meat is care..." |
| `King-Kong-Amazon_10-clip-13` | `Human Survival` | `Agriculture` | `3` | "A person is digging into the dark soil with their hands in a forest environment...." |
| `King-Kong-Amazon_6-clip-3` | `Human Survival` | `Agriculture` | `7` | "A person uses a stick to pound something in a small bamboo container on the grou..." |
| `King-Kong-Amazon_7-clip-3` | `Human Survival` | `Geography` | `12` | "A man scrapes the rim of a small bamboo cup with a knife. The man continues to s..." |
| `Military-Archive_1-clip-3` | `Military` | `Geography` | `1` | "A tan military vehicle moves across a dirt area. The military vehicle starts to ..." |
| `MilitaryNotes_3-manual` | `Military` | `Human Survival` | `0` | "Two military personnel prepare to attach a large piece of equipment to a hoist i..." |
| `MilitaryNotes_4-clip-0` | `Military` | `Human Survival` | `1` | "Title card introducing Corporal Benjamin Pitre at Camp Pendleton, California, de..." |

