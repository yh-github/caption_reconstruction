# Dataset Taxonomy & Content Audit

**Critical Finding**: significant content duplication. The dataset contains 95 files, but many appear to be duplicates of the same 5-6 base videos, widely mislabeled with unrelated filenames (e.g., `Climate-Change` files containing VR gaming content).

## Cluster 1: The "VR Headset" Video (8 duplicates)
**Content**: Person playing VR game, black headset, "The Future is Virtual".
**Filenames**:
- `Weathershot_7-clip-0.json` (Used in our T=1.5 analysis)
- `Climate-Change_2-clip-4.json`
- `Climate-Change_6-clip-7.json`
- `Gung-Ho-Vids_8-clip-0.json`
- `King-Kong-Amazon_5-clip-14.json`
- `Primal-Earth-Sounds_0-clip-44.json`
- `Survival-Instinct_11-clip-10.json`
- `Survival-Instinct_8-clip-4.json`

## Cluster 2: The "Freshly Baked Bread" Video (3 duplicates)
**Content**: Hands holding round loaf of bread, distinct crust.
**Filenames**:
- `Climate-Change_7-clip-3.json`
- `Gung-Ho-Vids_5-clip-2.json`
- `Survival-Skills-Primitive_3-clip-0.json`

## Cluster 3: The "Black Gridded Fabric" Video (4 duplicates)
**Content**: Hands smoothing black fabric with white grid, chalk pencil.
**Filenames**:
- `Climate-Change_0-clip-4.json`
- `Nick-Gaillard_1-clip-2.json`
- `Survival-Instinct_2-clip-1.json`
- `Survival-Instinct_7-clip-8.json`

## Cluster 4: The "Tearing Drawing" Video (3 duplicates)
**Content**: Hands tearing a paper drawing in half.
**Filenames**:
- `Hamiltonville-Farm_6-clip-18.json`
- `Survival-Instinct_9-clip-2.json`
- `Weathershot_3-clip-1.json`

## Cluster 5: The "Firefighter / Gas Mask" Video (1 instance seen so far)
- `AiirSource-Military_1-clip-0.json`

## Cluster 6: The "Explosion" Video (1 instance seen so far)
- `Army-military-2018_8-clip-73.json`

*(List continues with unique or less duplicated files...)*

## Implications for Analysis
The "Weathershot" result (T=1.5 winning) is highly suspect because this specific "VR" content appears in **8 different files** in the dataset (nearly 10% of the total 95 files). 
- If the model was trained/finetuned on this dataset, it saw this specific VR video 8 times more often than others.
- The high "uniqueness" of the T=1.5 reconstruction for `Weathershot` might be reacting to this over-represented concept, or conversely, the T=0.1 "generic" caption might be failing because the model is confused by the 8 different labels for this same video.
