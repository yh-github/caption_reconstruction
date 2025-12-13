# Data

The data is 100 videos, each 60 seconds long.  
the score is cosine similarity (which is 1-distance), between \-1 and 1\.  
Segment length is 1 second  
Videos are sampled at 1 FPS

The score is the mean of scores of all videos  
The score of a video is the minimum of the its cosine similarity over all reconstructed vectors

Score(method, maksing) \= MEAN(MIN(CosineSimilarity(original\_vector, reconstructed\_vector) ) )

The Z-Score normalized version is:

ScoreZ(method, masking) \= MEAN(MIN(Z,(CosineSimilarity(original\_vector, reconstructed\_vector) ) ) )

The global mean and std (,) are calculated over all maskings per method.

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
