# Future Optimization: Heterogeneous Batch Processing for Parameter Sweep

**Status**: Proposed / Deferred
**Goal**: Speed up hyperparameter grid search (Temperature, Repetition Penalty) by 4x-8x by running multiple configurations in parallel for a single video.

## Core Concept
Instead of running the pipeline $N$ times for $N$ different `(temp, penalty)` configurations, we run it **once** with a batch size of $N$. Each row in the batch corresponds to one configuration.

## Required Components

### 1. `HeterogeneousLogitsProcessor` (Already implemented in `src/new_code/logits_processor.py`)
This custom HuggingFace `LogitsProcessor` allows applying a vector of temperatures and penalties to a batch of logits.
*   Input: `temperatures: [B, 1]`, `penalties: [B]`
*   Logic: Applies $Temp_i$ and $Penalty_i$ to row $i$ of the logits tensor.

### 2. `MultiConfigState` Strategy
Since text reconstruction is iterative (gap $i$ depends on gap $i-1$), the different configurations will produce different texts, causing the "context" for each row to diverge.
We need a new Strategy class (e.g., `BatchGridSearchStrategy`) that maintains **multiple parallel video states**.

**Pseudocode Logic:**
```python
class BatchGridSearchStrategy:
    def __init__(self, configs: list[dict]):
        self.configs = configs
        self.batch_size = len(configs)
        # Processor initialized with vectors from configs
        self.logits_processor = HeterogeneousLogitsProcessor(...)

    def reconstruct(self, masked_video):
        # 1. Initialize N copies of the video
        video_states = [masked_video.copy() for _ in range(self.batch_size)]
        
        # 2. Iterate through clips (assuming synchronized masking)
        for clip_idx in range(len(masked_video.clips)):
            if not is_masked(clip_idx):
                continue
                
            # 3. Prepare Batch Inputs
            prompts = []
            for i in range(self.batch_size):
                # Each state has its own context history now!
                context = build_context(video_states[i], clip_idx)
                prompts.append(context)
                
            # 4. Run Batch Inference
            # The infiller must support passing a custom logits_processor
            generated_texts = self.infiller.generate_batch(
                prompts, 
                logits_processor=self.logits_processor
            )
            
            # 5. Update States
            for i in range(self.batch_size):
                video_states[i].clips[clip_idx].caption = generated_texts[i]
                
        # 6. Return List of Results
        return [Reconstructed(v) for v in video_states]
```

## Implementation Steps (When Activated)
1.  **Refactor `ClozeInfiller`**: Add `generate_batch` method that accepts a list of prompts and an optional `LogitsProcessorList`.
2.  **Create Strategy**: Implement `BatchGridSearchStrategy` in `text_reconstruction.py`.
3.  **Update Pipeline**: Modify `TextReconstructionStrategyBuilder` or `ExperimentPipeline` to handle a "Grid Search" mode where it returns multiple result objects for a single input video.
