import torch
from transformers import LogitsProcessor

class HeterogeneousLogitsProcessor(LogitsProcessor):
    """
    Applies DIFFERENT parameters to each row in a batch.
    Combines Temperature and Repetition Penalty.
    """
    def __init__(self, temperatures: list[float], penalties: list[float], device: str = "cuda"):
        # Temp: [Batch, 1]
        self.temperatures = torch.tensor(temperatures, device=device, dtype=torch.float16).unsqueeze(1)
        # Penalty: [Batch] - kept 1D for loop efficiency
        self.penalties = torch.tensor(penalties, device=device, dtype=torch.float16)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1. Apply Repetition Penalty (Per Row)
        # This is harder to fully vectorize without a massive scatter mask, 
        # so a tight loop over the batch (size 3-5) is actually faster/safer.
        
        # We iterate over the batch dimension
        for i in range(input_ids.shape[0]):
            penalty = self.penalties[i]
            if penalty == 1.0:
                continue # No penalty for this row
            
            # Find unique tokens used in this sequence so far
            unique_ids = torch.unique(input_ids[i])
            
            # Apply penalty to those indices in the scores matrix
            # Logic: If score < 0, multiply. If score > 0, divide.
            # (This pushes logits towards 0, lowering probability)
            
            # Grab the logits for the used tokens
            row_logits = scores[i, unique_ids]
            
            # Apply standard penalty logic
            # where logit < 0: logit * penalty
            # where logit > 0: logit / penalty
            # Note: HF implementation is: logit = torch.where(logit < 0, logit * penalty, logit / penalty)
            
            updated_logits = torch.where(
                row_logits < 0, 
                row_logits * penalty, 
                row_logits / penalty
            )
            
            # Write back
            scores[i, unique_ids] = updated_logits

        # 2. Apply Temperature (Vectorized)
        # We do this LAST to ensure the penalty is scaled correctly
        return scores / self.temperatures

# --------------------------------------------------------------------------------
# USAGE EXAMPLE WITH EXPERIMENT RUNNER
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Define your Spectrum Configurations
    # Format: (Temperature, Repetition_Penalty)
    configs = [
        (0.1, 1.0),  # The "Logician" (Greedy, allow repetition)
        (1.0, 1.1),  # The "Standard" (Natural flow)
        (1.5, 1.2),  # The "Explorer" (Creative, force new words)
        (0.5, 2.0)   # The "Strict Editor" (Logic, but strictly ban repetition)
    ]
    
    # Extract lists for the processor
    temps = [c[0] for c in configs]
    pens = [c[1] for c in configs]
    
    # ... In your Generate function ...
    # processor = HeterogeneousLogitsProcessor(temperatures=temps, penalties=pens, device="cuda")
    
    print("Configurations prepared:")
    for i, c in enumerate(configs):
        print(f"Batch Row {i}: Temp={c[0]}, RepPenalty={c[1]}")