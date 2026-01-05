import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # type: ignore
from typing import TypedDict, Any
from dataclasses import dataclass
from llm.local_llm import MODELS, ModelConfig

@dataclass
class SurprisalResult:
    index: int
    caption: str
    loss: float
    perplexity: float
    avg_attn_distance: float
    sink_allocation: float = 0.0

"""
Intended to be used as a a priori assessment of how hard it is to reconstruct a caption,
 aggregated to score a whole video. This will allow us to bin videos (+specific masking) by their difficulty.
 We could bin by video surprisal, or by the surprisal score of the masked segment.
"""
class PriorSurpriseScorer:
    def __init__(self, model_key: str = "phi-3") -> None:
        if model_key not in MODELS:
            raise ValueError(f"Model {model_key} not found in registry.")
        
        self.config: ModelConfig = MODELS[model_key]
        print(f"Loading {model_key} ({self.config['id']})...")

        bnb_config: BitsAndBytesConfig | None = None
        if self.config["load_in_4bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            self.config["id"], 
            trust_remote_code=self.config["trust_remote_code"]
        )
        
        # Pad token is needed for batching, though we are doing single-doc here.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model: Any = AutoModelForCausalLM.from_pretrained(
            self.config["id"],
            quantization_config=bnb_config,
            dtype=torch.float16 if not self.config["load_in_4bit"] else None,
            device_map="auto",
            trust_remote_code=self.config["trust_remote_code"],
            attn_implementation="eager"
        )

    def calculate_whole_log_surprisal(self, captions: list[str], calc_attn_dist: bool = False) -> list[SurprisalResult]:
        """
        Calculates the surprisal (loss) for every caption in the list 
        using a SINGLE forward pass (The Matrix Trick).
        """
        
        # 1. Prepare the full document
        # We join them with newlines to simulate the log structure
        full_text = "\n".join(captions)
        
        # 2. Tokenize with Offset Mapping
        # offset_mapping gives us the (char_start, char_end) for every token
        inputs = self.tokenizer(
            full_text, 
            return_tensors="pt", 
            return_offsets_mapping=True,
            add_special_tokens=True 
        )
        
        input_ids = inputs.input_ids.to("cuda")
        offsets = inputs.offset_mapping[0] # Move to CPU list for processing
        
        # 3. The "Matrix" Forward Pass
        # We run the model once on the huge sequence
        # Note: output_attentions is memory heavy, so we gate it.
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=calc_attn_dist)
            logits = outputs.logits
            attentions = outputs.attentions if calc_attn_dist else None

        # 4a. Calculate Loss Per Token
        # Shift: Logits at [i] predict label at [i+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # reduction='none' gives us a vector of losses [Loss_Token1, Loss_Token2, ...]
        loss_fct = CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # Cleanup large model outputs immediately
        # Cleanup large model outputs immediately
        del logits
        
        # 4b. Calculate Attention Distance Per Token (Optional)
        token_avg_dist = None
        
        if calc_attn_dist and attentions:
            # We aggregate attention across all layers and heads to get a single matrix (seq, seq)
            # To avoid OOM, we do this iteratively (Accumulator Mean)
            # attentions is a tuple of (batch, heads, seq, seq)
            
            num_layers = len(attentions)
            seq_len = input_ids.size(1)
            indices = torch.arange(seq_len, device="cuda")
            dist_matrix = indices.unsqueeze(1) - indices.unsqueeze(0) # (seq, seq)

            # Iterative mean to avoid holding all layers in VRAM
            avg_attn = torch.zeros((seq_len, seq_len), device="cuda")
            for layer_attn in attentions:
                # Mean across batch and heads
                avg_attn += layer_attn.squeeze(0).mean(dim=0)
            
            avg_attn /= num_layers
            
            # --- HANDLE ATTENTION SINKS ---
            # Most LLMs use the 0-th token (BOS) as a sink for "unneeded" attention.
            token_sink_fraction = None
            
            if seq_len > 1:
                # Capture how much was allocated to BOS before we zero it
                token_sink_fraction = avg_attn[:, 0].clone() # (seq,)
                
                avg_attn[:, 0] = 0.0
                row_sums = avg_attn.sum(dim=1, keepdim=True)
                # Avoid division by zero
                row_sums[row_sums == 0] = 1.0 
                avg_attn /= row_sums
            # ------------------------------

            # Multiply attention weights by distance
            token_avg_dist = (avg_attn * dist_matrix).sum(dim=1) # (seq,)
            
            del attentions
            del avg_attn
            del dist_matrix
        
        # Now we have token_avg_dist[i], which is the avg distance looked back from position i.
        # Position i corresponds to the generating state for input_ids[i+1].
        # So we align it similar to loss.
        # token_avg_dist[i] is effectively the "Memory usage" to predict token[i+1].
        
        # 5. Map Tokens back to Sentences
        # We iterate through our original captions and find which tokens belong to them
        results = []
        current_char_pos = 0
        
        # token_losses has length N-1 compared to input_ids
        
        for i, caption in enumerate(captions):
            start_char = current_char_pos
            end_char = start_char + len(caption)
            
            # Find tokens that fall within this character range
            # We look at the offsets. Note: offsets are usually [start, end)
            
            caption_token_indices = []
            # We also need the indices for attention. 
            # Loss at index `k` corresponds to prediction of token `k+1`.
            # This prediction was made by the state at index `k`. 
            # So token_avg_dist[k] matches token_losses[k].
            
            for idx, (tok_start, tok_end) in enumerate(offsets):
                # We skip the first token (BOS) for alignment with shift_labels
                if idx == 0: continue
                
                # Check if this token is largely inside our caption
                # We use overlap logic
                if tok_end > start_char and tok_start < end_char:
                    # The loss for this token is at index idx-1 because of the shift
                    caption_token_indices.append(idx - 1)
            
            if caption_token_indices:
                # Extract the losses for this specific sentence
                segment_loss = token_losses[caption_token_indices].mean().item()
                
                # Extract attention distance
                segment_att_dist = -1.0 
                segment_sink_alloc = 0.0
                
                if token_avg_dist is not None:
                     segment_att_dist = token_avg_dist[caption_token_indices].mean().item()
                     if token_sink_fraction is not None:
                         segment_sink_alloc = token_sink_fraction[caption_token_indices].mean().item()
                
                results.append(SurprisalResult(
                    index=i,
                    caption=caption,
                    loss=segment_loss,
                    perplexity=torch.exp(torch.tensor(segment_loss)).item(),
                    avg_attn_distance=segment_att_dist,
                    sink_allocation=segment_sink_alloc
                ))
            else:
                # Handle edge case (empty lines or tokenizer weirdness)
                results.append(SurprisalResult(
                    index=i, 
                    caption=caption, 
                    loss=0.0, 
                    perplexity=0.0,
                    avg_attn_distance=0.0
                ))

            # Update char pos (+1 for the newline we added)
            current_char_pos = end_char + 1
            
        torch.cuda.empty_cache()
        return results

# --------------------------------------------------------------------------------
# EXAMPLE USAGE
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate a 5-second video log
    # Note line 3 is deliberately weird ("Explodes")
    captions = [
        "[00:01] A man walks into a kitchen.",
        "[00:02] He opens the refrigerator door.",
        "[00:03] He grabs a bottle of water.",
        "[00:04] The bottle suddenly explodes into fireworks.",
        "[00:05] He drinks the water."
    ]
    
    scorer = PriorSurpriseScorer(model_key="mistral-v0.3")
    
    print("Calculating scores (One Pass)...")
    scores = scorer.calculate_whole_log_surprisal(captions)
    
    print("\n--- Outlier Detection Report ---")
    for item in scores:
        status = "OK"
        if item.loss > 4.0: status = "WEIRD/OUTLIER" # Threshold depends on model
        
        print(f"Time {item.caption[:7]} | Loss: {item.loss:.2f} | Status: {status}")
        print(f"   -> Text: {item.caption[8:]}")