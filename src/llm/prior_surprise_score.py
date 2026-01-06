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
    n_tokens: int = 0
    n_words: int = 0
    n_chars: int = 0
    caption_affinity: list[float] | None = None # Attention mass distribution over all captions

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
        self.model.eval()

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
        seq_len = input_ids.size(1)

        # 3. Memory-Optimized Manual Forward Pass
        # We process layers one by one to avoid OOM by keeping only the last layer's attention
        with torch.no_grad():
            if not calc_attn_dist:
                # Fast Path: Standard forward (no attentions)
                outputs = self.model(input_ids, output_attentions=False)
                logits = outputs.logits
                attentions = None
            else:
                # Manual Path: Iterate layers
                # A. Create Causal Mask (Triangular -inf)
                # Shape: (1, 1, seq_len, seq_len)
                attention_mask = torch.full(
                    (1, 1, seq_len, seq_len), 
                    float("-inf"), 
                    device=input_ids.device, 
                    dtype=self.model.dtype
                )
                attention_mask = torch.triu(attention_mask, diagonal=1)
                
                # B. Embeddings
                hidden_states = self.model.model.embed_tokens(input_ids)
                
                # C. Layers
                attentions = None
                for i, layer in enumerate(self.model.model.layers):
                    is_last = (i == len(self.model.model.layers) - 1)
                    
                    # Pass mask and position_ids (None usually infers seq_len)
                    layer_out = layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        position_ids=None,
                        output_attentions=is_last
                    )
                    
                    hidden_states = layer_out[0]
                    if is_last:
                        # Capture Last Layer Attention
                        # shape: (1, heads, seq, seq)
                        attentions = layer_out[1] 
                
                # D. Final Norm & Head
                hidden_states = self.model.model.norm(hidden_states)
                logits = self.model.lm_head(hidden_states)

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
        del logits
        if calc_attn_dist:
             del attention_mask
             del hidden_states
        
        # 4b. Calculate Attention Distance Per Token (Optional)
        token_avg_dist = None
        token_sink_fraction = None
        
        if calc_attn_dist and attentions is not None:
            # attentions is (1, heads, seq, seq) from Last Layer
            # Collapse heads -> (seq, seq)
            avg_attn = attentions.squeeze(0).mean(dim=0) # (seq, seq)
            
            indices = torch.arange(seq_len, device="cuda")
            dist_matrix = indices.unsqueeze(1) - indices.unsqueeze(0) # (seq, seq)

            # --- HANDLE ATTENTION SINKS ---
            # Most LLMs use the 0-th token (BOS) as a sink for "unneeded" attention.
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
        # 5. Map Tokens back to Sentences & Pre-compute Stats
        # We need to know which tokens belong to which caption to aggregate attention
        caption_token_map = {} # caption_idx -> list[token_indices]
        
        current_char_pos = 0
        for i, caption in enumerate(captions):
            start_char = current_char_pos
            end_char = start_char + len(caption)
            
            indices = []
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if idx == 0: continue
                # Overlap logic
                if tok_end > start_char and tok_start < end_char:
                    indices.append(idx - 1) # Shifted
            
            caption_token_map[i] = indices
            current_char_pos = end_char + 1

        # 6. Compute Caption Affinity (if available)
        caption_affinities = None # Dict[cap_idx, List[float]] - mass per target caption
        
        if calc_attn_dist and attentions:
            # avg_attn is (seq, seq) [renormalized, no sink]
            # We want to know for Source Caption S, how much mass falls on Target Caption T?
            caption_affinities = {}
            
            # Pre-compute target masks to avoid N^2 inner loop overhead on GPU?
            # Actually simpler: For each source, get the mean attention row.
            # Then sum up columns belonging to each target.
            
            # We can do this efficiently on CPU since N_caps is small (~100)
            avg_attn_cpu = avg_attn.float().cpu() # Move to CPU for complex reduction
            
            for s_idx in range(len(captions)):
                s_tokens = caption_token_map[s_idx]
                if not s_tokens:
                    caption_affinities[s_idx] = [0.0] * len(captions)
                    continue
                    
                # Get average attention profile for this caption's tokens
                # Shape: (len(s_tokens), seq_len) -> mean(0) -> (seq_len,)
                # Note: These are rows in the matrix
                source_profile = avg_attn_cpu[s_tokens, :].mean(dim=0)
                
                # Now bucket this profile into target captions
                row_distribution = []
                for t_idx in range(len(captions)):
                    t_tokens = caption_token_map[t_idx]
                    if not t_tokens:
                        row_distribution.append(0.0)
                    else:
                        # Sum mass falling on target tokens
                        mass = source_profile[t_tokens].sum().item()
                        row_distribution.append(mass)
                
                caption_affinities[s_idx] = row_distribution

            del avg_attn_cpu # Explicit cleanup of CPU copy
            
            # Cleanup GPU tensors
            del attentions
            del avg_attn
            del dist_matrix

        # 7. Build Results
        results = []
        for i, caption in enumerate(captions):
            indices = caption_token_map[i]
            
            if indices:
                segment_loss = token_losses[indices].mean().item()
                
                # Default Metrics
                segment_att_dist = -1.0
                segment_sink_alloc = 0.0
                segment_affinity = None
                
                if token_avg_dist is not None:
                    segment_att_dist = token_avg_dist[indices].mean().item()
                    
                if token_sink_fraction is not None:
                    segment_sink_alloc = token_sink_fraction[indices].mean().item()
                    
                if caption_affinities:
                    segment_affinity = caption_affinities.get(i)

                results.append(SurprisalResult(
                    index=i,
                    caption=caption,
                    loss=segment_loss,
                    perplexity=torch.exp(torch.tensor(segment_loss)).item(),
                    avg_attn_distance=segment_att_dist,
                    sink_allocation=segment_sink_alloc,
                    n_tokens=len(indices),
                    n_words=len(caption.split()),
                    n_chars=len(caption),
                    caption_affinity=segment_affinity
                ))
            else:
                results.append(SurprisalResult(
                    index=i, 
                    caption=caption, 
                    loss=0.0, 
                    perplexity=0.0,
                    avg_attn_distance=0.0,
                    n_tokens=0,
                    n_words=len(caption.split()),
                    n_chars=len(caption),
                    caption_affinity=None
                ))

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