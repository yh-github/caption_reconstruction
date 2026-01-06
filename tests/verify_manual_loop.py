
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm.local_llm import MODELS

def verify_manual_loop():
    print("Loading Phi-3...")
    model_key = "phi-3"
    config = MODELS[model_key]
    
    # Load model exactly as in PriorSurpriseScorer
    model = AutoModelForCausalLM.from_pretrained(
        config["id"],
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["id"])
    
    text = "The quick brown fox jumps over the lazy dog. The forest was green and the lake was blue."
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    
    print(f"Input shape: {input_ids.shape}")
    
    # 1. Standard Forward
    print("Running Standard Forward...")
    with torch.no_grad():
        out_std = model(input_ids)
        logits_std = out_std.logits
        
    # 2. Manual Loop (Replicating PriorSurpriseScorer logic)
    print("Running Manual Loop...")
    with torch.no_grad():
        base_model = model.model
        hidden_states = base_model.embed_tokens(input_ids)
        
        layers = base_model.layers
        for i, layer in enumerate(layers):
            is_last_layer = (i == len(layers) - 1)
            # LOGIC FROM CODE:
            # layer_outputs = layer(hidden_states, output_attentions=is_last_layer, position_ids=None)
            layer_outputs = layer(
                hidden_states, 
                output_attentions=is_last_layer,
                position_ids=None 
            )
            hidden_states = layer_outputs[0]
            
        hidden_states = base_model.norm(hidden_states)
        logits_manual = model.lm_head(hidden_states)
        
    # 3. Compare
    print("Comparing Logits...")
    diff = (logits_std - logits_manual).abs().max().item()
    print(f"Max Difference: {diff}")
    
    if diff > 1e-3:
        print("FAIL: Logits diverge significantly! Manual loop is unsafe (likely missing causal mask).")
    else:
        print("SUCCESS: Logits match.")

if __name__ == "__main__":
    verify_manual_loop()
