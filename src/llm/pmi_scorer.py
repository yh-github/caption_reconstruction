import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # type: ignore
from typing import TypedDict, Any

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
from llm.local_llm import MODELS, ModelConfig

class PMIScorer:
    def __init__(self, model_key: str = "mistral-v0.3") -> None:
        if model_key not in MODELS:
            raise ValueError(f"Model {model_key} not found in registry.")
        
        self.config: ModelConfig = MODELS[model_key]
        print(f"Loading {model_key} ({self.config['id']})...")
        
        # Load 4-bit to fit on T4
        bnb_config: BitsAndBytesConfig | None = None
        if self.config["load_in_4bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer: Any = AutoTokenizer.from_pretrained(self.config["id"], trust_remote_code=self.config["trust_remote_code"])
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            self.config["id"],
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if not self.config["load_in_4bit"] else None,
            device_map="auto",
            trust_remote_code=self.config["trust_remote_code"]
        )

    def _get_loss(self, prompt_text: str, target_text: str) -> float:
        """
        Helper to calculate the loss of the target_text given the prompt_text.
        """
        # Tokenize separately to find boundaries
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids.to("cuda")
        target_ids = self.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        
        # Concatenate
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Calculate loss only on target
        start_idx = prompt_ids.shape[1] - 1
        end_idx = input_ids.shape[1] - 1
        
        relevant_logits = logits[0, start_idx:end_idx, :]
        relevant_labels = target_ids[0, :]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(relevant_logits, relevant_labels)
        
        return loss.item()

    def calculate_informativeness(self, context_before: str, context_after: str, target_line: str) -> dict[str, float]:
        """
        Calculates how much the context HELPS the model predict the target.
        """
        
        # 1. Score with FULL Context (Past + Future)
        # We use the FIM-style prompt to allow the model to see the future legally
        prompt_context = f"""
        [INST] Analyze the sequence.
        Context Before: {context_before}
        Context After: {context_after}
        What happens in between? [/INST]
        Missing Line: """
        
        loss_context = self._get_loss(prompt_context, target_line)
        
        # 2. Score BLIND (No Context)
        # We just ask it to evaluate the sentence likelihood in a vacuum
        prompt_blind = f"""
        [INST] Write a generic video caption. [/INST]
        Caption: """
        
        loss_blind = self._get_loss(prompt_blind, target_line)
        
        # 3. Calculate PMI (Pointwise Mutual Information) approx
        # PMI = log(P(x|y)/P(x))
        # Since Loss = -log(P), then PMI = Loss_Blind - Loss_Context
        
        pmi_score = loss_blind - loss_context
        
        return {
            "caption": target_line,
            "loss_context": loss_context, # Lower is better
            "loss_blind": loss_blind,     # Lower is better
            "pmi_score": pmi_score        # HIGHER is better (Context was useful)
        }

# --------------------------------------------------------------------------------
# EXAMPLE USAGE
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    scorer = PMIScorer()
    
    context_before = "[00:10] The batter hits the ball high."
    context_after = "[00:12] The crowd cheers wildly."
    
    # Case A: Informative Fit (Specific to Baseball)
    # The context (batter, crowd) makes "homerun" very likely.
    target_good = "[00:11] The ball flies over the fence for a homerun."
    
    # Case B: Uninformative / Generic (Could happen anywhere)
    # This sentence is easy to predict blindly, so context doesn't help much.
    target_generic = "[00:11] It is a nice day."
    
    # Case C: Out of Context (Confusing)
    # The context actually makes this LESS likely than blind chance.
    target_bad = "[00:11] The batter sits down and eats a sandwich."

    print("\n--- Informativeness Report (Higher PMI = Better Fit) ---")
    
    res_a = scorer.calculate_informativeness(context_before, context_after, target_good)
    print(f"\nA: {target_good}")
    print(f"   Loss w/ Context: {res_a['loss_context']:.2f} (Low = Predictable)")
    print(f"   Loss Blind:      {res_a['loss_blind']:.2f}")
    print(f"   PMI Score:       {res_a['pmi_score']:.2f} (High Positive -> Context Helped!)")

    res_b = scorer.calculate_informativeness(context_before, context_after, target_generic)
    print(f"\nB: {target_generic}")
    print(f"   Loss w/ Context: {res_b['loss_context']:.2f}")
    print(f"   Loss Blind:      {res_b['loss_blind']:.2f}")
    print(f"   PMI Score:       {res_b['pmi_score']:.2f} (Near Zero -> Context didn't matter)")

    res_c = scorer.calculate_informativeness(context_before, context_after, target_bad)
    print(f"\nC: {target_bad}")
    print(f"   Loss w/ Context: {res_c['loss_context']:.2f} (High = Shocking)")
    print(f"   PMI Score:       {res_c['pmi_score']:.2f} (Negative -> Context contradicts target)")