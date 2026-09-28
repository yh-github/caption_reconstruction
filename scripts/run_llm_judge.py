import os
import json
import time
from tqdm import tqdm
from google import genai
from google.genai import types
from google.genai.errors import APIError

def build_prompt_a(record):
    domain = record.get("domain", "Unknown")
    question = record["question"]
    gt_answer = record["ground_truth_answer"]
    evidence_span = f"[{record['evidence_span'][0]}s - {record['evidence_span'][1]}s]"
    
    pre_gap = "\n".join([f"[{c['timestamp']}] {c['caption']}" for c in record["boundary_context"]["pre_gap"]])
    post_gap = "\n".join([f"[{c['timestamp']}] {c['caption']}" for c in record["boundary_context"]["post_gap"]])
    
    recon_data = record["gap_conditions"].get("reconstructed")
    if recon_data:
        recon = "\n".join([f"[{c['timestamp']}] {c['caption']}" for c in recon_data])
    else:
        recon = "NO RECONSTRUCTION AVAILABLE."
    
    sys_prompt = "You are a rigorous scientific evaluator auditing whether an AI temporal in-filling model successfully reconstructed missing video caption information."
    
    user_prompt = f"""You are evaluating a temporal reconstruction for a video where a segment of captions was masked out and in-filled by a language model.

### Context:
- Domain: {domain}
- Question: {question}
- Ground-Truth Answer(s): {gt_answer}
- Evidence Interval: seconds {evidence_span}

### Unmasked Surrounding Context:
- Pre-gap (seconds immediately preceding the masked window):
{pre_gap}
- Post-gap (seconds immediately following the masked window):
{post_gap}

### Reconstructed Captions (in-filled during the masked window):
{recon}

### Your Evaluation Task:
1. Boundary Leakage: Was the factual answer to the question ALREADY stated or plainly obvious in the unmasked pre-gap or post-gap captions?
2. Factual Recovery Category: Looking strictly at the reconstructed captions, how did the model perform relative to the ground-truth answer?
   - "EXACT_FACT": The reconstruction explicitly mentions the specific entity, action, or attribute required by the ground-truth answer (e.g., "blueberries").
   - "SEMANTIC_TYPE": The reconstruction mentions the correct general category or activity, but misses the specific factual detail (e.g., mentions "picking berries" or "harvesting fruit" instead of "blueberries").
   - "PLAUSIBLE_HALLUCINATION": The reconstruction describes a coherent event fitting the scene, but invents an incorrect factual detail (e.g., mentions "strawberries" or "pruning branches").
   - "MISSED": The reconstruction describes something completely unrelated to the true event (e.g., mentions trees/scenery instead of the specific action).

Respond strictly in valid JSON format:
{{
  "boundary_leakage": true | false,
  "leakage_explanation": "<brief explanation if true, else null>",
  "factual_recovery_category": "EXACT_FACT" | "SEMANTIC_TYPE" | "PLAUSIBLE_HALLUCINATION" | "MISSED",
  "explanation": "<brief rationale comparing reconstructed text against ground-truth answer>"
}}"""
    return sys_prompt, user_prompt

def build_prompt_b(record, condition):
    question = record["question"]
    candidates = record["top_k_candidates"].get(condition)
    if not candidates:
        return None, None
        
    top_k_text = "\n".join([f"[Rank {c['rank']}, Timestamp: {c['timestamp']}] {c['caption']}" for c in candidates])
    
    sys_prompt = "You are an objective question-answering evaluator. You must answer questions based STRICTLY on the provided retrieved text context. Do not use outside knowledge."
    
    user_prompt = f"""Answer the question based SOLELY on the retrieved video captions below.
If the retrieved captions do not contain sufficient evidence to answer the question, you MUST respond with "UNANSWERABLE".

Question: {question}

Retrieved Captions:
{top_k_text}

Respond strictly in valid JSON format:
{{
  "can_answer": true | false,
  "predicted_answer": "<your concise answer based only on context, or 'UNANSWERABLE'>",
  "supporting_timestamp": "<timestamp/second where evidence was found, or null>"
}}"""
    return sys_prompt, user_prompt

def call_gemini(client, sys_prompt, user_prompt):
    if not user_prompt:
        return None
        
    max_retries = 5
    for attempt in range(max_retries):
        try:
            time.sleep(4)  # Rate limit mitigation for free tier
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            return json.loads(response.text)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"\\nRate limit hit. Sleeping for 30 seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(30)
            elif attempt < max_retries - 1:
                print(f"\\nError: {e}. Retrying in 10s...")
                time.sleep(10)
            else:
                print(f"\\nFatal error calling Gemini: {e}")
                return None

def main():
    client = genai.Client()
    input_file = "results/qa_inspection/qa_inspection_dev_siglip.jsonl"
    output_file = "results/qa_inspection/judge_results_dev_siglip.jsonl"
    
    # Initialize output file if starting fresh, else resume
    evaluated_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        evaluated_ids.add(json.loads(line)["assignment_id"])
                    except:
                        pass
                    
    print(f"Found {len(evaluated_ids)} already evaluated records.")
    
    with open(input_file, "r") as f:
        records = [json.loads(line) for line in f]
        
    for record in tqdm(records, desc="Evaluating"):
        assignment_id = record["assignment_id"]
        if assignment_id in evaluated_ids:
            continue
            
        sys_a, user_a = build_prompt_a(record)
        res_a = call_gemini(client, sys_a, user_a) if record["recon_available"] else None
        
        sys_b_masked, user_b_masked = build_prompt_b(record, "masked")
        res_b_masked = call_gemini(client, sys_b_masked, user_b_masked)
        
        sys_b_recon, user_b_recon = build_prompt_b(record, "reconstructed")
        res_b_recon = call_gemini(client, sys_b_recon, user_b_recon) if record["recon_available"] else None
        
        eval_record = {
            "assignment_id": assignment_id,
            "video_id": record["video_id"],
            "question": record["question"],
            "ground_truth_answer": record["ground_truth_answer"],
            "recon_available": record["recon_available"],
            "intrinsic_recovery": res_a,
            "downstream_masked": res_b_masked,
            "downstream_reconstructed": res_b_recon
        }
        
        # Write incrementally
        mode = "a" if os.path.exists(output_file) else "w"
        with open(output_file, mode) as f:
            f.write(json.dumps(eval_record) + "\\n")
            
    print(f"\\nEvaluation complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
