import argparse
import pandas as pd
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from llm.local_llm import LocalLLM

JUDGE_PROMPT = """You are an expert evaluator assessing how well an AI model reconstructed missing video events.

### Ground Truth (What actually happened):
{ground_truth}

### Candidate Reconstruction (The model's guess):
{candidate}

### Task:
Evaluate the semantic alignment of the Candidate against the Ground Truth on a scale of 0 to 2.
- Score 0 (Leads Astray): The candidate hallucinates significant actions or entities that did not occur in the Ground Truth. It actively misrepresents reality.
- Score 1 (Safe but Stagnant): The candidate captures static/ongoing elements but misses the core dynamic event or change introduced in the Ground Truth.
- Score 2 (Meaningful Match): The candidate successfully captures the core new entities, actions, or physical state changes present in the Ground Truth, even if minor descriptive vocabulary differs.

Output ONLY a JSON object with two keys:
"reasoning": "your short explanation",
"score": <0, 1, or 2>
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", type=str, default="llama-3.1-8b")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing reconstructed JSONs (e.g. results/key_events_llama-3.1-8b/w=4)")
    args = parser.parse_args()
    
    print(f"Initializing Judge LLM: {args.judge_model}")
    judge = LocalLLM(args.judge_model)
    
    input_path = Path(args.input_dir)
    results = []
    
    for fpath in tqdm(list(input_path.glob("*.json"))):
        with open(fpath) as f:
            res = json.load(f)
            
        vid = res["video_id"]
        climax_start = res["climax_start"]
        w = res["width"]
        recon_dict = res.get("reconstructed_captions", {})
        
        # Load GT
        import glob
        gt_file = glob.glob(f"datasets/**/{vid}.json", recursive=True)
        if not gt_file:
            continue
        with open(gt_file[0]) as f:
            data = json.load(f)
            caps = data if isinstance(data, list) else data.get('data', data.get('captions', data.get('clips', [])))
            
        gt_texts = [caps[i]['caption'] for i in range(climax_start, climax_start + w) if i < len(caps)]
        gt_str = "\n".join([f"sec {climax_start+i}: {txt}" for i, txt in enumerate(gt_texts)])
        
        cand_str = "\n".join([f"sec {k}: {v}" for k, v in sorted(recon_dict.items(), key=lambda x: int(x[0]))])
        if not cand_str:
            cand_str = "[BLANK / FAILED TO GENERATE]"
            
        prompt = JUDGE_PROMPT.format(ground_truth=gt_str, candidate=cand_str)
        
        try:
            response = judge.generate(prompt)
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != 0:
                out = json.loads(response[start:end])
                score = out.get("score", 0)
                reasoning = out.get("reasoning", "")
            else:
                score = 0
                reasoning = "Failed to parse JSON"
                
            results.append({
                "video_id": vid,
                "score": score,
                "reasoning": reasoning
            })
        except Exception as e:
            print(f"Error evaluating {vid}: {e}")
            
    df = pd.DataFrame(results)
    out_csv = input_path / "judge_scores.csv"
    df.to_csv(out_csv, index=False)
    
    print(f"\nEvaluation Complete! Saved to {out_csv}")
    print(f"Mean Score: {df['score'].mean():.2f}")
    print(f"Score Distribution:\n{df['score'].value_counts(normalize=True)*100}")

if __name__ == "__main__":
    main()
