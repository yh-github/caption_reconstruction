
import re
import json
import numpy as np

log_file = "logs/e7a319f0157b4241800970dee87a894d.log"
metrics = []

with open(log_file, "r") as f:
    for line in f:
        if "Evaluation metrics {" in line:
            json_str = line.split("Evaluation metrics ", 1)[1].strip()
            try:
                m = json.loads(json_str)
                metrics.append(m)
            except json.JSONDecodeError:
                pass

if not metrics:
    print("No metrics found.")
else:
    keys = ["mean_rank", "mrr", "recall_at_1", "recall_at_5"]
    print(f"Found {len(metrics)} metric records.")
    for k in keys:
        if k in metrics[0]:
            vals = [m[k] for m in metrics]
            print(f"{k}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}")
