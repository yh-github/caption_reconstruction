
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

LOG_FILES = [
    # phi-3
    "logs/e7a319f0157b4241800970dee87a894d.log", 
    # vec_vid (baseline)
    "logs/f42cdde3d4a44dbda715dbc67e3ed6c4.log" 
]

# Regex to capture config: e.g. "RepeatClosestVector__fixed_fill(w=12, i=59)"
# And metrics JSON
EXPERIMENT_REGEX = re.compile(r"INFO - (.*?)__fixed_fill\(w=(\d+), i=(\d+)\) Logged aggregated metrics ({.*})")

def parse_logs():
    data = []
    
    for log_file in LOG_FILES:
        with open(log_file, 'r') as f:
            for line in f:
                match = EXPERIMENT_REGEX.search(line)
                if match:
                    exp_name = match.group(1)
                    width = int(match.group(2))
                    index = int(match.group(3))
                    json_str = match.group(4)
                    
                    try:
                        metrics = json.loads(json_str)
                        
                        # Determine method based on log file or exp name
                        method = "phi-3" if "phi-3" in exp_name or "e7a31" in log_file else "vec_vid"
                        if "vec_vid" in log_file: method = "vec_vid"

                        row = {
                            "method": method,
                            "width": width,
                            "index": index,
                            "mean_rank": metrics.get("mean_mean_rank_mean"), # Aggregate of aggregates? No wait.
                             # The log line says "Logged aggregated metrics".
                             # The keys are like "mean_mean_rank_mean". 
                             # This means it's an aggregation over the batch (100 instances?).
                             # Wait, the user wants deep analysis "variable index_masked ... start, middle, end".
                             # "w=12, i=59" -> width 12, index 59.
                             # If "num_of_instances": 100, then this row represents 100 videos for that specific (w, i) config.
                             # This is confusing. Is (w, i) fixed for 100 videos?
                             # Let's check the schema.
                             **metrics
                        }
                        data.append(row)
                    except json.JSONDecodeError:
                        continue

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = parse_logs()
    print(df.head())
    print(f"Total rows: {len(df)}")
    print(df.groupby(['method', 'width']).size())
    
    # Save for further analysis
    df.to_csv("results/combined_analysis_data.csv", index=False)
