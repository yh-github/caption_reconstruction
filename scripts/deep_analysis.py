
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("results/combined_analysis_data.csv")

# Ensure we have both methods for comparison
print("Data shape:", df.shape)
print("Methods found:", df['method'].unique())

# 1. Performance by num_masked (width)
# We use 'mean_mean_rank_mean' which seems to be the aggregated rank
# Lower rank is better.
# Let's pivot to compare
pivot_width = df.pivot_table(index='width', columns='method', values='mean_mean_rank_mean')
print("\n--- Mean Rank by Width ---")
print(pivot_width)

pivot_mrr = df.pivot_table(index='width', columns='method', values='mean_mrr_mean')
print("\n--- MRR by Width ---")
print(pivot_mrr)


# 2. Performance by index_masked (position)
pivot_idx = df.pivot_table(index='index', columns='method', values='mean_mrr_mean')
print("\n--- MRR by Index ---")
print(pivot_idx)

# 3. Categorize wins
# Since we don't have per-video rows here (only per-config aggregates),
# we can comparing strictly by config (same width, same index).
# 
merged = pd.merge(
    df[df['method'] == 'phi-3'][['width', 'index', 'mean_mrr_mean', 'mean_mean_rank_mean']],
    df[df['method'] == 'vec_vid'][['width', 'index', 'mean_mrr_mean', 'mean_mean_rank_mean']],
    on=['width', 'index'],
    suffixes=('_phi', '_vec')
)

merged['mrr_delta'] = merged['mean_mrr_mean_phi'] - merged['mean_mrr_mean_vec']
merged['rank_delta'] = merged['mean_mean_rank_mean_vec'] - merged['mean_mean_rank_mean_phi'] # Pos delta = phi better

print("\n--- Head-to-Head Config Comparison ---")
print(f"Total configs compared: {len(merged)}")
print(f"Phi-3 better MRR: {sum(merged['mrr_delta'] > 0)}")
print(f"Vec_vid better MRR: {sum(merged['mrr_delta'] < 0)}")

# Best wins for Phi-3 (by Rank Delta - lower is better for rank, so higher delta is good)
print("\n--- Top 5 Phi-3 Wins (Rank Delta) ---")
print(merged.sort_values('rank_delta', ascending=False).head(5))

# Worst losses
print("\n--- Top 5 Phi-3 Losses (Rank Delta) ---")
print(merged.sort_values('rank_delta', ascending=True).head(5))

# Save deep analysis
merged.to_csv("results/deep_analysis_config_comparison.csv", index=False)
