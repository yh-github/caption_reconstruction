import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_csv('results/video_surprisal_scores.csv')

# map categories
def map_cat(c):
    if c in ['Scenery', 'Nature/Doc']:
        return 'Nature & Scenery'
    elif c == 'Military':
        return 'Military'
    return c

df['mapped_cat'] = df['category'].apply(map_cat)
df['continuity'] = 1.0 - df['video_avg_dist']

mil = df[df['mapped_cat'] == 'Military']['continuity']
nat = df[df['mapped_cat'] == 'Nature & Scenery']['continuity']

print(f"Military count: {len(mil)}, Nature & Scenery count: {len(nat)}")
print(f"Military mean: {mil.mean():.4f} +/- {mil.std():.4f}")
print(f"Nature mean: {nat.mean():.4f} +/- {nat.std():.4f}")

u, p = mannwhitneyu(nat, mil)
print(f"U={u}, p={p:.4e}")
