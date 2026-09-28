import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_csv('results/unified_benchmark_master.csv')
vid_df = df.groupby(['video_id', 'category']).first().reset_index()

mil_vids = vid_df[vid_df['category'] == 'Military']['video_id'].unique()
nat_vids = vid_df[vid_df['category'] == 'Nature & Scenery']['video_id'].unique()

print(f"Military vids: {len(mil_vids)}, Nature & Scenery vids: {len(nat_vids)}")

# I don't see continuity in this file. Let me check the columns.
print("Columns:", df.columns)
