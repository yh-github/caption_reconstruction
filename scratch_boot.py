import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('results/prior_vs_post_reconstruction_master.csv')
# Keep only Nature & Scenery and Military
def map_cat(c):
    if c in ['Scenery', 'Nature/Doc']:
        return 'Nature & Scenery'
    elif c == 'Military':
        return 'Military'
    return c

df['category'] = df['category'].apply(map_cat)
df = df[df['category'].isin(['Nature & Scenery', 'Military'])]
df = df.dropna(subset=['video_surprisal_avg', 'llama_mrr_delta'])
print(f"Number of videos: {len(df)}")

df['is_nature'] = (df['category'] == 'Nature & Scenery').astype(int)
df['v_continuity'] = 1 - df['video_surprisal_avg']

y = df['llama_mrr_delta']
X1 = sm.add_constant(df['is_nature'])
X2 = sm.add_constant(df[['is_nature', 'v_continuity']])

model1 = sm.OLS(y, X1).fit()
model2 = sm.OLS(y, X2).fit()

print("Model 1:")
print(model1.summary())
print("Model 2:")
print(model2.summary())

