import numpy as np
from scipy.stats import mannwhitneyu

np.random.seed(42)
nat = np.random.normal(0.9497, 0.0292, 42)
mil = np.random.normal(0.9225, 0.0344, 63)

u, p = mannwhitneyu(nat, mil)
print(f"U = {u}, p = {p}")
