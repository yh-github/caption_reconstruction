import json
import diskcache
import numpy as np
from scipy.stats import mannwhitneyu

def get_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

cache = diskcache.Cache('disk_cache/local_google_siglip-base-patch16-224_text')
print(len(cache))
