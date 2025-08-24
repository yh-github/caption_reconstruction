import yaml
from collections import Counter
from pathlib import Path
import numpy as np

from utils import build_safe_dict_gen
from video_link_loader import load_wild_dataset

path = Path("/home/yoavh/code/research/caption_reconstruction/datasets/wildQA/more/test.json")
items = [(x.to_link(),x.duration) for x in load_wild_dataset(path) if x.duration >= 60]
links = build_safe_dict_gen(items)

domains = build_safe_dict_gen([(x.to_link(),x.domain) for x in load_wild_dataset(path) if x.duration >= 60])

# Calculate histogram data
durations = [v for v in links.values()]
domains_counter = Counter(domains.values())

hist, bin_edges = np.histogram(durations, bins=10)

# Print textual histogram
print("\nDuration Histogram:")
print("-" * 50)
for i in range(len(hist)):
    bin_start = f"{bin_edges[i]:6.1f}"
    bin_end = f"{bin_edges[i + 1]:6.1f}"
    bar = "#" * int(hist[i] * 50 / max(hist))
    print(f"{bin_start} - {bin_end} | {bar} ({hist[i]})")


print()
print("## Domains:")
d=dict(domains_counter)
d['Total']=domains_counter.total()
print(yaml.dump(d, sort_keys=False))



"""
DEV:
Duration Histogram:
--------------------------------------------------
  60.0 -   76.1 | ################################################## (75)
  76.1 -   92.1 | ######## (12)
  92.1 -  108.2 | ## (3)
 108.2 -  124.2 | # (2)
 124.2 -  140.2 |  (1)
 140.2 -  156.3 | # (2)
 156.3 -  172.3 |  (0)
 172.3 -  188.4 |  (1)
 188.4 -  204.4 | ## (3)
 204.4 -  220.5 |  (1)

## Domains:
Geography: 12
Military: 19
Human Survival: 28
Natural Disaster: 17
Agriculture: 24
Total: 100

# TEST:
Duration Histogram:
--------------------------------------------------
  60.0 -   71.0 | ################################################## (168)
  71.0 -   82.1 | ####### (25)
  82.1 -   93.1 | ### (12)
  93.1 -  104.1 | # (6)
 104.1 -  115.1 | # (6)
 115.1 -  126.2 | # (4)
 126.2 -  137.2 |  (3)
 137.2 -  148.2 |  (2)
 148.2 -  159.2 | # (4)
 159.2 -  170.3 | # (5)

## Domains:
Geography: 29
Military: 44
Human Survival: 66
Natural Disaster: 36
Agriculture: 60
Total: 235


"""