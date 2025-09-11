import yaml
from collections import Counter
from pathlib import Path
import numpy as np
from urllib.parse import urlparse

from utils import build_safe_dict
from data.video_link_loader import load_wild_dataset


def extract_filename(url: str) -> str:
    """Extract filename from Dropbox URL"""
    path = urlparse(url).path
    return path.split('/')[-1]


def print_stats(path:Path):
    items = [(x.to_link(),x.duration) for x in load_wild_dataset(path) if x.duration >= 60]
    links = build_safe_dict(items)

    domains = build_safe_dict([(x.to_link(),x.domain) for x in load_wild_dataset(path) if x.duration >= 60])

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


def dropbox_download_list(path:Path)->list[str]:
    items = [(x.video_link, x.video_id) for x in load_wild_dataset(path) if x.duration >= 60]
    links = build_safe_dict(items)
    print(len(links))
    id_to_link = build_safe_dict([(v,k) for k,v in links.items()])
    assert len(id_to_link) == len(links)
    return list(links.keys())
    # for k,v in id_to_link.items():
    #     filename = extract_filename(v)
    #     print(f"{v}\t{filename}")


import os
import requests

def download_urls(urls: list[str], output_directory: str):
    """Download multiple URLs to a specified directory"""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for url in urls:
        try:
            # Extract a filename from the URL
            filename = url.split('/')[-1].split('?')[0]
            save_path = os.path.join(output_directory, filename)

            print(f'Downloading {filename} to {save_path}...')

            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes

            # Save the file in chunks
            with open(save_path, 'wb') as video_file:
                for chunk in response.iter_content(chunk_size=8192):
                    video_file.write(chunk)

            print(f'Successfully downloaded {filename}.')

        except requests.exceptions.RequestException as e:
            print(f'Failed to download {url}. Error: {e}')

print('\nAll downloads complete!')


json_path = Path("/home/yoavh/code/research/caption_reconstruction/datasets/wildQA/dev.json")
for x in dropbox_download_list(json_path):
    print(x)


