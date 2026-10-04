import json

files = [
    "datasets/wildQA/captions__wild4/Millennial-Farmer_1-clip-11.json",
    "datasets/wildQA/captions__wild4/Survival-Skills-Primitive_3-clip-0.json",
    "datasets/wildQA/captions__wild4/Nick-Gaillard_4-clip-2.json"
]

for fpath in files:
    try:
        with open(fpath) as f:
            data = json.load(f)
            
            # extract captions list
            caps = []
            if isinstance(data, list):
                caps = data
            elif isinstance(data, dict):
                if 'data' in data: caps = data['data']
                elif 'captions' in data: caps = data['captions']
                elif 'clips' in data: caps = data['clips']
            
            if not caps or 'caption' not in caps[0]:
                print(f"Skipping {fpath}, unknown format")
                continue
                
            print(f"\n### {fpath.split('/')[-1]}")
            idx = 29
            w = 4 if 'Millennial' in fpath else 2
            
            print(f"Context Before: {caps[idx-1]['caption']}")
            for i in range(idx, idx+w):
                print(f"Masked {i}: {caps[i]['caption']}")
            print(f"Context After: {caps[idx+w]['caption']}")
    except Exception as e:
        print(f"Error on {fpath}: {e}")
