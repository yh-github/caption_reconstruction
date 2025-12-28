import json
from pathlib import Path
from collections import defaultdict

def categorize_videos(captions_dir: str):
    """
    Categorizes videos based on their filename prefixes (e.g., 'AiirSource-Military').
    """
    captions_path = Path(captions_dir)
    categories = defaultdict(list)
    
    # Metadata map for higher-level categories
    # Based on inspection of filenames
    high_level_map = {
        'AiirSource-Military': 'Military',
        'Army-military': 'Military',
        'MilitaryNotes': 'Military',
        'Sandboxx': 'Military',
        'USA-Military-Channel': 'Military',
        'WarLeaks-Military-Blog': 'Military',
        
        'How-Farms-Work': 'Farming',
        'Hamiltonville-Farm': 'Farming',
        'John-Suscovich': 'Farming', # Chicken farming
        'Millennial-Farmer': 'Farming',
        'Olly\'s-Farm': 'Farming',
        'Peterson-Farm-Bros': 'Farming',
        'Welker-Farms-Inc': 'Farming',
        'RealAgriculture': 'Farming',
        
        'BC-Bushcraft': 'Survival',
        'Bertram-Craft': 'Survival',
        'Joe-Robinet': 'Survival',
        'Primitive-Technology': 'Survival',
        'Survival-Instinct': 'Survival',
        'Survival-Skills-Primitive': 'Survival',
        'Chad-Zuber': 'Survival',
        'Primal-Earth-Sounds': 'Survival', # Often bushcraft sounds/building
        
        'Climate-Change': 'Nature/Doc',
        'Natural-Disaster': 'Nature/Doc',
        'Tornado-Trackers': 'Nature/Doc',
        'Weathershot': 'Nature/Doc',
        'King-Kong-Amazon': 'Nature/Doc',
        
        'TreadmillTV': 'Scenery',
        '4k-Relaxation': 'Scenery',
        
        'Gung-Ho-Vids': 'Action/Vehicle',
        'Ultimate-Chase': 'Action/Vehicle', # Storm chasing usually? Or police?
        'Nick-Gaillard': 'Action/Vehicle', # Often skiing/gopro
        'Dan-Robinson': 'Action/Vehicle',
        
        'TK-Hinshaw': 'Farming', # Heavy equipment/farming
        'Army-military-2018': 'Military',
    }
    
    # Process files
    video_categories = {}
    
    files = list(captions_path.glob('*.json'))
    print(f"Found {len(files)} caption files.")
    
    for f in files:
        # distinct name is usually the part before the first underscore
        # But some have hyphens. The "Series" name is usually the prefix.
        # e.g. "AiirSource-Military_1-clip-0.json" -> "AiirSource-Military"
        
        filename = f.stem
        # Heuristic: split by '_' and take the first part as the series name
        series_name = filename.split('_')[0]
        
        category = high_level_map.get(series_name, 'Other')
        video_categories[filename] = {
            'series': series_name,
            'category': category
        }
        categories[category].append(filename)
        
    # Print stats
    print("\nCategory Counts:")
    for cat, items in categories.items():
        print(f"  {cat}: {len(items)}")
        
    # Save to file
    output_path = Path("results/video_categories.json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        json.dump(video_categories, f, indent=2)
    print(f"\nSaved categorization to {output_path}")
        
    return video_categories

if __name__ == "__main__":
    # Assuming run from project root
    base_path = Path("datasets/wildQA/captions__wild2/")
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist.")
    else:
        cats = categorize_videos(base_path)
        # Verify we classified most things
        others = [k for k, v in cats.items() if v['category'] == 'Other']
        if others:
            print(f"\nUncategorized videos ({len(others)}):")
            for o in others[:10]:
                print(f"  {o} (Series: {cats[o]['series']})")
