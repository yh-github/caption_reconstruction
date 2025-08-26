import numpy as np
from pathlib import Path

NPY_FILE_PATTERN = "*.npy"


FRAME_COUNT_PATTERN = r'Extracted (\d+) frames from ([^\.]+)\.mp4'
def parse_processing_log(log_text: str) -> dict[str, int]:
    """Extract frame counts from processing log text."""
    frame_counts = {}
    for match in re.finditer(FRAME_COUNT_PATTERN, log_text):
        count, filename = match.groups()
        frame_counts[filename] = int(count)
    return frame_counts

# frames_by_id = parse_processing_log(proc)
frames_by_id = {'4k-Relaxation_12-clip-6': 190, '4k-Relaxation_3-clip-4': 118, 'AiirSource-Military_1-clip-0': 70, 'AiirSource-Military_12-manual': 66, 'AiirSource-Military_7-clip-1': 65, 'Army-military-2018_8-clip-73': 78, 'BC-Bushcraft_10-clip-8': 86, 'BC-Bushcraft_11-clip-31': 204, 'BC-Bushcraft_2-clip-2': 66, 'BC-Bushcraft_9-clip-5': 64, 'Bertram-Craft_12-clip-25': 66, 'Bertram-Craft_2-clip-3': 66, 'Bertram-Craft_5-clip-33': 88, 'Chad-Zuber_10-clip-20': 66, 'Chad-Zuber_4-clip-2': 71, 'Chad-Zuber_5-clip-0': 61, 'Climate-Change_0-clip-4': 65, 'Climate-Change_2-clip-4': 61, 'Climate-Change_6-clip-7': 64, 'Climate-Change_7-clip-3': 62, 'Dan-Robinson_3-clip-3': 61, 'Disaster-Compilations_10-clip-0': 66, 'Disaster-Compilations_5-clip-0': 61, 'Disaster-Compilations_9-clip-2': 68, 'Gung-Ho-Vids_0-clip-2': 90, 'Gung-Ho-Vids_12-clip-1': 61, 'Gung-Ho-Vids_2-clip-2': 64, 'Gung-Ho-Vids_5-clip-2': 62, 'Gung-Ho-Vids_8-clip-0': 61, 'Hamiltonville-Farm_1-clip-4': 65, 'Hamiltonville-Farm_2-clip-0': 63, 'Hamiltonville-Farm_6-clip-18': 68, 'Hamiltonville-Farm_8-clip-3': 62, 'How-Farms-Work_10-clip-1': 147, 'How-Farms-Work_3-clip-2': 64, 'How-Farms-Work_5-clip-0': 61, 'How-Farms-Work_8-clip-4': 72, 'How-Farms-Work_9-manual': 141, 'Joe-Robinet_0-clip-23': 65, 'Joe-Robinet_2-clip-5': 75, 'Joe-Robinet_4-clip-0': 178, 'Joe-Robinet_5-clip-23': 61, 'John-Suscovich_0-clip-1': 76, 'John-Suscovich_10-manual': 124, 'John-Suscovich_12-clip-0': 79, 'John-Suscovich_2-clip-3': 95, 'John-Suscovich_3-manual': 76, 'King-Kong-Amazon_11-clip-7': 65, 'King-Kong-Amazon_5-clip-14': 89, 'MilitaryNotes_10-clip-0': 62, 'MilitaryNotes_11-clip-1': 62, 'MilitaryNotes_8-clip-0': 70, 'Millennial-Farmer_0-clip-11': 66, 'Millennial-Farmer_1-clip-11': 61, 'Millennial-Farmer_7-clip-13': 61, 'Millennial-Farmer_8-clip-16': 95, 'Natural-Disaster_10-clip-0': 62, 'Nick-Gaillard_1-clip-2': 70, 'Nick-Gaillard_4-clip-2': 84, 'Olly_s-Farm_1-clip-5': 65, 'Olly_s-Farm_6-clip-1': 61, 'Peterson-Farm-Bros_5-clip-2': 78, 'Peterson-Farm-Bros_6-clip-4': 144, 'Primal-Earth-Sounds_0-clip-44': 221, 'Primal-Earth-Sounds_9-clip-26': 63, 'Primitive-Technology_11-clip-8': 62, 'Primitive-Technology_3-clip-1': 64, 'Primitive-Technology_5-clip-0': 70, 'Primitive-Technology_6-clip-3': 61, 'Primitive-Technology_7-clip-4': 66, 'RealAgriculture_9-clip-1': 82, 'Sandboxx_0-clip-4': 72, 'Sandboxx_7-clip-0': 61, 'Sandboxx_9-clip-4': 64, 'Survival-Instinct_11-clip-10': 64, 'Survival-Instinct_2-clip-1': 61, 'Survival-Instinct_7-clip-8': 61, 'Survival-Instinct_8-clip-4': 62, 'Survival-Instinct_9-clip-2': 65, 'Survival-Skills-Primitive_11-clip-0': 61, 'Survival-Skills-Primitive_3-clip-0': 65, 'TK-Hinshaw_5-clip-0': 86, 'TK-Hinshaw_7-clip-0': 64, 'Tornado-Trackers_1-clip-0': 61, 'Tornado-Trackers_10-clip-4': 61, 'Tornado-Trackers_6-clip-2': 68, 'TreadmillTV_2-clip-1': 68, 'TreadmillTV_3-clip-12': 62, 'TreadmillTV_6-clip-2': 84, 'TreadmillTV_9-clip-2': 192, 'USA-Military-Channel_0-clip-6': 67, 'USA-Military-Channel_10-clip-2': 63, 'USA-Military-Channel_11-clip-2': 62, 'Ultimate-Chase_3-clip-4': 63, 'Ultimate-Chase_8-clip-0': 61, 'Ultimate-Chase_9-clip-0': 94, 'WarLeaks-Military-Blog_9-clip-0': 72, 'Weathershot_3-clip-1': 67, 'Weathershot_7-clip-0': 90, 'Welker-Farms-Inc_3-clip-4': 61}

def find_numpy_files(directory: Path, file_pattern: str = NPY_FILE_PATTERN) -> list[Path]:
    """Find all files matching the given pattern in the directory recursively."""
    return list(directory.rglob(file_pattern))

def load_numpy_file_and_print_shape(file_path: Path) -> None:
    """Load a NumPy file and print its shape."""
    data = np.load(file_path)
    mark = ""
    if data.shape[0]<60:
        mark = "***"
    file_id = file_path.stem
    assert frames_by_id[file_id] == data.shape[0], f"{file_id} {data.shape[0]} {frames_by_id[file_id]}"
    print(f"File: {file_path}   Shape: {data.shape} {mark} frames: {frames_by_id[file_id]}")

def process_numpy_files(npy_files: list[Path]) -> None:
    """Process and print information for a list of NumPy files."""
    for file_path in npy_files:
        load_numpy_file_and_print_shape(file_path)

def bump_path(name_to_bump: str, paths: list[Path]) -> list[Path]:
    for i, path in enumerate(paths):
        if path.name == name_to_bump:
            paths.insert(0, paths.pop(i))
            break
    return paths

def main(directory: Path) -> None:
    """Main function to process NumPy files in the given directory."""
    npy_files = find_numpy_files(directory)
    if not npy_files:
        print(f"No .npy files found in {directory}")
        return
    process_numpy_files(npy_files)



# Example usage:
main(Path("/home/yoavh/code/research/caption_reconstruction/local/wild_videos_embs"))
# Parse remote listing:
# file_sizes = parse_remote_listing(REMOTE_LL)


remote_ll="""
    291968  2025-08-24T20:24:22Z  gs://yh-ai/wild_videos_embs/4k-Relaxation_12-clip-6.npy
    181376  2025-08-24T20:24:25Z  gs://yh-ai/wild_videos_embs/4k-Relaxation_3-clip-4.npy
    107648  2025-08-24T20:24:28Z  gs://yh-ai/wild_videos_embs/AiirSource-Military_1-clip-0.npy
    101504  2025-08-24T20:24:30Z  gs://yh-ai/wild_videos_embs/AiirSource-Military_12-manual.npy
     99968  2025-08-24T20:24:32Z  gs://yh-ai/wild_videos_embs/AiirSource-Military_7-clip-1.npy
    119936  2025-08-24T20:24:35Z  gs://yh-ai/wild_videos_embs/Army-military-2018_8-clip-73.npy
    132224  2025-08-24T20:24:38Z  gs://yh-ai/wild_videos_embs/BC-Bushcraft_10-clip-8.npy
    313472  2025-08-24T20:24:43Z  gs://yh-ai/wild_videos_embs/BC-Bushcraft_11-clip-31.npy
    101504  2025-08-24T20:24:46Z  gs://yh-ai/wild_videos_embs/BC-Bushcraft_2-clip-2.npy
     98432  2025-08-24T20:24:48Z  gs://yh-ai/wild_videos_embs/BC-Bushcraft_9-clip-5.npy
    101504  2025-08-24T20:24:50Z  gs://yh-ai/wild_videos_embs/Bertram-Craft_12-clip-25.npy
    101504  2025-08-24T20:24:53Z  gs://yh-ai/wild_videos_embs/Bertram-Craft_2-clip-3.npy
    135296  2025-08-24T20:24:55Z  gs://yh-ai/wild_videos_embs/Bertram-Craft_5-clip-33.npy
    101504  2025-08-24T20:24:58Z  gs://yh-ai/wild_videos_embs/Chad-Zuber_10-clip-20.npy
    109184  2025-08-24T20:25:00Z  gs://yh-ai/wild_videos_embs/Chad-Zuber_4-clip-2.npy
     93824  2025-08-24T20:25:03Z  gs://yh-ai/wild_videos_embs/Chad-Zuber_5-clip-0.npy
     99968  2025-08-24T20:25:06Z  gs://yh-ai/wild_videos_embs/Climate-Change_0-clip-4.npy
     93824  2025-08-24T20:25:09Z  gs://yh-ai/wild_videos_embs/Climate-Change_2-clip-4.npy
     98432  2025-08-24T20:25:12Z  gs://yh-ai/wild_videos_embs/Climate-Change_6-clip-7.npy
     95360  2025-08-24T20:25:15Z  gs://yh-ai/wild_videos_embs/Climate-Change_7-clip-3.npy
     93824  2025-08-24T20:25:18Z  gs://yh-ai/wild_videos_embs/Dan-Robinson_3-clip-3.npy
    101504  2025-08-24T20:25:20Z  gs://yh-ai/wild_videos_embs/Disaster-Compilations_10-clip-0.npy
     93824  2025-08-24T20:25:23Z  gs://yh-ai/wild_videos_embs/Disaster-Compilations_5-clip-0.npy
    104576  2025-08-24T20:25:26Z  gs://yh-ai/wild_videos_embs/Disaster-Compilations_9-clip-2.npy
    138368  2025-08-24T20:25:28Z  gs://yh-ai/wild_videos_embs/Gung-Ho-Vids_0-clip-2.npy
     93824  2025-08-24T20:25:30Z  gs://yh-ai/wild_videos_embs/Gung-Ho-Vids_12-clip-1.npy
     98432  2025-08-24T20:25:32Z  gs://yh-ai/wild_videos_embs/Gung-Ho-Vids_2-clip-2.npy
     95360  2025-08-24T20:25:34Z  gs://yh-ai/wild_videos_embs/Gung-Ho-Vids_5-clip-2.npy
     93824  2025-08-24T20:25:37Z  gs://yh-ai/wild_videos_embs/Gung-Ho-Vids_8-clip-0.npy
     99968  2025-08-24T20:25:39Z  gs://yh-ai/wild_videos_embs/Hamiltonville-Farm_1-clip-4.npy
     96896  2025-08-24T20:25:42Z  gs://yh-ai/wild_videos_embs/Hamiltonville-Farm_2-clip-0.npy
    104576  2025-08-24T20:25:44Z  gs://yh-ai/wild_videos_embs/Hamiltonville-Farm_6-clip-18.npy
     95360  2025-08-24T20:25:47Z  gs://yh-ai/wild_videos_embs/Hamiltonville-Farm_8-clip-3.npy
    225920  2025-08-24T20:25:51Z  gs://yh-ai/wild_videos_embs/How-Farms-Work_10-clip-1.npy
     98432  2025-08-24T20:25:53Z  gs://yh-ai/wild_videos_embs/How-Farms-Work_3-clip-2.npy
     93824  2025-08-24T20:25:55Z  gs://yh-ai/wild_videos_embs/How-Farms-Work_5-clip-0.npy
    110720  2025-08-24T20:25:58Z  gs://yh-ai/wild_videos_embs/How-Farms-Work_8-clip-4.npy
    216704  2025-08-24T20:26:02Z  gs://yh-ai/wild_videos_embs/How-Farms-Work_9-manual.npy
     99968  2025-08-24T20:26:04Z  gs://yh-ai/wild_videos_embs/Joe-Robinet_0-clip-23.npy
    115328  2025-08-24T20:26:06Z  gs://yh-ai/wild_videos_embs/Joe-Robinet_2-clip-5.npy
    273536  2025-08-24T20:26:10Z  gs://yh-ai/wild_videos_embs/Joe-Robinet_4-clip-0.npy
     93824  2025-08-24T20:26:13Z  gs://yh-ai/wild_videos_embs/Joe-Robinet_5-clip-23.npy
    116864  2025-08-24T20:26:16Z  gs://yh-ai/wild_videos_embs/John-Suscovich_0-clip-1.npy
    190592  2025-08-24T20:26:19Z  gs://yh-ai/wild_videos_embs/John-Suscovich_10-manual.npy
    121472  2025-08-24T20:26:22Z  gs://yh-ai/wild_videos_embs/John-Suscovich_12-clip-0.npy
    146048  2025-08-24T20:26:24Z  gs://yh-ai/wild_videos_embs/John-Suscovich_2-clip-3.npy
    116864  2025-08-24T20:26:27Z  gs://yh-ai/wild_videos_embs/John-Suscovich_3-manual.npy
     99968  2025-08-24T20:26:30Z  gs://yh-ai/wild_videos_embs/King-Kong-Amazon_11-clip-7.npy
    136832  2025-08-24T20:26:34Z  gs://yh-ai/wild_videos_embs/King-Kong-Amazon_5-clip-14.npy
     95360  2025-08-24T20:26:36Z  gs://yh-ai/wild_videos_embs/MilitaryNotes_10-clip-0.npy
     95360  2025-08-24T20:26:38Z  gs://yh-ai/wild_videos_embs/MilitaryNotes_11-clip-1.npy
    107648  2025-08-24T20:26:40Z  gs://yh-ai/wild_videos_embs/MilitaryNotes_8-clip-0.npy
    101504  2025-08-24T20:26:42Z  gs://yh-ai/wild_videos_embs/Millennial-Farmer_0-clip-11.npy
     93824  2025-08-24T20:26:45Z  gs://yh-ai/wild_videos_embs/Millennial-Farmer_1-clip-11.npy
     93824  2025-08-24T20:26:47Z  gs://yh-ai/wild_videos_embs/Millennial-Farmer_7-clip-13.npy
    146048  2025-08-24T20:26:50Z  gs://yh-ai/wild_videos_embs/Millennial-Farmer_8-clip-16.npy
     95360  2025-08-24T20:26:51Z  gs://yh-ai/wild_videos_embs/Natural-Disaster_10-clip-0.npy
    107648  2025-08-24T20:26:53Z  gs://yh-ai/wild_videos_embs/Nick-Gaillard_1-clip-2.npy
    129152  2025-08-24T20:26:56Z  gs://yh-ai/wild_videos_embs/Nick-Gaillard_4-clip-2.npy
     99968  2025-08-24T20:27:00Z  gs://yh-ai/wild_videos_embs/Olly_s-Farm_1-clip-5.npy
     93824  2025-08-24T20:27:02Z  gs://yh-ai/wild_videos_embs/Olly_s-Farm_6-clip-1.npy
    119936  2025-08-24T20:27:05Z  gs://yh-ai/wild_videos_embs/Peterson-Farm-Bros_5-clip-2.npy
    221312  2025-08-24T20:27:11Z  gs://yh-ai/wild_videos_embs/Peterson-Farm-Bros_6-clip-4.npy
    339584  2025-08-24T20:27:24Z  gs://yh-ai/wild_videos_embs/Primal-Earth-Sounds_0-clip-44.npy
     96896  2025-08-24T20:27:28Z  gs://yh-ai/wild_videos_embs/Primal-Earth-Sounds_9-clip-26.npy
     95360  2025-08-24T20:27:32Z  gs://yh-ai/wild_videos_embs/Primitive-Technology_11-clip-8.npy
     98432  2025-08-24T20:27:35Z  gs://yh-ai/wild_videos_embs/Primitive-Technology_3-clip-1.npy
    107648  2025-08-24T20:27:39Z  gs://yh-ai/wild_videos_embs/Primitive-Technology_5-clip-0.npy
     93824  2025-08-24T20:27:42Z  gs://yh-ai/wild_videos_embs/Primitive-Technology_6-clip-3.npy
    101504  2025-08-24T20:27:46Z  gs://yh-ai/wild_videos_embs/Primitive-Technology_7-clip-4.npy
    126080  2025-08-24T20:27:50Z  gs://yh-ai/wild_videos_embs/RealAgriculture_9-clip-1.npy
    110720  2025-08-24T20:27:54Z  gs://yh-ai/wild_videos_embs/Sandboxx_0-clip-4.npy
     93824  2025-08-24T20:27:57Z  gs://yh-ai/wild_videos_embs/Sandboxx_7-clip-0.npy
     98432  2025-08-24T20:28:01Z  gs://yh-ai/wild_videos_embs/Sandboxx_9-clip-4.npy
     98432  2025-08-24T20:28:05Z  gs://yh-ai/wild_videos_embs/Survival-Instinct_11-clip-10.npy
     93824  2025-08-24T20:28:09Z  gs://yh-ai/wild_videos_embs/Survival-Instinct_2-clip-1.npy
     93824  2025-08-24T20:28:11Z  gs://yh-ai/wild_videos_embs/Survival-Instinct_7-clip-8.npy
     95360  2025-08-24T20:28:15Z  gs://yh-ai/wild_videos_embs/Survival-Instinct_8-clip-4.npy
     99968  2025-08-24T20:28:19Z  gs://yh-ai/wild_videos_embs/Survival-Instinct_9-clip-2.npy
     93824  2025-08-24T20:28:22Z  gs://yh-ai/wild_videos_embs/Survival-Skills-Primitive_11-clip-0.npy
     99968  2025-08-24T20:28:27Z  gs://yh-ai/wild_videos_embs/Survival-Skills-Primitive_3-clip-0.npy
    132224  2025-08-24T20:28:31Z  gs://yh-ai/wild_videos_embs/TK-Hinshaw_5-clip-0.npy
     98432  2025-08-24T20:28:34Z  gs://yh-ai/wild_videos_embs/TK-Hinshaw_7-clip-0.npy
     93824  2025-08-24T20:28:37Z  gs://yh-ai/wild_videos_embs/Tornado-Trackers_1-clip-0.npy
     93824  2025-08-24T20:28:40Z  gs://yh-ai/wild_videos_embs/Tornado-Trackers_10-clip-4.npy
    104576  2025-08-24T20:28:44Z  gs://yh-ai/wild_videos_embs/Tornado-Trackers_6-clip-2.npy
    104576  2025-08-24T20:28:47Z  gs://yh-ai/wild_videos_embs/TreadmillTV_2-clip-1.npy
     95360  2025-08-24T20:28:51Z  gs://yh-ai/wild_videos_embs/TreadmillTV_3-clip-12.npy
    129152  2025-08-24T20:28:56Z  gs://yh-ai/wild_videos_embs/TreadmillTV_6-clip-2.npy
    295040  2025-08-24T20:29:04Z  gs://yh-ai/wild_videos_embs/TreadmillTV_9-clip-2.npy
    103040  2025-08-24T20:29:08Z  gs://yh-ai/wild_videos_embs/USA-Military-Channel_0-clip-6.npy
     96896  2025-08-24T20:29:11Z  gs://yh-ai/wild_videos_embs/USA-Military-Channel_10-clip-2.npy
     95360  2025-08-24T20:29:15Z  gs://yh-ai/wild_videos_embs/USA-Military-Channel_11-clip-2.npy
     96896  2025-08-24T20:29:17Z  gs://yh-ai/wild_videos_embs/Ultimate-Chase_3-clip-4.npy
     93824  2025-08-24T20:29:21Z  gs://yh-ai/wild_videos_embs/Ultimate-Chase_8-clip-0.npy
    144512  2025-08-24T20:29:25Z  gs://yh-ai/wild_videos_embs/Ultimate-Chase_9-clip-0.npy
    110720  2025-08-24T20:29:28Z  gs://yh-ai/wild_videos_embs/WarLeaks-Military-Blog_9-clip-0.npy
    103040  2025-08-24T20:29:32Z  gs://yh-ai/wild_videos_embs/Weathershot_3-clip-1.npy
    138368  2025-08-24T20:29:36Z  gs://yh-ai/wild_videos_embs/Weathershot_7-clip-0.npy
     93824  2025-08-24T20:29:40Z  gs://yh-ai/wild_videos_embs/Welker-Farms-Inc_3-clip-4.npy
"""

import re
rex = re.compile(r'^\s*(\d+)\s+\S+\s+gs:.*/([^/]+)\.npy')
d_remote={}
for x in remote_ll.splitlines():
    if not x.strip():
        continue
    m = rex.match(x)
    if m:
        d_remote[m.group(2)] = m.group(1)


print(d_remote)



local_ll ="""
total 11948
drwxrwxr-x 2 yoavh yoavh  12288 Aug 24 23:30 ./
drwxrwxr-x 4 yoavh yoavh  12288 Aug 25 21:03 ../
-rw-rw-r-- 1 yoavh yoavh 291968 Aug 24 23:30 4k-Relaxation_12-clip-6.npy
-rw-rw-r-- 1 yoavh yoavh 181376 Aug 24 23:30 4k-Relaxation_3-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh 101504 Aug 24 23:30 AiirSource-Military_12-manual.npy
-rw-rw-r-- 1 yoavh yoavh 107648 Aug 24 23:30 AiirSource-Military_1-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  99968 Aug 24 23:30 AiirSource-Military_7-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh 119936 Aug 24 23:30 Army-military-2018_8-clip-73.npy
-rw-rw-r-- 1 yoavh yoavh 132224 Aug 24 23:30 BC-Bushcraft_10-clip-8.npy
-rw-rw-r-- 1 yoavh yoavh 313472 Aug 24 23:30 BC-Bushcraft_11-clip-31.npy
-rw-rw-r-- 1 yoavh yoavh 101504 Aug 24 23:30 BC-Bushcraft_2-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh  98432 Aug 24 23:30 BC-Bushcraft_9-clip-5.npy
-rw-rw-r-- 1 yoavh yoavh 101504 Aug 24 23:30 Bertram-Craft_12-clip-25.npy
-rw-rw-r-- 1 yoavh yoavh 101504 Aug 24 23:30 Bertram-Craft_2-clip-3.npy
-rw-rw-r-- 1 yoavh yoavh 135296 Aug 24 23:30 Bertram-Craft_5-clip-33.npy
-rw-rw-r-- 1 yoavh yoavh 101504 Aug 24 23:30 Chad-Zuber_10-clip-20.npy
-rw-rw-r-- 1 yoavh yoavh 109184 Aug 24 23:30 Chad-Zuber_4-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Chad-Zuber_5-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  99968 Aug 24 23:30 Climate-Change_0-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Climate-Change_2-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh  98432 Aug 24 23:30 Climate-Change_6-clip-7.npy
-rw-rw-r-- 1 yoavh yoavh  95360 Aug 24 23:30 Climate-Change_7-clip-3.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Dan-Robinson_3-clip-3.npy
-rw-rw-r-- 1 yoavh yoavh 101504 Aug 24 23:30 Disaster-Compilations_10-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Disaster-Compilations_5-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 104576 Aug 24 23:30 Disaster-Compilations_9-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh 138368 Aug 24 23:30 Gung-Ho-Vids_0-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Gung-Ho-Vids_12-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh  98432 Aug 24 23:30 Gung-Ho-Vids_2-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh  95360 Aug 24 23:30 Gung-Ho-Vids_5-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Gung-Ho-Vids_8-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  99968 Aug 24 23:30 Hamiltonville-Farm_1-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh  96896 Aug 24 23:30 Hamiltonville-Farm_2-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 104576 Aug 24 23:30 Hamiltonville-Farm_6-clip-18.npy
-rw-rw-r-- 1 yoavh yoavh  95360 Aug 24 23:30 Hamiltonville-Farm_8-clip-3.npy
-rw-rw-r-- 1 yoavh yoavh 225920 Aug 24 23:30 How-Farms-Work_10-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh  98432 Aug 24 23:30 How-Farms-Work_3-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 How-Farms-Work_5-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 110720 Aug 24 23:30 How-Farms-Work_8-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh 216704 Aug 24 23:30 How-Farms-Work_9-manual.npy
-rw-rw-r-- 1 yoavh yoavh  99968 Aug 24 23:30 Joe-Robinet_0-clip-23.npy
-rw-rw-r-- 1 yoavh yoavh 115328 Aug 24 23:30 Joe-Robinet_2-clip-5.npy
-rw-rw-r-- 1 yoavh yoavh 273536 Aug 24 23:30 Joe-Robinet_4-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Joe-Robinet_5-clip-23.npy
-rw-rw-r-- 1 yoavh yoavh 116864 Aug 24 23:30 John-Suscovich_0-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh 190592 Aug 24 23:30 John-Suscovich_10-manual.npy
-rw-rw-r-- 1 yoavh yoavh 121472 Aug 24 23:30 John-Suscovich_12-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 146048 Aug 24 23:30 John-Suscovich_2-clip-3.npy
-rw-rw-r-- 1 yoavh yoavh 116864 Aug 24 23:30 John-Suscovich_3-manual.npy
-rw-rw-r-- 1 yoavh yoavh  99968 Aug 24 23:30 King-Kong-Amazon_11-clip-7.npy
-rw-rw-r-- 1 yoavh yoavh 136832 Aug 24 23:30 King-Kong-Amazon_5-clip-14.npy
-rw-rw-r-- 1 yoavh yoavh  95360 Aug 24 23:30 MilitaryNotes_10-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  95360 Aug 24 23:30 MilitaryNotes_11-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh 107648 Aug 24 23:30 MilitaryNotes_8-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 101504 Aug 24 23:30 Millennial-Farmer_0-clip-11.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Millennial-Farmer_1-clip-11.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Millennial-Farmer_7-clip-13.npy
-rw-rw-r-- 1 yoavh yoavh 146048 Aug 24 23:30 Millennial-Farmer_8-clip-16.npy
-rw-rw-r-- 1 yoavh yoavh  95360 Aug 24 23:30 Natural-Disaster_10-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 107648 Aug 24 23:30 Nick-Gaillard_1-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh 129152 Aug 24 23:30 Nick-Gaillard_4-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh  99968 Aug 24 23:30 Olly_s-Farm_1-clip-5.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Olly_s-Farm_6-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh 119936 Aug 24 23:30 Peterson-Farm-Bros_5-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh 221312 Aug 24 23:30 Peterson-Farm-Bros_6-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh 339584 Aug 24 23:30 Primal-Earth-Sounds_0-clip-44.npy
-rw-rw-r-- 1 yoavh yoavh  96896 Aug 24 23:30 Primal-Earth-Sounds_9-clip-26.npy
-rw-rw-r-- 1 yoavh yoavh  95360 Aug 24 23:30 Primitive-Technology_11-clip-8.npy
-rw-rw-r-- 1 yoavh yoavh  98432 Aug 24 23:30 Primitive-Technology_3-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh 107648 Aug 24 23:30 Primitive-Technology_5-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Primitive-Technology_6-clip-3.npy
-rw-rw-r-- 1 yoavh yoavh 101504 Aug 24 23:30 Primitive-Technology_7-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh 126080 Aug 24 23:30 RealAgriculture_9-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh 110720 Aug 24 23:30 Sandboxx_0-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Sandboxx_7-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  98432 Aug 24 23:30 Sandboxx_9-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh  98432 Aug 24 23:30 Survival-Instinct_11-clip-10.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Survival-Instinct_2-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Survival-Instinct_7-clip-8.npy
-rw-rw-r-- 1 yoavh yoavh  95360 Aug 24 23:30 Survival-Instinct_8-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh  99968 Aug 24 23:30 Survival-Instinct_9-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Survival-Skills-Primitive_11-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  99968 Aug 24 23:30 Survival-Skills-Primitive_3-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 132224 Aug 24 23:30 TK-Hinshaw_5-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  98432 Aug 24 23:30 TK-Hinshaw_7-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Tornado-Trackers_10-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Tornado-Trackers_1-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 104576 Aug 24 23:30 Tornado-Trackers_6-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh 104576 Aug 24 23:30 TreadmillTV_2-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh  95360 Aug 24 23:30 TreadmillTV_3-clip-12.npy
-rw-rw-r-- 1 yoavh yoavh 129152 Aug 24 23:30 TreadmillTV_6-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh 295040 Aug 24 23:30 TreadmillTV_9-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh  96896 Aug 24 23:30 Ultimate-Chase_3-clip-4.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Ultimate-Chase_8-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 144512 Aug 24 23:30 Ultimate-Chase_9-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 103040 Aug 24 23:30 USA-Military-Channel_0-clip-6.npy
-rw-rw-r-- 1 yoavh yoavh  96896 Aug 24 23:30 USA-Military-Channel_10-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh  95360 Aug 24 23:30 USA-Military-Channel_11-clip-2.npy
-rw-rw-r-- 1 yoavh yoavh 110720 Aug 24 23:30 WarLeaks-Military-Blog_9-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh 103040 Aug 24 23:30 Weathershot_3-clip-1.npy
-rw-rw-r-- 1 yoavh yoavh 138368 Aug 24 23:30 Weathershot_7-clip-0.npy
-rw-rw-r-- 1 yoavh yoavh  93824 Aug 24 23:30 Welker-Farms-Inc_3-clip-4.npy
"""

rex2 = re.compile(r'(\d+)\s+(?:\S+\s+){3}(\S+)\.npy$')
d_local={}
for x in local_ll.splitlines():
    if not x.strip():
        continue
    m = rex2.search(x)
    if m:
        d_local[m.group(2)] = m.group(1)

print(d_local)

assert len(d_local) == len(d_remote)
assert set(d_local.keys()) == set(d_remote.keys())
assert set(d_local.values()) == set(d_remote.values())
assert d_local == d_remote


proc="""
--- Processing: 4k-Relaxation_12-clip-6.mp4 ---
Extracted 190 frames from 4k-Relaxation_12-clip-6.mp4 at 1 FPS.
Saved 190 vectors to /mnt/gcs/wild_videos_embs/4k-Relaxation_12-clip-6.npy with shape (190, 384)

--- Processing: 4k-Relaxation_3-clip-4.mp4 ---
Extracted 118 frames from 4k-Relaxation_3-clip-4.mp4 at 1 FPS.
Saved 118 vectors to /mnt/gcs/wild_videos_embs/4k-Relaxation_3-clip-4.npy with shape (118, 384)

--- Processing: AiirSource-Military_1-clip-0.mp4 ---
Extracted 70 frames from AiirSource-Military_1-clip-0.mp4 at 1 FPS.
Saved 70 vectors to /mnt/gcs/wild_videos_embs/AiirSource-Military_1-clip-0.npy with shape (70, 384)

--- Processing: AiirSource-Military_12-manual.mp4 ---
Extracted 66 frames from AiirSource-Military_12-manual.mp4 at 1 FPS.
Saved 66 vectors to /mnt/gcs/wild_videos_embs/AiirSource-Military_12-manual.npy with shape (66, 384)

--- Processing: AiirSource-Military_7-clip-1.mp4 ---
Extracted 65 frames from AiirSource-Military_7-clip-1.mp4 at 1 FPS.
Saved 65 vectors to /mnt/gcs/wild_videos_embs/AiirSource-Military_7-clip-1.npy with shape (65, 384)

--- Processing: Army-military-2018_8-clip-73.mp4 ---
Extracted 78 frames from Army-military-2018_8-clip-73.mp4 at 1 FPS.
Saved 78 vectors to /mnt/gcs/wild_videos_embs/Army-military-2018_8-clip-73.npy with shape (78, 384)

--- Processing: BC-Bushcraft_10-clip-8.mp4 ---
Extracted 86 frames from BC-Bushcraft_10-clip-8.mp4 at 1 FPS.
Saved 86 vectors to /mnt/gcs/wild_videos_embs/BC-Bushcraft_10-clip-8.npy with shape (86, 384)

--- Processing: BC-Bushcraft_11-clip-31.mp4 ---
Extracted 204 frames from BC-Bushcraft_11-clip-31.mp4 at 1 FPS.
Saved 204 vectors to /mnt/gcs/wild_videos_embs/BC-Bushcraft_11-clip-31.npy with shape (204, 384)

--- Processing: BC-Bushcraft_2-clip-2.mp4 ---
Extracted 66 frames from BC-Bushcraft_2-clip-2.mp4 at 1 FPS.
Saved 66 vectors to /mnt/gcs/wild_videos_embs/BC-Bushcraft_2-clip-2.npy with shape (66, 384)

--- Processing: BC-Bushcraft_9-clip-5.mp4 ---
Extracted 64 frames from BC-Bushcraft_9-clip-5.mp4 at 1 FPS.
Saved 64 vectors to /mnt/gcs/wild_videos_embs/BC-Bushcraft_9-clip-5.npy with shape (64, 384)

--- Processing: Bertram-Craft_12-clip-25.mp4 ---
Extracted 66 frames from Bertram-Craft_12-clip-25.mp4 at 1 FPS.
Saved 66 vectors to /mnt/gcs/wild_videos_embs/Bertram-Craft_12-clip-25.npy with shape (66, 384)

--- Processing: Bertram-Craft_2-clip-3.mp4 ---
Extracted 66 frames from Bertram-Craft_2-clip-3.mp4 at 1 FPS.
Saved 66 vectors to /mnt/gcs/wild_videos_embs/Bertram-Craft_2-clip-3.npy with shape (66, 384)

--- Processing: Bertram-Craft_5-clip-33.mp4 ---
Extracted 88 frames from Bertram-Craft_5-clip-33.mp4 at 1 FPS.
Saved 88 vectors to /mnt/gcs/wild_videos_embs/Bertram-Craft_5-clip-33.npy with shape (88, 384)

--- Processing: Chad-Zuber_10-clip-20.mp4 ---
Extracted 66 frames from Chad-Zuber_10-clip-20.mp4 at 1 FPS.
Saved 66 vectors to /mnt/gcs/wild_videos_embs/Chad-Zuber_10-clip-20.npy with shape (66, 384)

--- Processing: Chad-Zuber_4-clip-2.mp4 ---
Extracted 71 frames from Chad-Zuber_4-clip-2.mp4 at 1 FPS.
Saved 71 vectors to /mnt/gcs/wild_videos_embs/Chad-Zuber_4-clip-2.npy with shape (71, 384)

--- Processing: Chad-Zuber_5-clip-0.mp4 ---
Extracted 61 frames from Chad-Zuber_5-clip-0.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Chad-Zuber_5-clip-0.npy with shape (61, 384)

--- Processing: Climate-Change_0-clip-4.mp4 ---
Extracted 65 frames from Climate-Change_0-clip-4.mp4 at 1 FPS.
Saved 65 vectors to /mnt/gcs/wild_videos_embs/Climate-Change_0-clip-4.npy with shape (65, 384)

--- Processing: Climate-Change_2-clip-4.mp4 ---
Extracted 61 frames from Climate-Change_2-clip-4.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Climate-Change_2-clip-4.npy with shape (61, 384)

--- Processing: Climate-Change_6-clip-7.mp4 ---
Extracted 64 frames from Climate-Change_6-clip-7.mp4 at 1 FPS.
Saved 64 vectors to /mnt/gcs/wild_videos_embs/Climate-Change_6-clip-7.npy with shape (64, 384)

--- Processing: Climate-Change_7-clip-3.mp4 ---
Extracted 62 frames from Climate-Change_7-clip-3.mp4 at 1 FPS.
Saved 62 vectors to /mnt/gcs/wild_videos_embs/Climate-Change_7-clip-3.npy with shape (62, 384)

--- Processing: Dan-Robinson_3-clip-3.mp4 ---
Extracted 61 frames from Dan-Robinson_3-clip-3.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Dan-Robinson_3-clip-3.npy with shape (61, 384)

--- Processing: Disaster-Compilations_10-clip-0.mp4 ---
Extracted 66 frames from Disaster-Compilations_10-clip-0.mp4 at 1 FPS.
Saved 66 vectors to /mnt/gcs/wild_videos_embs/Disaster-Compilations_10-clip-0.npy with shape (66, 384)

--- Processing: Disaster-Compilations_5-clip-0.mp4 ---
Extracted 61 frames from Disaster-Compilations_5-clip-0.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Disaster-Compilations_5-clip-0.npy with shape (61, 384)

--- Processing: Disaster-Compilations_9-clip-2.mp4 ---
Extracted 68 frames from Disaster-Compilations_9-clip-2.mp4 at 1 FPS.
Saved 68 vectors to /mnt/gcs/wild_videos_embs/Disaster-Compilations_9-clip-2.npy with shape (68, 384)

--- Processing: Gung-Ho-Vids_0-clip-2.mp4 ---
Extracted 90 frames from Gung-Ho-Vids_0-clip-2.mp4 at 1 FPS.
Saved 90 vectors to /mnt/gcs/wild_videos_embs/Gung-Ho-Vids_0-clip-2.npy with shape (90, 384)

--- Processing: Gung-Ho-Vids_12-clip-1.mp4 ---
Extracted 61 frames from Gung-Ho-Vids_12-clip-1.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Gung-Ho-Vids_12-clip-1.npy with shape (61, 384)

--- Processing: Gung-Ho-Vids_2-clip-2.mp4 ---
Extracted 64 frames from Gung-Ho-Vids_2-clip-2.mp4 at 1 FPS.
Saved 64 vectors to /mnt/gcs/wild_videos_embs/Gung-Ho-Vids_2-clip-2.npy with shape (64, 384)

--- Processing: Gung-Ho-Vids_5-clip-2.mp4 ---
Extracted 62 frames from Gung-Ho-Vids_5-clip-2.mp4 at 1 FPS.
Saved 62 vectors to /mnt/gcs/wild_videos_embs/Gung-Ho-Vids_5-clip-2.npy with shape (62, 384)

--- Processing: Gung-Ho-Vids_8-clip-0.mp4 ---
Extracted 61 frames from Gung-Ho-Vids_8-clip-0.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Gung-Ho-Vids_8-clip-0.npy with shape (61, 384)

--- Processing: Hamiltonville-Farm_1-clip-4.mp4 ---
Extracted 65 frames from Hamiltonville-Farm_1-clip-4.mp4 at 1 FPS.
Saved 65 vectors to /mnt/gcs/wild_videos_embs/Hamiltonville-Farm_1-clip-4.npy with shape (65, 384)

--- Processing: Hamiltonville-Farm_2-clip-0.mp4 ---
Extracted 63 frames from Hamiltonville-Farm_2-clip-0.mp4 at 1 FPS.
Saved 63 vectors to /mnt/gcs/wild_videos_embs/Hamiltonville-Farm_2-clip-0.npy with shape (63, 384)

--- Processing: Hamiltonville-Farm_6-clip-18.mp4 ---
Extracted 68 frames from Hamiltonville-Farm_6-clip-18.mp4 at 1 FPS.
Saved 68 vectors to /mnt/gcs/wild_videos_embs/Hamiltonville-Farm_6-clip-18.npy with shape (68, 384)

--- Processing: Hamiltonville-Farm_8-clip-3.mp4 ---
Extracted 62 frames from Hamiltonville-Farm_8-clip-3.mp4 at 1 FPS.
Saved 62 vectors to /mnt/gcs/wild_videos_embs/Hamiltonville-Farm_8-clip-3.npy with shape (62, 384)

--- Processing: How-Farms-Work_10-clip-1.mp4 ---
Extracted 147 frames from How-Farms-Work_10-clip-1.mp4 at 1 FPS.
Saved 147 vectors to /mnt/gcs/wild_videos_embs/How-Farms-Work_10-clip-1.npy with shape (147, 384)

--- Processing: How-Farms-Work_3-clip-2.mp4 ---
Extracted 64 frames from How-Farms-Work_3-clip-2.mp4 at 1 FPS.
Saved 64 vectors to /mnt/gcs/wild_videos_embs/How-Farms-Work_3-clip-2.npy with shape (64, 384)

--- Processing: How-Farms-Work_5-clip-0.mp4 ---
Extracted 61 frames from How-Farms-Work_5-clip-0.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/How-Farms-Work_5-clip-0.npy with shape (61, 384)

--- Processing: How-Farms-Work_8-clip-4.mp4 ---
Extracted 72 frames from How-Farms-Work_8-clip-4.mp4 at 1 FPS.
Saved 72 vectors to /mnt/gcs/wild_videos_embs/How-Farms-Work_8-clip-4.npy with shape (72, 384)

--- Processing: How-Farms-Work_9-manual.mp4 ---
Extracted 141 frames from How-Farms-Work_9-manual.mp4 at 1 FPS.
Saved 141 vectors to /mnt/gcs/wild_videos_embs/How-Farms-Work_9-manual.npy with shape (141, 384)

--- Processing: Joe-Robinet_0-clip-23.mp4 ---
Extracted 65 frames from Joe-Robinet_0-clip-23.mp4 at 1 FPS.
Saved 65 vectors to /mnt/gcs/wild_videos_embs/Joe-Robinet_0-clip-23.npy with shape (65, 384)

--- Processing: Joe-Robinet_2-clip-5.mp4 ---
Extracted 75 frames from Joe-Robinet_2-clip-5.mp4 at 1 FPS.
Saved 75 vectors to /mnt/gcs/wild_videos_embs/Joe-Robinet_2-clip-5.npy with shape (75, 384)

--- Processing: Joe-Robinet_4-clip-0.mp4 ---
Extracted 178 frames from Joe-Robinet_4-clip-0.mp4 at 1 FPS.
Saved 178 vectors to /mnt/gcs/wild_videos_embs/Joe-Robinet_4-clip-0.npy with shape (178, 384)

--- Processing: Joe-Robinet_5-clip-23.mp4 ---
Extracted 61 frames from Joe-Robinet_5-clip-23.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Joe-Robinet_5-clip-23.npy with shape (61, 384)

--- Processing: John-Suscovich_0-clip-1.mp4 ---
Extracted 76 frames from John-Suscovich_0-clip-1.mp4 at 1 FPS.
Saved 76 vectors to /mnt/gcs/wild_videos_embs/John-Suscovich_0-clip-1.npy with shape (76, 384)

--- Processing: John-Suscovich_10-manual.mp4 ---
Extracted 124 frames from John-Suscovich_10-manual.mp4 at 1 FPS.
Saved 124 vectors to /mnt/gcs/wild_videos_embs/John-Suscovich_10-manual.npy with shape (124, 384)

--- Processing: John-Suscovich_12-clip-0.mp4 ---
Extracted 79 frames from John-Suscovich_12-clip-0.mp4 at 1 FPS.
Saved 79 vectors to /mnt/gcs/wild_videos_embs/John-Suscovich_12-clip-0.npy with shape (79, 384)

--- Processing: John-Suscovich_2-clip-3.mp4 ---
Extracted 95 frames from John-Suscovich_2-clip-3.mp4 at 1 FPS.
Saved 95 vectors to /mnt/gcs/wild_videos_embs/John-Suscovich_2-clip-3.npy with shape (95, 384)

--- Processing: John-Suscovich_3-manual.mp4 ---
Extracted 76 frames from John-Suscovich_3-manual.mp4 at 1 FPS.
Saved 76 vectors to /mnt/gcs/wild_videos_embs/John-Suscovich_3-manual.npy with shape (76, 384)

--- Processing: King-Kong-Amazon_11-clip-7.mp4 ---
Extracted 65 frames from King-Kong-Amazon_11-clip-7.mp4 at 1 FPS.
Saved 65 vectors to /mnt/gcs/wild_videos_embs/King-Kong-Amazon_11-clip-7.npy with shape (65, 384)

--- Processing: King-Kong-Amazon_5-clip-14.mp4 ---
Extracted 89 frames from King-Kong-Amazon_5-clip-14.mp4 at 1 FPS.
Saved 89 vectors to /mnt/gcs/wild_videos_embs/King-Kong-Amazon_5-clip-14.npy with shape (89, 384)

--- Processing: MilitaryNotes_10-clip-0.mp4 ---
Extracted 62 frames from MilitaryNotes_10-clip-0.mp4 at 1 FPS.
Saved 62 vectors to /mnt/gcs/wild_videos_embs/MilitaryNotes_10-clip-0.npy with shape (62, 384)

--- Processing: MilitaryNotes_11-clip-1.mp4 ---
Extracted 62 frames from MilitaryNotes_11-clip-1.mp4 at 1 FPS.
Saved 62 vectors to /mnt/gcs/wild_videos_embs/MilitaryNotes_11-clip-1.npy with shape (62, 384)

--- Processing: MilitaryNotes_8-clip-0.mp4 ---
Extracted 70 frames from MilitaryNotes_8-clip-0.mp4 at 1 FPS.
Saved 70 vectors to /mnt/gcs/wild_videos_embs/MilitaryNotes_8-clip-0.npy with shape (70, 384)

--- Processing: Millennial-Farmer_0-clip-11.mp4 ---
Extracted 66 frames from Millennial-Farmer_0-clip-11.mp4 at 1 FPS.
Saved 66 vectors to /mnt/gcs/wild_videos_embs/Millennial-Farmer_0-clip-11.npy with shape (66, 384)

--- Processing: Millennial-Farmer_1-clip-11.mp4 ---
Extracted 61 frames from Millennial-Farmer_1-clip-11.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Millennial-Farmer_1-clip-11.npy with shape (61, 384)

--- Processing: Millennial-Farmer_7-clip-13.mp4 ---
Extracted 61 frames from Millennial-Farmer_7-clip-13.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Millennial-Farmer_7-clip-13.npy with shape (61, 384)

--- Processing: Millennial-Farmer_8-clip-16.mp4 ---
Extracted 95 frames from Millennial-Farmer_8-clip-16.mp4 at 1 FPS.
Saved 95 vectors to /mnt/gcs/wild_videos_embs/Millennial-Farmer_8-clip-16.npy with shape (95, 384)

--- Processing: Natural-Disaster_10-clip-0.mp4 ---
Extracted 62 frames from Natural-Disaster_10-clip-0.mp4 at 1 FPS.
Saved 62 vectors to /mnt/gcs/wild_videos_embs/Natural-Disaster_10-clip-0.npy with shape (62, 384)

--- Processing: Nick-Gaillard_1-clip-2.mp4 ---
Extracted 70 frames from Nick-Gaillard_1-clip-2.mp4 at 1 FPS.
Saved 70 vectors to /mnt/gcs/wild_videos_embs/Nick-Gaillard_1-clip-2.npy with shape (70, 384)

--- Processing: Nick-Gaillard_4-clip-2.mp4 ---
Extracted 84 frames from Nick-Gaillard_4-clip-2.mp4 at 1 FPS.
Saved 84 vectors to /mnt/gcs/wild_videos_embs/Nick-Gaillard_4-clip-2.npy with shape (84, 384)

--- Processing: Olly_s-Farm_1-clip-5.mp4 ---
Extracted 65 frames from Olly_s-Farm_1-clip-5.mp4 at 1 FPS.
Saved 65 vectors to /mnt/gcs/wild_videos_embs/Olly_s-Farm_1-clip-5.npy with shape (65, 384)

--- Processing: Olly_s-Farm_6-clip-1.mp4 ---
Extracted 61 frames from Olly_s-Farm_6-clip-1.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Olly_s-Farm_6-clip-1.npy with shape (61, 384)

--- Processing: Peterson-Farm-Bros_5-clip-2.mp4 ---
Extracted 78 frames from Peterson-Farm-Bros_5-clip-2.mp4 at 1 FPS.
Saved 78 vectors to /mnt/gcs/wild_videos_embs/Peterson-Farm-Bros_5-clip-2.npy with shape (78, 384)

--- Processing: Peterson-Farm-Bros_6-clip-4.mp4 ---
Extracted 144 frames from Peterson-Farm-Bros_6-clip-4.mp4 at 1 FPS.
Saved 144 vectors to /mnt/gcs/wild_videos_embs/Peterson-Farm-Bros_6-clip-4.npy with shape (144, 384)

--- Processing: Primal-Earth-Sounds_0-clip-44.mp4 ---
Extracted 221 frames from Primal-Earth-Sounds_0-clip-44.mp4 at 1 FPS.
Saved 221 vectors to /mnt/gcs/wild_videos_embs/Primal-Earth-Sounds_0-clip-44.npy with shape (221, 384)

--- Processing: Primal-Earth-Sounds_9-clip-26.mp4 ---
Extracted 63 frames from Primal-Earth-Sounds_9-clip-26.mp4 at 1 FPS.
Saved 63 vectors to /mnt/gcs/wild_videos_embs/Primal-Earth-Sounds_9-clip-26.npy with shape (63, 384)

--- Processing: Primitive-Technology_11-clip-8.mp4 ---
Extracted 62 frames from Primitive-Technology_11-clip-8.mp4 at 1 FPS.
Saved 62 vectors to /mnt/gcs/wild_videos_embs/Primitive-Technology_11-clip-8.npy with shape (62, 384)

--- Processing: Primitive-Technology_3-clip-1.mp4 ---
Extracted 64 frames from Primitive-Technology_3-clip-1.mp4 at 1 FPS.
Saved 64 vectors to /mnt/gcs/wild_videos_embs/Primitive-Technology_3-clip-1.npy with shape (64, 384)

--- Processing: Primitive-Technology_5-clip-0.mp4 ---
Extracted 70 frames from Primitive-Technology_5-clip-0.mp4 at 1 FPS.
Saved 70 vectors to /mnt/gcs/wild_videos_embs/Primitive-Technology_5-clip-0.npy with shape (70, 384)

--- Processing: Primitive-Technology_6-clip-3.mp4 ---
Extracted 61 frames from Primitive-Technology_6-clip-3.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Primitive-Technology_6-clip-3.npy with shape (61, 384)

--- Processing: Primitive-Technology_7-clip-4.mp4 ---
Extracted 66 frames from Primitive-Technology_7-clip-4.mp4 at 1 FPS.
Saved 66 vectors to /mnt/gcs/wild_videos_embs/Primitive-Technology_7-clip-4.npy with shape (66, 384)

--- Processing: RealAgriculture_9-clip-1.mp4 ---
Extracted 82 frames from RealAgriculture_9-clip-1.mp4 at 1 FPS.
Saved 82 vectors to /mnt/gcs/wild_videos_embs/RealAgriculture_9-clip-1.npy with shape (82, 384)

--- Processing: Sandboxx_0-clip-4.mp4 ---
Extracted 72 frames from Sandboxx_0-clip-4.mp4 at 1 FPS.
Saved 72 vectors to /mnt/gcs/wild_videos_embs/Sandboxx_0-clip-4.npy with shape (72, 384)

--- Processing: Sandboxx_7-clip-0.mp4 ---
Extracted 61 frames from Sandboxx_7-clip-0.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Sandboxx_7-clip-0.npy with shape (61, 384)

--- Processing: Sandboxx_9-clip-4.mp4 ---
Extracted 64 frames from Sandboxx_9-clip-4.mp4 at 1 FPS.
Saved 64 vectors to /mnt/gcs/wild_videos_embs/Sandboxx_9-clip-4.npy with shape (64, 384)

--- Processing: Survival-Instinct_11-clip-10.mp4 ---
Extracted 64 frames from Survival-Instinct_11-clip-10.mp4 at 1 FPS.
Saved 64 vectors to /mnt/gcs/wild_videos_embs/Survival-Instinct_11-clip-10.npy with shape (64, 384)

--- Processing: Survival-Instinct_2-clip-1.mp4 ---
Extracted 61 frames from Survival-Instinct_2-clip-1.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Survival-Instinct_2-clip-1.npy with shape (61, 384)

--- Processing: Survival-Instinct_7-clip-8.mp4 ---
Extracted 61 frames from Survival-Instinct_7-clip-8.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Survival-Instinct_7-clip-8.npy with shape (61, 384)

--- Processing: Survival-Instinct_8-clip-4.mp4 ---
Extracted 62 frames from Survival-Instinct_8-clip-4.mp4 at 1 FPS.
Saved 62 vectors to /mnt/gcs/wild_videos_embs/Survival-Instinct_8-clip-4.npy with shape (62, 384)

--- Processing: Survival-Instinct_9-clip-2.mp4 ---
Extracted 65 frames from Survival-Instinct_9-clip-2.mp4 at 1 FPS.
Saved 65 vectors to /mnt/gcs/wild_videos_embs/Survival-Instinct_9-clip-2.npy with shape (65, 384)

--- Processing: Survival-Skills-Primitive_11-clip-0.mp4 ---
Extracted 61 frames from Survival-Skills-Primitive_11-clip-0.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Survival-Skills-Primitive_11-clip-0.npy with shape (61, 384)

--- Processing: Survival-Skills-Primitive_3-clip-0.mp4 ---
Extracted 65 frames from Survival-Skills-Primitive_3-clip-0.mp4 at 1 FPS.
Saved 65 vectors to /mnt/gcs/wild_videos_embs/Survival-Skills-Primitive_3-clip-0.npy with shape (65, 384)

--- Processing: TK-Hinshaw_5-clip-0.mp4 ---
Extracted 86 frames from TK-Hinshaw_5-clip-0.mp4 at 1 FPS.
Saved 86 vectors to /mnt/gcs/wild_videos_embs/TK-Hinshaw_5-clip-0.npy with shape (86, 384)

--- Processing: TK-Hinshaw_7-clip-0.mp4 ---
Extracted 64 frames from TK-Hinshaw_7-clip-0.mp4 at 1 FPS.
Saved 64 vectors to /mnt/gcs/wild_videos_embs/TK-Hinshaw_7-clip-0.npy with shape (64, 384)

--- Processing: Tornado-Trackers_1-clip-0.mp4 ---
Extracted 61 frames from Tornado-Trackers_1-clip-0.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Tornado-Trackers_1-clip-0.npy with shape (61, 384)

--- Processing: Tornado-Trackers_10-clip-4.mp4 ---
Extracted 61 frames from Tornado-Trackers_10-clip-4.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Tornado-Trackers_10-clip-4.npy with shape (61, 384)

--- Processing: Tornado-Trackers_6-clip-2.mp4 ---
Extracted 68 frames from Tornado-Trackers_6-clip-2.mp4 at 1 FPS.
Saved 68 vectors to /mnt/gcs/wild_videos_embs/Tornado-Trackers_6-clip-2.npy with shape (68, 384)

--- Processing: TreadmillTV_2-clip-1.mp4 ---
Extracted 68 frames from TreadmillTV_2-clip-1.mp4 at 1 FPS.
Saved 68 vectors to /mnt/gcs/wild_videos_embs/TreadmillTV_2-clip-1.npy with shape (68, 384)

--- Processing: TreadmillTV_3-clip-12.mp4 ---
Extracted 62 frames from TreadmillTV_3-clip-12.mp4 at 1 FPS.
Saved 62 vectors to /mnt/gcs/wild_videos_embs/TreadmillTV_3-clip-12.npy with shape (62, 384)

--- Processing: TreadmillTV_6-clip-2.mp4 ---
Extracted 84 frames from TreadmillTV_6-clip-2.mp4 at 1 FPS.
Saved 84 vectors to /mnt/gcs/wild_videos_embs/TreadmillTV_6-clip-2.npy with shape (84, 384)

--- Processing: TreadmillTV_9-clip-2.mp4 ---
Extracted 192 frames from TreadmillTV_9-clip-2.mp4 at 1 FPS.
Saved 192 vectors to /mnt/gcs/wild_videos_embs/TreadmillTV_9-clip-2.npy with shape (192, 384)

--- Processing: USA-Military-Channel_0-clip-6.mp4 ---
Extracted 67 frames from USA-Military-Channel_0-clip-6.mp4 at 1 FPS.
Saved 67 vectors to /mnt/gcs/wild_videos_embs/USA-Military-Channel_0-clip-6.npy with shape (67, 384)

--- Processing: USA-Military-Channel_10-clip-2.mp4 ---
Extracted 63 frames from USA-Military-Channel_10-clip-2.mp4 at 1 FPS.
Saved 63 vectors to /mnt/gcs/wild_videos_embs/USA-Military-Channel_10-clip-2.npy with shape (63, 384)

--- Processing: USA-Military-Channel_11-clip-2.mp4 ---
Extracted 62 frames from USA-Military-Channel_11-clip-2.mp4 at 1 FPS.
Saved 62 vectors to /mnt/gcs/wild_videos_embs/USA-Military-Channel_11-clip-2.npy with shape (62, 384)

--- Processing: Ultimate-Chase_3-clip-4.mp4 ---
Extracted 63 frames from Ultimate-Chase_3-clip-4.mp4 at 1 FPS.
Saved 63 vectors to /mnt/gcs/wild_videos_embs/Ultimate-Chase_3-clip-4.npy with shape (63, 384)

--- Processing: Ultimate-Chase_8-clip-0.mp4 ---
Extracted 61 frames from Ultimate-Chase_8-clip-0.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Ultimate-Chase_8-clip-0.npy with shape (61, 384)

--- Processing: Ultimate-Chase_9-clip-0.mp4 ---
Extracted 94 frames from Ultimate-Chase_9-clip-0.mp4 at 1 FPS.
Saved 94 vectors to /mnt/gcs/wild_videos_embs/Ultimate-Chase_9-clip-0.npy with shape (94, 384)

--- Processing: WarLeaks-Military-Blog_9-clip-0.mp4 ---
Extracted 72 frames from WarLeaks-Military-Blog_9-clip-0.mp4 at 1 FPS.
Saved 72 vectors to /mnt/gcs/wild_videos_embs/WarLeaks-Military-Blog_9-clip-0.npy with shape (72, 384)

--- Processing: Weathershot_3-clip-1.mp4 ---
Extracted 67 frames from Weathershot_3-clip-1.mp4 at 1 FPS.
Saved 67 vectors to /mnt/gcs/wild_videos_embs/Weathershot_3-clip-1.npy with shape (67, 384)

--- Processing: Weathershot_7-clip-0.mp4 ---
Extracted 90 frames from Weathershot_7-clip-0.mp4 at 1 FPS.
Saved 90 vectors to /mnt/gcs/wild_videos_embs/Weathershot_7-clip-0.npy with shape (90, 384)

--- Processing: Welker-Farms-Inc_3-clip-4.mp4 ---
Extracted 61 frames from Welker-Farms-Inc_3-clip-4.mp4 at 1 FPS.
Saved 61 vectors to /mnt/gcs/wild_videos_embs/Welker-Farms-Inc_3-clip-4.npy with shape (61, 384)
"""

rex3 = re.compile(rex2.pattern, re.MULTILINE)
d_local2 = {m.group(2):m.group(1) for m in rex3.finditer(local_ll)}

assert len(d_local2) == 100, f'Expected 100 videos, got {len(d_local2)=}'
assert d_local2 == d_local