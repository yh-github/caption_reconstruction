import sys
from collections import defaultdict
from pathlib import Path

from config_loader import load_config
from dev_qa import QAData
from video_link_loader import load_wild_dataset

input_text="""
run_name='llm_wild_text1' Millennial-Farmer_1-clip-11__2 bs_f1=0.8236556053161621
run_name='llm_wild_text1' Olly's-Farm_1-clip-5__2 bs_f1=0.7603437900543213
run_name='llm_wild_text1' John-Suscovich_10-manual__2 bs_f1=0.833636462688446
run_name='llm_wild_text1' John-Suscovich_10-manual__3 bs_f1=0.6068527698516846
run_name='llm_wild_text1' Hamiltonville-Farm_8-clip-3__2 bs_f1=0.5516985654830933
run_name='llm_wild_text1' How-Farms-Work_9-manual__2 bs_f1=0.7696927785873413
run_name='llm_wild_text1' How-Farms-Work_3-clip-2__2 bs_f1=0.8102477192878723
run_name='llm_wild_text1' John-Suscovich_2-clip-3__2 bs_f1=0.6686551570892334
run_name='llm_wild_text1' Peterson-Farm-Bros_6-clip-4__2 bs_f1=0.7062968015670776
run_name='llm_wild_text1' Peterson-Farm-Bros_6-clip-4__3 bs_f1=0.5418146252632141
run_name='llm_wild_text1' John-Suscovich_12-clip-0__2 bs_f1=0.7316266894340515
run_name='llm_wild_text1' John-Suscovich_12-clip-0__3 bs_f1=0.6318220496177673
run_name='llm_wild_text1' How-Farms-Work_8-clip-4__2 bs_f1=0.8248878121376038
run_name='llm_wild_text1' Hamiltonville-Farm_6-clip-18__2 bs_f1=0.548923134803772
run_name='llm_wild_text1' Olly's-Farm_6-clip-1__2 bs_f1=0.8221946358680725
run_name='llm_wild_text1' Olly's-Farm_6-clip-1__3 bs_f1=0.6636063456535339
run_name='llm_wild_text1' Olly's-Farm_6-clip-1__4 bs_f1=0.5715035796165466
run_name='llm_wild_text1' John-Suscovich_0-clip-1__2 bs_f1=0.7052331566810608
run_name='llm_wild_text1' Millennial-Farmer_8-clip-16__2 bs_f1=0.6215484738349915
run_name='llm_wild_text1' How-Farms-Work_10-clip-1__2 bs_f1=0.5939114689826965
run_name='llm_wild_text1' Hamiltonville-Farm_2-clip-0__2 bs_f1=0.6873508095741272
run_name='llm_wild_text1' How-Farms-Work_5-clip-0__2 bs_f1=0.7787784337997437
run_name='llm_wild_text1' Hamiltonville-Farm_1-clip-4__2 bs_f1=0.7792831659317017
run_name='llm_wild_text1' Hamiltonville-Farm_1-clip-4__3 bs_f1=0.5712787508964539
run_name='llm_wild_text1' Hamiltonville-Farm_1-clip-4__4 bs_f1=0.5822743773460388
run_name='llm_wild_text1' Millennial-Farmer_7-clip-13__2 bs_f1=0.5341479778289795
run_name='llm_wild_text1' RealAgriculture_9-clip-1__2 bs_f1=0.8053471446037292
run_name='llm_wild_text1' Welker-Farms-Inc_3-clip-4__2 bs_f1=0.7533809542655945
run_name='llm_wild_text1' Peterson-Farm-Bros_5-clip-2__2 bs_f1=0.5694451332092285
run_name='llm_wild_text1' Peterson-Farm-Bros_2-clip-13__2 bs_f1=0.5504590272903442
run_name='llm_wild_text1' Peterson-Farm-Bros_2-clip-13__3 bs_f1=0.7544609904289246
run_name='llm_wild_text1' John-Suscovich_3-manual__2 bs_f1=0.8945643901824951
run_name='vlm_wild_video1' Millennial-Farmer_1-clip-11__2 bs_f1=0.7702653408050537
run_name='vlm_wild_video1' Olly's-Farm_1-clip-5__2 bs_f1=0.8790715336799622
run_name='vlm_wild_video1' John-Suscovich_10-manual__2 bs_f1=0.833636462688446
run_name='vlm_wild_video1' John-Suscovich_10-manual__3 bs_f1=0.5885676145553589
run_name='vlm_wild_video1' Hamiltonville-Farm_8-clip-3__2 bs_f1=0.8204329013824463
run_name='vlm_wild_video1' How-Farms-Work_9-manual__2 bs_f1=0.7777180671691895
run_name='vlm_wild_video1' How-Farms-Work_3-clip-2__2 bs_f1=0.7545867562294006
run_name='vlm_wild_video1' John-Suscovich_2-clip-3__2 bs_f1=0.5954087972640991
run_name='vlm_wild_video1' Peterson-Farm-Bros_6-clip-4__2 bs_f1=0.677699863910675
run_name='vlm_wild_video1' Peterson-Farm-Bros_6-clip-4__3 bs_f1=0.6004921197891235
run_name='vlm_wild_video1' John-Suscovich_12-clip-0__2 bs_f1=0.6984461545944214
run_name='vlm_wild_video1' John-Suscovich_12-clip-0__3 bs_f1=0.5778902769088745
run_name='vlm_wild_video1' How-Farms-Work_8-clip-4__2 bs_f1=0.7771240472793579
run_name='vlm_wild_video1' Hamiltonville-Farm_6-clip-18__2 bs_f1=0.7708947062492371
run_name='vlm_wild_video1' Olly's-Farm_6-clip-1__2 bs_f1=0.6670897603034973
run_name='vlm_wild_video1' Olly's-Farm_6-clip-1__3 bs_f1=0.6537163853645325
run_name='vlm_wild_video1' Olly's-Farm_6-clip-1__4 bs_f1=0.6683306694030762
run_name='vlm_wild_video1' John-Suscovich_0-clip-1__2 bs_f1=0.6296950578689575
run_name='vlm_wild_video1' Millennial-Farmer_8-clip-16__2 bs_f1=0.6634336709976196
run_name='vlm_wild_video1' How-Farms-Work_10-clip-1__2 bs_f1=0.8420235514640808
run_name='vlm_wild_video1' Hamiltonville-Farm_2-clip-0__2 bs_f1=0.7074334025382996
run_name='vlm_wild_video1' How-Farms-Work_5-clip-0__2 bs_f1=0.8193525075912476
run_name='vlm_wild_video1' Hamiltonville-Farm_1-clip-4__2 bs_f1=0.7792832851409912
run_name='vlm_wild_video1' Hamiltonville-Farm_1-clip-4__3 bs_f1=0.7829139232635498
run_name='vlm_wild_video1' Hamiltonville-Farm_1-clip-4__4 bs_f1=0.6009327173233032
run_name='vlm_wild_video1' Millennial-Farmer_7-clip-13__2 bs_f1=0.7560682892799377
run_name='vlm_wild_video1' RealAgriculture_9-clip-1__2 bs_f1=0.8023850917816162
run_name='vlm_wild_video1' Welker-Farms-Inc_3-clip-4__2 bs_f1=0.9187568426132202
run_name='vlm_wild_video1' Peterson-Farm-Bros_5-clip-2__2 bs_f1=0.6016191840171814
run_name='vlm_wild_video1' Peterson-Farm-Bros_2-clip-13__2 bs_f1=0.7447953820228577
run_name='vlm_wild_video1' Peterson-Farm-Bros_2-clip-13__3 bs_f1=0.851232647895813
run_name='vlm_wild_video1' John-Suscovich_3-manual__2 bs_f1=0.9121108055114746
"""

import re
import pandas as pd
import numpy as np

def load_questions():
    qa_by_id: defaultdict[str, list[QAData]] = defaultdict(list)
    config = load_config(sys.argv[1])
    for v in load_wild_dataset(Path(config['data_config']['path'])):
        qa_by_id[v.video_id].append(QAData.model_validate(v.model_dump()))

    for video_id, qa_data in qa_by_id.items():
        for qi, qa in enumerate(qa_data, start=1):
            yield {
                'video__q_id': f'{video_id}__{qi}',
                'question': qa.question,
                'question_type': qa.question_type
            }

qs_df = pd.DataFrame(load_questions())

def data_as_dicts(data):
    rex = re.compile(r"run_name='(?P<method>.*?)' (?P<video__q_id>(?P<vid_id>\S+?)__(?P<q_id>\S+)) bs_f1=(?P<bs_f1>\S+)$")
    for line in input_text.split("\n"):
        m = rex.match(line)
        if m:
            yield m.groupdict()
            

df = pd.DataFrame(data_as_dicts(input_text))
df['bs_f1'] = df['bs_f1'].astype(float)

# Calculate statistics per method
stats = df.groupby('method')['bs_f1'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).round(4)

print("\nStatistics per method:")
print(stats)

# Create comparison dataframe
video_comparison = df.pivot(index='video__q_id', columns='method', values='bs_f1')

# Calculate rankings (1 is best)

rankings = video_comparison.rank(axis=1, ascending=False)
avg_ranks = rankings.mean()

print("\nAverage ranks (lower is better):")
for method, rank in avg_ranks.items():
    print(f"{method}: {rank:.2f}")

# Count number of times each method is best with minimum delta difference
# delta = 0.01
#
#
# def check_significant_win(row):
#     sorted_scores = sorted(row.items(), key=lambda x: x[1], reverse=True)
#     if len(sorted_scores) < 2:
#         return {method: False for method in row.keys()}
#
#     result = {}
#     for method, score in sorted_scores:
#         # Method is best and difference vs second best is >= delta
#         result[method] = (score == sorted_scores[0][1] and
#                           score - sorted_scores[1][1] >= delta)
#     return result
#
#
# significant_wins = video_comparison.apply(check_significant_win, axis=1)
# best_counts = significant_wins.sum()
#
# total_comparisons = len(rankings)
# print("\nNumber of times each method performs best (with min delta=0.01):")
# for method, count in best_counts.items():
#     percentage = (count / total_comparisons) * 100
#     print(f"{method}: {count} ({percentage:.1f}%)")



# Compute significant wins without storing dictionaries
def check_significant_win(row):
    sorted_scores = row.sort_values(ascending=False)
    if len(sorted_scores) < 2:
        return pd.Series({method: False for method in row.index})

    # Create a mask for the best method based on the delta
    result = {method: False for method in row.index}
    if sorted_scores.iloc[0] - sorted_scores.iloc[1] >= delta:
        result[sorted_scores.index[0]] = True
    return pd.Series(result)

for delta in [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
    significant_wins = video_comparison.apply(check_significant_win, axis=1)
    best_counts = significant_wins.sum()

    # Print results
    total_comparisons = len(rankings)
    print(f"\nNumber of times each method performs best (with min {delta=}):")
    for method, count in best_counts.items():
        percentage = (count / total_comparisons) * 100
        print(f"{method}: {count} ({percentage:.1f}%)")


# Calculate standard deviation and mean of bs_f1 per method
group_stats = df.groupby('method')['bs_f1'].agg(['mean', 'std'])

# Calculate a baseline delta as one standard deviation
baseline_delta = group_stats['std'].mean()
print(f"\nBaseline delta (mean of std): {baseline_delta:.4f}")

# Optionally, calculate 95% confidence intervals

group_stats['ci95_min'] = group_stats['mean'] - 1.96 * group_stats['std'] / np.sqrt(group_stats['std'])
group_stats['ci95_max'] = group_stats['mean'] + 1.96 * group_stats['std'] / np.sqrt(group_stats['std'])
print("\nConfidence intervals (95%) per method:")
print(group_stats[['mean', 'ci95_min', 'ci95_max']])

print('#################')

# print(qs_df.columns)
# print(qs_df.head())

qs_flattened = qs_df.explode('question_type')
merged_df = df.merge(qs_flattened, on='video__q_id')

results = merged_df.groupby(['method', 'question_type']).agg(
    mean_bs_f1=('bs_f1', 'mean'),
    std_bs_f1=('bs_f1', 'std'),
    question_count=('bs_f1', 'count')
).reset_index()

# Round results for clarity
results = results.round(4)

print("Results per method per question_type:")
print(results)


