import yaml
import logging
from common_utils.jsonables import dump_model_compact_json
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from pydantic import BaseModel

from data.data_loaders import WildLoader
from data_models.captions_only import CaptionedVideo

class AnalysisArgs(BaseModel):
    method1: str = 'CaptionedVideo__pro_d_one_shot_v1__t=1'
    method2: str = 'video_embeddings__MeanClosestVectors'
    metric: str = 'cos_sim_mean'
    use_z_score: bool = False
    experiments_csvs_dir: Path = Path('results/for_analysis/')
    plot_output_dir: Path = Path('results/plots/ranking_stability/')

def create_bin_labels(bins: list[int], open_ended_last: bool = False) -> list[str]:
    """
    Creates a list of string labels from a list of numerical bin edges.

    Args:
        bins: A list of numerical bin edges.
        open_ended_last: If True, the last label will be "X+". Otherwise, it will be a range.

    Example:
        `create_bin_labels([0, 5, 10, 50])` -> `['0-5', '5-10', '50+']`
        `create_bin_labels([0, 20, 40, 60], open_ended_last=False)` -> `['0-20', '20-40', '40-60']`
    """
    labels = []
    num_labels = len(bins) - 1
    if num_labels <= 0:
        return []

    for i in range(num_labels):
        if i == num_labels - 1 and open_ended_last:
            labels.append(f"{bins[i]}+")
        else:
            labels.append(f"{bins[i]}-{bins[i+1]}")
    return labels


def prep(df):
    filter_out = [
        'CaptionedVideo__BaselineRepeatStrategy',
    ]
    df['method'] = df['data_type'] + '__' + df['recon_strategy']
    df['num_masked'] = df['masked'].apply(lambda x: len(eval(x)))
    df['first_masked'] = df['masked'].apply(lambda x: min(eval(x)))
    bins = [0, 20, 40, 60]
    df['first_masked_bin'] = pd.cut(df['first_masked'], bins=bins, right=False, labels=create_bin_labels(bins))
    filtered_df = df[~df['method'].isin(filter_out)].copy()
    return filtered_df

def load_dfs(dir:Path|str):
    df_list = []
    df_list_z = []

    paths = list(Path(dir).glob("*.csv"))
    assert paths, f"*.csv not found in {dir}"

    for path in paths:
        df = pd.read_csv(path, index_col=0)
        if 'z_score' in str(path):
            df_list_z.append(df)
        else:
            #   df['data_type'] = path.stem
            df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Total length: {len(combined_df)=}")

    combined_df_z = pd.concat(df_list_z, ignore_index=True)
    logging.info(f"Total length: {len(combined_df_z)=}")
    return prep(combined_df), prep(combined_df_z)

def get_high_diff_ranks(args:AnalysisArgs) -> dict[str, list[str]]:
    df_1, df_z = load_dfs(args.experiments_csvs_dir)
    combined_df = df_1 if not args.use_z_score else df_z

    selected_methods = [args.method1, args.method2]
    ranking_column = args.metric
    assert ranking_column in combined_df.columns, f"Metric must be {combined_df.columns = }"
    filtered_df = combined_df[combined_df['method'].isin(selected_methods)].copy()

    grouped_df = filtered_df.groupby(['method', 'video_id'])[ranking_column].mean().reset_index()

    ranked_videos = grouped_df.groupby('method')[['video_id', ranking_column]].apply(
        lambda x: x.sort_values(by=ranking_column, ascending=True).reset_index(drop=True)
    )

    ranked_videos['rank'] = ranked_videos.groupby('method').cumcount() + 1

    logging.info(ranked_videos.head())

    merged_ranks = pd.merge(
        ranked_videos[ranked_videos.index.get_level_values('method') == selected_methods[0]],
        ranked_videos[ranked_videos.index.get_level_values('method') == selected_methods[1]],
        on='video_id',
        suffixes=('_method1', '_method2')
    )

    merged_ranks['rank_difference'] = merged_ranks['rank_method1'] - merged_ranks['rank_method2']
    merged_ranks['rank_ratio'] = merged_ranks['rank_method1'] / merged_ranks['rank_method2']

    logging.info(merged_ranks.head())

    def get_id_list(df:DataFrame) -> list[str]:
        return df['video_id'].tolist()

    method1_high_method2_low = merged_ranks.sort_values(by='rank_difference', ascending=False).head(5)
    method1_low_method2_high = merged_ranks.sort_values(by='rank_difference', ascending=True).head(5)

    return {
        args.method1: get_id_list(method1_high_method2_low),
        args.method2: get_id_list(method1_low_method2_high)
    }

class TwoVideoLists(BaseModel):
    captioned_video_list1: list[CaptionedVideo]
    captioned_video_list2: list[CaptionedVideo]

class CompareResult(BaseModel):
    characteristics1: list[str]
    characteristics2: list[str]
    key_differences: list[str]

def build_prompt(lists: TwoVideoLists) -> str:
    return f"""\
We have two lists of captioned videos that differ in some systematic way. Your task is to identify these differences.

**Analysis Framework:**
1. **Within-group similarities:** What patterns do videos within each list share?
2. **Between-group differences:** How do the two groups differ from each other?

**Focus Areas:**
* Video type/genre characteristics
* Pacing and rhythm patterns  
* Structural elements and organization
* Narrative techniques or visual surprises
* Continuity vs. disconnected segments
* Production style or technical aspects
* Movement and action: fast/slow, a lot of action or hardly any, etc.
* Narrative: complex or straightforward
* Audience: what age or prior knowledge should someone be/have to understand the video?
* Any other glaring difference

**Output Format:**
* **List 1 Characteristics:** [Common patterns within list 1]
* **List 2 Characteristics:** [Common patterns within list 2]  
* **Key Differences:** [How the groups contrast with each other]

If unsure about any analysis point, you may include up to 2 alternative suggestions.

Here are the lists:

{dump_model_compact_json(lists, code_block=True)}    
"""

def main(args:AnalysisArgs):
    dataloader = WildLoader("datasets/wildQA/captions__wild2/", limit=100)

    def get_vid_data(ids: list[str]):
        return [dataloader.find(vid_id) for vid_id in ids]

    def struct(ids_dict: dict[str, list[str]]) -> TwoVideoLists:
        assert len(ids_dict.keys()) == 2
        vids = [get_vid_data(ids) for ids in ids_dict.values()]
        return TwoVideoLists(captioned_video_list1=vids[0], captioned_video_list2=vids[1])

    ids = get_high_diff_ranks(args)

    lists = struct(ids)
    prompt=build_prompt(lists)

    print("ARGS:")
    print(yaml.dump(args.model_dump(), sort_keys=False))
    print()
    print("IDS:")
    print(yaml.dump(ids))
    print()

    with open("prompt.txt", 'w') as f:
        f.write(prompt)

import sys
if __name__ == "__main__":
    dargs = dict(enumerate(sys.argv[1:], start=1))
    main(AnalysisArgs(metric=dargs[1], use_z_score=dargs.get(2) == '--z'))