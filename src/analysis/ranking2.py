import logging

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pathlib import Path

from analysis.llm_based import load_dfs, AnalysisArgs
from data_models.exec_args import get_dargs


def calculate_rank_differences(df: DataFrame, method1: str, method2: str, metric: str) -> DataFrame:
    """
    Core function to calculate rank differences for all videos across all conditions.
    This version groups by the specific combination of masked captions to provide
    the most detailed data for all downstream plots.
    """
    selected_methods = [method1, method2]
    filtered_df = df[df['method'].isin(selected_methods)].copy()

    # Create a hashable version of the masked list to group by
    filtered_df['masked_tuple'] = filtered_df['masked'].apply(lambda x: tuple(sorted(eval(x))))

    # Group by the specific masked tuple, method, and video_id.
    grouped = filtered_df.groupby(['masked_tuple', 'method', 'video_id'], observed=False)[metric].mean().reset_index()

    # Define the grouping keys for ranking
    rank_groups = ['masked_tuple', 'method']

    # --- Rank Calculation Logic ---
    grouped['rank'] = grouped.groupby(rank_groups, observed=False)[metric].rank(method='first', ascending=False)

    # --- Merging and Difference Calculation ---
    method1_ranks = grouped[grouped['method'] == method1]
    method2_ranks = grouped[grouped['method'] == method2]

    merged_ranks = pd.merge(
        method1_ranks,
        method2_ranks,
        on=['video_id', 'masked_tuple'],
        suffixes=('_m1', '_m2')
    )

    merged_ranks['rank_difference'] = merged_ranks['rank_m1'] - merged_ranks['rank_m2']
    # Add num_masked back for convenience in other plots
    merged_ranks['num_masked'] = merged_ranks['masked_tuple'].apply(len)
    return merged_ranks


def get_high_diff_ranks(rank_df: DataFrame, method1: str, method2: str, top_n: int = 5) -> dict[str, list[str]]:
    """
    Extracts top N videos with the highest/lowest rank differences based ONLY
    on the lowest num_masked value.
    """
    min_num_masked = rank_df['num_masked'].min()
    logging.info(f"Selecting top videos based on the lowest number of masked captions: {min_num_masked}")
    lowest_masked_df = rank_df[rank_df['num_masked'] == min_num_masked]

    # Group by video_id and take the mean to handle cases where a video might
    # have multiple entries for the same min_num_masked. This prevents duplicates.
    agg_ranks = lowest_masked_df.groupby('video_id')['rank_difference'].mean().reset_index()

    method1_better = agg_ranks.sort_values(by='rank_difference', ascending=True).head(top_n)
    method2_better = agg_ranks.sort_values(by='rank_difference', ascending=False).head(top_n)

    return {
        method1: method1_better['video_id'].tolist(),
        method2: method2_better['video_id'].tolist()
    }


def _shorten_method_name(name: str) -> str:
    """Shortens a long method name for cleaner plot labels."""
    return name.split('__')[0]


def plot_impact_trajectories(
        rank_df: pd.DataFrame,
        output_path: Path,
        videos_to_plot: list[str],
        metric: str,
        method1: str,
        method2: str
):
    """
    Creates a line plot showing the average rank difference for each video,
    conditioned on which specific caption index was masked.
    """
    print(f"Generating impact trajectory plot for {len(videos_to_plot)} selected videos...")

    filtered_df = rank_df[rank_df['video_id'].isin(videos_to_plot)].copy()
    exploded_df = filtered_df.explode('masked_tuple')
    exploded_df = exploded_df.rename(columns={'masked_tuple': 'masked_index'})

    # Aggregate the data: for each (video_id, masked_index) pair, get the mean rank difference
    plot_df = exploded_df.groupby(['video_id', 'masked_index'])['rank_difference'].mean().reset_index()

    # Determine the starting group for coloring
    min_masked = rank_df['num_masked'].min()
    start_ranks = rank_df[rank_df['num_masked'] == min_masked]
    label_neg = f"{_shorten_method_name(method1)} better (initially)"
    label_pos = f"{_shorten_method_name(method2)} better (initially)"
    video_to_group = {
        video_id: label_neg if group_df['rank_difference'].mean() < 0 else label_pos
        for video_id, group_df in start_ranks.groupby('video_id')
    }
    plot_df['start_group'] = plot_df['video_id'].map(video_to_group)

    # --- Create the Line Plot ---
    plt.figure(figsize=(20, 8))
    ax = sns.lineplot(
        data=plot_df,
        x='masked_index',
        y='rank_difference',
        hue='start_group',
        units='video_id',  # This is the key to drawing separate lines per video
        estimator=None,  # We've already aggregated, so don't do it again
        palette={label_neg: 'red', label_pos: 'blue'},
        alpha=0.6,
        linewidth=1.8
    )

    ax.axhline(0, color='black', linestyle='--', lw=1.5)
    ax.set_title(
        f"Impact of Specific Masked Captions on Rank Difference\n(for {len(videos_to_plot)} selected videos, metric: {metric})")
    ax.set_xlabel("Caption Index That Was Masked")
    ax.set_ylabel(f"Average Rank Difference ({_shorten_method_name(method1)} - {_shorten_method_name(method2)})")

    plt.legend(title="Initial Rank Difference")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_signed_max_impact(
        rank_df: pd.DataFrame,
        output_path: Path,
        videos_to_plot: list[str],
        metric: str,
        method1: str,
        method2: str
):
    """
    Creates a bar plot showing the signed maximum absolute rank difference
    caused by masking each specific caption index.
    """
    print(f"Generating signed max impact plot for {len(videos_to_plot)} selected videos...")

    filtered_df = rank_df[rank_df['video_id'].isin(videos_to_plot)].copy()
    exploded_df = filtered_df.explode('masked_tuple')
    exploded_df = exploded_df.rename(columns={'masked_tuple': 'masked_index'})

    # --- Correct Aggregation Logic ---
    # 1. Take the absolute value of the rank_difference column first.
    abs_rank_diff = exploded_df['rank_difference'].abs()
    # 2. Group by the masked_index and find the original index of the max absolute value.
    idx = abs_rank_diff.groupby(exploded_df['masked_index']).idxmax()
    # 3. Use .loc to select these "most extreme" rows from the original exploded_df
    most_extreme_cases = exploded_df.loc[idx]

    # Create the color mapping column
    most_extreme_cases['color'] = ['red' if x < 0 else 'blue' for x in most_extreme_cases['rank_difference']]

    # --- Create the Bar Plot ---
    plt.figure(figsize=(20, 8))
    ax = sns.barplot(
        data=most_extreme_cases,
        x='masked_index',
        y='rank_difference',
        hue='color',  # Use the 'color' column to determine the bar color
        palette={'red': 'red', 'blue': 'blue'},  # Map color names to actual colors
        dodge=False,  # Prevent bars from being side-by-side
        legend=False  # The legend is redundant here
    )

    ax.axhline(0, color='black', lw=1)
    ax.set_title(
        f"Most Extreme Impact of Specific Masked Captions\n(for {len(videos_to_plot)} selected videos, metric: {metric})")
    ax.set_xlabel("Caption Index That Was Masked")
    ax.set_ylabel(f"Signed Max Abs Rank Difference ({_shorten_method_name(method1)} - {_shorten_method_name(method2)})")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def main(args: AnalysisArgs):
    df, df_z = load_dfs("results/upload/")
    combined_df = df if not args.use_z_score else df_z

    # --- Generate the detailed data once ---
    rank_df = calculate_rank_differences(combined_df, args.method1, args.method2, args.metric)

    # --- Select videos and print IDs ---
    ids_dict = get_high_diff_ranks(rank_df, args.method1, args.method2, top_n=5)
    print("Top difference videos (selected from lowest num_masked):")
    import yaml
    print(yaml.dump(ids_dict))

    results_dir = Path("results/plots/rank_stability")
    results_dir.mkdir(exist_ok=True, parents=True)

    videos_to_plot = sorted(list(set(ids_dict[args.method1] + ids_dict[args.method2])))

    # --- Generate the plots ---
    plot_impact_trajectories(
        rank_df,
        results_dir / "impact_by_masked_index_lines.png",
        videos_to_plot=videos_to_plot,
        metric=args.metric,
        method1=args.method1,
        method2=args.method2
    )

    plot_signed_max_impact(
        rank_df,
        results_dir / "impact_by_masked_index_bars.png",
        videos_to_plot=videos_to_plot,
        metric=args.metric,
        method1=args.method1,
        method2=args.method2
    )


if __name__ == "__main__":
    dargs = get_dargs()
    main(AnalysisArgs(metric=dargs.get(1, 'cos_sim_mean')))

