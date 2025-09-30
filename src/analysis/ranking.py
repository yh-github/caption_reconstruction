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
    """
    selected_methods = [method1, method2]
    filtered_df = df[df['method'].isin(selected_methods)].copy()

    # Group by the conditions AND the method to rank within each slice.
    # `first_masked_bin` has been removed from the grouping.
    grouped = filtered_df.groupby(['num_masked', 'method', 'video_id'], observed=False)[metric].mean().reset_index()

    # Define the grouping keys for ranking
    rank_groups = ['num_masked', 'method']

    # --- Rank Calculation Logic ---
    # For similarity metrics, a higher score is better, so it should get the #1 rank.
    # Therefore, we use ascending=False.
    grouped['rank'] = grouped.groupby(rank_groups, observed=False)[metric].rank(method='first', ascending=False)

    # --- Merging and Difference Calculation ---
    method1_ranks = grouped[grouped['method'] == method1]
    method2_ranks = grouped[grouped['method'] == method2]

    merged_ranks = pd.merge(
        method1_ranks,
        method2_ranks,
        on=['video_id', 'num_masked'],  # `first_masked_bin` removed from merge key
        suffixes=('_m1', '_m2')
    )

    # A negative value means method1 has a better (lower) rank number than method2.
    merged_ranks['rank_difference'] = merged_ranks['rank_m1'] - merged_ranks['rank_m2']
    return merged_ranks


def get_high_diff_ranks(rank_df: DataFrame, method1: str, method2: str, top_n: int = 5) -> dict[str, list[str]]:
    """
    Takes the detailed rank difference DataFrame and extracts the top N videos
    with the highest and lowest rank differences, based ONLY on the lowest
    num_masked value.
    """
    # 1. Find the minimum num_masked value in the dataset
    min_num_masked = rank_df['num_masked'].min()
    logging.info(f"Selecting top videos based on the lowest number of masked captions: {min_num_masked}")

    # 2. Filter the DataFrame to this specific slice of data
    lowest_masked_df = rank_df[rank_df['num_masked'] == min_num_masked]

    # 3. Sort by rank difference to find the most extreme videos in this slice
    # A large negative difference means method1 is better.
    method1_better = lowest_masked_df.sort_values(by='rank_difference', ascending=True).head(top_n)
    # A large positive difference means method2 is better.
    method2_better = lowest_masked_df.sort_values(by='rank_difference', ascending=False).head(top_n)

    return {
        method1: method1_better['video_id'].tolist(),
        method2: method2_better['video_id'].tolist()
    }


def plot_rank_stability(
        rank_df: pd.DataFrame,
        output_path: Path,
        top_n: int = 5,
        aggregation_method: str = 'mean',
        max_masked: int = 30,
):
    """
    Creates a line plot showing rank difference for the most extreme videos.
    """
    print(f"Generating rank stability plot for top {top_n} videos by '{aggregation_method}' difference...")

    # 1. Filter the DataFrame by max_masked
    plot_df = rank_df[rank_df['num_masked'] <= max_masked].copy()

    # 2. Select the videos to plot based on the aggregation method
    if aggregation_method == 'mean':
        # Find videos with the highest and lowest average rank difference
        agg = plot_df.groupby('video_id')['rank_difference'].mean()
        title_suffix = f"Top {top_n} by Mean Difference"
    elif aggregation_method == 'max':
        # Find videos with the single largest positive or negative rank difference
        indices_of_max = plot_df.loc[plot_df.groupby('video_id', observed=False)['rank_difference'].abs().idxmax()]
        agg = indices_of_max.set_index('video_id')['rank_difference']
        title_suffix = f"Top {top_n} by Max Difference"
    else:
        raise ValueError("aggregation_method must be 'mean' or 'max'")

    # Sort the aggregated values
    sorted_agg = agg.sort_values()
    # Get top N lowest (most negative) and top N highest (most positive)
    top_n_lowest = sorted_agg.head(top_n)
    top_n_highest = sorted_agg.tail(top_n)
    # Combine their indices (which are the video_ids)
    top_videos_to_plot = pd.concat([top_n_lowest, top_n_highest]).index

    # Filter the main plot DataFrame to only these videos
    plot_df = plot_df[plot_df['video_id'].isin(top_videos_to_plot)]

    # 3. Create the plot
    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(
        data=plot_df, x='num_masked', y='rank_difference', hue='video_id',
        palette='viridis'
    )
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.set_title(f'Rank Difference Stability\n(up to {max_masked} masked, {title_suffix})')
    ax.set_xlabel("Number of Masked Captions")
    ax.set_ylabel("Rank Difference (m1 - m2)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def main(args: AnalysisArgs):
    df, df_z = load_dfs("results/upload/")
    combined_df = df if not args.use_z_score else df_z

    rank_df = calculate_rank_differences(combined_df, args.method1, args.method2, args.metric)

    # Get the top N videos based on the new logic (lowest num_masked)
    ids = get_high_diff_ranks(rank_df, args.method1, args.method2, top_n=5)
    print("Top difference videos (selected from lowest num_masked):")
    # This import was missing from the original file
    import yaml
    print(yaml.dump(ids))

    results_dir = Path("results/plots/rank_stability")
    results_dir.mkdir(exist_ok=True, parents=True)

    # Generate the simplified stability plot
    plot_rank_stability(
        rank_df,
        results_dir / "rank_stability_by_mean_diff.png",
        top_n=5,
        aggregation_method='mean',
        max_masked=30
    )

    plot_rank_stability(
        rank_df,
        results_dir / "rank_stability_by_max_diff.png",
        top_n=5,
        aggregation_method='max',
        max_masked=30
    )


if __name__ == "__main__":
    dargs = get_dargs()
    main(AnalysisArgs(metric=dargs.get(1, 'cos_sim_mean')))

