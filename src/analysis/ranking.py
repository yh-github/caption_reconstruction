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
        videos_to_plot: list[str],
        metric: str,
        max_masked: int = 30,
):
    """
    Creates a line plot showing the rank difference trajectory for a specific
    list of videos, colored by their initial rank difference.
    """
    min_masked = rank_df['num_masked'].min()
    title_suffix = f"Top videos selected at num_masked={min_masked}"
    print(f"Generating rank stability plot for {len(videos_to_plot)} selected videos...")

    # 1. Filter the DataFrame by max_masked and the selected videos
    plot_df = rank_df[
        (rank_df['num_masked'] <= max_masked) &
        (rank_df['video_id'].isin(videos_to_plot))
        ].copy()

    # 2. Determine the starting group for each video based on its sign at min_masked
    start_ranks = plot_df[plot_df['num_masked'] == min_masked]
    video_to_group = {
        row['video_id']: 'Method1 Better (starts negative)' if row[
                                                                   'rank_difference'] < 0 else 'Method2 Better (starts positive)'
        for _, row in start_ranks.iterrows()
    }
    plot_df['start_group'] = plot_df['video_id'].map(video_to_group)

    # 3. Create the plot
    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(
        data=plot_df,
        x='num_masked',
        y='rank_difference',
        hue='start_group',  # Color lines based on the starting group
        units='video_id',  # Draw a separate line for each video
        estimator=None,  # Don't aggregate, draw the raw lines
        palette={'Method1 Better (starts negative)': 'blue', 'Method2 Better (starts positive)': 'red'},
        alpha=0.5,  # Make lines semi-transparent to show overlaps
        linewidth=1.8,
        # marker='o',
        # markersize=8
    )

    # 4. Add guidelines for readability
    ax.yaxis.grid(True, linestyle=':', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', lw=1.5)  # Pronounced zero line

    # 5. Final Formatting
    ax.set_title(f'Rank Difference Stability for metric: {metric}\n(up to {max_masked} masked, {title_suffix})')
    ax.set_xlabel("Number of Masked Captions")
    ax.set_ylabel("Rank Difference (m1 - m2)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Initial Rank Difference")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def main(args: AnalysisArgs):
    df, df_z = load_dfs("results/upload/")
    combined_df = df if not args.use_z_score else df_z

    rank_df = calculate_rank_differences(combined_df, args.method1, args.method2, args.metric)

    # 1. Get the top N videos based on the lowest num_masked condition.
    ids_dict = get_high_diff_ranks(rank_df, args.method1, args.method2, top_n=5)
    print("Top difference videos (selected from lowest num_masked):")
    # This import was missing from the original file
    import yaml
    print(yaml.dump(ids_dict))

    results_dir = Path("results/plots/rank_stability")
    results_dir.mkdir(exist_ok=True, parents=True)

    # 2. Create a flat, unique list of these video IDs to plot.
    videos_to_plot = sorted(list(set(ids_dict[args.method1] + ids_dict[args.method2])))

    # 3. Generate the stability plot using the selected IDs.
    plot_rank_stability(
        rank_df,
        results_dir / "rank_stability_of_initial_diffs.png",
        videos_to_plot=videos_to_plot,
        metric=args.metric,
        max_masked=45
    )


if __name__ == "__main__":
    dargs = get_dargs()
    main(AnalysisArgs(metric=dargs.get(1, 'cos_sim_min')))

