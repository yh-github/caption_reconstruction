import logging

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import numpy as np

from analysis.llm_based import load_dfs, AnalysisArgs
from data_models.exec_args import get_dargs


def calculate_rank_differences_by_num_masked(df: DataFrame, method1: str, method2: str, metric: str) -> DataFrame:
    """
    Calculates rank differences by grouping by the *number* of masked captions.
    This provides the correct data structure for the first stability plot.
    """
    selected_methods = [method1, method2]
    filtered_df = df[df['method'].isin(selected_methods)].copy()

    # --- New Filtering Logic ---
    # For each 'num_masked', count how many methods each 'video_id' appears in.
    video_counts = filtered_df.groupby(['num_masked', 'video_id'])['method'].nunique()
    # Identify the video_ids that appear in both methods (count == 2).
    common_videos = video_counts[video_counts == 2].reset_index()[['num_masked', 'video_id']]
    # Filter the original dataframe to keep only these common videos for a fair ranking.
    filtered_df = pd.merge(filtered_df, common_videos, on=['num_masked', 'video_id'])

    # Group by the number of masked items, method, and video_id, averaging the metric.
    grouped = filtered_df.groupby(['num_masked', 'method', 'video_id'], observed=False)[metric].mean().reset_index()

    rank_groups = ['num_masked', 'method']
    grouped['rank'] = grouped.groupby(rank_groups, observed=False)[metric].rank(method='first', ascending=False)

    method1_ranks = grouped[grouped['method'] == method1]
    method2_ranks = grouped[grouped['method'] == method2]

    merged_ranks = pd.merge(
        method1_ranks,
        method2_ranks,
        on=['video_id', 'num_masked'],
        suffixes=('_m1', '_m2')
    )
    merged_ranks['rank_difference'] = merged_ranks['rank_m1'] - merged_ranks['rank_m2']
    return merged_ranks


def calculate_rank_differences_by_tuple(df: DataFrame, method1: str, method2: str, metric: str) -> DataFrame:
    """
    Calculates rank differences by grouping by the *specific tuple* of masked captions.
    This provides the detailed data needed for the second impact analysis plot.
    """
    selected_methods = [method1, method2]
    filtered_df = df[df['method'].isin(selected_methods)].copy()

    filtered_df['masked_tuple'] = filtered_df['masked'].apply(lambda x: tuple(sorted(eval(x))))

    # --- New Filtering Logic ---
    # For each 'masked_tuple', find the intersection of video_ids.
    video_counts = filtered_df.groupby(['masked_tuple', 'video_id'])['method'].nunique()
    common_videos = video_counts[video_counts == 2].reset_index()[['masked_tuple', 'video_id']]
    filtered_df = pd.merge(filtered_df, common_videos, on=['masked_tuple', 'video_id'])

    grouped = filtered_df.groupby(['masked_tuple', 'method', 'video_id'], observed=False)[metric].mean().reset_index()

    rank_groups = ['masked_tuple', 'method']
    grouped['rank'] = grouped.groupby(rank_groups, observed=False)[metric].rank(method='first', ascending=False)

    method1_ranks = grouped[grouped['method'] == method1]
    method2_ranks = grouped[grouped['method'] == method2]

    merged_ranks = pd.merge(
        method1_ranks,
        method2_ranks,
        on=['video_id', 'masked_tuple'],
        suffixes=('_m1', '_m2')
    )
    merged_ranks['rank_difference'] = merged_ranks['rank_m1'] - merged_ranks['rank_m2']
    merged_ranks['num_masked'] = merged_ranks['masked_tuple'].apply(len)
    return merged_ranks


def get_video_id_ordering(rank_df: DataFrame, base_num_masked: int) -> list[str]:
    """
    Get the canonical ordering of video IDs based on method1's rank at a specific num_masked.
    This ordering will be used consistently across all plots.
    """
    base_df = rank_df[rank_df['num_masked'] == base_num_masked].copy()
    base_df = base_df.sort_values(by='rank_m1', ascending=True)
    return base_df['video_id'].tolist()


def get_high_diff_ranks(rank_df: DataFrame, method1: str, method2: str, top_n: int = 5) -> dict[str, list[str]]:
    """
    Extracts top N videos with the highest/lowest rank differences based ONLY
    on the lowest num_masked value.
    """
    min_num_masked = rank_df['num_masked'].min()
    logging.info(f"Selecting top videos based on the lowest number of masked captions: {min_num_masked}")
    lowest_masked_df = rank_df[rank_df['num_masked'] == min_num_masked]

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


def plot_rank_stability(
        rank_df: pd.DataFrame,
        output_path: Path,
        videos_to_plot: list[str],
        metric: str,
        method1: str,
        method2: str,
        max_masked: int = 45,
):
    """
    Creates a line plot showing the rank difference trajectory vs. num_masked.
    """
    min_masked = rank_df['num_masked'].min()
    title_suffix = f"Top videos selected at num_masked={min_masked}"
    print(f"Generating rank stability plot for {len(videos_to_plot)} selected videos...")

    plot_df = rank_df[
        (rank_df['num_masked'] <= max_masked) &
        (rank_df['video_id'].isin(videos_to_plot))
        ].copy()

    start_ranks = plot_df[plot_df['num_masked'] == min_masked]
    label_neg = f"{_shorten_method_name(method1)} better (starts negative)"
    label_pos = f"{_shorten_method_name(method2)} better (starts positive)"

    video_to_group = {
        video_id: label_neg if group_df['rank_difference'].mean() < 0 else label_pos
        for video_id, group_df in start_ranks.groupby('video_id')
    }
    plot_df['start_group'] = plot_df['video_id'].map(video_to_group)

    plt.figure(figsize=(14, 9))
    ax = sns.lineplot(
        data=plot_df,
        x='num_masked',
        y='rank_difference',
        hue='start_group',
        units='video_id',
        estimator=None,
        palette={label_neg: 'red', label_pos: 'blue'},
        alpha=0.5,
        linewidth=1.8
    )

    ax.yaxis.grid(True, linestyle=':', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', lw=1.5)
    ax.set_title(f'Rank Difference Stability for metric: {metric}\n(up to {max_masked} masked, {title_suffix})')
    ax.set_xlabel("Number of Masked Captions")
    ax.set_ylabel(f"Rank Diff ({_shorten_method_name(method1)} - {_shorten_method_name(method2)})")

    ax.legend(
        bbox_to_anchor=(0.5, 1.15),
        loc='upper center',
        borderaxespad=0.,
        ncol=2,
        title="Initial Rank Difference"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


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


def plot_rank_comparison(
        rank_df: pd.DataFrame,
        output_path: Path,
        num_masked: int,
        video_id_order: list[str],
        metric: str,
        method1: str,
        method2: str,
        tolerance: int = 10
):
    """
    Creates a dumbbell plot comparing the ranks of videos for two methods
    at a specific num_masked value, using a fixed video ID ordering.
    """
    print(f"Generating rank comparison dumbbell plot for num_masked={num_masked}...")

    plot_df = rank_df[rank_df['num_masked'] == num_masked].copy()

    if plot_df.empty:
        print(f"No data found for num_masked={num_masked}. Skipping plot.")
        return

    # Use the provided ordering, but only include videos that exist in this data
    available_videos = set(plot_df['video_id'])
    ordered_videos = [vid for vid in video_id_order if vid in available_videos]

    # Filter and reorder the dataframe
    plot_df = plot_df[plot_df['video_id'].isin(ordered_videos)].copy()
    plot_df['video_order'] = plot_df['video_id'].map({vid: i for i, vid in enumerate(ordered_videos)})
    plot_df = plot_df.sort_values('video_order')

    short_m1 = _shorten_method_name(method1)
    short_m2 = _shorten_method_name(method2)

    plt.figure(figsize=(20, 10))
    ax = plt.gca()

    x_coords = np.arange(len(plot_df))

    # For the tolerance band, use method1's rank as the reference
    # (since x-axis order is based on method1's rank at num_masked=6)
    diagonal_line = plot_df['rank_m1'].values

    # --- Add the Tolerance Band ---
    ax.fill_between(
        x_coords,
        diagonal_line - tolerance,
        diagonal_line + tolerance,
        color='gray',
        alpha=0.2,
        label=f'Tolerance Band (±{tolerance} ranks from {short_m1})',
        zorder=1
    )

    # Create the vertical "dumbbell" lines connecting the two ranks
    for i, (y1, y2) in enumerate(zip(plot_df['rank_m1'], plot_df['rank_m2'])):
        ax.plot([i, i], [y1, y2], color='lightgray', linestyle='-', linewidth=1.5, zorder=2)

    # Plot the points for each method
    ax.scatter(x=x_coords, y=plot_df['rank_m1'], color='red', s=50, label=short_m1, zorder=3, alpha=0.5)
    ax.scatter(x=x_coords, y=plot_df['rank_m2'], color='blue', s=50, label=short_m2, zorder=3, alpha=0.5)

    # Format the plot with video IDs
    ax.set_xticks(x_coords)
    ax.set_xticklabels(plot_df['video_id'], rotation=90, fontsize=8)
    ax.set_title(
        f"Rank Comparison at num_masked={num_masked} (metric: {metric})\n(x-axis order fixed from num_masked=6)")
    ax.set_xlabel("Video ID (ordered by base ranking)")
    ax.set_ylabel("Rank")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Invert y-axis so Rank #1 is at the top
    ax.invert_yaxis()
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def main(args: AnalysisArgs):
    df, df_z = load_dfs("results/upload/")
    combined_df = df if not args.use_z_score else df_z

    # --- Generate data for both plots ---
    rank_df_by_num = calculate_rank_differences_by_num_masked(combined_df, args.method1, args.method2, args.metric)
    rank_df_by_tuple = calculate_rank_differences_by_tuple(combined_df, args.method1, args.method2, args.metric)

    # --- Get fixed video ID ordering from num_masked=6 ---
    base_num_masked = 6
    video_id_order = get_video_id_ordering(rank_df_by_num, base_num_masked)
    print(f"Fixed video ID ordering established from num_masked={base_num_masked}")
    print(f"Total videos in base ordering: {len(video_id_order)}")

    # --- Select videos and print IDs ---
    ids_dict = get_high_diff_ranks(rank_df_by_num, args.method1, args.method2, top_n=5)
    print("Top difference videos (selected from lowest num_masked):")
    import yaml
    print(yaml.dump(ids_dict))

    results_dir = Path("results/plots/rank_stability")
    results_dir.mkdir(exist_ok=True, parents=True)

    videos_to_plot = sorted(list(set(ids_dict[args.method1] + ids_dict[args.method2])))

    # --- Generate the plots ---
    plot_rank_stability(
        rank_df_by_num,
        results_dir / "rank_stability_of_initial_diffs.png",
        videos_to_plot=videos_to_plot,
        metric=args.metric,
        method1=args.method1,
        method2=args.method2,
        max_masked=45
    )

    plot_impact_trajectories(
        rank_df_by_tuple,
        results_dir / "impact_by_masked_index_lines.png",
        videos_to_plot=videos_to_plot,
        metric=args.metric,
        method1=args.method1,
        method2=args.method2
    )

    # --- Generate comparison plots for multiple num_masked values ---
    num_masked_values = sorted(rank_df_by_num['num_masked'].unique())
    print(f"\nGenerating comparison plots for num_masked values: {num_masked_values}")

    for num_masked in num_masked_values:
        plot_rank_comparison(
            rank_df_by_num,
            results_dir / f"rank_comparison_at_{num_masked}_masked.png",
            num_masked=num_masked,
            video_id_order=video_id_order,
            metric=args.metric,
            method1=args.method1,
            method2=args.gita.method2,
            tolerance=10
        )

    print(f"\nAll plots saved to {results_dir}")
    print(f"You can now compare plots across different num_masked values with consistent x-axis ordering")


if __name__ == "__main__":
    dargs = get_dargs()
    main(AnalysisArgs(metric=dargs.get(1, 'cos_sim_mean')))

