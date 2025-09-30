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
    This version groups by the specific combination of masked captions.
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
    Takes the detailed rank difference DataFrame and extracts the top N videos
    with the highest and lowest rank differences, based ONLY on the lowest
    num_masked value.
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
    Creates a line plot showing the rank difference trajectory for a specific
    list of videos, colored by their initial rank difference.
    """
    min_masked = rank_df['num_masked'].min()
    title_suffix = f"Top videos selected at num_masked={min_masked}"
    print(f"Generating rank stability plot for {len(videos_to_plot)} selected videos...")

    plot_df = rank_df[
        (rank_df['num_masked'] <= max_masked) &
        (rank_df['video_id'].isin(videos_to_plot))
        ].copy()

    # For this plot, we need to aggregate by num_masked since there can be
    # multiple masked_tuples for the same number of masks.
    plot_df = plot_df.groupby(['video_id', 'num_masked'])['rank_difference'].mean().reset_index()

    start_ranks = plot_df[plot_df['num_masked'] == min_masked]
    label_neg = f"{method1} better (starts negative)"
    label_pos = f"{method2} better (starts positive)"

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
    ax.set_ylabel(f"Rank Difference ({method1} - {method2})")

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

    # Filter for only the videos we care about
    filtered_df = rank_df[rank_df['video_id'].isin(videos_to_plot)].copy()

    # "Explode" the DataFrame so each row represents one video and one masked index
    exploded_df = filtered_df.explode('masked_tuple')
    exploded_df = exploded_df.rename(columns={'masked_tuple': 'masked_index'})

    # For this plot, we need to aggregate the rank difference for each video and masked_index
    # This gives us the average impact of masking a specific index for a specific video
    plot_df = exploded_df.groupby(['video_id', 'masked_index'])['rank_difference'].mean().reset_index()

    # Determine the starting group for coloring, same as the other plot
    min_masked = rank_df['num_masked'].min()
    start_ranks = rank_df[rank_df['num_masked'] == min_masked]
    label_neg = f"{method1} better (initially)"
    label_pos = f"{method2} better (initially)"
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
        units='video_id',
        estimator=None,
        palette={label_neg: 'red', label_pos: 'blue'},
        alpha=0.5,
        linewidth=1.8
    )

    ax.axhline(0, color='black', linestyle='--', lw=1.5)
    ax.set_title(
        f"Impact of Specific Masked Captions on Rank Difference\n(for {len(videos_to_plot)} selected videos, metric: {metric})")
    ax.set_xlabel("Caption Index That Was Masked")
    ax.set_ylabel("Average Rank Difference (m1 - m2)")

    plt.legend(title="Initial Rank Difference")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def main(args: AnalysisArgs):
    df, df_z = load_dfs("results/upload/")
    combined_df = df if not args.use_z_score else df_z

    rank_df = calculate_rank_differences(combined_df, args.method1, args.method2, args.metric)

    ids_dict = get_high_diff_ranks(rank_df, args.method1, args.method2, top_n=5)
    print("Top difference videos (selected from lowest num_masked):")
    import yaml
    print(yaml.dump(ids_dict))

    results_dir = Path("results/plots/rank_stability")
    results_dir.mkdir(exist_ok=True, parents=True)

    videos_to_plot = sorted(list(set(ids_dict[args.method1] + ids_dict[args.method2])))

    # Generate the stability plot
    plot_rank_stability(
        rank_df,
        results_dir / "rank_stability_of_initial_diffs.png",
        videos_to_plot=videos_to_plot,
        metric=args.metric,
        method1=args.method1,
        method2=args.method2,
        max_masked=45
    )

    # Generate the new impact line plot
    plot_impact_trajectories(
        rank_df,
        results_dir / "impact_by_masked_index_lines.png",
        videos_to_plot=videos_to_plot,
        metric=args.metric,
        method1=args.method1,
        method2=args.method2
    )


if __name__ == "__main__":
    dargs = get_dargs()
    main(AnalysisArgs(metric=dargs.get(1, 'cos_sim_mean')))

