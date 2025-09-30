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
    This version groups by the number of masked captions.
    """
    selected_methods = [method1, method2]
    filtered_df = df[df['method'].isin(selected_methods)].copy()

    # We need the specific masked list later, so we evaluate it here.
    filtered_df['masked_tuple'] = filtered_df['masked'].apply(lambda x: tuple(sorted(eval(x))))

    # Group by the number of masked items, method, and video_id.
    grouped = filtered_df.groupby(['num_masked', 'method', 'video_id'], observed=False)[metric].mean().reset_index()

    # Define the grouping keys for ranking
    rank_groups = ['num_masked', 'method']

    # --- Rank Calculation Logic ---
    grouped['rank'] = grouped.groupby(rank_groups, observed=False)[metric].rank(method='first', ascending=False)

    # --- Merging and Difference Calculation ---
    method1_ranks = grouped[grouped['method'] == method1]
    method2_ranks = grouped[grouped['method'] == method2]

    merged_ranks = pd.merge(
        method1_ranks,
        method2_ranks,
        on=['video_id', 'num_masked'],
        suffixes=('_m1', '_m2')
    )

    merged_ranks['rank_difference'] = merged_ranks['rank_m1'] - merged_ranks['rank_m2']
    # We need to bring the original masked_tuple back for the second plot
    merged_ranks = merged_ranks.merge(
        filtered_df[['video_id', 'num_masked', 'masked_tuple']].drop_duplicates(),
        on=['video_id', 'num_masked']
    )
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

    start_ranks = plot_df[plot_df['num_masked'] == min_masked]
    label_neg = f"{method1} better (starts negative)"
    label_pos = f"{method2} better (starts positive)"

    # Because there can be multiple rows for the same video at min_masked, we take the mean.
    video_to_group = {
        video_id: label_neg if group_df['rank_difference'].mean() < 0 else label_pos
        for video_id, group_df in start_ranks.groupby('video_id')
    }
    plot_df['start_group'] = plot_df['video_id'].map(video_to_group)

    plt.figure(figsize=(14, 9))  # Increased height for legend
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

    # Position legend above the plot
    ax.legend(
        bbox_to_anchor=(0.5, 1.15),
        loc='upper center',
        borderaxespad=0.,
        ncol=2,
        title="Initial Rank Difference"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent title overlap

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_impact_as_lineplot(
        rank_df: pd.DataFrame,
        output_path: Path,
        videos_to_plot: list[str],
        target_index: int,
        metric: str,
        method1: str,
        method2: str
):
    """
    Creates a line plot showing how rank difference changes when a specific
    caption index is masked.
    """
    print(f"Generating impact line plot for target_index={target_index}...")

    # Filter for only the videos we care about
    filtered_df = rank_df[rank_df['video_id'].isin(videos_to_plot)].copy()

    # Create the new boolean column for the x-axis
    filtered_df['target_is_masked'] = filtered_df['masked_tuple'].apply(lambda x: target_index in x)

    # Aggregate the data: for each video, get the mean rank_difference when the
    # target is masked vs. when it is not.
    agg_df = filtered_df.groupby(['video_id', 'target_is_masked'])['rank_difference'].mean().reset_index()

    # Determine the starting group for coloring, same as the other plot
    min_masked = rank_df['num_masked'].min()
    start_ranks = rank_df[rank_df['num_masked'] == min_masked]
    label_neg = f"{method1} better (starts negative)"
    label_pos = f"{method2} better (starts positive)"
    video_to_group = {
        video_id: label_neg if group_df['rank_difference'].mean() < 0 else label_pos
        for video_id, group_df in start_ranks.groupby('video_id')
    }
    agg_df['start_group'] = agg_df['video_id'].map(video_to_group)

    # --- Create the Line Plot ---
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(
        data=agg_df,
        x='target_is_masked',
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
        f"Impact of Masking Index {target_index} on Rank Difference\n(for {len(videos_to_plot)} selected videos, metric: {metric})")
    ax.set_xlabel(f"Was Caption at Index {target_index} Masked?")
    ax.set_ylabel("Average Rank Difference (m1 - m2)")
    ax.set_xticks([False, True])  # Ensure x-axis only shows False and True
    ax.set_xticklabels(['Not Masked', 'Masked'])

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

    # Generate the new impact plot
    plot_impact_as_lineplot(
        rank_df,
        results_dir / "impact_by_mask_index_1.png",
        videos_to_plot=videos_to_plot,
        target_index=1,
        metric=args.metric,
        method1=args.method1,
        method2=args.method2
    )


if __name__ == "__main__":
    dargs = get_dargs()
    main(AnalysisArgs(metric=dargs.get(1, 'cos_sim_mean')))

