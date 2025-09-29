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

    # Group by the conditions AND the method to rank within each slice
    grouped = filtered_df.groupby(['num_masked', 'first_masked_bin', 'method', 'video_id'], observed=False)[metric].mean().reset_index()

    # Calculate ranks within each group
    grouped['rank'] = grouped.groupby(['num_masked', 'first_masked_bin', 'method'], observed=False)[metric].rank(method='first',
                                                                                                 ascending=True)

    # Pivot and merge to compare ranks
    method1_ranks = grouped[grouped['method'] == method1]
    method2_ranks = grouped[grouped['method'] == method2]

    merged_ranks = pd.merge(
        method1_ranks,
        method2_ranks,
        on=['video_id', 'num_masked', 'first_masked_bin'],
        suffixes=('_m1', '_m2')
    )

    merged_ranks['rank_difference'] = merged_ranks['rank_m1'] - merged_ranks['rank_m2']
    return merged_ranks


def get_high_diff_ranks(rank_df: DataFrame, method1: str, method2: str, top_n: int = 5) -> dict[str, list[str]]:
    """
    Takes the detailed rank difference DataFrame and extracts the top N videos
    with the highest and lowest rank differences (aggregated across all conditions).
    """
    # Aggregate by taking the mean rank difference across all conditions for each video
    agg_ranks = rank_df.groupby('video_id')['rank_difference'].mean().reset_index()

    method1_high_method2_low = agg_ranks.sort_values(by='rank_difference', ascending=False).head(top_n)
    method1_low_method2_high = agg_ranks.sort_values(by='rank_difference', ascending=True).head(top_n)

    return {
        method1: method1_high_method2_low['video_id'].tolist(),
        method2: method1_low_method2_high['video_id'].tolist()
    }


def plot_rank_stability(
        rank_df: pd.DataFrame,
        output_path: Path,
        top_n: int = 5,
        aggregation_method: str = 'mean',
        max_masked: int = 30
):
    """
    Creates a line plot showing rank difference for the most extreme videos.

    Args:
        rank_df: DataFrame with rank differences.
        output_path: Path to save the plot.
        top_n: The number of top positive and top negative videos to show.
        aggregation_method: 'mean' for average difference, 'max' for peak difference.
        max_masked: The maximum number of masked captions to show on the x-axis.
    """
    print(f"Generating rank stability plot for top {top_n} videos by '{aggregation_method}' difference...")

    # 1. Filter the DataFrame by max_masked
    plot_df = rank_df[rank_df['num_masked'] <= max_masked].copy()

    # 2. Select the videos to plot based on the aggregation method
    if aggregation_method == 'mean':
        # Find videos with the highest and lowest average rank difference
        agg = plot_df.groupby('video_id')['rank_difference'].mean()
        title_suffix = f"Top {top_n} Highest & Lowest by Mean Difference"
    elif aggregation_method == 'max':
        # Find videos with the single largest positive or negative rank difference
        agg = plot_df.loc[plot_df.groupby('video_id')['rank_difference'].abs().idxmax()]
        agg = agg.set_index('video_id')['rank_difference']
        title_suffix = f"Top {top_n} Highest & Lowest by Max Difference"
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
    g = sns.relplot(
        data=plot_df, x='num_masked', y='rank_difference', hue='video_id',
        col='first_masked_bin', kind='line', col_wrap=3,
        height=6,  # Increased height for better visibility
        aspect=1.2,
        legend=False
    )
    g.figure.suptitle(f'Rank Difference Stability (up to {max_masked} masked)\n({title_suffix})', y=1.05)
    g.set_axis_labels("Number of Masked Captions", "Rank Difference (Method1 - Method2)")
    g.map(plt.axhline, y=0, color='k', linestyle='--', lw=1)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_aggregate_stability_heatmap(rank_df: DataFrame, output_path: Path):
    """
    Creates a heatmap showing the average absolute rank difference across all conditions.
    """
    print(f"Generating aggregate stability heatmap at {output_path}...")

    # Pivot the data to create a matrix for the heatmap
    heatmap_data = rank_df.pivot_table(
        index='first_masked_bin',
        columns='num_masked',
        values='rank_difference',
        observed=False,
        aggfunc=lambda x: x.abs().mean()  # Aggregate by mean absolute difference
    )

    plt.figure(figsize=(16, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,  # Show the values in the cells
        fmt=".1f",  # Format values to one decimal place
        cmap="viridis",
        linewidths=.5
    )
    plt.title("Aggregate Stability: Average Absolute Rank Difference", pad=20)
    plt.xlabel("Number of Masked Captions")
    plt.ylabel("First Masked Caption Index (Bin)")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")


def main(args: AnalysisArgs):
    df, df_z = load_dfs("results/upload/")
    combined_df = df if not args.use_z_score else df_z

    rank_df = calculate_rank_differences(combined_df, args.method1, args.method2, args.metric)

    results_dir = Path("results/plots/rank_stability")
    results_dir.mkdir(exist_ok=True, parents=True)
    # plot_rank_stability(rank_df, results_dir / "rank_stability_top_movers.png", aggregation_method='std')
    # plot_rank_stability(rank_df, results_dir / "rank_stability_top_diff.png", aggregation_method='mean_abs_diff')

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

    plot_aggregate_stability_heatmap(rank_df, results_dir / "aggregate_stability_heatmap.png")


if __name__ == "__main__":
    dargs = get_dargs()
    main(AnalysisArgs(metric=dargs.get(1, 'cos_sim_mean')))

