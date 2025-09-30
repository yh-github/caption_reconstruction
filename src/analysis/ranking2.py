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
    For each video and each individual masked caption index, we calculate the rank difference.
    """
    selected_methods = [method1, method2]
    filtered_df = df[df['method'].isin(selected_methods)].copy()

    # Parse the masked list and explode it so each row represents one masked index
    filtered_df['masked_list'] = filtered_df['masked'].apply(lambda x: eval(x))
    exploded_df = filtered_df.explode('masked_list').reset_index(drop=True)
    exploded_df = exploded_df.rename(columns={'masked_list': 'masked_index'})

    # Group by method, video_id, and the specific masked_index to get the metric value
    grouped = exploded_df.groupby(['method', 'video_id', 'masked_index'], observed=False)[metric].mean().reset_index()

    # Rank videos within each method and masked_index combination
    # Higher metric values get better (lower) ranks
    grouped['rank'] = grouped.groupby(['method', 'masked_index'], observed=False)[metric].rank(
        method='first', ascending=False
    )

    # Pivot to get method1 and method2 ranks side by side
    method1_ranks = grouped[grouped['method'] == method1][['video_id', 'masked_index', 'rank']].rename(
        columns={'rank': 'rank_m1'}
    )
    method2_ranks = grouped[grouped['method'] == method2][['video_id', 'masked_index', 'rank']].rename(
        columns={'rank': 'rank_m2'}
    )

    merged_ranks = pd.merge(
        method1_ranks,
        method2_ranks,
        on=['video_id', 'masked_index'],
        how='inner'
    )

    # Calculate the signed rank difference (positive means method2 ranks better)
    merged_ranks['rank_difference'] = merged_ranks['rank_m1'] - merged_ranks['rank_m2']

    return merged_ranks


def get_high_diff_ranks(rank_df: DataFrame, method1: str, method2: str, top_n: int = 5) -> dict[str, list[str]]:
    """
    Extracts top N videos with the highest/lowest maximum rank differences.
    For each video, we find its maximum absolute rank difference across all masked indices,
    keeping the sign to determine which method is better.
    """

    # For each video, find the signed max absolute difference
    def get_signed_max_diff(group):
        max_abs_idx = group['rank_difference'].abs().idxmax()
        return group.loc[max_abs_idx, 'rank_difference']

    max_diffs = rank_df.groupby('video_id').apply(get_signed_max_diff).reset_index()
    max_diffs.columns = ['video_id', 'max_rank_difference']

    # Videos where method1 is much better (large negative differences)
    method1_better = max_diffs.sort_values(by='max_rank_difference', ascending=True).head(top_n)
    # Videos where method2 is much better (large positive differences)
    method2_better = max_diffs.sort_values(by='max_rank_difference', ascending=False).head(top_n)

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
    Creates a line plot showing the rank difference trajectory for each video
    as different caption indices are masked.
    """
    print(f"Generating impact trajectory plot for {len(videos_to_plot)} selected videos...")

    filtered_df = rank_df[rank_df['video_id'].isin(videos_to_plot)].copy()

    # Determine which group each video belongs to based on its max absolute difference
    def get_signed_max_diff(group):
        max_abs_idx = group['rank_difference'].abs().idxmax()
        return group.loc[max_abs_idx, 'rank_difference']

    video_max_diffs = filtered_df.groupby('video_id').apply(get_signed_max_diff).to_dict()

    label_neg = f"{_shorten_method_name(method1)} better (max diff)"
    label_pos = f"{_shorten_method_name(method2)} better (max diff)"

    filtered_df['start_group'] = filtered_df['video_id'].map(
        lambda vid: label_neg if video_max_diffs[vid] < 0 else label_pos
    )

    # --- Create the Line Plot ---
    plt.figure(figsize=(20, 8))
    ax = sns.lineplot(
        data=filtered_df,
        x='masked_index',
        y='rank_difference',
        hue='start_group',
        units='video_id',  # Draw separate lines per video
        estimator=None,  # Don't aggregate
        palette={label_neg: 'red', label_pos: 'blue'},
        alpha=0.6,
        linewidth=1.8
    )

    ax.axhline(0, color='black', linestyle='--', lw=1.5)
    ax.set_title(
        f"Rank Stability: Impact of Masking Different Captions\n"
        f"({len(videos_to_plot)} videos with highest max rank differences, metric: {metric})"
    )
    ax.set_xlabel("Caption Index That Was Masked")
    ax.set_ylabel(f"Rank Difference ({_shorten_method_name(method1)} - {_shorten_method_name(method2)})")

    plt.legend(title="Video Group (by max difference)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def main(args: AnalysisArgs):
    df, df_z = load_dfs("results/upload/")
    combined_df = df if not args.use_z_score else df_z

    # --- Generate the detailed data ---
    rank_df = calculate_rank_differences(combined_df, args.method1, args.method2, args.metric)

    # --- Select videos based on maximum rank differences ---
    ids_dict = get_high_diff_ranks(rank_df, args.method1, args.method2, top_n=5)
    print("Top difference videos (based on maximum absolute rank difference):")
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


if __name__ == "__main__":
    dargs = get_dargs()
    main(AnalysisArgs(metric=dargs.get(1, 'cos_sim_mean')))