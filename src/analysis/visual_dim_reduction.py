import sys
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any
from pydantic import BaseModel


class Labeler(BaseModel):
    metric: str
    method_names: tuple[str, str]
    lists: tuple[list[str], list[str]]

    def to_label_dict(self) -> dict[str, str]:
        assert set(self.lists[0]).isdisjoint(set(self.lists[1])) #TODO handle intersections
        return {
            k: f"{method} with high {self.metric}"
            for method, ids in zip(self.method_names, self.lists)
            for k in ids
        }


cos_sim_residual_mean = Labeler(
    metric="cos_sim_residual_mean",
    method_names=("CaptionedVideo__pro_d_one_shot_v1__t=1", "video_embeddings__MeanClosestVectors"),
    lists=(
        [
            "Gung-Ho-Vids_5-clip-2",
            "Climate-Change_7-clip-3",
            "Welker-Farms-Inc_3-clip-4",
            "Climate-Change_0-clip-4",
            "Primitive-Technology_3-clip-1"
        ],
        [
            "4k-Relaxation_3-clip-4",
            "4k-Relaxation_12-clip-6",
            "King-Kong-Amazon_11-clip-7",
            "King-Kong-Amazon_5-clip-14",
            "Primal-Earth-Sounds_0-clip-44"
        ]
    )
)

cos_sim_mean = Labeler(
    metric="cos_sim_mean",
    method_names=("CaptionedVideo__pro_d_one_shot_v1__t=1", "video_embeddings__MeanClosestVectors"),
    lists=(
        [
            "Welker-Farms-Inc_3-clip-4",
            "Climate-Change_7-clip-3",
            "How-Farms-Work_3-clip-2",
            "Sandboxx_0-clip-4",
            "Natural-Disaster_10-clip-0"
        ],
        [
            "Millennial-Farmer_8-clip-16",
            "Survival-Instinct_8-clip-4",
            "Survival-Instinct_11-clip-10",
            "King-Kong-Amazon_5-clip-14",
            "Joe-Robinet_2-clip-5"
        ]
    )
)



def to_meta_path(path: Path) -> Path:
    return path.parent / f"{path.stem}_meta.yaml"


def plot_results(
        csv_path: Path,
        label_mapping: dict[str, Any] | None = None,
        output_dir: Path | None = None
):
    """
    Loads reduced vector data and metadata to generate a 2D scatter plot.

    This function visualizes the output of the feature extraction pipeline. It can
    optionally highlight specific data points by color-coding them based on a
    provided label mapping.

    Args:
        csv_path: Path to the CSV file containing 'id', 'x', and 'y' columns.
        label_mapping: An optional dictionary mapping IDs to a specific label.
                       Points with a label will be colored differently.
        output_dir: If provided, the plot will be saved to this directory as a PNG.
    """
    # 1. Load Data and Metadata
    assert csv_path.exists(), f"CSV file not found at {csv_path}"

    # Infer the metadata path from the CSV path
    meta_path = to_meta_path(csv_path)
    assert meta_path.exists(), f"Metadata file not found at {meta_path}"
    with meta_path.open('r') as f:
        metadata = yaml.safe_load(f)

    df = pd.read_csv(csv_path)

    # 2. Create the Plot Figure
    plt.figure(figsize=(12, 9)) # Increased height to make room for legend

    # 3. Plotting Logic
    real_labels = []
    if label_mapping:
        df['label'] = df['id'].map(label_mapping).fillna('unlabeled')

        unlabeled_df = df[df['label'] == 'unlabeled']
        labeled_df = df[df['label'] != 'unlabeled']

        # Plot unlabeled points first (in gray)
        sns.scatterplot(
            data=unlabeled_df, x='x', y='y',
            color='lightgray', alpha=0.7, s=50, label='unlabeled'
        )

        # Create a high-contrast palette for labeled points
        real_labels = sorted([label for label in df['label'].unique() if label != 'unlabeled'])
        strong_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
        palette = {label: strong_colors[i % len(strong_colors)] for i, label in enumerate(real_labels)}

        # Plot labeled points on top, with more pronounced style
        sns.scatterplot(
            data=labeled_df, x='x', y='y',
            hue='label',
            palette=palette,
            s=150,          # Larger size
            marker='X',     # Different shape
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7       # Add transparency to show overlaps
        )
    else:
        # If no labels, just plot all points normally
        sns.scatterplot(data=df, x='x', y='y', alpha=0.8, s=50)

    # 4. Format and Title the Plot
    dataset_name = metadata.get("dataset_name", "Unknown Dataset")
    feature_strat = metadata.get("feature_extraction_strategy", "Unknown Feature Strategy")
    reduction_strat_full = metadata.get("dimensionality_reduction_strategy", "Unknown Reduction Strategy")
    # Shorten the reduction strategy name for the title and axis labels
    reduction_strat_short = reduction_strat_full.split('(')[0]

    plt.title(f"'{dataset_name}': {feature_strat} -> {reduction_strat_short}", fontsize=16, pad=20)
    # Use dynamic axis labels
    plt.xlabel(f"{reduction_strat_short} 1", fontsize=12)
    plt.ylabel(f"{reduction_strat_short} 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 5. Position legend above the plot
    # The legend will be horizontal with a number of columns equal to the labels.
    if label_mapping:
        plt.legend(
            bbox_to_anchor=(0.5, 1.15), # Positioned above the plot title
            loc='upper center',
            borderaxespad=0.,
            ncol=len(real_labels) + 1 # +1 for the 'unlabeled' category
        )

    # 6. Save or Show the Plot
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Use the same base filename as the CSV for easy association
        output_filename = csv_path.with_suffix(".png").name
        output_path = output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')  # bbox_inches ensures legend is saved
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()  # Close the figure to free up memory


def create_mock_results_files(results_dir: Path):
    """Helper function to create fake data for demonstration."""
    results_dir.mkdir(exist_ok=True)

    # Mock Data for File 1
    df1 = pd.DataFrame({
        'id': [f'vid_{i}' for i in range(10)],
        'x': np.random.randn(10) * 2 + 10,
        'y': np.random.randn(10) * 2 + 10,
    })
    meta1 = {
        'dataset_name': 'Videos',
        'feature_extraction_strategy': 'MeanStrategy',
        'dimensionality_reduction_strategy': 'PCA(n_components=2)'
    }
    df1.to_csv(results_dir / "MeanStrategy_PCA(n_components_2).csv", index=False)
    with open(results_dir / "MeanStrategy_PCA(n_components_2)_meta.yaml", 'w') as f:
        yaml.dump(meta1, f)

    # Mock Data for File 2
    df2 = pd.DataFrame({
        'id': [f'txt_{i}' for i in range(10)],
        'x': np.random.randn(10) * 3 - 5,
        'y': np.random.randn(10) * 3 - 5,
    })
    meta2 = {
        'dataset_name': 'Text Captions',
        'feature_extraction_strategy': 'TemporalMeanDiffStrategy',
        'dimensionality_reduction_strategy': 'TSNE(n_components=2, perplexity=5)'
    }
    df2.to_csv(results_dir / "TemporalMeanDiffStrategy_TSNE(n_components_2_perplexity_5).csv", index=False)
    with open(results_dir / "TemporalMeanDiffStrategy_TSNE(n_components_2_perplexity_5)_meta.yaml", 'w') as f:
        yaml.dump(meta2, f)


if __name__ == "__main__":
    # Find all CSV files in the results directory
    data_dir = Path(sys.argv[1])
    image_dir = Path("results/plots/" + data_dir.stem)
    csv_files_to_plot = sorted(data_dir.glob("*.csv"))

    my_label_hypothesis = cos_sim_residual_mean.to_label_dict()

    if not csv_files_to_plot:
        print("No CSV files found in the 'mock_results' directory to plot.")
    else:
        print(f"Found {len(csv_files_to_plot)} CSV files to plot...")
        for csv_file in csv_files_to_plot:
            plot_results(
                csv_path=csv_file,
                label_mapping=my_label_hypothesis,
                output_dir=image_dir
            )

