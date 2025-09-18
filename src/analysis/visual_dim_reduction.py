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

    def to_label_dict(self) -> dict[str,str]:
        return {
            k:f"high {self.metric} for {method}"
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


def to_meta_path(path:Path) -> Path:
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
    assert csv_path.exists(),f"CSV file not found at {csv_path}"

    # Infer the metadata path from the CSV path
    meta_path = to_meta_path(csv_path)
    assert meta_path.exists(), f"Metadata file not found at {meta_path}"
    with meta_path.open('r') as f:
        metadata = yaml.safe_load(f)

    df = pd.read_csv(csv_path)

    # 2. Prepare for Plotting (Handle Labels and Colors)
    hue_column = None
    palette = None
    if label_mapping:
        hue_column = 'label'
        # Map the IDs to their labels. Unmapped IDs will get a default value.
        df[hue_column] = df['id'].map(label_mapping).fillna('unlabeled')

        # Create a color palette
        unique_labels = df[hue_column].unique()
        # Use a nice color palette from seaborn
        colors = sns.color_palette("husl", n_colors=len(unique_labels) - 1)

        palette = {label: color for label, color in zip(unique_labels, colors)}
        # Ensure 'unlabeled' points are a neutral gray
        palette['unlabeled'] = 'lightgray'

    # 3. Create the Scatter Plot
    plt.figure(figsize=(12, 8))

    plot = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue=hue_column,
        palette=palette,
        alpha=0.8,
        s=50  # marker size
    )

    # 4. Format and Title the Plot
    dataset_name = metadata.get("dataset_name")
    feature_strat = metadata.get("feature_extraction_strategy", "Unknown Feature Strategy")
    reduction_strat = metadata.get("dimensionality_reduction_strategy", "Unknown Reduction Strategy")

    plt.title(f"'{dataset_name}: {feature_strat}' with '{reduction_strat}'", fontsize=14)
    # plt.xlabel("Component 1", fontsize=12)
    # plt.ylabel("Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    if hue_column:
        plt.legend(title="Labels")

    # 5. Save or Show the Plot
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Use the same base filename as the CSV for easy association
        output_filename = csv_path.with_suffix(".png").name
        output_path = output_dir / output_filename
        plt.savefig(output_path, dpi=150)
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
        'feature_extraction_strategy': 'MeanStrategy',
        'dimensionality_reduction_strategy': 'PCA(n_components_2)'
    }
    df1.to_csv(results_dir / "MeanStrategy_PCA(n_components_2).csv", index=False)
    with open(results_dir / "MeanStrategy_PCA(n_components_2).yaml", 'w') as f:
        yaml.dump(meta1, f)

    # Mock Data for File 2
    df2 = pd.DataFrame({
        'id': [f'txt_{i}' for i in range(10)],
        'x': np.random.randn(10) * 3 - 5,
        'y': np.random.randn(10) * 3 - 5,
    })
    meta2 = {
        'feature_extraction_strategy': 'TemporalMeanDiffStrategy',
        'dimensionality_reduction_strategy': 'TSNE(n_components_2_perplexity_5)'
    }
    df2.to_csv(results_dir / "TemporalMeanDiffStrategy_TSNE(n_components_2_perplexity_5).csv", index=False)
    with open(results_dir / "TemporalMeanDiffStrategy_TSNE(n_components_2_perplexity_5).yaml", 'w') as f:
        yaml.dump(meta2, f)


if __name__ == "__main__":
    # # --- Setup ---
    # # Create a directory with some fake result files to plot
    # mock_results_dir = Path("mock_results")
    # create_mock_results_files(mock_results_dir)
    #
    # # Define a sample label mapping. We are hypothesizing that certain videos
    # # belong to "Group A" and "Group B".
    # my_label_hypothesis = {
    #     'vid_1': 'Group A',
    #     'vid_3': 'Group A',
    #     'vid_8': 'Group B',
    #     'txt_2': 'Group B'  # Labels can apply across datasets
    # }

    # --- Main Logic ---
    # Find all CSV files in the results directory
    data_dir = Path(sys.argv[1])
    image_dir = Path("results/plots/"+data_dir.stem)
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
