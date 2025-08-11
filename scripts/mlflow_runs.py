import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

class ExperimentInfo(BaseModel):
    name:str
    path: Path
    subdir_count: int
    start_time: int

def find_experiment_hierarchy(root_dir: str = ".") -> list[ExperimentInfo]:
    """
    Finds all directories containing meta.yaml files and displays them in a hierarchy.

    Args:
        root_dir: The root directory to start searching from (default: current directory)
    """
    experiments:list[ExperimentInfo] = []

    def find_experiment_dirs(current_path: Path) -> None:
        """Recursively find directories with meta.yaml files."""
        try:
            # Check if current directory has meta.yaml
            meta_file = current_path / "meta.yaml"
            if meta_file.exists():
                # Try to read the meta.yaml file
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = yaml.safe_load(f)

                    # Check if it has a 'name' key
                    if isinstance(meta_data, dict) and 'name' in meta_data:
                        # Count subdirectories
                        subdirs = [item for item in current_path.iterdir()
                                   if item.is_dir()]

                        experiments.append(ExperimentInfo(
                            name=meta_data['name'],
                            path=Path(current_path),
                            subdir_count=len(subdirs),
                            start_time=meta_data.get('start_time',0) or meta_data.get('creation_time')
                        ))
                        # Don't recurse deeper when we find a meta.yaml
                        return

                except (yaml.YAMLError, IOError) as e:
                    raise Exception(f"Warning: Could not read {meta_file}: {e}")

            # If no meta.yaml found, continue recursively
            for item in current_path.iterdir():
                if item.is_dir():
                    find_experiment_dirs(item)

        except PermissionError:
            pass

    # Start the search
    root_path = Path(root_dir).resolve()
    find_experiment_dirs(root_path)

    # Sort experiments by name
    experiments.sort(key=lambda x: x.start_time)
    return experiments


def parent_name_to_config(parent_name:str) -> str:
    return f"config/{parent_name}.yaml"

def get_run_hierarchy(root_path: str) -> tuple[dict[str, str], defaultdict[str, list], set[str]] | None:
    """
    Parses an MLflow experiment directory to identify and print the
    hierarchical relationship between parent and child runs.

    Args:
        root_path (str): The path to the MLflow experiment directory
                         (e.g., 'mlruns/1').
    """
    # --- 1. Argument Validation ---
    if not os.path.isdir(root_path):
        print(f"Error: The provided path '{root_path}' does not exist or is not a directory.")
        return

    # --- 2. Data Structure Initialization ---
    # Maps a run's ID to its given name.
    id_to_name = {}
    # Maps a parent's run ID to a list of its children's run IDs.
    # defaultdict simplifies appending to lists.
    parent_to_children = defaultdict(list)
    # Keep track of all run IDs found in the directory.
    all_run_ids = set()

    # print(f"Scanning directory: {root_path}\n")

    # --- 3. Data Collection ---
    # Iterate through each entry in the root_path. We assume each subdirectory
    # is an MLflow run.
    for entry in os.scandir(root_path):
        if entry.is_dir():
            run_id = entry.name
            run_path = entry.path
            all_run_ids.add(run_id)

            parent_id = None
            # Default name if the tag is not found.
            run_name = "Unnamed Run"

            # Attempt to read the parent run ID from the tags.
            parent_id_file = os.path.join(run_path, "tags", "mlflow.parentRunId")
            try:
                with open(parent_id_file, 'r') as f:
                    parent_id = f.read().strip()
            except FileNotFoundError:
                # This is expected for runs that are not children.
                pass
            except IOError as e:
                print(f"Warning: Could not read {parent_id_file}: {e}")

            # Attempt to read the run name from the tags.
            run_name_file = os.path.join(run_path, "tags", "mlflow.runName")
            try:
                with open(run_name_file, 'r') as f:
                    run_name = f.read().strip()
            except FileNotFoundError:
                # This run might not have been explicitly named.
                pass
            except IOError as e:
                print(f"Warning: Could not read {run_name_file}: {e}")

            # Populate our dictionaries with the collected data.
            id_to_name[run_id] = run_name
            if parent_id:
                parent_to_children[parent_id].append(run_id)

    return id_to_name, parent_to_children, all_run_ids


def display_run_hierarchy(id_to_name, parent_to_children, all_run_ids):

    # --- 4. Output Generation ---
    # Identify which runs are parents and which are children.
    parent_ids = set(parent_to_children.keys())
    child_ids = {child for children in parent_to_children.values() for child in children}
    
    # Runs that are parents should be printed with their children.
    d = {}
    if parent_ids:
        print("--- Run Hierarchy ---")
        # Sort parent runs by their name.
        sorted_parent_ids = sorted(list(parent_ids), key=lambda pid: id_to_name.get(pid, "Parent Run (Not in this directory)"))
        for parent_id in sorted_parent_ids:
            d[len(d)] = parent_id
            parent_name = id_to_name.get(parent_id, "Parent Run (Not in this directory)")
            print(f"{len(d)-1}. {parent_id}\t{parent_name}")

            # Sort child runs by their name.
            sorted_child_ids = sorted(parent_to_children[parent_id], key=lambda cid: id_to_name.get(cid, "Unnamed Child Run"))
            for child_id in sorted_child_ids:
                d[len(d)] = child_id
                child_name = id_to_name.get(child_id, "Unnamed Child Run")
                print(f"\t{len(d)-1}. {child_id}\t{child_name}")
            print()  # Add a blank line for readability between groups.

    # Identify and print "orphan" runs (runs that are not parents and not children).
    orphan_runs = all_run_ids - parent_ids - child_ids
    if orphan_runs:
        print("--- Standalone Runs ---")
        # Sort standalone runs by their name.
        sorted_orphan_ids = sorted(list(orphan_runs), key=lambda rid: id_to_name.get(rid, "Unnamed Run"))
        for run_id in sorted_orphan_ids:
            run_name = id_to_name.get(run_id, "Unnamed Run")
            print(f"{run_id}\t{run_name}")

def main():
    """Main function to parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Display a hierarchy of MLflow runs from a given experiment directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "path",
        type=str,
        help="The path to the MLflow experiment directory (e.g., './mlruns/0')."
    )
    args = parser.parse_args()
    exps = find_experiment_hierarchy(args.path)

    if len(exps) == 0:
        print("No experiments found in the specified directory.")
        return

    elif len(exps) == 1:
        exp = exps[0]
        _, parent_to_children, _ = get_run_hierarchy(str(exp.path))
        print(f"Single Experiment: {exp.name} {exp.path.name} ({len(parent_to_children)})")
        print()
        display_run_hierarchy(*get_run_hierarchy(exp.path)) # <<<<
        return
    else:
        print(f"Experiments found in {args.path}:")
        for i, exp in enumerate(exps):
            _, parent_to_children, _ = get_run_hierarchy(str(exp.path))
            print(f"{i+1}. {exp.name} {exp.path.name} ({len(parent_to_children)})")
        exp_idx = -1
        while exp_idx < 0 or exp_idx >= len(exps):
            print()
            exp_idx = int(input("Enter the number of the experiment to display the hierarchy for: ")) - 1
            if exp_idx < 0 or exp_idx >= len(exps):
                print("Invalid experiment number.")

        print()
        display_run_hierarchy(*get_run_hierarchy(str(exps[exp_idx].path)))

if __name__ == "__main__":
    print()
    try:
        main()
    except KeyboardInterrupt:
        print("\n^C\n")
    print()
