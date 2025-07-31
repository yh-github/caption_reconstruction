import os
import argparse
from collections import defaultdict

def display_run_hierarchy(root_path: str):
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

    print(f"Scanning directory: {root_path}\n")

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

    # --- 4. Output Generation ---
    # Identify which runs are parents and which are children.
    parent_ids = set(parent_to_children.keys())
    child_ids = {child for children in parent_to_children.values() for child in children}
    
    # Runs that are parents should be printed with their children.
    if parent_ids:
        print("--- Run Hierarchy ---")
        # Sort parent runs by their name.
        sorted_parent_ids = sorted(list(parent_ids), key=lambda pid: id_to_name.get(pid, "Parent Run (Not in this directory)"))
        for parent_id in sorted_parent_ids:
            parent_name = id_to_name.get(parent_id, "Parent Run (Not in this directory)")
            print(f"{parent_id}\t{parent_name}")

            # Sort child runs by their name.
            sorted_child_ids = sorted(parent_to_children[parent_id], key=lambda cid: id_to_name.get(cid, "Unnamed Child Run"))
            for child_id in sorted_child_ids:
                child_name = id_to_name.get(child_id, "Unnamed Child Run")
                print(f"\t{child_id}\t{child_name}")
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
    display_run_hierarchy(args.path)

if __name__ == "__main__":
    main()
