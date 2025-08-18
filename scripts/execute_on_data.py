#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script to execute a Python code string against data from JSON or YAML files.

This script takes a Python code string and a list of file paths as arguments.
For each file, it loads the data (detecting JSON/YAML automatically), makes
the data available in a 'data' variable (and the path in a 'file_path' Path object),
and then executes the provided code.

This allows for quick, one-line data processing and inspection from the shell,
similar to `perl -e` or `python -c`.

Requires the PyYAML package for YAML support:
    pip install PyYAML

Usage:
    python execute_on_data.py "<code>" <file1> <file2> ...

Examples:
    # Print the name of the first user from a JSON file
    python execute_on_data.py "print(data[0]['name'])" users.json

    # Print the number of items in each file, using the Path object's name attribute
    python execute_on_data.py "print(f'{file_path.name}: {len(data)} items')" data.json config.yml

    # Modify data and print it as YAML (does not save back to file)
    python execute_on_data.py "data['new_key'] = 'new_value'; print(yaml.dump(data))" config.yml
"""

import json
import sys
from pathlib import Path

# Try to import yaml, but don't fail if it's not installed.
# We'll handle the error gracefully later.
try:
    import yaml
except ImportError:
    yaml = None

def process_file(file_path: Path, code_string: str):
    """
    Loads a data file and executes a code string with the data.

    Args:
        file_path (Path): The path object for the JSON or YAML file.
        code_string (str): The Python code to execute.
    """
    try:
        # Determine the file type by its extension from the Path object
        file_ext = file_path.suffix.lower()

        # Open and load the data file
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_ext == '.json':
                data = json.load(f)
            elif file_ext in ['.yml', '.yaml']:
                if yaml is None:
                    print(f"❌ Error: PyYAML is not installed for file '{file_path}'. Please run 'pip install PyYAML'.", file=sys.stderr)
                    return
                data = yaml.safe_load(f)
            else:
                print(f"❌ Error: Unsupported file type '{file_ext}' for file '{file_path}'.", file=sys.stderr)
                return

        # Set up the execution context for the user's code string.
        # The code will have access to these variables.
        execution_context = {
            'data': data,
            'json': json,
            'yaml': yaml,
            'file_path': file_path,
            'Path': Path
        }

        # Execute the user's code string in the prepared context
        exec(code_string, {'__builtins__': __builtins__}, execution_context)

    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"❌ Error: The file '{file_path}' is not a valid JSON file.", file=sys.stderr)
    except (yaml.YAMLError if yaml else Exception) as e:
        if 'YAMLError' in str(type(e)):
             print(f"❌ Error: The file '{file_path}' is not a valid YAML file.", file=sys.stderr)
        else:
             print(f"❌ An unexpected error occurred while processing '{file_path}': {e}", file=sys.stderr)
    except Exception as e:
        print(f"❌ An error occurred while executing code on '{file_path}': {e}", file=sys.stderr)


if __name__ == "__main__":
    # Check for the correct number of arguments
    if len(sys.argv) < 3:
        print("Usage: python execute_on_data.py \"<code>\" <file1> <file2> ...", file=sys.stderr)
        sys.exit(1)

    # The first argument is the code string
    code_to_execute = sys.argv[1]
    
    # The rest of the arguments are file paths, converted to Path objects
    file_paths = [Path(p) for p in sys.argv[2:]]

    # Process each file
    for path in file_paths:
        process_file(path, code_to_execute)
