#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script to load a JSON or YAML file and start a Python REPL with its contents.

This script takes one command-line argument: the path to a data file.
It automatically detects the file type based on the extension (.json, .yml, .yaml).
It loads the data into a variable called 'data' and then starts an
interactive Python session (REPL), allowing you to inspect and manipulate
the data directly from your terminal.

The 'json' and 'yaml' modules are pre-imported and available in the REPL.

Requires the PyYAML package for YAML support:
    pip install PyYAML

Usage:
    python load_data_repl.py your_data_file.json
    python load_data_repl.py your_config_file.yml

Example:
    $ python load_data_repl.py users.json
    >>> JSON data loaded into the 'data' variable.
    >>> type(data)
    <class 'list'>
"""

import json
import sys
import code
import os

# Try to import yaml, but don't fail if it's not installed
# We'll handle the error gracefully later.
try:
    import yaml
except ImportError:
    yaml = None

def load_and_repl(file_path):
    """
    Loads data from a JSON or YAML file and starts an interactive REPL session.

    Args:
        file_path (str): The path to the data file to load.
    """
    try:
        # Determine the file type by its extension
        _, file_ext = os.path.splitext(file_path)
        file_ext = file_ext.lower()

        # Open and load the data file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_ext == '.json':
                data = json.load(f)
                file_type = "JSON"
            elif file_ext in ['.yml', '.yaml']:
                if yaml is None:
                    print("❌ Error: PyYAML is not installed. Please run 'pip install PyYAML' to use YAML files.", file=sys.stderr)
                    sys.exit(1)
                data = yaml.safe_load(f)
                file_type = "YAML"
            else:
                print(f"❌ Error: Unsupported file type '{file_ext}'. Please use .json, .yml, or .yaml.", file=sys.stderr)
                sys.exit(1)
        
        # Create a banner message to display when the REPL starts
        banner = (
            f"✅ {file_type} data from '{file_path}' is loaded into the 'data' variable.\n"
            f"   The 'json' and 'yaml' modules are also available.\n"
            f"   You can now interact with it. Type exit() or press Ctrl+D to quit."
        )
        
        # Define the local environment for the REPL
        local_vars = {
            'data': data,
            'json': json,
            'yaml': yaml
        }
        
        # Start the interactive console
        code.interact(banner=banner, local=local_vars)

    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: The file '{file_path}' is not a valid JSON file.", file=sys.stderr)
        sys.exit(1)
    except (yaml.YAMLError if yaml else Exception) as e:
        # Handle YAML parsing errors specifically if PyYAML is installed
        if 'YAMLError' in str(type(e)):
             print(f"❌ Error: The file '{file_path}' is not a valid YAML file.", file=sys.stderr)
        else:
             print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python load_data_repl.py <path_to_data_file>", file=sys.stderr)
        sys.exit(1)
    
    # Get the file path from the first argument
    data_file_path = sys.argv[1]
    
    # Run the main function
    load_and_repl(data_file_path)

