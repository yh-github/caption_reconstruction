from pathlib import Path

def add_suffix_to_path(path:Path, suffix:str) -> Path:
    new_filename = f"{path.stem}{suffix}{path.suffix}"
    return path.parent / new_filename
