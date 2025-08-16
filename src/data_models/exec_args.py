from pathlib import Path
from pydantic import BaseModel

class ExecArgs(BaseModel):
    config_path: Path
    verbose: bool
    dry_run: bool
    validate_cache: bool
