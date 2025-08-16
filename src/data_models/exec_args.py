from pydantic import BaseModel

class ExecArgs(BaseModel):
    config_path: str
    command: str
    verbose: bool


