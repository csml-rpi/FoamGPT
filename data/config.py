# config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    max_loop: int = 10
    
    batchsize: int = 10
    searchdocs: int = 2
    run_times: int = 1  # current run number (for directory naming)
    database_path: str = Path(__file__).resolve().parent.parent / "database"
    run_directory: str = Path(__file__).resolve().parent.parent / "runs"
    case_dir: str = ""
    max_time_limit = 36000 # Max time limit after which the openfoam run will be terminated
    model_provider: str = "openai"
    model_version: str = "gpt4o"
    temperature: float = 0.6
    
