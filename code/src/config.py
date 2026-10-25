# config.py
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    max_loop: int = 10
    
    batchsize: int = 10
    searchdocs: int = 2 # max(10, searchdocs)
    run_times: int = 1  # current run number (for directory naming)
    database_path: str = Path(__file__).resolve().parent.parent / "database"
    run_directory: str = Path(__file__).resolve().parent.parent / "runs"
    case_dir: str = ""
    max_time_limit: int = 360 # Max time limit after which the openfoam run will be terminated, in seconds
    recursion_limit: int = 100  # LangGraph recursion limit
    # Input writer generation mode:
    # - "sequential_dependency": generate files sequentially; use already-generated files as context to enforce consistency.
    # - "parallel_no_context": generate files in parallel without cross-file context (faster, may need more reviewer iterations).
    input_writer_generation_mode: str = "parallel_no_context" # for testing fine-tuned models

    # Dataset logging (for fine-tuning data extraction)
    dataset_log_path: str = ""  # Path to per-case dataset.jsonl (empty = disabled)
    case_id: str = ""           # e.g. "Basic/Cavity/1" or "Advanced/Cavity_LES"

    # Embedding Configuration
    embedding_provider: str = "huggingface" # [openai, huggingface, ollama]
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B" # e.g. "text-embedding-3-small", "text-embedding-3-large", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-8B"

    models: dict = field(default_factory=lambda: {
        "plan": {
            "model_provider": "bedrock",
            # "model_version": "us.anthropic.claude-3-5-sonnet-20241022-v2:0", # sonnet 3.5
            "model_version": "us.anthropic.claude-opus-4-6-v1", # opus 4.6
            "temperature": 0.0
        },
        "write": {
            "model_provider": "bedrock",
            # "model_version": "us.anthropic.claude-3-5-sonnet-20241022-v2:0", # sonnet 3.5
            "model_version": "us.anthropic.claude-opus-4-6-v1", # opus 4.6
            "temperature": 0.0
        },
        "review": {
            "model_provider": "bedrock",
            # "model_version": "us.anthropic.claude-3-5-sonnet-20241022-v2:0", # sonnet 3.5
            "model_version": "us.anthropic.claude-opus-4-6-v1", # opus 4.6
            "temperature": 0.0
        },
    })

    # Maps substep name → {"model_provider", "model_version", "temperature"}.
    # Any substep listed here overrides the service-level model for that substep only.
    substep_model_overrides: dict = field(default_factory=dict)

