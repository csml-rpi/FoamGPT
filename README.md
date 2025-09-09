# FoamGPT
Paper: https://arxiv.org/pdf/2505.04997
Dataset: https://huggingface.co/datasets/LeoYML/FoamGPT
## Finetuning Requirements
Before you begin, please take note that the finetuning_script code is written and ran the following library versions:

- **Python 3.12.7**: Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **Pip**: Python package installer. It usually comes with Python installations. You can check if you have it by running:
  ```bash
  pip --version
  ```
    - **transformers 4.52.3**: A library for state-of-the-art natural language processing
    - **trl 0.18.0**: A library for reinforcement learning with transformers
    - **peft 0.17.0**: library for parameter-efficient fine-tuning
    - **jinja2 3.1.4**: A templating engine for Python, used for rendering templates 
    - **bitsandbytes 0.45.5**: A library for efficient training of large models
    - **tf-keras 2.19.0**: A high-level API for building and training deep learning models with TensorFlow

## Finetuning Call
Please type out the following command into command prompt or other terminals and click enter:
  ```bash
  python finetuning_script.py
  ```

## Environment Setup

To run everything (including all parsing, preprocessing, FAISS generation, and model generation scripts), create a Conda environment with the following configuration:
  ```bash
  conda env create --file enviornment.yml
```
## Data Parsing Pipeline

To process and prepare OpenFOAM tutorials for LLM finetuning:
```bash
Step-by-step Script Execution:
python data/script/tutorial_parser.py --wm_project_dir="$WM_PROJECT_DIR" --output_dir="data/raw"
python data/foamgpt/foamgpt_parser.py --char-limit=1500
python data/script/faiss_allrun_scripts.py  
python data/script/faiss_command_help.py  
python data/script/faiss_tutorials_details.py  
python data/script/faiss_tutorials_structure.py
python data/foamgpt/foamgpt_gen.py
python data/foamgpt/foamgpt_data.py
python data/foamgpt/foamgpt_huggingface.py  
python data/foamgpt/foamgpt_openai.py
python finetune/finetuning_script.py
```
## Modifications Made

Added config.py to ./data

Added util.py to ./data

Moved data/database → data/fiass/database

Changed stats = llm_service.get_stats() → stats = llm_service.get_statistics() in data/foamgpt/foamgpt_data.py

Changed some aspects of finetuning_script.py to align with workstation