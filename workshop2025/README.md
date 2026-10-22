# FoamGPT
Paper: https://arxiv.org/pdf/2505.04997
Dataset: https://huggingface.co/datasets/LeoYML/FoamGPT
CFD-LLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics: https://arxiv.org/abs/2509.20374

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
```

## Finetuning Call
Please type out the following command into command prompt or other terminals and click enter:
  ```bash
  python train.py
  ```

Ensure that you have gone through the Data Parsing Pipeline before type out the following command into command prompt or other terminals.

## Using bfloat16 vs float16
You may recieve an error relating to bfloat16 not being compatable with your graphics card(s). If required, you can make the following changes in finetune/finetuning_script:

### Quantifcation Configuration

```bash
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

to 

```bash
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### Model

```bash
md = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
```

to 

```bash
md = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
```

### Training Arguments

```bash
training_args = SFTConfig(
    ...
    fp16=False,
    bf16=True,
    ...
)
```

to

```bash
training_args = SFTConfig(
    ...
    fp16=True,
    bf16=False,
    ...
)
```

## CUDA Data
You may run into an issue with your GPU(s) running out of memory. If so you can make the following changes in finetune/fineing_script.py:

### Change bloat16 to float16
See Using bfloat16 vs float16 Section above

### Change bloat16 to float32
See Using bfloat16 vs float16 Section above except change 
```bash
training_args = SFTConfig(
    ...
    fp16=False,
    bf16=True,
    ...
)
```

to

```bash
training_args = SFTConfig(
    ...
    fp16=False,
    bf16=False,
    ...
)
```

### Changing CUDA Memory Variables

```bash
training_args = SFTConfig(
    ...
    num_train_epochs=7,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    ...
)
```

```bash
training_args = SFTConfig(
    ...
    num_train_epochs=7,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    ...
)
```

If problems still arise, try lowering the arguments above and/or combining these changes with any of the above 2 methods.
Open a pull request if you wish to contribute to this repo.
