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
