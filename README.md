# From Zero to Babel : Building a Custom Language Translation Model with RAKWV State Tuning

## Introduction

In the rapidly evolving landscape of artificial intelligence, large language models (LLMs) have traditionally been dominated by Transformer-based architectures. However, a new contender is reshaping the efficiency frontier: **RWKV** (Receptance Weighted Key Value). Often described as a Transformer alternative that combines the parallelizable training of Transformers with the efficient inference of RNNs, RWKV offers a compelling path for developers who need high performance without the prohibitive computational costs of traditional models.

But having a base model is only half the battle. To make a model truly useful for a specific task like language translation it must be adapted to understand the nuances between source and target languages. This is where **State Tuning** (or more specifically, *RAKWV State Tuning*) comes in. Unlike full fine-tuning, which updates all model weights and requires massive GPU memory, state tuning allows us to efficiently adjust the model's "state" or attention mechanisms, drastically reducing hardware requirements while maintaining high translation quality.

I will show you how to build a translation model by using an English-to-Chinese dataset. Specifically, for this tutorial, I trained the model using an RTX 4060 with the RWKV7 1.5 billion parameter model. However, you can still follow this tutorial accordingly, even if you are using different hardware or a different language model. If you follow these instructions, you can apply this workflow to your own custom dataset. Whether you're working with Spanish, Japanese, Arabic, or any other language pair, the principles and code we'll cover will adapt to your needs

In this tutorial, we will go From Zero to Babel starting with no specialized model and ending with a custom, high-performance translation pipeline. Here is what we will cover:

1.  **Setting Up the Environment:** We will configure a Python environment and install the necessary dependencies to work with RWKV models.
2.  **Preparing the Data:** We will curate and preprocess a parallel corpus (source-target text pairs) to format it for training.
3.  **Fine-Tuning with RAKWV State Tuning:** We will implement the fine-tuning process, focusing on how to apply state tuning efficiently to teach the model a new language pair.
4.  **Saving and Running the Model:** Finally, we will save our fine-tuned adapter weights and run inference with our custom translation model.

Whether you are a developer with limited GPU resources or a researcher looking for faster iteration cycles, this guide will equip you with the tools to build a specialized translation model that punches above its weight class. Let’s get started.

## Setting Up the Environment

Before we start building our custom translation model, we need to set up a robust training environment. RWKV training relies on specific versions of PyTorch, DeepSpeed, and PyTorch Lightning to achieve optimal performance. The following steps will guide you through configuring a Conda environment, installing the necessary dependencies, and verifying that your GPU (CUDA) is ready for training.

### 1. Configure the Conda Virtual Environment

We recommend using **Conda** (Miniconda or Anaconda) to manage the environment and avoid dependency conflicts. If you don’t have Conda installed on your Linux system (or WSL), start by installing Miniconda:

```bash
# Download the latest Miniconda installation package
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installation script (follow the prompts and type "yes" when asked)
sh Miniconda3-latest-Linux-x86_64.sh -u

# Restart your shell or source the bashrc to activate Conda
source ~/.bashrc
```

Now create a dedicated Conda environment named `rwkv` with Python 3.10 (the version recommended for RWKV training):

```bash
# Create the environment
conda create -n rwkv python=3.10

# Activate it
conda activate rwkv
```

You should see the prompt change to `(rwkv)`, indicating that you are now working inside the new environment.
### 2. Install the Required Software

With the environment activated, install the core packages. The following software stack is considered best practice for RWKV training:

- **PyTorch** 2.1.2+cu121 (or a later version with CUDA 12.1 support)
- **DeepSpeed** (latest)
- **PyTorch Lightning** 1.9.5
- Additional tools: `wandb`, `ninja`, `bitsandbytes`, `einops`, `triton`, `rwkv-fla`, `transformers`, `GPUtil`, `plotly`, `gradio`, `datasets`

Run these commands in sequence:

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Lightning (specific version), DeepSpeed, and other common ML tools
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade

# Install additional packages used by RWKV-PEFT and the translation pipeline
pip install bitsandbytes einops triton rwkv-fla rwkv transformers GPUtil plotly gradio datasets
```

> **Tip:** While the versions above are the recommended ones, you may use newer versions as long as they remain compatible with RWKV and DeepSpeed.
### 3. Verify the CUDA Environment

After installation, it’s crucial to confirm that PyTorch can detect your GPU(s) and that CUDA is properly configured. Run the following Python commands in your terminal:

```bash
python3
```

Then inside the Python interpreter:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

If you see `CUDA available: True`, everything is set up correctly. If it returns `False`, check that the PyTorch CUDA version matches your installed NVIDIA drivers. You may need to reinstall CUDA 12.1 or later (see [NVIDIA’s official documentation](https://developer.nvidia.com/cuda-downloads) for installation steps).

## 2. Preparing the Data

With our environment ready, the next step is to prepare a parallel corpus for translation. The data should be structured so the model learns to map source language sentences to target language translations. We will cover three substeps:

1. Converting a raw CSV dataset (with source and target columns) into a JSONL format.
2. Formatting the data as a single‑round question–answer (Q&A) structure, which RWKV handles effectively.
3. Transforming the JSONL into `binidx` files—the efficient binary format required for RWKV training.

> **Note:** If you already have a dataset in the single‑round Q&A JSONL format, you can skip directly to **Step 3**.


### Step 1: CSV to JSONL

A common way to store parallel texts is in a CSV file with two columns: source language and target language. For this tutorial, we assume a CSV with columns `Column1` (source) and `Column2` (target). For example:

| Column1 | Column2 |
|---------|--------|
| We're at a tipping point in human history, a species poised between gaining the stars and losing the planet we call home. | 我们身处人类历史的转折点上， 人类处在想获得其他星球， 同时也在失去地球家园的尴尬境地。 |
| Even in just the past few years, we've greatly expanded our knowledge of how Earth fits within the context of our universe. | 即使是在过去短短几年的时间里， 我们对地球 是如何在宇宙中存在的认识 已经有了大幅度的提升。 |
| NASA's Kepler mission has discovered thousands of potential planets around other stars, indicating that Earth is but one of billions of planets in our galaxy. | 美国国家航空航天局的 开普勒任务已经发现了 围绕着其他恒星的数千颗 潜在的行星， 这也表明了地球只是银河系中 数十亿行星中的一颗。 |

We provide a helper script `convert_csv_to_json.py` (included in the repository) that reads such a CSV and outputs a JSONL file where each line is a dictionary containing the source and reference.

**How to use it:**

1. Place your CSV file (e.g., `chinese_to_english.csv`) in your working directory.
2. Edit the script to set the correct file paths:

```python
# Define file paths
input_csv = Path(r"chinese_to_english.csv")   # Your CSV file
output_jsonl = Path(r"dataset.jsonl")         # Desired JSONL output path
```

3. Run the script:

```bash
python convert_csv_to_json.py
```

The script will generate a JSONL file with one JSON object per line, each containing the source and target. For the next step, we will reformat these objects into the single‑round Q&A style.

---

### Step 2: Formatting as Single‑Round Q&A

RWKV is often fine‑tuned using a conversational format that includes roles like `User:` and `Assistant:`. For translation, we can treat the source sentence as the user’s input and the translation as the assistant’s response. The model learns to produce the appropriate translation when prompted.

**Data format:**

```json
{"text": "User: {source_sentence}\n\nAssistant: {target_sentence}"}
```

For example:

```json
{"text": "User: We're at a tipping point in human history, a species poised between gaining the stars and losing the planet we call home.\n\nAssistant: 我们身处人类历史的转折点上， 人类处在想获得其他星球， 同时也在失去地球家园的尴尬境地。"}
```

> **Tip:** You can also include a `System:` role to set the context (e.g., “You are a helpful translator”), but for translation tasks the simple User–Assistant pair is often sufficient.

If your CSV‑to‑JSONL output is already a list of objects with `source` and `target` fields, you can easily convert it to the required format using a short Python script or by modifying the original conversion script to produce this format directly.

**Example conversion snippet:**

```python
import json

with open("dataset.jsonl", "r", encoding="utf-8") as infile, \
     open("dataset_qna.jsonl", "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        source = data["Column1"]   # adjust field name as needed
        target = data["Column2"]
        text_entry = {"text": f"User: {source}\n\nAssistant: {target}"}
        outfile.write(json.dumps(text_entry, ensure_ascii=False) + "\n")
```

Now you have a JSONL file ready for the final conversion.


### Step 3: Converting JSONL to `binidx` Files

RWKV training uses a custom binary format (`bin` + `idx`) for efficient data loading. We will use the `json2binidx_tool` to perform this conversion.

#### 3.1 Clone the Tool Repository

```bash
git clone https://github.com/Abel2076/json2binidx_tool.git
cd json2binidx_tool
```

#### 3.2 Place Your JSONL File

Copy your Q&A‑formatted JSONL file into the `data` folder inside the tool directory:

```bash
cp /path/to/your/dataset_qna.jsonl ./data/
```

#### 3.3 Run the Conversion

Execute the preprocessing script with the following command (modify the input and output paths accordingly):

```bash
python3 tools/preprocess_data.py \
  --input ./data/dataset_qna.jsonl \
  --output-prefix ./data/dataset \
  --vocab ./rwkv_vocab_v20230424.txt \
  --dataset-impl mmap \
  --tokenizer-type RWKVTokenizer \
  --append-eod
```

**Explanation of parameters:**

- `--input`: path to your JSONL file.
- `--output-prefix`: prefix for the output files. The tool will create `dataset_text_document.bin` and `dataset_text_document.idx`.
- `--vocab`: path to the RWKV vocabulary file (provided in the repository).
- `--dataset-impl mmap`: memory‑mapped implementation for speed.
- `--tokenizer-type RWKVTokenizer`: use the RWKV tokenizer.
- `--append-eod`: append end‑of‑document tokens.

After successful execution, you will find two new files in the `data` folder:  
`dataset_text_document.bin` and `dataset_text_document.idx`. These are ready to be used as input for RWKV fine‑tuning.


