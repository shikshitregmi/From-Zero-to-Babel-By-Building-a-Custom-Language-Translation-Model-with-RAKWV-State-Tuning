# From Zero to Babel : Building a Custom Language Translation Model with RAKWV State Tuning

## Introduction

In the rapidly evolving landscape of artificial intelligence, large language models (LLMs) have traditionally been dominated by Transformer-based architectures. However, a new contender is reshaping the efficiency frontier: **RWKV** (Receptance Weighted Key Value). Often described as a Transformer alternative that combines the parallelizable training of Transformers with the efficient inference of RNNs, RWKV offers a compelling path for developers who need high performance without the prohibitive computational costs of traditional models.

But having a base model is only half the battle. To make a model truly useful for a specific task like language translation it must be adapted to understand the nuances between source and target languages. This is where **State Tuning** (or more specifically, *RAKWV State Tuning*) comes in. Unlike full fine-tuning, which updates all model weights and requires massive GPU memory, state tuning allows us to efficiently adjust the model's "state" or attention mechanisms, drastically reducing hardware requirements while maintaining high translation quality.

I will show you how to build a translation model by using an English-to-Chinese dataset. Specifically, for this tutorial, I trained the model using an RTX 4060 with the RWKV-V7 1.5 billion parameter model. However, you can still follow this tutorial accordingly, even if you are using different hardware or a different language model. If you follow these instructions, you can apply this workflow to your own custom dataset. Whether you're working with Spanish, Japanese, Arabic, or any other language pair, the principles and code we'll cover will adapt to your needs

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
from pathlib import Path

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

####  Clone the Tool Repository

```bash
git clone https://github.com/Abel2076/json2binidx_tool.git
cd json2binidx_tool
```

#### Place Your JSONL File

Copy your Q&A‑formatted JSONL file into the `data` folder inside the tool directory:

```bash
cp /path/to/your/dataset_qna.jsonl ./data/
```

#### Run the Conversion

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
Here's an improved, clearer, and more accurate version of **Section 3: Fine-Tuning with RWKV State Tuning**. I've corrected typos, clarified steps, added warnings about critical parameters (especially the learning rate), and improved the overall structure.


## 3 Fine-Tuning with RWKV State Tuning

After preparing your dataset, we move to the most important step: **fine-tuning the RWKV model** using **state tuning** to teach it a new language pair.  
We’ll use an **RTX 4060** and the **RWKV-v7 1.5B** model, but the guide works for other setups too.

###  Clone the RWKV-PEFT Repository

First, clone the official PEFT (Parameter-Efficient Fine-Tuning) repository for RWKV:

```bash
git clone https://github.com/JL-er/RWKV-PEFT.git
cd RWKV-PEFT
pip install -r requirements.txt
```

> **Note:** Works on Linux or WSL. If you’re on Windows, use WSL2 for best compatibility.

### Configure the Training Script

The main training script is `train.py`. We’ll configure it via the shell script `scripts/state_tuning.sh`.  
Open that file in a text editor (VS Code, nano, etc.) and adjust the following parameters.

#### Set File Paths

```bash
load_model="/path/to/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"
proj_dir="/path/to/output/dir"               # logs & trained state files
data_file="/path/to/dataset_prefix"          # no .bin/.idx suffix
```

- `load_model` – Base RWKV model (`.pth` file)
- `proj_dir` – Where to save training logs and the final state file
- `data_file` – Prefix of your dataset (e.g., `/data/my_dataset` → `my_dataset.bin` & `my_dataset.idx`)

#### Model Architecture Parameters (`n_layer`, `n_embd`)

Match these to your base model:

| Model Size | `n_layer` | `n_embd` |
|------------|-----------|----------|
| 0.1B       | 12        | 768      |
| 0.4B       | 24        | 1024     |
| **1.5B**   | **24**    | **2048** |
| 2.9B       | 32        | 2560     |
| 7B         | 32        | 4096     |
| 14B        | 61        | 4096     |

For the 1.5B model:  
```bash
n_layer=24
n_embd=2048
```

#### Critical Training Parameters

| Parameter | Recommended Value (State Tuning) | Explanation |
|-----------|----------------------------------|-------------|
| `micro_bsz` | Start from 1, increase if VRAM allows | Batch size per GPU |
| `ctx_len` | 512 (start short) | Context length – lower saves VRAM |
| `epoch_steps` | 200–1000 | Steps per epoch |
| `epoch_save` | 1 | Save state every N epochs |
| `epoch_count` | 5–20 | Total epochs |

> ** Very Important for State Tuning**  
> Unlike full fine‑tuning, **state tuning requires a high learning rate**.  
> Set `--lr_init 1e-2` and `--lr_final 1e-4`.  
> *The example script incorrectly shows `1e-5` – do not use that!*

#### Other Important Flags

| Flag | Value | Notes |
|------|-------|-------|
| `--data_type` | `binidx` (or `jsonl`) | Use `binidx` for efficient binary format |
| `--vocab_size` | `65536` | Default for RWKV World models |
| `--accelerator` | `gpu` | CPU not supported for training |
| `--devices` | `1` | Number of GPUs |
| `--precision` | `bf16` | Balanced speed/stability on RTX 4060 |
| `--strategy` | `deepspeed_stage_1` | Saves memory |
| `--grad_cp` | `1` | Enable gradient checkpointing (saves VRAM) |
| `--peft` | `state` | **Must** be `state` for state tuning |
| `--my_testing` | `"x070"` | RWKV version (v7) |
| `--op` | `fla` | Required for state tuning |
| `--lr_schedule` | `wsd` or `cos_decay` | Warmup‑stable‑decay (optional) |
| `--wandb` | (optional) | Set to project name for logging |

### Final Example Script (`state_tuning.sh`)

Here is a corrected script for **RWKV 1.5B state tuning**:

```bash
load_model="/mnt/g/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"
proj_dir="/mnt/g/RWKV-PEFT/Output"
data_file="/mnt/g/RWKV-PEFT/Data/output_text_document"

n_layer=24
n_embd=2048

micro_bsz=2          # start small, increase after testing
epoch_save=1
epoch_steps=200
ctx_len=512

python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--vocab_size 65536 \
--data_type binidx \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 10 \
--lr_init 1e-2 --lr_final 1e-4 \
--accelerator gpu --precision bf16 \
--devices 1 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x070" \
--peft state --op fla
```

> **Save the file after editing.**

### Start Training

Run the following command inside the `RWKV-PEFT` directory:

```bash
sh scripts/state_tuning.sh
```

###  Monitor and Find Outputs

- **Training logs** – Printed to console and saved in `proj_dir` as `.txt` files.
- **Trained state file** – Saved in `proj_dir` with a name like `rwkv-{step}.pth`. This file contains only the **state** (learned parameters), not the full model – it’s small and can be reloaded later.

###  Common Issues & Tips

| Problem | Solution |
|---------|----------|
| Out of memory (CUDA OOM) | Reduce `micro_bsz` to 1, reduce `ctx_len` to 256, or set `grad_cp=1` |
| Loss not decreasing | Increase learning rate (e.g., `--lr_init 2e-2`). State tuning needs high LR. |
| Dataset not found | Double-check `data_file` path – no `.bin` or `.idx` extension. |
| Slow training | Use `--precision bf16` and `--strategy deepspeed_stage_1`. Also increase `micro_bsz` if VRAM allows. |
| Want to resume training | Add `--load_state /path/to/previous_state.pth` |
Here's a polished, improved version of the final section **"Saving and Running the Model"** — clearer structure, corrected grammar, and more actionable steps.

## 4 Saving and Running the Model

After fine-tuning, you have a **state file** (`.pth`) containing the adapter weights. Now you’ll learn how to use it for translation inference. There are two recommended approaches:

1. **Merge** the state file into the base model (produces a standalone fine-tuned model).
2. **Mount** the state file separately at runtime using inference tools like **RWKV Runner** or **Ai00** (easier, more flexible).

> **Critical:** The state file must be used with the **exact same base RWKV model** that was used during training.  
> *Example:* If you trained with `RWKV-x070-World-1.5B-v3`, you must load that same base model when mounting or merging the state file.


### Option 1: Merge State into Base Model (Permanent)

Use the provided script `demo-state-merge.sh` to merge the adapter weights directly into the base model, creating a single, self-contained fine-tuned model.

```bash
bash scripts/demo-state-merge.sh
```

After merging, you’ll have a new `.pth` file that includes both the original weights and your fine-tuned adaptations. This file can be used like any regular RWKV model.

###  Option 2: Mount State Separately (Recommended)

Most users prefer **mounting** – loading the state file alongside the base model at inference time. This keeps the original model intact and allows easy swapping of different state files.

#### Using RWKV Runner:
- Load your base model (e.g., `RWKV-x070-World-1.5B-v3.pth`)
- In the settings, specify the path to your trained state file (`.pth`)
- The runner will automatically apply the state during generation

#### Using Ai00:
- Configure the model path and state file path in the startup configuration
- Launch the server – the state will be active for all requests



### Example Translation Results

Our sample dataset was built with short contexts and heavy use of emojis. After state tuning, the model produces outputs like this:

(Replace with your actual language pair and examples.)

The state file captures style, tone, and translation patterns even with limited training data.

### Sharing Your Fine-Tuned State

One of the best features of state tuning is that **others can enhance their RWKV models simply by mounting your state file** – no need to retrain. Share your `.pth` file along with a note about the base model version used.

>  **Pro tip:** State files are typically very small (often < 50 MB), making them easy to distribute and version‑control.



### Next Steps

You’ve now:
- Prepared a parallel corpus
- Fine‑tuned RWKV‑v7 with state tuning
- Saved and deployed your custom translation model

Congratulations! You’ve built a custom language translation model from scratch using RWKV State Tuning. Experiment with different datasets, context lengths, or learning rates to further improve quality.