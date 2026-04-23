# SciImpact: Evaluation Code and Data

This repository contains the released data and experiment code for **SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction**.

SciImpact studies scientific impact prediction as a pairwise ranking problem. Given two comparable artifacts, a model must decide which one has higher impact under a specific definition of impact. Rather than treating citations as the only proxy for impact, SciImpact benchmarks seven complementary dimensions:

- Citation
- Award
- Patent
- Media
- Code
- Dataset
- Model

Across the full release, SciImpact contains **215,928 contrastive pairs** covering **19 scientific fields** and heterogeneous input types including paper titles and abstracts, GitHub README files, Hugging Face dataset cards, and Hugging Face model cards.

## Benchmark Overview

Each benchmark instance is a contrastive pair with a constrained target answer. The design keeps evaluation simple while making it possible to compare model behavior across very different impact signals.

### Impact Dimensions

| Dimension | Input | Target signal | Pairs |
| --- | --- | --- | ---: |
| Citation | Paper title + abstract | Which paper has more citations | 43,309 |
| Award | Paper title + abstract | Which paper won a major award | 42,003 |
| Patent | Paper title + abstract | Which paper is cited in more patents | 45,746 |
| Media | Paper title + abstract | Which paper receives more media mentions | 52,739 |
| Code | GitHub README | Which repository has more stars | 9,193 |
| Dataset | Hugging Face dataset card | Which dataset has more downloads | 10,475 |
| Model | Hugging Face model card | Which model has more downloads | 12,463 |

The `Award` dimension is implemented with three sub-benchmarks:

- `cs_best_paper`: computer science best-paper prediction
- `MDPI_best_paper`: MDPI best-paper prediction
- `nobel`: Nobel-winning paper prediction

Their released pair counts are:

- `cs_best_paper`: 7,926
- `MDPI_best_paper`: 10,343
- `nobel`: 23,734

### Field Coverage

SciImpact covers 19 fields:

- art
- biology
- business
- chemistry
- computer science
- economics
- engineering
- environmental science
- geography
- geology
- history
- materials science
- mathematics
- medicine
- philosophy
- physics
- political science
- psychology
- sociology

Depending on the underlying source, computer science may appear in code as `csrankings` or `CSCombined`.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `LLaMA-Factory/` | Fine-tuning configuration, dataset registration, launch scripts, and DeepSpeed config |
| `sft_data/` | Released supervised fine-tuning train/validation/test JSONL splits |
| `citation/` | Citation prediction evaluation scripts |
| `cs_best_paper/` | CS best-paper award evaluation scripts |
| `MDPI_best_paper/` | MDPI best-paper evaluation scripts |
| `nobel/` | Nobel paper evaluation scripts |
| `patent/` | Patent-impact evaluation scripts |
| `media/` | Media-impact evaluation scripts |
| `github/` | GitHub README star-prediction evaluation scripts |
| `huggingface_dataset/` | Hugging Face dataset card download-prediction scripts |
| `huggingface_model/` | Hugging Face model card download-prediction scripts |

Each task directory typically contains two entry points:

- `run_vllm_*.py`: evaluate base instruction-tuned models with vLLM
- `run_sft_*.py`: evaluate locally fine-tuned SciImpact checkpoints

## Released Data

### Unified SFT Splits

The current public release includes a unified supervised fine-tuning dataset in `sft_data/`:

| File | Split size |
| --- | ---: |
| `paper_impact_sft_train_clean.jsonl` | 88,515 |
| `paper_impact_sft_val_clean.jsonl` | 64,206 |
| `paper_impact_sft_test_clean.jsonl` | 63,207 |

Total: **215,928** examples.

`LLaMA-Factory/dataset_info.json` registers:

- `paper_impact_sft_train`
- `paper_impact_sft_val`

in `sharegpt` format for fine-tuning. The released test split is kept as a separate JSONL file for held-out evaluation.

### Data Format

Each example is stored as a conversation with a task-specific system instruction, a pairwise comparison prompt, and a constrained gold answer:

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are an impartial judge deciding which of two research papers has more citations."
    },
    {
      "from": "human",
      "value": "Paper A: ... Paper B: ..."
    },
    {
      "from": "gpt",
      "value": "Paper B has more citations."
    }
  ]
}
```

This format unifies all SciImpact tasks under one SFT dataset while keeping the target dimension explicit in the instruction.

### Task-Specific Evaluation Data

The repo does not store every raw pair file directly under the top-level task directories. Instead, many evaluation scripts download their task-specific files on demand from the Hugging Face dataset repository configured inside the script through constants such as:

- `HF_REPO_ID`
- `PAIR_FILE`
- `PAIR_BASE`
- `BASE_PATH`
- `README_FILE`

For example:

- citation scripts load paper pairs by field
- GitHub scripts load repository pairs plus README text
- dataset/model scripts load paired cards and download counts
- patent/media scripts load field-specific sampled pair files

If you are adapting the release, inspect the constants at the top of each script to locate the exact upstream file names.

## Installation

The evaluation scripts rely on a standard Python stack built around:

- `vllm`
- `transformers`
- `huggingface_hub`
- `tqdm`

The training pipeline additionally relies on:

- `LLaMA-Factory`
- `deepspeed`
- `wandb`

A typical setup is:

```bash
pip install vllm transformers huggingface_hub tqdm wandb deepspeed
```

and then use the vendored `LLaMA-Factory/` code in this repository for fine-tuning.

## Training

SciImpact uses a vendored `LLaMA-Factory` setup for supervised fine-tuning.

### What the provided training scripts do

The provided shell scripts in `LLaMA-Factory/` launch full-parameter SFT with:

- a 4096-token cutoff length
- bf16 training
- DeepSpeed ZeRO-2 via `ds_zero2.json`
- Weights & Biases logging
- model backbones such as `meta-llama/Llama-3.2-3B-Instruct` and `Qwen/Qwen3-4B-Instruct-2507`

### Required Environment

The training scripts expect:

- a working Python environment for `LLaMA-Factory`
- GPUs for multi-GPU training
- a Hugging Face token, either in `HF_TOKEN` or `../.hf_token`
- a Weights & Biases API key, either in `WANDB_API_KEY` or `../.wandb_api_key`

Useful environment variables used throughout the repo include:

- `HF_TOKEN`
- `WANDB_API_KEY`
- `VISIBLE_GPUS`
- `VLLM_TENSOR_PARALLEL_SIZE`
- `VLLM_GPU_MEM_UTIL`
- `MISTRAL_BATCH_SIZE`
- `MISTRAL_MAX_MODEL_LEN`

### Example Training Commands

Run the provided training scripts from inside `LLaMA-Factory/` so the relative token paths resolve correctly:

```bash
cd LLaMA-Factory
bash run_llama_sft.sh
```

or

```bash
cd LLaMA-Factory
bash run_qwen_sft.sh
```

By default, checkpoints are written under `LLaMA-Factory/saves/`.

## Evaluation

Each task directory contains task-specific prompt construction, data loading, constrained output parsing, and accuracy computation.

### Task Map

| Directory | Dimension | Input text | Prediction target |
| --- | --- | --- | --- |
| `citation/` | Citation | title + abstract | which paper has more citations |
| `cs_best_paper/` | Award | title + abstract | which paper won the best paper award |
| `MDPI_best_paper/` | Award | title + abstract | which paper won the MDPI Best Paper Award |
| `nobel/` | Award | title + abstract | which paper is the Nobel-winning paper |
| `patent/` | Patent | title + abstract | which paper could be cited in more patents |
| `media/` | Media | title + abstract | which paper could get more media mentions |
| `github/` | Code | README | which repository has more GitHub stars |
| `huggingface_dataset/` | Dataset | dataset card | which dataset has more downloads |
| `huggingface_model/` | Model | model card | which model has more downloads |

### Base Model Evaluation

The `run_vllm_*.py` scripts evaluate instruction-tuned base models. The model lists are declared directly in each script and currently include combinations of:

- Meta Llama 3 / 3.1 / 3.2
- Qwen 2.5 / Qwen 3
- Ministral 3
- NVIDIA Nemotron

These scripts:

1. fetch or load the benchmark pair files
2. build a dimension-specific prompt
3. run batched inference with vLLM
4. parse the answer into a binary decision
5. compute pairwise accuracy and save predictions

### Fine-Tuned Model Evaluation

The `run_sft_*.py` scripts evaluate local checkpoints produced by SciImpact fine-tuning. By default they reference checkpoints such as:

- `../LLaMA-Factory/saves/Llama3.2-3B_Paper_Impact_SFT`
- `../LLaMA-Factory/saves/Qwen3-4B_Paper_Impact_SFT`
- `../LLaMA-Factory/saves/Qwen3-4B_Paper_Impact_SFT_1ep`

These scripts are useful for reproducing the main finding in the paper that task-specific SFT can substantially strengthen compact open-weight models on scientific impact prediction.

### Running an Evaluation Script

Run evaluation scripts from the corresponding task directory so their relative paths resolve correctly. For example:

```bash
cd citation
python run_vllm_citations_suite.py
```

or

```bash
cd github
python run_sft_github_suite.py
```

Most scripts require a Hugging Face token and will exit with an error if `HF_TOKEN` is missing.

Result JSON files are typically written in the current task directory with filenames declared in each script's `MODELS` list.

## Implementation Notes

- Paper-based tasks usually truncate abstracts to keep prompt length bounded.
- GitHub README evaluation truncates long READMEs to a fixed budget.
- Hugging Face dataset/model tasks truncate long cards before inference.
- Several scripts use constrained output spaces to stabilize decoding and scoring.
- GPU defaults differ by task; override them with environment variables when needed.

## Citation

If you use this repository or the SciImpact benchmark, please cite:

```bibtex
@misc{zhu2026sciimpactmultidimensionalmultifieldbenchmark,
  title={SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction},
  author={Hangxiao Zhu and Yuyu Zhang and Ping Nie and Yu Zhang},
  year={2026},
  eprint={2604.17141},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.17141}
}
```
