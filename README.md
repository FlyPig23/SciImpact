# paper_impact_code

This repo contains data preparation and evaluation code for multiple "paper impact" style tasks (best-paper selection, citations, awards, media mentions, patents, model/dataset cards, and related benchmarks).

## LLaMA-Factory usage
- The LLaMA-Factory codebase is vendored in `LLaMA-Factory/` and is used for SFT and evaluation.
- SFT datasets are stored in `sft_data/`.

## Test suites by task
- Best paper (`best_paper/`)
- MDPI best paper (`MDPI_best_paper/`)
- Nobel prizes (`nobel/`)
- Patents (`patent/`)
- Media (`media/`)
- Citations (`citation/`)
- Code (`github/`)
- Dataset cards (`huggingface_dataset/`) 
- Model cards (`huggingface_model/`)
