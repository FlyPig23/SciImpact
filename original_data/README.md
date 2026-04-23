# Original Data

This directory stages selected "more raw" source files from the Hugging Face dataset repo `FlyPig23/paper-impact-data` into the main `SciImpact` repository.

The goal is to keep the GitHub repo closer to the actual benchmark construction inputs, while still avoiding obvious non-release artifacts such as logs, debug outputs, giant negative pools, and full external snapshots.

## What Is Included

The sync script in this folder pulls these groups:

- field-organized `MAPLE_sampled/*/papers.json`
- field-organized citation pairs from `construct_citations_dataset/sampled_all_field_pairs_433/`
- field-organized media pairs from `sciscinet/combined_pair_sampled/`
- field-organized patent pairs from `sciscinet/patents_pair_sampled/`
- CS best-paper `_new` splits
- MDPI best-paper source/eval files without the giant negative pool
- Nobel `_new` sampled pair files
- GitHub pair data plus `readme_data.json`
- Hugging Face dataset-card pair files
- Hugging Face model-card pair files

Large data files under `original_data/` are tracked with Git LFS.

## What Is Deliberately Excluded

Even with Git LFS enabled, these are still excluded by default:

- `openalex-snapshot/*`
- `logs/*`
- `meh/*`
- `MDPI_best_paper/mdpi_negatives_enriched_with_source.json`
- full raw `huggingface_dataset/Dataset_Card/*`
- full raw Hugging Face model-card corpora
- other intermediate/debug files that are not needed to run the released benchmark tasks

`sampled_maple_openalex_matches/*` is a reasonable LFS candidate for a later follow-up, but it is not required by the current released evaluation scripts, so it is left out of the default sync.

## Layout

```text
original_data/
  by_field/
    <field>/
      maple_sampled/
      citation/
      media/
      patent/
  awards/
    cs_best_paper/
    mdpi_best_paper/
    nobel/
  non_field/
    github/
    huggingface_dataset_pairs/
    huggingface_model_pairs_limit_2/
```

## Computer-Science Naming

Different upstream sources use different aliases for computer science:

- `computer_science`
- `csrankings`
- `cscombined`

This directory keeps `csrankings` as its own field bucket for MAPLE/citation data. For downstream pair files:

- media uses `cscombined` and is normalized into `by_field/computer_science/media/`
- patent uses `csrankings` and stays under `by_field/csrankings/patent/`

## Sync

Run:

```bash
python original_data/sync_from_hf.py
```

Use `--dry-run` to inspect the transfer plan first, or `--force` to re-download files that already exist.
