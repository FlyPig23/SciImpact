#!/usr/bin/env python3
"""Download the selected original-data files from the HF dataset repo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HF_BASE = "https://huggingface.co/datasets/FlyPig23/paper-impact-data/resolve/main"
ROOT = Path(__file__).resolve().parent


MAPLE_FIELDS = {
    "Art": "art",
    "Biology": "biology",
    "Business": "business",
    "Chemistry": "chemistry",
    "Computer_Science": "computer_science",
    "CSRankings": "csrankings",
    "Economics": "economics",
    "Engineering": "engineering",
    "Environmental_Science": "environmental_science",
    "Geography": "geography",
    "Geology": "geology",
    "History": "history",
    "Materials_Science": "materials_science",
    "Mathematics": "mathematics",
    "Medicine": "medicine",
    "Philosophy": "philosophy",
    "Physics": "physics",
    "Political_Science": "political_science",
    "Psychology": "psychology",
    "Sociology": "sociology",
}

CITATION_FIELDS = [
    "art",
    "biology",
    "business",
    "chemistry",
    "computer_science",
    "csrankings",
    "economics",
    "engineering",
    "environmental_science",
    "geography",
    "geology",
    "history",
    "materials_science",
    "mathematics",
    "medicine",
    "philosophy",
    "physics",
    "political_science",
    "psychology",
    "sociology",
]

MEDIA_FIELDS = {
    "art": "art",
    "biology": "biology",
    "business": "business",
    "chemistry": "chemistry",
    "cscombined": "computer_science",
    "economics": "economics",
    "engineering": "engineering",
    "environmentalscience": "environmental_science",
    "geography": "geography",
    "geology": "geology",
    "history": "history",
    "materialsscience": "materials_science",
    "mathematics": "mathematics",
    "medicine": "medicine",
    "philosophy": "philosophy",
    "physics": "physics",
    "politicalscience": "political_science",
    "psychology": "psychology",
    "sociology": "sociology",
}

PATENT_FIELDS = {
    "art": "art",
    "biology": "biology",
    "business": "business",
    "chemistry": "chemistry",
    "csrankings": "csrankings",
    "economics": "economics",
    "engineering": "engineering",
    "environmentalscience": "environmental_science",
    "geography": "geography",
    "geology": "geology",
    "history": "history",
    "materialsscience": "materials_science",
    "mathematics": "mathematics",
    "medicine": "medicine",
    "philosophy": "philosophy",
    "physics": "physics",
    "politicalscience": "political_science",
    "psychology": "psychology",
    "sociology": "sociology",
}

NOBEL_FIELDS = ["Chemistry", "Medicine", "Physics"]
SPLITS = ["train", "val", "test"]


def build_manifest() -> list[tuple[str, Path]]:
    manifest: list[tuple[str, Path]] = []

    for source_field, target_field in MAPLE_FIELDS.items():
        manifest.append(
            (
                f"MAPLE_sampled/{source_field}/papers.json",
                ROOT / "by_field" / target_field / "maple_sampled" / "papers.json",
            )
        )

    for field in CITATION_FIELDS:
        for split in SPLITS:
            manifest.append(
                (
                    f"construct_citations_dataset/sampled_all_field_pairs_433/{field}_{split}_pairs.jsonl",
                    ROOT / "by_field" / field / "citation" / f"{split}_pairs.jsonl",
                )
            )

    for source_field, target_field in MEDIA_FIELDS.items():
        for split in SPLITS:
            manifest.append(
                (
                    f"sciscinet/combined_pair_sampled/{source_field}_{split}_pairs.json",
                    ROOT / "by_field" / target_field / "media" / f"{split}_pairs.json",
                )
            )

    for source_field, target_field in PATENT_FIELDS.items():
        for split in SPLITS:
            manifest.append(
                (
                    f"sciscinet/patents_pair_sampled/{source_field}_{split}_pairs.json",
                    ROOT / "by_field" / target_field / "patent" / f"{split}_pairs.json",
                )
            )

    for split in SPLITS:
        manifest.append(
            (
                f"best_paper/split_dataset/{split}_pairs_new.json",
                ROOT / "awards" / "cs_best_paper" / f"{split}_pairs_new.json",
            )
        )

    manifest.append(
        (
            "MDPI_best_paper/mdpi_awards_enriched_with_source.json",
            ROOT / "awards" / "mdpi_best_paper" / "mdpi_awards_enriched_with_source.json",
        )
    )
    for split in SPLITS:
        manifest.append(
            (
                f"MDPI_best_paper/mdpi_{split}_new.json",
                ROOT / "awards" / "mdpi_best_paper" / f"mdpi_{split}_new.json",
            )
        )

    for field in NOBEL_FIELDS:
        target_field = field.lower()
        for split in SPLITS:
            manifest.append(
                (
                    f"nobel/sampled_nobel_pairs/{field}_{split}_new.json",
                    ROOT / "awards" / "nobel" / target_field / f"{split}_pairs_new.json",
                )
            )

    manifest.append(
        (
            "github_data/readme_data.json",
            ROOT / "non_field" / "github" / "readme_data.json",
        )
    )
    for split in SPLITS:
        manifest.append(
            (
                f"github_data/github_pairs_by_repo/{split}_pairs.json",
                ROOT / "non_field" / "github" / "pairs_by_repo" / f"{split}_pairs.json",
            )
        )

    for split in SPLITS:
        manifest.append(
            (
                f"huggingface_dataset/datasetcard_pairs/datasetcard_{split}_pairs.jsonl",
                ROOT / "non_field" / "huggingface_dataset_pairs" / f"{split}_pairs.jsonl",
            )
        )
        manifest.append(
            (
                f"huggingface_model/modelcard_pairs_limit_2/modelcard_{split}_pairs.jsonl",
                ROOT / "non_field" / "huggingface_model_pairs_limit_2" / f"{split}_pairs.jsonl",
            )
        )

    return manifest


def download_file(source_path: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"skip {destination.relative_to(ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{source_path}"
    print(f"get  {source_path}")
    subprocess.run(
        ["curl", "-L", "-o", str(destination), url],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest()
    print(f"planned files: {len(manifest)}")

    if args.dry_run:
        for source_path, destination in manifest:
            print(f"{source_path}\t{destination.relative_to(ROOT)}")
        return 0

    for source_path, destination in manifest:
        download_file(source_path, destination, force=args.force)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
