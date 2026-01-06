"""
Run Nobel prize paper classification evals for local fine-tuned models with vLLM.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import HfFolder, hf_hub_download
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

HF_REPO_ID = "/paper-impact-data"
PAIR_FILES = [
    "nobel/sampled_nobel_pairs/Chemistry_test_new.json",
    "nobel/sampled_nobel_pairs/Medicine_test_new.json",
    "nobel/sampled_nobel_pairs/Physics_test_new.json",
]

MAX_ABSTRACT_LEN = 1800
BATCH_SIZE = 16
DEFAULT_VISIBLE_GPUS = os.environ.get("VISIBLE_GPUS", "0,1,6,7")
TP_DEFAULT = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "4"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90"))

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.9,
    max_tokens=32,
)


def get_sampling_params(family: str) -> SamplingParams:
    if family == "qwen":
        return SamplingParams(
            temperature=0.0,
            top_p=0.9,
            max_tokens=32,
            structured_outputs=StructuredOutputsParams(
                choice=[
                    "Paper A is the Nobel prize-winning paper.",
                    "Paper B is the Nobel prize-winning paper.",
                ],
                disable_fallback=True,
            ),
        )
    return sampling_params


@dataclass
class ModelSpec:
    name: str
    family: str  # "llama", "qwen"
    output_file: str
    llm_kwargs: dict[str, Any] | None = None


MODELS: list[ModelSpec] = [
    ModelSpec(
        "../LLaMA-Factory/saves/Llama3.2-3B_Paper_Impact_SFT",
        "llama",
        "llama32_sft_vllm_nobel_new_results.json",
    ),
    ModelSpec(
        "../LLaMA-Factory/saves/Qwen3-4B_Paper_Impact_SFT",
        "qwen",
        "qwen3_sft_vllm_nobel_new_results.json",
    ),
    ModelSpec(
        "../LLaMA-Factory/saves/Qwen3-4B_Paper_Impact_SFT_1ep",
        "qwen",
        "qwen3_sft_vllm_nobel_1ep_results.json",
    ),
]


def get_hf_token() -> str | None:
    local_file = Path("../.hf_token")
    if local_file.exists():
        return local_file.read_text().strip() or None
    token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
    if token:
        return token
    return None


def download_from_hf(file_path: str, token: str | None) -> Path:
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=file_path,
        repo_type="dataset",
        token=token,
    )
    return Path(local_path)


def truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "... [truncated]"


def format_paper(paper: dict) -> str:
    title = paper.get("Title") or paper.get("title") or "Untitled"
    abstract = truncate(
        paper.get("abstract") or paper.get("maple_abstract") or paper.get("ss_abstract") or paper.get("scraped_abstract") or "",
        MAX_ABSTRACT_LEN,
    )
    return f"Title: {title}\nAbstract:\n{abstract}\n"


def build_prompt(family: str, paper_a: str, paper_b: str) -> str:
    system_prompt = (
        "You are an impartial judge deciding which of two research papers is the Nobel prize-winning paper. "
        "Your reply MUST be exactly one sentence and must be one of these two options:\n"
        "- Paper A is the Nobel prize-winning paper.\n"
        "- Paper B is the Nobel prize-winning paper.\n"
        "You are not allowed to output anything else—no explanations, no extra words."
    )
    user_prompt = (
        f"Paper A:\n{paper_a}\n"
        f"Paper B:\n{paper_b}\n"
        "Based on the information above, which paper is the Nobel prize-winning paper?\n\n"
        "Answer with exactly one of the allowed sentences. No explanation. No extra text."
    )

    if family == "llama":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    # Qwen-friendly prompt
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_model_response(response: str | None) -> str | None:
    if not response:
        return None
    clean_resp = response.strip()
    low = clean_resp.lower()
    if re.search(r"paper a is the nobel prize-winning paper", low, re.IGNORECASE):
        return "first"
    if re.search(r"paper b is the nobel prize-winning paper", low, re.IGNORECASE):
        return "second"
    if "paper a" in low and "paper b" not in low:
        return "first"
    if "paper b" in low and "paper a" not in low:
        return "second"
    if "first" in low and "second" not in low:
        return "first"
    if "second" in low and "first" not in low:
        return "second"
    return None


def strip_think_tags(text: str | None) -> str | None:
    if text is None:
        return None
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def load_pairs(file_path: Path) -> List[dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def get_field_name(file_path: str) -> str:
    stem = Path(file_path).stem
    suffix = "_test"
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def process_batch(llm: LLM, batch: list[Dict], results: list[Dict], sampling_params: SamplingParams) -> int:
    prompts = [item["prompt"] for item in batch]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    used = 0
    for item, output in zip(batch, outputs):
        text = output.outputs[0].text.strip() if output.outputs else None
        text = strip_think_tags(text)
        if text is not None:
            used += 1
        pred = parse_model_response(text)
        pred_first_prize = None
        if pred == "first":
            pred_first_prize = True
        elif pred == "second":
            pred_first_prize = False
        is_correct = pred_first_prize == item["first_is_prize"] if pred_first_prize is not None else False
        results.append(
            {
                "field": item["field"],
                "file": item["file"],
                "first_is_prize": item["first_is_prize"],
                "predicted_prize_is_first": pred_first_prize,
                "is_correct": is_correct,
                "raw_response": text,
                "prize_title": item["prize_title"],
                "non_prize_title": item["non_prize_title"],
            }
        )
    return used


def run_model(spec: ModelSpec, hf_token: str, output_dir: Path) -> dict:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_VISIBLE_GPUS)

    llm_args: dict[str, Any] = {
        "model": spec.name,
        "trust_remote_code": True,
        "dtype": "auto",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "tensor_parallel_size": TP_DEFAULT,
        "enforce_eager": True,
    }
    if spec.llm_kwargs:
        llm_args.update(spec.llm_kwargs)

    print(f"\n=== Running {spec.name} ({spec.family}) ===")
    llm = LLM(**llm_args)
    sp = get_sampling_params(spec.family)

    all_pairs: list[Dict] = []
    pairs_count_by_field: dict[str, int] = {}
    field_order: List[str] = []
    for file_path in PAIR_FILES:
        local_path = download_from_hf(file_path, token=hf_token)
        pairs = load_pairs(local_path)
        field_name = get_field_name(file_path)
        field_order.append(field_name)
        pairs_count_by_field[field_name] = len(pairs)
        print(f"Loaded {len(pairs)} pairs from {file_path}")
        for i, sample in enumerate(pairs):
            prize_paper = sample["prize_paper"]
            non_prize_paper = sample["non_prize_paper"]
            first_is_prize = i % 2 == 0
            paper_a = format_paper(prize_paper if first_is_prize else non_prize_paper)
            paper_b = format_paper(non_prize_paper if first_is_prize else prize_paper)
            prompt = build_prompt(spec.family, paper_a, paper_b)
            all_pairs.append(
                {
                    "prompt": prompt,
                    "first_is_prize": first_is_prize,
                    "file": file_path,
                    "field": field_name,
                    "prize_title": prize_paper.get("Title"),
                    "non_prize_title": non_prize_paper.get("Title"),
                }
            )

    batch: list[Dict] = []
    results: list[Dict] = []
    model_calls = 0

    for item in tqdm(all_pairs, desc="Nobel pairs"):
        batch.append(item)
        if len(batch) >= BATCH_SIZE:
            model_calls += process_batch(llm, batch, results, sp)
            batch = []
    if batch:
        model_calls += process_batch(llm, batch, results, sp)

    evaluated = len(results)
    correct_predictions = sum(r["is_correct"] for r in results)
    accuracy = correct_predictions / evaluated if evaluated else 0

    evaluated_by_field: dict[str, int] = {}
    correct_by_field: dict[str, int] = {}
    for record in results:
        field = record["field"]
        evaluated_by_field[field] = evaluated_by_field.get(field, 0) + 1
        if record["is_correct"]:
            correct_by_field[field] = correct_by_field.get(field, 0) + 1

    field_stats: list[Dict] = []
    for field in field_order:
        pairs_count = pairs_count_by_field.get(field, 0)
        evaluated_count = evaluated_by_field.get(field, 0)
        correct_count = correct_by_field.get(field, 0)
        field_accuracy = correct_count / evaluated_count if evaluated_count else 0
        field_stats.append(
            {
                "field": field,
                "pairs_count": pairs_count,
                "evaluated_count": evaluated_count,
                "correct_predictions": correct_count,
                "accuracy": field_accuracy,
            }
        )

    out_path = output_dir / spec.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "model": spec.name,
        "total_pairs": len(all_pairs),
        "evaluated": evaluated,
        "correct": correct_predictions,
        "accuracy": accuracy,
        "elapsed_seconds": None,  # filled below
        "results": results,
        "field_stats": field_stats,
        "output_path": out_path,
        "model_calls": model_calls,
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensor_parallel_size": TP_DEFAULT,
    }


def main():
    hf_token = get_hf_token()
    if hf_token is None:
        raise SystemExit("Missing HF token. Run `huggingface-cli login` or set HF_TOKEN before running.")

    output_dir = Path(__file__).resolve().parent
    summaries = []

    for spec in MODELS:
        start = time.time()
        payload = run_model(spec, hf_token, output_dir)
        payload["elapsed_seconds"] = time.time() - start
        out_path = payload.pop("output_path")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(
            f"Saved {spec.name} results to {out_path} | "
            f"acc={payload['accuracy']:.2%} | "
            f"elapsed={payload['elapsed_seconds']:.1f}s | "
            f"samples={payload['evaluated']}"
        )
        summaries.append(
            {
                "model": spec.name,
                "output": str(out_path),
                "accuracy": payload["accuracy"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "evaluated": payload["evaluated"],
            }
        )

    print("\n=== Summary ===")
    for s in summaries:
        print(
            f"{s['model']}: acc={s['accuracy']:.2%}, eval={s['evaluated']}, "
            f"time={s['elapsed_seconds']:.1f}s -> {s['output']}"
        )


if __name__ == "__main__":
    main()
