"""
Run patent-impact pair evals for local fine-tuned models with vLLM.
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
BASE_PATH = "sciscinet/patents_pair_sampled"
FIELDS = [
    "CSRankings",
    "Chemistry",
    "Physics",
    "Medicine",
    "Economics",
    "Art",
    "Biology",
    "Business",
    "Engineering",
    "Environmental_Science",
    "Geography",
    "Geology",
    "History",
    "Materials_Science",
    "Mathematics",
    "Philosophy",
    "Political_Science",
    "Psychology",
    "Sociology",
]

MAX_ABSTRACT_LEN = 1800
BATCH_SIZE = 16
DEFAULT_VISIBLE_GPUS = os.environ.get("VISIBLE_GPUS", "0,1,6,7")
TP_DEFAULT = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "4"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.80"))

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
                    "Paper A could be cited in more patents.",
                    "Paper B could be cited in more patents.",
                ],
                disable_fallback=True,
            ),
        )
    return sampling_params


@dataclass
class ModelSpec:
    name: str
    family: str 
    output_file: str
    llm_kwargs: dict[str, Any] | None = None


MODELS: list[ModelSpec] = [
    ModelSpec(
        "../LLaMA-Factory/saves/Llama3.2-3B_Paper_Impact_SFT",
        "llama",
        "llama32_sft_vllm_patents_results.json",
    ),
    ModelSpec(
        "../LLaMA-Factory/saves/Qwen3-4B_Paper_Impact_SFT",
        "qwen",
        "qwen3_sft_vllm_patents_results.json",
    ),
    ModelSpec(
        "../LLaMA-Factory/saves/Qwen3-4B_Paper_Impact_SFT_1ep",
        "qwen",
        "qwen3_sft_vllm_patents_1ep_results.json",
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


def format_paper(title: str, abstract: str) -> str:
    abstract = truncate(abstract, MAX_ABSTRACT_LEN)
    return f"Title: {title}\nAbstract:\n{abstract}\n"


def field_to_filename(field: str) -> str:
    return re.sub(r"[^a-z0-9]", "", field.lower()) + "_test_pairs.json"


def build_prompt(family: str, paper_a: str, paper_b: str) -> str:
    system_prompt = (
        "You are an impartial judge deciding which of two research papers would be cited in more patents. "
        "Your reply MUST be exactly one sentence and must be one of these two options:\n"
        "- Paper A could be cited in more patents.\n"
        "- Paper B could be cited in more patents.\n"
        "You are not allowed to output anything else—no explanations, no extra words."
    )
    user_prompt = (
        f"Paper A:\n{paper_a}\n"
        f"Paper B:\n{paper_b}\n"
        "Based on the information above, which paper could be cited in more patents?\n\n"
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
    low = response.strip().lower()
    if re.search(r"paper a .*more patents", low):
        return "first"
    if re.search(r"paper b .*more patents", low):
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


def load_pairs(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def process_batch(
    llm: LLM,
    batch: list[Dict],
    results: list[Dict],
    sampling_params: SamplingParams,
    evaluated_by_field: dict[str, int],
    correct_by_field: dict[str, int],
) -> int:
    prompts = [item["prompt"] for item in batch]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    used = 0
    for item, output in zip(batch, outputs):
        text = output.outputs[0].text.strip() if output.outputs else None
        text = strip_think_tags(text)
        if text is not None:
            used += 1
        pred = parse_model_response(text)
        pred_a_more = None
        if pred == "first":
            pred_a_more = item["order_a_first"]
        elif pred == "second":
            pred_a_more = not item["order_a_first"]
        is_correct = pred_a_more == item["paper_a_higher"] if pred_a_more is not None else False
        if pred_a_more is not None:
            fld = item["field"]
            evaluated_by_field[fld] = evaluated_by_field.get(fld, 0) + 1
            if is_correct:
                correct_by_field[fld] = correct_by_field.get(fld, 0) + 1
        results.append(
            {
                "field": item["field"],
                "paper_a_id": item["paper_a_id"],
                "paper_b_id": item["paper_b_id"],
                "order_a_first": item["order_a_first"],
                "paper_a_higher": item["paper_a_higher"],
                "predicted_a_more": pred_a_more,
                "is_correct": is_correct,
                "raw_response": text,
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

    batch: list[Dict] = []
    results: list[Dict] = []
    model_calls = 0
    evaluated_by_field: dict[str, int] = {}
    correct_by_field: dict[str, int] = {}

    for field in FIELDS:
        file_path = f"{BASE_PATH}/{field_to_filename(field)}"
        try:
            local_path = download_from_hf(file_path, token=hf_token)
        except Exception as e:
            print(f"⚠️  Skipping {field}: failed to download {file_path} ({e})")
            continue
        pairs = load_pairs(local_path)
        print(f"{field}: loaded {len(pairs)} pairs")

        for idx, pair in enumerate(tqdm(pairs, desc=f"{field} pairs")):
            title_a = pair.get("paper_a_title") or ""
            title_b = pair.get("paper_b_title") or ""
            abs_a = (pair.get("paper_a_abstract") or "").strip()
            abs_b = (pair.get("paper_b_abstract") or "").strip()
            if not abs_a or not abs_b:
                continue

            paper_a_higher = pair.get("paper_a_count", 0) >= pair.get("paper_b_count", 0)
            order_a_first = idx % 2 == 0
            paper_a_text = format_paper(title_a if order_a_first else title_b, abs_a if order_a_first else abs_b)
            paper_b_text = format_paper(title_b if order_a_first else title_a, abs_b if order_a_first else abs_a)
            prompt = build_prompt(spec.family, paper_a_text, paper_b_text)

            batch.append(
                {
                    "prompt": prompt,
                    "order_a_first": order_a_first,
                    "paper_a_id": pair.get("paper_a_id"),
                    "paper_b_id": pair.get("paper_b_id"),
                    "paper_a_higher": paper_a_higher,
                    "field": field,
                }
            )

            if len(batch) >= BATCH_SIZE:
                model_calls += process_batch(llm, batch, results, sp, evaluated_by_field, correct_by_field)
                batch = []

    if batch:
        model_calls += process_batch(llm, batch, results, sp, evaluated_by_field, correct_by_field)

    evaluated = len(results)
    correct_predictions = sum(r["is_correct"] for r in results)
    accuracy = correct_predictions / evaluated if evaluated else 0
    field_stats: list[Dict] = []
    for f in FIELDS:
        evaluated_count = evaluated_by_field.get(f, 0)
        correct_count = correct_by_field.get(f, 0)
        field_accuracy = correct_count / evaluated_count if evaluated_count else 0
        field_stats.append(
            {
                "field": f,
                "evaluated_count": evaluated_count,
                "correct_predictions": correct_count,
                "accuracy": field_accuracy,
            }
        )

    excluded_fields = {"CSRankings", "CSCombined", "Chemistry", "Physics", "Medicine"}
    filtered_fields = [f for f in field_stats if f["field"] not in excluded_fields]
    other_total_correct = sum(f["correct_predictions"] for f in filtered_fields)
    other_total_evaluated = sum(f["evaluated_count"] for f in filtered_fields)
    other_weighted_accuracy = other_total_correct / other_total_evaluated if other_total_evaluated else 0.0
    unweighted_accs = [f["accuracy"] for f in filtered_fields if f["evaluated_count"] > 0]
    other_unweighted_accuracy = sum(unweighted_accs) / len(unweighted_accs) if unweighted_accs else 0.0

    out_path = output_dir / spec.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "model": spec.name,
        "total_pairs": len(results),
        "evaluated": evaluated,
        "correct": correct_predictions,
        "accuracy": accuracy,
        "elapsed_seconds": None,  # filled below
        "field_stats": field_stats,
        "other_fields": {
            "weighted_accuracy": other_weighted_accuracy,
            "unweighted_accuracy": other_unweighted_accuracy,
            "total_correct": other_total_correct,
            "total_evaluated": other_total_evaluated,
            "excluded_fields": sorted(excluded_fields),
        },
        "results": results,
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
