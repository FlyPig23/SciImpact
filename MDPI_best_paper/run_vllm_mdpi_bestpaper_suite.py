"""
Run MDPI Best Paper prediction evals for a list of models with vLLM, one model at a time.
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
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

HF_REPO_ID = "/paper-impact-data"
NEG_FILE = "MDPI_best_paper/mdpi_test_new.json"
POS_FILE = "MDPI_best_paper/mdpi_awards_enriched_with_source.json"

MAX_ABSTRACT_LEN = 1800
BATCH_SIZE = 16
MISTRAL_BATCH_SIZE = int(os.environ.get("MISTRAL_BATCH_SIZE", "4"))
MISTRAL_MAX_BATCHED_TOKENS = os.environ.get("MISTRAL_MAX_BATCHED_TOKENS")
MISTRAL_MAX_MODEL_LEN = os.environ.get("MISTRAL_MAX_MODEL_LEN")
DEFAULT_VISIBLE_GPUS = os.environ.get("VISIBLE_GPUS", "0,1,6,7")
TP_DEFAULT = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "4"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.80"))
NEMOTRON_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "nothink.jinja"

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
                    "Paper A won the MDPI Best Paper Award",
                    "Paper B won the MDPI Best Paper Award",
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
    tensor_parallel_size: int | None = None
    llm_kwargs: dict[str, Any] | None = None
    env_vars: dict[str, str] | None = None


# Add/remove entries here to change which models run.
MODELS: list[ModelSpec] = [
    ModelSpec("meta-llama/Meta-Llama-3-8B-Instruct", "llama", "llama3_vllm_mdpi_bestpaper_results.json"),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama", "llama31_vllm_mdpi_bestpaper_results.json"),
    ModelSpec("meta-llama/Llama-3.2-3B-Instruct", "llama", "llama32_vllm_mdpi_bestpaper_results.json"),
    ModelSpec("Qwen/Qwen3-4B-Instruct-2507", "qwen", "qwen3_vllm_mdpi_bestpaper_results.json"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", "qwen", "qwen25_7b_vllm_mdpi_bestpaper_results.json"),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", "qwen", "qwen25_14b_vllm_mdpi_bestpaper_results.json"),
    ModelSpec(
        "mistralai/Ministral-3-3B-Instruct-2512",
        "mistral",
        "ministral3_3b_vllm_mdpi_bestpaper_results.json",
        llm_kwargs={"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"},
    ),
    ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "nemotron",
        "nemotron3_vllm_mdpi_bestpaper_results.json",
        llm_kwargs={
            "kv_cache_dtype": "fp8",
        },
        env_vars={
            "VLLM_USE_FLASHINFER_MOE_FP8": "1",
            "VLLM_FLASHINFER_MOE_BACKEND": "throughput",
        },
    ),
]


def get_hf_token() -> str | None:
    # ../.hf_token is in path
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
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "... [truncated]"


def format_paper(entry: Dict) -> str:
    title = entry.get("title") or entry.get("Title") or "Untitled"
    abstract = (
        entry.get("abstract")
        or entry.get("maple_abstract")
        or entry.get("ss_abstract")
        or entry.get("scraped_abstract")
        or ""
    )
    abstract = truncate(abstract, MAX_ABSTRACT_LEN)
    return f"Title: {title}\nAbstract: {abstract}"


def build_prompt(family: str, first_paper: str, second_paper: str, tokenizer=None) -> str:
    system_prompt = (
            "You are an impartial judge deciding which of two MDPI papers won the MDPI Best Paper Award. "
            "Your reply MUST be exactly one sentence and must be one of these two options:\n"
            "- Paper A won the MDPI Best Paper Award\n"
            "- Paper B won the MDPI Best Paper Award\n"
            "You are not allowed to output anything else—no explanations, no extra words."
    )
    user_prompt = (
            f"Paper A:\n{first_paper}\n\n"
            f"Paper B:\n{second_paper}\n\n"
            "Based on the information above, which paper won the MDPI Best Paper Award?\n\n"
            "Reply with exactly one sentence following the system instruction."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if tokenizer is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if family == "llama":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    if family == "mistral":
        return (
            "<s>[INST] <<SYS>>\n"
            f"{system_prompt}\n"
            "<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )
    if family == "qwen":
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    if family == "nemotron":
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think></think>"
        )

    return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"


def parse_model_response(response: str | None) -> str | None:
    if not response:
        return None
    clean_resp = response.strip()
    low = clean_resp.lower()
    if re.search(r"Paper A won", low, re.IGNORECASE):
        return "first"
    if re.search(r"Paper B won", low, re.IGNORECASE):
        return "second"
    if re.search(r"paper a won the best paper award", low, re.IGNORECASE):
        return "first"
    if re.search(r"paper b won the best paper award", low, re.IGNORECASE):
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


def load_json(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def process_batch(llm: LLM, batch: list[Dict], results: list[Dict], sampling_params: SamplingParams, family: str) -> int:
    prompts = [item["prompt"] for item in batch]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    used = 0
    for item, output in zip(batch, outputs):
        text = output.outputs[0].text.strip() if output.outputs else None
        if text is not None:
            used += 1
        pred = parse_model_response(text)
        pred_positive_first = None
        if pred == "first":
            pred_positive_first = True
        elif pred == "second":
            pred_positive_first = False
        is_correct = pred_positive_first == item["positive_first"] if pred_positive_first is not None else False
        results.append(
            {
                "positive_id": item["positive_id"],
                "negative_id": item["negative_id"],
                "positive_first": item["positive_first"],
                "predicted": pred,
                "is_correct": is_correct,
                "raw_response": text,
            }
        )
    return used


def run_model(spec: ModelSpec, hf_token: str, output_dir: Path) -> dict:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_VISIBLE_GPUS

    start_time = time.time()

    env_backup: dict[str, str | None] = {}
    llm: LLM | None = None
    tokenizer = None
    try:
        if spec.env_vars:
            for key, value in spec.env_vars.items():
                env_backup[key] = os.environ.get(key)
                os.environ[key] = value

        print(f"\n=== Running {spec.name} ({spec.family}) ===")
        llm_args: dict[str, Any] = {
            "model": spec.name,
            "trust_remote_code": True,
            "dtype": "auto",
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "tensor_parallel_size": spec.tensor_parallel_size or TP_DEFAULT,
            "enforce_eager": True,
        }
        if spec.family == "mistral":
            if MISTRAL_MAX_BATCHED_TOKENS:
                llm_args["max_num_batched_tokens"] = int(MISTRAL_MAX_BATCHED_TOKENS)
            if MISTRAL_MAX_MODEL_LEN:
                llm_args["max_model_len"] = int(MISTRAL_MAX_MODEL_LEN)
        if spec.llm_kwargs:
            llm_args.update(spec.llm_kwargs)
        llm = LLM(**llm_args)
        if spec.family == "nemotron":
            tokenizer = AutoTokenizer.from_pretrained(spec.name, trust_remote_code=True)
            if not NEMOTRON_TEMPLATE_PATH.exists():
                raise FileNotFoundError(f"Missing Nemotron chat template: {NEMOTRON_TEMPLATE_PATH}")
            tokenizer.chat_template = NEMOTRON_TEMPLATE_PATH.read_text()

        sp = get_sampling_params(spec.family)

        # Load data
        neg_path = download_from_hf(NEG_FILE, token=hf_token)
        pos_path = download_from_hf(POS_FILE, token=hf_token)
        negatives = load_json(neg_path)
        positives = load_json(pos_path)
        pos_lookup = {p["openalex_id"]: p for p in positives}

        missing = {
            n["matched_positive_openalex_id"]
            for n in negatives
            if n["matched_positive_openalex_id"] not in pos_lookup
        }
        if missing:
            raise SystemExit(f"Missing {len(missing)} positive papers referenced by test set.")

        batch: list[Dict] = []
        results: list[Dict] = []
        model_calls = 0

        for idx, sample in enumerate(tqdm(negatives, desc=f"MDPI negatives ({spec.family})")):
            pos_entry = pos_lookup.get(sample["matched_positive_openalex_id"])
            if not pos_entry:
                continue

            pos_abs = (
                pos_entry.get("abstract")
                or pos_entry.get("maple_abstract")
                or pos_entry.get("ss_abstract")
                or pos_entry.get("scraped_abstract")
                or ""
            ).strip()
            neg_abs = (
                sample.get("abstract")
                or sample.get("maple_abstract")
                or sample.get("ss_abstract")
                or sample.get("scraped_abstract")
                or ""
            ).strip()
            if not pos_abs or not neg_abs:
                continue

            positive_first = idx % 2 == 0
            first_paper = format_paper(pos_entry if positive_first else sample)
            second_paper = format_paper(sample if positive_first else pos_entry)
            prompt = build_prompt(spec.family, first_paper, second_paper, tokenizer=tokenizer)

            batch.append(
                {
                    "prompt": prompt,
                    "positive_first": positive_first,
                    "positive_id": pos_entry["openalex_id"],
                    "negative_id": sample["openalex_id"],
                }
            )

            if len(batch) >= (MISTRAL_BATCH_SIZE if spec.family == "mistral" else BATCH_SIZE):
                model_calls += process_batch(llm, batch, results, sp, spec.family)
                batch = []

        if batch:
            model_calls += process_batch(llm, batch, results, sp, spec.family)

        correct_predictions = sum(r["is_correct"] for r in results)
        accuracy = correct_predictions / len(results) if results else 0
        elapsed = time.time() - start_time

        output_payload = {
            "model": spec.name,
            "total_pairs": len(negatives),
            "evaluated": len(results),
            "correct": correct_predictions,
            "accuracy": accuracy,
            "baseline_accuracy": 0.49,
            "elapsed_seconds": elapsed,
            "results": results,
        }

        out_path = output_dir / spec.output_file
        out_path.parent.mkdir(parents=True, exist_ok=True)

        return output_payload, out_path
    finally:
        try:
            if llm is not None:
                llm.__del__()  # type: ignore[attr-defined]
        except Exception:
            pass
        for key, old_val in env_backup.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


def main():
    hf_token = get_hf_token()
    if hf_token is None:
        raise SystemExit("Missing HF token. Run `huggingface-cli login` or set HF_TOKEN before running.")

    output_dir = Path(__file__).resolve().parent
    summaries = []

    for spec in MODELS:
        start_time = time.time()
        payload, out_path = run_model(spec, hf_token, output_dir)
        payload["elapsed_seconds"] = payload.get("elapsed_seconds") or 0
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
