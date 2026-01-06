"""
Run CS Best Paper prediction evals for a list of models with vLLM, one model at a time.
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
SPLIT_FILE = "best_paper/split_dataset/test_pairs_new.json"
MAX_ABSTRACT_LEN = 1800
BATCH_SIZE = 16
NEMOTRON_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "nothink.jinja"

DEFAULT_VISIBLE_GPUS = os.environ.get("VISIBLE_GPUS", "0,1,6,7")
TP_DEFAULT = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "4"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90"))

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.9,
    max_tokens=192,
)

def get_sampling_params(family: str) -> SamplingParams:
    if family == "qwen":
        return SamplingParams(
            temperature=0.0,
            top_p=0.9,
            max_tokens=64,
            structured_outputs=StructuredOutputsParams(
                choice=[
                    "Paper A won the best paper award.",
                    "Paper B won the best paper award.",
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
    env_vars: dict[str, str] | None = None


# Add/remove entries here to change which models run.
MODELS: list[ModelSpec] = [
    ModelSpec("meta-llama/Meta-Llama-3-8B-Instruct", "llama", "llama3_vllm_bestpaper_results.json"),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama", "llama31_vllm_bestpaper_results.json"),
    ModelSpec("meta-llama/Llama-3.2-3B-Instruct", "llama", "llama32_vllm_bestpaper_results.json"),
    ModelSpec("Qwen/Qwen3-4B-Instruct-2507", "qwen", "qwen3_vllm_bestpaper_results.json"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", "qwen", "qwen25_7b_vllm_bestpaper_results.json"),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", "qwen", "qwen25_14b_vllm_bestpaper_results.json"),
    ModelSpec(
        "mistralai/Ministral-3-3B-Instruct-2512",
        "mistral",
        "ministral3_3b_vllm_bestpaper_results.json",
        {"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"},
    ),
    ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "nemotron",
        "nemotron3_vllm_bestpaper_results.json",
        {
            "kv_cache_dtype": "fp8",
        },
        {
            "VLLM_USE_FLASHINFER_MOE_FP8": "1",
            "VLLM_FLASHINFER_MOE_BACKEND": "throughput",
        },
    )
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
    title = entry.get("title") or entry.get("title_raw") or "Untitled"
    abstract = entry.get("abstract_raw") or entry.get("abstract") or entry.get("scraped_abstract") or ""
    abstract = truncate(abstract, MAX_ABSTRACT_LEN)
    return f"Title: {title}\nAbstract: {abstract}"


def build_prompt(family: str, paper_a_text: str, paper_b_text: str, tokenizer=None) -> str:
    system_prompt = (
            "You are an impartial paper reviewer. Given the titles and abstracts of two papers, "
            "identify which paper won the Best Paper award. Your reply must be either "
            "'Paper A won the best paper award.' or 'Paper B won the best paper award.'"
    )
    user_prompt = (
            f"Paper A:\n{paper_a_text}\n\n"
            f"Paper B:\n{paper_b_text}\n\n"
            "Based on the information above, which paper should win the Best Paper award?\n\n"
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
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if family == "mistral":
        return (
            "<s>[INST] <<SYS>>\n"
            f"{system_prompt}\n"
            "<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )
    if family == "nemotron":
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think></think>"
        )
    if family == "qwen":
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"


def parse_model_response(response: str | None) -> str | None:
    if not response:
        return None
    clean_resp = response.strip()
    low = clean_resp.lower()
    if re.search(r"Paper A won", clean_resp, re.IGNORECASE):
        return "first"
    if re.search(r"Paper B won", clean_resp, re.IGNORECASE):
        return "second"
    if re.search(r"paper a won the best paper award", low, re.IGNORECASE):
        return "first"
    if re.search(r"paper b won the best paper award", low, re.IGNORECASE):
        return "second"
    low = clean_resp.lower()
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
    # Remove <think>...</think> blocks that some models emit.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def load_pairs(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


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


def run_model(spec: ModelSpec, pairs: list[Dict], hf_token: str, output_dir: Path) -> dict:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_VISIBLE_GPUS

    env_backup: dict[str, str | None] = {}
    llm: LLM | None = None
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
            "tensor_parallel_size": TP_DEFAULT,
            "enforce_eager": True,
        }
        if spec.llm_kwargs:
            llm_args.update(spec.llm_kwargs)
        llm = LLM(**llm_args)
        sp = get_sampling_params(spec.family)
        if spec.family == "nemotron":
            tokenizer = AutoTokenizer.from_pretrained(spec.name, trust_remote_code=True)
            if spec.family == "nemotron":
                if not NEMOTRON_TEMPLATE_PATH.exists():
                    raise FileNotFoundError(
                        f"Missing Nemotron chat template: {NEMOTRON_TEMPLATE_PATH}"
                    )
                tokenizer.chat_template = NEMOTRON_TEMPLATE_PATH.read_text()
        else:
            tokenizer = None

        batch: list[Dict] = []
        results: list[Dict] = []
        model_calls = 0

        for idx, pair in enumerate(tqdm(pairs, desc="CS best-paper pairs")):
            pos = pair["positive"]
            neg = pair["negative"]

            pos_abs = (pos.get("abstract_raw") or pos.get("abstract") or pos.get("scraped_abstract") or "").strip()
            neg_abs = (neg.get("abstract_raw") or neg.get("abstract") or neg.get("scraped_abstract") or "").strip()
            if not pos_abs or not neg_abs:
                continue

            positive_first = idx % 2 == 0
            paper_a_text = format_paper(pos if positive_first else neg)
            paper_b_text = format_paper(neg if positive_first else pos)
            prompt = build_prompt(spec.family, paper_a_text, paper_b_text, tokenizer=tokenizer)

            batch.append(
                {
                    "prompt": prompt,
                    "positive_first": positive_first,
                    "positive_id": pos.get("paper"),
                    "negative_id": neg.get("paper"),
                }
            )

            if len(batch) >= BATCH_SIZE:
                model_calls += process_batch(llm, batch, results, sp)
                batch = []

        if batch:
            model_calls += process_batch(llm, batch, results, sp)

        correct_predictions = sum(r["is_correct"] for r in results)
        accuracy = correct_predictions / len(results) if results else 0
        output_payload = {
            "model": spec.name,
            "total_pairs": len(pairs),
            "evaluated": len(results),
            "correct": correct_predictions,
            "accuracy": accuracy,
            "elapsed_seconds": None,  # filled below
            "results": results,
        }

        # Clean up engine to free GPUs before the next model
        try:
            if llm is not None:
                llm.__del__()  # type: ignore[attr-defined]
        except Exception:
            pass

        out_path = output_dir / spec.output_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return output_payload, out_path
    finally:
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
    split_path = download_from_hf(SPLIT_FILE, token=hf_token)
    pairs = load_pairs(split_path)

    summaries = []
    for spec in MODELS:
        start = time.time()
        payload, out_path = run_model(spec, pairs, hf_token, output_dir)
        payload["elapsed_seconds"] = time.time() - start
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
