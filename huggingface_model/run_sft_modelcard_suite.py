"""
Run Hugging Face model card download-prediction evals for local fine-tuned models with vLLM.

- Uses visible GPUs (set VISIBLE_GPUS or CUDA_VISIBLE_DEVICES).
- MODELS lists the local checkpoints to evaluate.
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

# --- DATA SOURCES ---
HF_REPO_ID = "FlyPig23/paper-impact-data"
PAIR_FILE = "huggingface_model/modelcard_pairs_limit_2/modelcard_test_pairs.jsonl"

# --- EVAL CONFIG ---
MAX_CARD_LEN = 4000
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
                    "Model A has more downloads.",
                    "Model B has more downloads.",
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
        "llama32_sft_vllm_modelcard_results.json",
    ),
    ModelSpec(
        "../LLaMA-Factory/saves/Qwen3-4B_Paper_Impact_SFT",
        "qwen",
        "qwen3_sft_vllm_modelcard_results.json",
    ),
    ModelSpec(
        "../LLaMA-Factory/saves/Qwen3-4B_Paper_Impact_SFT_1ep",
        "qwen",
        "qwen3_sft_vllm_modelcard_1ep_results.json",
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


def load_pairs(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(family: str, card_a: str, card_b: str) -> str:
    system_prompt = (
        "You are an impartial judge deciding which of two Hugging Face models has more downloads. "
        "Your reply MUST be exactly one sentence and must be one of these two options:\n"
        "- Model A has more downloads.\n"
        "- Model B has more downloads.\n"
        "You are not allowed to output anything else—no explanations, no extra words."
    )
    user_prompt = (
        f"Model A:\n{card_a}\n"
        f"Model B:\n{card_b}\n"
        "Based on the information above, which model has more downloads?\n\n"
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
    if re.search(r"model a .*downloads", low):
        return "first"
    if re.search(r"model b .*downloads", low):
        return "second"
    if "model a" in low and "model b" not in low:
        return "first"
    if "model b" in low and "model a" not in low:
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
        pred_a_more = None
        if pred == "first":
            pred_a_more = item["order_a_first"]
        elif pred == "second":
            pred_a_more = not item["order_a_first"]
        actual_a_more = item["downloads_a"] > item["downloads_b"]
        is_correct = pred_a_more == actual_a_more if pred_a_more is not None else False
        results.append(
            {
                "model_a": item["model_a"],
                "model_b": item["model_b"],
                "downloads_a": item["downloads_a"],
                "downloads_b": item["downloads_b"],
                "order_a_first": item["order_a_first"],
                "predicted": pred,
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

    pair_path = download_from_hf(PAIR_FILE, token=hf_token)
    samples = load_pairs(pair_path)
    print(f"Loaded {len(samples)} pairs")

    batch: list[Dict] = []
    results: list[Dict] = []
    model_calls = 0

    for i, sample in enumerate(tqdm(samples, desc=f"modelcard pairs ({spec.family})")):
        card_a = truncate(sample.get("card_a", ""), MAX_CARD_LEN)
        card_b = truncate(sample.get("card_b", ""), MAX_CARD_LEN)
        downloads_a = sample.get("downloads_a", 0)
        downloads_b = sample.get("downloads_b", 0)

        order_a_first = i % 2 == 0
        prompt = build_prompt(
            spec.family,
            card_a if order_a_first else card_b,
            card_b if order_a_first else card_a,
        )

        batch.append(
            {
                "prompt": prompt,
                "order_a_first": order_a_first,
                "model_a": sample.get("model_a"),
                "model_b": sample.get("model_b"),
                "downloads_a": downloads_a,
                "downloads_b": downloads_b,
            }
        )

        if len(batch) >= BATCH_SIZE:
            model_calls += process_batch(llm, batch, results, sp)
            batch = []

    if batch:
        model_calls += process_batch(llm, batch, results, sp)

    evaluated = len(results)
    correct_predictions = sum(r["is_correct"] for r in results)
    accuracy = correct_predictions / evaluated if evaluated else 0

    out_path = output_dir / spec.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "model": spec.name,
        "total_pairs": len(samples),
        "evaluated": evaluated,
        "correct": correct_predictions,
        "accuracy": accuracy,
        "elapsed_seconds": None,  # filled below
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
