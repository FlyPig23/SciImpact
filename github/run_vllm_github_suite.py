"""
Run GitHub repo star prediction evals for a list of models with vLLM, one model at a time.
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
PAIR_FILE = "github_data/github_pairs_by_repo/test_pairs.json"
README_FILE = "github_data/readme_data.json"

MAX_README_LEN = 8000
TARGET_README_LEN = 4000
BATCH_SIZE = 16
MISTRAL_BATCH_SIZE = int(os.environ.get("MISTRAL_BATCH_SIZE", "4"))
MISTRAL_MAX_BATCHED_TOKENS = os.environ.get("MISTRAL_MAX_BATCHED_TOKENS")
MISTRAL_MAX_MODEL_LEN = os.environ.get("MISTRAL_MAX_MODEL_LEN")
DEFAULT_VISIBLE_GPUS = os.environ.get("VISIBLE_GPUS", "0,1,6,7")
TP_DEFAULT = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "4"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90"))
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
                    "GitHub repo A has more stars.",
                    "GitHub repo B has more stars.",
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


MODELS: list[ModelSpec] = [
    ModelSpec("meta-llama/Meta-Llama-3-8B-Instruct", "llama", "llama3_vllm_github_results.json"),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama", "llama31_vllm_github_results.json"),
    ModelSpec("meta-llama/Llama-3.2-3B-Instruct", "llama", "llama32_vllm_github_results.json"),
    ModelSpec("Qwen/Qwen3-4B-Instruct-2507", "qwen", "qwen3_vllm_github_results.json"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", "qwen", "qwen25_7b_vllm_github_results.json"),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", "qwen", "qwen25_14b_vllm_github_results.json"),
    ModelSpec(
        "mistralai/Ministral-3-3B-Instruct-2512",
        "mistral",
        "ministral3_3b_vllm_github_results.json",
        llm_kwargs={"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"},
    ),
    ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "nemotron",
        "nemotron3_vllm_github_results.json",
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


def truncate_readme(readme: str) -> str:
    if len(readme) <= TARGET_README_LEN:
        return readme
    if len(readme) <= MAX_README_LEN:
        return readme
    first_len = int(TARGET_README_LEN * 0.7)
    last_len = TARGET_README_LEN - first_len
    separator = "\n\n... [content truncated for length] ...\n\n"
    truncated = readme[:first_len] + separator + readme[-last_len:]
    if len(truncated) > MAX_README_LEN:
        truncated = readme[: MAX_README_LEN - 100] + "\n\n... [content truncated] ..."
    return truncated


def load_readme_lookup(path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    total = 0
    truncated = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            repo_url = data.get("repo_url")
            readme = data.get("readme") or ""
            if not repo_url:
                continue
            total += 1
            if len(readme) > MAX_README_LEN:
                truncated += 1
                readme = truncate_readme(readme)
            lookup[repo_url] = readme
    print(f"Loaded {len(lookup)} READMEs | total={total} | truncated={truncated}")
    return lookup


def build_prompt(family: str, readme_a: str, readme_b: str, tokenizer=None) -> str:
    system_prompt = (
        "You are an impartial judge deciding which of two GitHub repositories has more stars. "
        "Your reply MUST be exactly one sentence and must be one of these two options:\n"
        "- GitHub repo A has more stars.\n"
        "- GitHub repo B has more stars.\n"
        "You are not allowed to output anything else—no explanations, no extra words."
    )
    user_prompt = (
        f"GitHub repo A README:\n{readme_a}\n\n"
        f"GitHub repo B README:\n{readme_b}\n\n"
        "Based on the information above, which repository has more stars?\n\n"
        "Answer with exactly one of the allowed sentences. No explanation. No extra text."
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
    low = response.strip().lower()
    if re.search(r"repo a .*more stars", low):
        return "first"
    if re.search(r"repo b .*more stars", low):
        return "second"
    if "repo a" in low and "repo b" not in low:
        return "first"
    if "repo b" in low and "repo a" not in low:
        return "second"
    if "first" in low and "second" not in low:
        return "first"
    if "second" in low and "first" not in low:
        return "second"
    return None


def load_pairs(path: Path) -> List[Dict]:
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
        pred_a_more = None
        if pred == "first":
            pred_a_more = item["order_a_first"]
        elif pred == "second":
            pred_a_more = not item["order_a_first"]
        actual_a_more = item["stars_a"] > item["stars_b"]
        is_correct = pred_a_more == actual_a_more if pred_a_more is not None else False
        results.append(
            {
                "repo_a": item["repo_a"],
                "repo_b": item["repo_b"],
                "stars_a": item["stars_a"],
                "stars_b": item["stars_b"],
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
        if spec.family in {"nemotron", "qwen"}:
            tokenizer = AutoTokenizer.from_pretrained(spec.name, trust_remote_code=True)
            if spec.family == "nemotron":
                if not NEMOTRON_TEMPLATE_PATH.exists():
                    raise FileNotFoundError(f"Missing Nemotron chat template: {NEMOTRON_TEMPLATE_PATH}")
                tokenizer.chat_template = NEMOTRON_TEMPLATE_PATH.read_text()

        sp = get_sampling_params(spec.family)

        pair_path = download_from_hf(PAIR_FILE, token=hf_token)
        readme_path = download_from_hf(README_FILE, token=hf_token)
        pairs = load_pairs(pair_path)
        readme_lookup = load_readme_lookup(readme_path)
        print(f"Loaded {len(pairs)} pairs")

        batch: list[Dict] = []
        results: list[Dict] = []
        model_calls = 0

        for idx, sample in enumerate(tqdm(pairs, desc=f"GitHub pairs ({spec.family})")):
            repo_a = sample.get("repo_url_a") or sample.get("repo_a")
            repo_b = sample.get("repo_url_b") or sample.get("repo_b")
            stars_a = sample.get("stars_a", 0)
            stars_b = sample.get("stars_b", 0)
            readme_a = readme_lookup.get(repo_a, "")
            readme_b = readme_lookup.get(repo_b, "")
            if not readme_a or not readme_b:
                continue

            order_a_first = (idx % 2 == 0)
            prompt = build_prompt(
                spec.family,
                readme_a if order_a_first else readme_b,
                readme_b if order_a_first else readme_a,
                tokenizer=tokenizer,
            )

            batch.append(
                {
                    "prompt": prompt,
                    "order_a_first": order_a_first,
                    "repo_a": repo_a,
                    "repo_b": repo_b,
                    "stars_a": stars_a,
                    "stars_b": stars_b,
                }
            )

            if len(batch) >= (MISTRAL_BATCH_SIZE if spec.family == "mistral" else BATCH_SIZE):
                model_calls += process_batch(llm, batch, results, sp, spec.family)
                batch = []

        if batch:
            model_calls += process_batch(llm, batch, results, sp, spec.family)

        evaluated = len(results)
        correct_predictions = sum(r["is_correct"] for r in results)
        accuracy = correct_predictions / evaluated if evaluated else 0
        elapsed = time.time() - start_time

        out_path = output_dir / spec.output_file
        out_path.parent.mkdir(parents=True, exist_ok=True)

        return {
            "model": spec.name,
            "total_pairs": len(pairs),
            "evaluated": evaluated,
            "correct": correct_predictions,
            "accuracy": accuracy,
            "elapsed_seconds": elapsed,
            "results": results,
            "output_path": out_path,
            "model_calls": model_calls,
        }
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
        payload = run_model(spec, hf_token, output_dir)
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
