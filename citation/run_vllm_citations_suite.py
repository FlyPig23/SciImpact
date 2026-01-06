"""
Run citation-prediction evals for a list of models with vLLM, one model at a time.
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

# --- DATA SOURCES ---
HF_REPO_ID = "/paper-impact-data"
PAIR_BASE = "construct_citations_dataset/sampled_all_field_pairs_433"
MATCH_BASE = "sampled_maple_openalex_matches"
FIELDS = [
    "art",
    "biology",
    "business",
    "chemistry",
    #"computer_science",
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
LARGE_FIELDS_FOR_OTHER_AVG = {"csrankings", "chemistry", "physics", "medicine"}

# --- EVAL CONFIG ---
MAX_ABSTRACT_LEN = 1800
BATCH_SIZE = 16
NEMOTRON_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "nothink.jinja"
MISTRAL_BATCH_SIZE = int(os.environ.get("MISTRAL_BATCH_SIZE", "4"))
MISTRAL_MAX_BATCHED_TOKENS = os.environ.get("MISTRAL_MAX_BATCHED_TOKENS")
MISTRAL_MAX_MODEL_LEN = os.environ.get("MISTRAL_MAX_MODEL_LEN")
DEFAULT_VISIBLE_GPUS = os.environ.get("VISIBLE_GPUS", "4,5,6,7")
TP_DEFAULT = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "4"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.80"))

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.9,
    max_tokens=32,
)


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
    ModelSpec("meta-llama/Meta-Llama-3-8B-Instruct", "llama", "llama3_vllm_citations_results.json"),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama", "llama31_vllm_citations_results.json"),
    ModelSpec("meta-llama/Llama-3.2-3B-Instruct", "llama", "llama32_vllm_citations_results.json"),
    ModelSpec("Qwen/Qwen3-4B-Instruct-2507", "qwen", "qwen3_vllm_citations_results.json"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", "qwen", "qwen25_7b_vllm_citations_results.json"),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", "qwen", "qwen25_14b_vllm_citations_results.json"),
    ModelSpec(
        "mistralai/Ministral-3-3B-Instruct-2512",
        "mistral",
        "ministral3_3b_vllm_citations_results.json",
        llm_kwargs={"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"},
    ),
    ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "nemotron",
        "nemotron3_vllm_citations_results.json",
        llm_kwargs={
            "kv_cache_dtype": "fp8",
        },
        env_vars={
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
    title = entry.get("title") or entry.get("Title") or "Untitled"
    abstract = truncate(entry.get("abstract") or entry.get("abstract_raw") or "", MAX_ABSTRACT_LEN)
    return f"Title: {title}\nAbstract:\n{abstract}\n"


def build_prompt(family: str, first_paper: str, second_paper: str, tokenizer=None) -> str:
    system_prompt = (
            "You are an impartial judge deciding which of two research papers has more citations. "
            "Your reply MUST be exactly one sentence and must be one of these two options:\n"
            "- Paper A has more citations\n"
            "- Paper B has more citations\n"
            "You are not allowed to output anything else—no explanations, no extra words."
        )
    user_prompt = (
            f"Paper A:\n{first_paper}\n"
            f"Paper B:\n{second_paper}\n\n"
            "Based solely on the information above, which paper do you think has more citations?\n\n"
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
        # Mistral v0.1/v0.2 style (v0.3 supports chat template above)
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

    # Fallback for unknown models (NOT recommended for Qwen or Nemotron)
    return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"


def parse_model_response(response: str | None) -> str | None:
    if not response:
        return None
    clean_resp = response.strip()
    if re.search(r"Paper A has more citations", clean_resp, re.IGNORECASE):
        return "first"
    if re.search(r"Paper B has more citations", clean_resp, re.IGNORECASE):
        return "second"
    low = clean_resp.lower()
    if "paper a has more citations" in low:
        return "first"
    if "paper b has more citations" in low:
        return "second"
    if "a" in low and "b" not in low:
        return "first"
    if "b" in low and "a" not in low:
        return "second"
    if "first" in low and "second" not in low:
        return "first"
    if "second" in low and "first" not in low:
        return "second"
    return None


def load_pairs(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_match_dict(path: Path) -> Dict[str, Dict]:
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            data[str(entry["paper"])] = entry
    return data


def _generate_with_retry(
    llm: LLM, prompts: list[str], family: str, max_retries: int = 2
) -> tuple[list[str | None], dict]:
    """Generate texts, retrying missing/empty outputs for Mistral models."""
    retry_stats = {"attempts": 0, "pending_counts": []}
    if family != "mistral":
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        retry_stats["attempts"] = 1
        retry_stats["pending_counts"] = [0]
        return [out.outputs[0].text.strip() if out.outputs else None for out in outputs], retry_stats

    texts: list[str | None] = [None] * len(prompts)
    pending: list[tuple[int, str]] = list(enumerate(prompts))
    for attempt in range(max_retries + 1):
        retry_stats["attempts"] = attempt + 1
        retry_stats["pending_counts"].append(len(pending))
        if not pending:
            break
        # On the final attempt, run one by one to reduce drop rate.
        if attempt == max_retries:
            current_prompts = pending
            pending = []
            for orig_idx, prompt in current_prompts:
                output = llm.generate([prompt], sampling_params=sampling_params)[0]
                text = output.outputs[0].text.strip() if output.outputs else None
                if text:
                    texts[orig_idx] = text
                else:
                    pending.append((orig_idx, prompt))
            continue

        current_prompts = [p for _, p in pending]
        outputs = llm.generate(current_prompts, sampling_params=sampling_params)
        new_pending: list[tuple[int, str]] = []
        for (orig_idx, prompt), output in zip(pending, outputs):
            text = output.outputs[0].text.strip() if output.outputs else None
            if text:
                texts[orig_idx] = text
            else:
                new_pending.append((orig_idx, prompt))
        pending = new_pending
    retry_stats["pending_counts"].append(len(pending))
    return texts, retry_stats


def process_batch(llm: LLM, batch: list[Dict], results: list[Dict], family: str) -> int:
    prompts = [item["prompt"] for item in batch]
    texts, retry_stats = _generate_with_retry(llm, prompts, family)
    if family == "mistral" and retry_stats["pending_counts"]:
        last_pending = retry_stats["pending_counts"][-1]
        mid_pending = max(retry_stats["pending_counts"][1:], default=0)
        if last_pending > 0 or mid_pending > 0:
            print(
                f"[mistral retry] attempts={retry_stats['attempts']} "
                f"pending_after_attempts={retry_stats['pending_counts']} "
                f"final_pending={last_pending}"
            )
    used = 0
    for item, text in zip(batch, texts):
        if text is not None:
            used += 1
        pred = parse_model_response(text)
        predicted_a_more = None
        if pred is not None:
            predicted_a_more = (pred == "first") if item["order_a_first"] else (pred == "second")
        actual_a_more = item["citations_a"] > item["citations_b"]
        is_correct = predicted_a_more == actual_a_more if predicted_a_more is not None else False
        results.append(
            {
                "field": item["field"],
                "paper_a_id": item["paper_a_id"],
                "paper_b_id": item["paper_b_id"],
                "citations_a": item["citations_a"],
                "citations_b": item["citations_b"],
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

        batch: list[Dict] = []
        results: list[Dict] = []
        model_calls = 0
        total_pairs = 0
        pairs_count_by_field: dict[str, int] = {}

        batch_limit = MISTRAL_BATCH_SIZE if spec.family == "mistral" else BATCH_SIZE

        for field in FIELDS:
            pair_path = download_from_hf(f"{PAIR_BASE}/{field}_test_pairs.jsonl", token=hf_token)
            match_path = download_from_hf(f"{MATCH_BASE}/maple_openalex_{field}_match.json", token=hf_token)

            pairs = load_pairs(pair_path)
            match_dict = load_match_dict(match_path)
            total_pairs += len(pairs)
            pairs_count_by_field[field] = len(pairs)
            print(f"Field {field}: loaded {len(pairs)} pairs")

            for idx, sample in enumerate(pairs):
                paper_a = match_dict.get(str(sample["paper_a_id"]), {})
                paper_b = match_dict.get(str(sample["paper_b_id"]), {})
                abs_a = (paper_a.get("abstract") or "").strip()
                abs_b = (paper_b.get("abstract") or "").strip()
                if not abs_a or not abs_b:
                    continue

                order_a_first = (idx % 2 == 0)
                first_paper = format_paper(paper_a if order_a_first else paper_b)
                second_paper = format_paper(paper_b if order_a_first else paper_a)
                prompt = build_prompt(spec.family, first_paper, second_paper, tokenizer=tokenizer)

                batch.append(
                    {
                        "prompt": prompt,
                        "order_a_first": order_a_first,
                        "field": field,
                        "paper_a_id": sample["paper_a_id"],
                        "paper_b_id": sample["paper_b_id"],
                        "citations_a": sample["citations_a"],
                        "citations_b": sample["citations_b"],
                    }
                )

                if len(batch) >= batch_limit:
                    model_calls += process_batch(llm, batch, results, spec.family)
                    batch = []

        if batch:
            model_calls += process_batch(llm, batch, results, spec.family)

        correct_predictions = sum(r["is_correct"] for r in results)
        evaluated = len(results)
        accuracy = correct_predictions / evaluated if evaluated else 0

        evaluated_by_field: dict[str, int] = {}
        correct_by_field: dict[str, int] = {}
        for record in results:
            field = record["field"]
            evaluated_by_field[field] = evaluated_by_field.get(field, 0) + 1
            if record["is_correct"]:
                correct_by_field[field] = correct_by_field.get(field, 0) + 1

        field_stats: list[Dict] = []
        for field in FIELDS:
            pairs_count = pairs_count_by_field.get(field, 0)
            evaluated_count = evaluated_by_field.get(field, 0)
            correct_count = correct_by_field.get(field, 0)
            skipped_count = max(pairs_count - evaluated_count, 0)
            field_accuracy = correct_count / evaluated_count if evaluated_count else 0
            field_stats.append(
                {
                    "field": field,
                    "pairs_count": pairs_count,
                    "evaluated_count": evaluated_count,
                    "skipped_count": skipped_count,
                    "correct_predictions": correct_count,
                    "accuracy": field_accuracy,
                }
            )

        other_fields_correct = sum(
            correct_by_field.get(field, 0) for field in FIELDS if field not in LARGE_FIELDS_FOR_OTHER_AVG
        )
        other_fields_evaluated = sum(
            evaluated_by_field.get(field, 0) for field in FIELDS if field not in LARGE_FIELDS_FOR_OTHER_AVG
        )
        other_fields_avg = other_fields_correct / other_fields_evaluated if other_fields_evaluated else 0

        summary_head = {
            "overall_accuracy": accuracy,
            "other_fields_avg": other_fields_avg,
            "elapsed_seconds": None,  # filled below
        }

        out_path = output_dir / spec.output_file
        out_path.parent.mkdir(parents=True, exist_ok=True)

        return {
            "summary": summary_head,
            "field_stats": field_stats,
            "model": spec.name,
            "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tensor_parallel_size": spec.tensor_parallel_size or TP_DEFAULT,
            "total_fields": len(FIELDS),
            "total_pairs": total_pairs,
            "evaluated": evaluated,
            "correct": correct_predictions,
            "accuracy": accuracy,
            "results": results,
            "output_path": out_path,
        }
    finally:
        # Clean up engine to free GPUs before the next model
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
        start = time.time()
        payload = run_model(spec, hf_token, output_dir)
        payload["summary"]["elapsed_seconds"] = time.time() - start
        out_path = payload.pop("output_path")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(
            f"Saved {spec.name} results to {out_path} | "
            f"acc={payload['accuracy']:.2%} | "
            f"elapsed={payload['summary']['elapsed_seconds']:.1f}s | "
            f"samples={payload['evaluated']}"
        )
        summaries.append(
            {
                "model": spec.name,
                "output": str(out_path),
                "accuracy": payload["accuracy"],
                "elapsed_seconds": payload["summary"]["elapsed_seconds"],
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
