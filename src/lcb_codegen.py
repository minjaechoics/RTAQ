#!/usr/bin/env python3
"""LiveCodeBench code generation for run_sampled_eval_vllm.py: frozen problem set, official prompt, official grader.

Everything benchmark-specific comes from the official repository vendored at third_party/LiveCodeBench (commit in
LCB_COMMIT), so prompts, code extraction and test execution match `lcb_runner` exactly:

  prompt     single user turn = SYSTEM_MESSAGE_GENERIC + "\\n\\n" + get_generic_question_template_answer(problem)
             (the reasoning-model layout LCB uses for o1/Grok), then the model's own chat template
  extraction extract_code: the last ``` fenced block, taken from the text after </think>; an unfinished sample
             (no </think>) falls back to the whole text, as AIME falls back to the last \\boxed{}
  grading    run_test on public + private tests with the official per-test timeout (6 s) and global timeout
             ((timeout + 1) * n_tests + 5 s); a sample is correct iff every test passes (np.all(results > 0))

Two guards are added around run_test, neither of which changes a verdict for a well-behaved program: the child
process runs inside an empty scratch directory that is deleted afterwards (reliability_guard does not block open() for
writing), and its address space may grow at most --lcb_memory_gb (default 8) beyond what the interpreter already maps
(the official guard passes no limit at all; judge limits are ~1 GB).

Problem set (build once):
  python lcb_codegen.py build --start 2025-02-01 --end 2025-05-01 \\
      --out data/livecodebench/lcb_codegen_v6_2502_2505.jsonl
The window matches Qwen's "LiveCodeBench v6 (25.02-25.05)". Rows are stored verbatim (private tests stay
compressed) with a manifest next to the file, so every model is scored on byte-identical problems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

TWLA = Path(__file__).resolve().parent
RTAQ_ROOT = Path(os.environ.get("RTAQ_ROOT") or TWLA.parent)
LCB_ROOT = RTAQ_ROOT / "third_party" / "LiveCodeBench"   # scripts/08_evaluate.sh clones it at LCB_COMMIT
LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LCB_DATASET = "livecodebench/code_generation_lite"
LCB_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")
DEFAULT_PROBLEMS = RTAQ_ROOT / "data" / "livecodebench" / "lcb_codegen_v6_2502_2505.jsonl"

if str(LCB_ROOT) not in sys.path:
    sys.path.insert(0, str(LCB_ROOT))


@contextmanager
def _lcb_cwd():
    """lcb_runner.prompts opens its few-shot files by repo-relative path at import time."""
    previous = os.getcwd()
    os.chdir(LCB_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def load_problems(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prompt_messages(problem: dict) -> list[dict]:
    with _lcb_cwd():
        from lcb_runner.prompts.code_generation import PromptConstants, get_generic_question_template_answer

    question = SimpleNamespace(question_content=problem["question_content"], starter_code=problem["starter_code"])
    return [{"role": "user",
             "content": PromptConstants.SYSTEM_MESSAGE_GENERIC + "\n\n" + get_generic_question_template_answer(question)}]


def extract_solution(text: str) -> tuple[str, bool]:
    """Return (code, think_closed). Code is "" when no fenced block exists."""
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    closed = "</think>" in text
    answer = text.rsplit("</think>", 1)[1] if closed else text
    return extract_code(answer, LMStyle.OpenAIReasonPreview), closed


def evaluation_sample(problem: dict) -> dict:
    """Official CodeGenerationProblem.get_evaluation_sample(), decoding the private tests the same way."""
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    return CodeGenerationProblem(**problem).get_evaluation_sample()


# ---------------------------------------------------------------- grading (runs in worker processes)

_WORKER: dict = {}


def _init_worker(problems_path: str) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.set_int_max_str_digits(50000)
    _WORKER["problems"] = {p["question_id"]: p for p in load_problems(Path(problems_path))}
    from lcb_runner.evaluation.testing_util import run_test  # imported once; forked test children inherit it

    _WORKER["run_test"] = run_test


@lru_cache(maxsize=4)
def _sample_for(question_id: str) -> dict:
    return evaluation_sample(_WORKER["problems"][question_id])


def _guarded_run(sample, code, timeout, memory_bytes, workdir, result, metadata_list):
    import resource

    os.chdir(workdir)
    if memory_bytes:
        # headroom on top of the interpreter's own mappings (~4 GB virtual after the lcb_runner imports), so an
        # honest solution within the ~1 GB judge limits is never refused while a runaway allocation fails fast
        with open("/proc/self/status") as status:
            baseline = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmSize:"))
        limit = baseline + memory_bytes
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    res, metadata = _WORKER["run_test"](sample, test=code, debug=False, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def _check_correctness(sample: dict, code: str, timeout: int, memory_bytes: int):
    """lcb_runner check_correctness with the scratch-directory and memory guards (same global timeout)."""
    context = multiprocessing.get_context("fork")  # grading workers hold no CUDA state
    manager = context.Manager()
    workdir = tempfile.TemporaryDirectory(prefix="lcb_exec_")
    try:
        result, metadata_list = manager.list(), manager.list()
        process = context.Process(target=_guarded_run,
                                  args=(sample, code, timeout, memory_bytes, workdir.name, result, metadata_list))
        process.start()
        process.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)
        if process.is_alive():
            process.kill()
            process.join()
        if not result:
            return [-1] * len(json.loads(sample["input_output"])["inputs"]), {"error_code": -1,
                                                                            "error_message": "Global Timeout"}
        return list(result[0]), dict(metadata_list[0]) if metadata_list else {}
    finally:
        manager.shutdown()
        workdir.cleanup()


def _grade_task(question_id: str, code: str, timeout: int, memory_bytes: int) -> dict:
    import numpy as np

    if not code.strip():
        return {"correct": False, "code_found": False, "error_code": -5, "error_message": "No code block"}
    sample = _sample_for(question_id)
    try:
        results, metadata = _check_correctness(sample, code, timeout, memory_bytes)
    except Exception as error:  # mirrors evaluate_generations_by_problem: a runner exception is a failure
        results, metadata = [-5], {"error_code": -5, "error_message": f"TestRunnerError {error!r}"}
    fixed = []
    for value in results:
        if isinstance(value, np.ndarray):
            value = value.item(0)
        if isinstance(value, np.bool_):
            value = bool(value)
        fixed.append(value)
    n_tests = len(json.loads(sample["input_output"])["inputs"])
    return {"correct": bool(np.all(np.array(fixed) > 0)), "code_found": True,
            "tests_passed": sum(1 for v in fixed if v is True), "tests_total": n_tests,
            "error_code": metadata.get("error_code"), "error_message": str(metadata.get("error_message", ""))[:300]}


def grade_samples(problems_path: Path, samples: list[dict], cache_dir: Path, workers: int = 16, timeout: int = 6,
                  memory_gb: float = 8.0) -> dict[str, dict]:
    """Grade {"key", "question_id", "text"} samples in parallel; results are cached per key in cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    memory_bytes = int(memory_gb * 2**30) if memory_gb else 0
    grades, todo = {}, []
    for sample in samples:
        code, closed = extract_solution(sample["text"])
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        path = cache_dir / f"{sample['key']}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("code_sha256") == code_hash and cached.get("timeout") == timeout:
                grades[sample["key"]] = cached
                continue
        todo.append((sample, code, code_hash, closed))
    print(f"[lcb] {len(samples) - len(todo)} grades cached, {len(todo)} to run with {workers} workers", flush=True)
    if not todo:
        return grades
    # spawn: the caller may hold a vLLM engine (CUDA context + threads), which must not be forked
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker,
                             initargs=(str(problems_path),)) as pool:
        futures = {pool.submit(_grade_task, sample["question_id"], code, timeout, memory_bytes):
                   (sample, code_hash, closed) for sample, code, code_hash, closed in todo}
        done = 0
        for future in as_completed(futures):
            sample, code_hash, closed = futures[future]
            grade = {**future.result(), "think_closed": closed, "code_sha256": code_hash, "timeout": timeout}
            path = cache_dir / f"{sample['key']}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(grade))
            os.replace(temporary, path)
            grades[sample["key"]] = grade
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"[lcb] graded {done}/{len(todo)}", flush=True)
    return grades


# ---------------------------------------------------------------- problem-set build

def build(args) -> None:
    from huggingface_hub import hf_hub_download

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    kept, sources = [], {}
    for name in args.files:
        path = Path(hf_hub_download(LCB_DATASET, name, repo_type="dataset"))
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        count = 0
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                date = datetime.fromisoformat(row["contest_date"])
                if start <= date <= end:  # same inclusive filter as load_code_generation_dataset
                    kept.append(row)
                    count += 1
        print(f"{name}: kept {count}", flush=True)
    kept.sort(key=lambda row: (row["contest_date"], row["question_id"]))
    ids = [row["question_id"] for row in kept]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate question_id in the selected window")
    for row in kept:  # decode once so a broken row fails here, not mid-evaluation
        evaluation_sample(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": LCB_DATASET, "files_sha256": sources, "lcb_runner_commit": LCB_COMMIT,
        "start_date": args.start, "end_date": args.end, "problems": len(kept),
        "difficulty": dict(Counter(row["difficulty"] for row in kept)),
        "platform": dict(Counter(row["platform"] for row in kept)),
        "functional": sum(1 for row in kept if json.loads(row["metadata"]).get("func_name")),
        "contest_date_range": [kept[0]["contest_date"], kept[-1]["contest_date"]] if kept else None,
        "problems_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "question_ids": ids,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "question_ids"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--files", nargs="+", default=list(LCB_FILES))
    b.add_argument("--start", default="2025-02-01")
    b.add_argument("--end", default="2025-05-01")
    b.add_argument("--out", default=str(DEFAULT_PROBLEMS))
    args = parser.parse_args()
    if args.command == "build":
        build(args)


if __name__ == "__main__":
    main()
