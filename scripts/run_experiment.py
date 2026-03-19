#!/usr/bin/env python3
"""
Run an agent specification N times across different models,
saving each trace as a conversation file with token usage tracking.

Usage:
    python scripts/run_experiment.py --spec examples/benchpress/spec.md --runs 10
    python scripts/run_experiment.py --spec examples/tmlr_audit/spec.md --runs 5 --models opus-4.6
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from agent_core import AgentState, agent_loop


MODELS = {
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-opus-4-6-reliability": "claude-opus-4-6",
    "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
}

DEFAULT_ENTRY_ID = "nse-bench"


def find_existing_runs(results_file):
    """Scan results file for successfully completed runs. Returns set of (model_key, run_idx)."""
    completed = set()
    if not os.path.exists(results_file):
        return completed
    with open(results_file) as f:
        for line in f:
            try:
                r = json.loads(line)
                # Count as completed if it has output OR used tokens (real LLM call happened)
                has_real_run = r.get("has_output") or r.get("usage", {}).get("input_tokens", 0) > 0
                if has_real_run and "error" not in r:
                    completed.add((r["model_key"], r["run_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def run_single(spec_text, model_key, model_name, run_idx,
               project_dir, max_parts, timeout, quiet,
               entry_id=None, name_prefix="Experiment", data_dir=".",
               spec_filename="specification.md"):
    """Run a single agent trace and return summary."""
    run_project = os.path.join(project_dir, f"{model_key}_run{run_idx:02d}")
    os.makedirs(run_project, exist_ok=True)

    # Copy all data files into agent's working directory
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            src = os.path.join(data_dir, fname)
            dst = os.path.join(run_project, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    spec_dst = os.path.join(run_project, spec_filename)
    if not os.path.exists(spec_dst):
        with open(spec_dst, "w") as f:
            f.write(spec_text)

    # Only include code and markdown files in context — data files are accessed via scratch.py
    context_extensions = {'.py', '.md', '.txt', '.yml', '.yaml'}
    agent_files = [os.path.join(run_project, f) for f in sorted(os.listdir(run_project))
                   if os.path.isfile(os.path.join(run_project, f))
                   and os.path.splitext(f)[1].lower() in context_extensions]

    label = f"{model_key} run {run_idx}"

    if quiet:
        on_chunk = lambda c: None
        on_status = lambda m: print(f"  [{label}] {m}", file=sys.stderr)
    else:
        on_chunk = lambda c: print(c, end="", flush=True)
        on_status = lambda m: print(f"\n  [{label}] {m}", file=sys.stderr)

    state = AgentState(
        project_dir=run_project,
        selected_files=agent_files,
        model_provider="Anthropic",
        api_key_type="research",
        model_name=model_name,
        mode="base",
        max_parts=max_parts,
        max_output_chars=25000,
        exec_timeout=timeout,
        max_history=50,
        on_assistant_chunk=on_chunk,
        on_status=on_status,
    )

    state.conversation.start_new_conversation(
        start_entry_id=entry_id,
        code_files=[],
        name=f"{name_prefix}: {model_key} run {run_idx:02d}",
    )
    # Sync conversation manager metadata with actual run parameters
    state.conversation.max_parts = max_parts
    state.conversation.exec_timeout = timeout
    state.conversation.selected_project = run_project

    # Frame the spec as a task to execute, not a document to discuss
    task_prompt = (
        "Execute the following specification. Work through each step using scratch.py, "
        "saving all required output files to your working directory. "
        "Do not ask clarifying questions — make reasonable choices and document them.\n\n"
        + spec_text
    )

    t0 = time.time()
    result = agent_loop(state, task_prompt)
    elapsed = time.time() - t0

    return {
        "model_key": model_key,
        "model_name": model_name,
        "run_idx": run_idx,
        "conversation_id": result["conversation_id"],
        "turn_count": result["turn_count"],
        "parts": len(result["parts"]),
        "usage": result["usage"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "elapsed_seconds": round(elapsed, 1),
        "has_output": bool(result["final_output"]),
        "final_output_preview": (result["final_output"] or "")[:500],
        "project_dir": run_project,
        "timestamp": datetime.now().isoformat(),
    }


def _execute_one(args_tuple):
    """Module-level wrapper for ProcessPoolExecutor (must be picklable)."""
    job, kwargs = args_tuple
    model_key, model_name, run_idx = job
    return run_single(
        kwargs["spec_text"], model_key, model_name, run_idx,
        kwargs["project_dir"], kwargs["max_parts"], kwargs["timeout"],
        quiet=(kwargs["quiet"] or kwargs["batch_size"] > 1),
        entry_id=kwargs["entry_id"], name_prefix=kwargs["name_prefix"],
        data_dir=kwargs["data_dir"],
        spec_filename=kwargs["spec_filename"],
    )


def main():
    parser = argparse.ArgumentParser(description="Run agent specification across models")
    parser.add_argument("--spec", required=True, help="Path to specification.md")
    parser.add_argument("--runs", type=int, default=10, help="Runs per model (default: 10)")
    parser.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                        help="Models to test (default: all)")
    parser.add_argument("--max-parts", type=int, default=12, help="Max parts per agent (default: 12)")
    parser.add_argument("--timeout", type=int, default=120, help="Exec timeout per scratch.py run (default: 120)")
    parser.add_argument("--project", default="./results",
                        help="Base project dir")
    parser.add_argument("--data-dir", default=".",
                        help="Directory containing llm_benchmark_data.json and canonical_mask.json")
    parser.add_argument("--entry-id", default=DEFAULT_ENTRY_ID,
                        help=f"Entry ID to link conversations to (default: {DEFAULT_ENTRY_ID})")
    parser.add_argument("--name-prefix", default="NSE-Bench",
                        help="Conversation name prefix (default: 'NSE-Bench')")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of agents to run concurrently (default: 1)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress streaming")
    args = parser.parse_args()

    with open(args.spec) as f:
        spec_text = f.read()

    if not os.path.isdir(args.data_dir):
        print(f"WARNING: --data-dir {args.data_dir} not found, no files will be copied", file=sys.stderr)

    model_configs = []
    for m in args.models:
        if m in MODELS:
            model_configs.append((m, MODELS[m]))
        else:
            model_configs.append((m, m))

    os.makedirs(args.project, exist_ok=True)
    results_file = os.path.join(args.project, "experiment_results.jsonl")

    # Check for existing completed runs
    existing = find_existing_runs(results_file)
    if existing:
        print(f"Found {len(existing)} existing completed run(s), will skip them", file=sys.stderr)

    total_runs = len(model_configs) * args.runs
    print(f"Experiment: {len(model_configs)} models x {args.runs} runs = {total_runs} total", file=sys.stderr)
    print(f"Results: {results_file}", file=sys.stderr)

    # Build list of all (model_key, model_name, run_idx) to execute
    pending = []
    for model_key, model_name in model_configs:
        for run_idx in range(1, args.runs + 1):
            if (model_key, run_idx) in existing:
                print(f"  SKIP {model_key} run {run_idx} (already completed)", file=sys.stderr)
                continue
            pending.append((model_key, model_name, run_idx))

    print(f"\n{len(pending)} runs to execute (batch size: {args.batch_size})", file=sys.stderr)

    all_results = []

    # Build shared kwargs for _execute_one (must be picklable for ProcessPoolExecutor)
    shared_kwargs = {
        "spec_text": spec_text,
        "project_dir": args.project,
        "max_parts": args.max_parts,
        "timeout": args.timeout,
        "quiet": args.quiet,
        "entry_id": args.entry_id,
        "name_prefix": args.name_prefix,
        "data_dir": args.data_dir,
        "spec_filename": os.path.basename(args.spec),
        "batch_size": args.batch_size,
    }

    if args.batch_size <= 1:
        # Sequential execution (original behavior)
        for job in pending:
            model_key, model_name, run_idx = job
            print(f"\n  {model_key} run {run_idx}...", file=sys.stderr)
            try:
                summary = _execute_one((job, shared_kwargs))
                all_results.append(summary)
                with open(results_file, "a") as f:
                    f.write(json.dumps(summary) + "\n")
                u = summary["usage"]
                cost_str = f"${summary['estimated_cost_usd']:.3f}"
                tok_str = f"{u['input_tokens']}in/{u['output_tokens']}out"
                print(f"  Done: {summary['turn_count']} turns, {cost_str}, {tok_str}, {summary['elapsed_seconds']}s",
                      file=sys.stderr)
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                err = {"model_key": model_key, "run_idx": run_idx,
                       "error": str(e), "timestamp": datetime.now().isoformat()}
                all_results.append(err)
                with open(results_file, "a") as f:
                    f.write(json.dumps(err) + "\n")
    else:
        # Parallel execution
        print(f"  Running up to {args.batch_size} agents concurrently", file=sys.stderr)
        with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
            future_to_job = {executor.submit(_execute_one, (job, shared_kwargs)): job for job in pending}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                model_key, _, run_idx = job
                try:
                    summary = future.result()
                    all_results.append(summary)
                    with open(results_file, "a") as f:
                        f.write(json.dumps(summary) + "\n")
                    u = summary["usage"]
                    cost_str = f"${summary['estimated_cost_usd']:.3f}"
                    tok_str = f"{u['input_tokens']}in/{u['output_tokens']}out"
                    print(f"  ✓ {model_key} run {run_idx}: {summary['turn_count']} turns, "
                          f"{cost_str}, {tok_str}, {summary['elapsed_seconds']}s", file=sys.stderr)
                except Exception as e:
                    print(f"  ✗ {model_key} run {run_idx} FAILED: {e}", file=sys.stderr)
                    err = {"model_key": model_key, "run_idx": run_idx,
                           "error": str(e), "timestamp": datetime.now().isoformat()}
                    all_results.append(err)
                    with open(results_file, "a") as f:
                        f.write(json.dumps(err) + "\n")

    # Summary
    print(f"\n{'='*60}", file=sys.stderr)
    for model_key, _ in model_configs:
        runs = [r for r in all_results if r.get("model_key") == model_key and "error" not in r]
        errors = [r for r in all_results if r.get("model_key") == model_key and "error" in r]
        if not runs:
            print(f"  {model_key}: 0 completed, {len(errors)} failed", file=sys.stderr)
            continue
        costs = [r["estimated_cost_usd"] for r in runs]
        turns = [r["turn_count"] for r in runs]
        total_cost = sum(costs)
        avg_turns = sum(turns) / len(turns)
        with_output = sum(1 for r in runs if r["has_output"])
        print(f"  {model_key}: {len(runs)} ok, {len(errors)} failed, "
              f"avg {avg_turns:.1f} turns, total ${total_cost:.2f}, "
              f"{with_output}/{len(runs)} produced output", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()