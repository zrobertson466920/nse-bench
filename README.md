# NSE-Bench: Measuring Nonstandard Errors in AI Agents

AI agents given identical data and instructions produce different answers.
How bad is the disagreement? Does it have structure? Can you measure it?

This repo provides infrastructure to run K independent agents on any
analysis task, then measure whether their disagreement is noise or signal.

## The problem

[Gao & Xiao (2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6427518) coined
"nonstandard errors in AI agents" (AI NSE): 150 Claude agents given the same
financial dataset produce effect sizes ranging from +6%/yr to −5%/yr. The
disagreement is real and large.

But measuring *dispersion* (IQR of outputs) is not the same as measuring
*structure* (do methodological forks predict output patterns?). This repo does both.

## What's here

| Component | Purpose |
|:----------|:--------|
| `scripts/` | Agent execution infrastructure |
| `evaluate.py` | Response matrix → TVD-MI → permutation test → fork report |
| `examples/` | Self-contained studies with specs, data, traces, and results |

Each example is a self-contained study:

```
examples/benchpress/
├── spec.md              # What analysis agents receive
├── reliability_spec.md  # What evaluator agents receive
├── data/                # All files agents need (copied to workspace)
├── traces/              # Published conversation traces
└── results/             # Evaluation outputs
```

## Quick start

```bash
pip install -r scripts/requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."

# 1. Run K agents on a task
python scripts/run_experiment.py \
    --spec examples/benchpress/spec.md \
    --runs 50 --batch-size 5 \
    --data-dir examples/benchpress/data \
    --project results/benchpress \
    --models claude-opus-4-6

# 2. Run reliability evaluators
python scripts/run_experiment.py \
    --spec examples/benchpress/reliability_spec.md \
    --runs 2 \
    --data-dir examples/benchpress/data \
    --project results/benchpress \
    --models claude-opus-4-6-reliability

# 3. Evaluate cross-agent structure
python evaluate.py --results-dir results/benchpress
```

## What it measures

| Metric | What it tells you |
|:-------|:-----------------|
| IQR / MAE | Agents disagree (dispersion) |
| TVD-MI welfare | The disagreement has structure (mutual information) |
| Permutation p-value | The structure is distinguishable from chance |
| Fork ΔW | Which methodological choices carry the signal |

**IQR tells you agents disagree. TVD-MI tells you *why*.**

## Published examples

| Study | K | Domain | Pre-registered | Traces | Key finding |
|:------|:-:|:-------|:--------------:|:------:|:------------|
| [BenchPress audit](https://github.com/zrobertson466920/benchpress-reliability-audit) | 50 | Benchmark prediction | [Yes (OSF)](https://osf.io/x36uk) | ✅ 48/50 | Median MAE 15.6; all 5 hypotheses fail; cross-agent structure p=0.002 |
| [TMLR audit](https://zrobertson466920.github.io/AuditReliability/) | 50 | Editorial process | Yes | ✅ 36/50 | W=0.14; spec ambiguity is the discriminating fork |

## How this compares

| | Gao & Xiao (2026) | This repo |
|:--|:-------------------|:----------|
| What it measures | Dispersion (IQR of effect sizes) | Dispersion + structure (TVD-MI) |
| Pre-registration | No | Yes (OSF + SHA256 checksums) |
| Fork identification | Post-hoc R² decomposition | Pre-registered tiered queries |
| Reliability of measurement | Not assessed | Measured recursively (3 independent evaluators) |
| Cross-evaluator validation | None | Two evaluators, welfare r=0.60 |
| Permutation null | None | Yes (5000 permutations) |
| Traces published | Conversion CSVs | Full agent conversation traces |
| Reproducibility | Notebook reprocesses outputs | Two commands reproduce everything |
| Studies | 1 (NYSE data) | 2 (editorial process + benchmark prediction), both with traces |

## Writing your own spec

A specification is a markdown file that tells agents what to do. See `examples/benchpress/spec.md`
for an example. Key requirements:

1. **Self-contained task description** — the agent receives only this file plus any data files
2. **Required outputs** — specify exact filenames so `evaluate.py` can find results
3. **No solution hints** — the spec defines the task, not the approach

## Checksums

```
# BenchPress data
DATA_SHA256 = 255ea00914119403032f90f9568e9b6236483ff8b6858f18c22b62fd5bebe449
CANONICAL_MASK_SHA256 = 2a572935a067134718f207d59a1de29cb4f0aefbe94f81148da6b79bb896091c
```

## Citation

If you use this infrastructure, please cite:

```bibtex
@misc{robertson2026nsebench,
  author = {Robertson, Zachary},
  title = {NSE-Bench: Measuring Nonstandard Errors in AI Agents},
  year = {2026},
  url = {https://github.com/zrobertson466920/nse-bench}
}
```

## License

MIT