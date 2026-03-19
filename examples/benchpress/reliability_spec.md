# /reliability_specification.md
# BenchPress Reliability Audit: Replication Reliability Analysis (Mutual Evaluation) (v1.0)

**Goal:** Assess reliability of K independent agent analyses of the same BenchPress task *without human/subjective judging* by measuring shared information across outputs using a TVD-based mutual evaluation mechanism.

**Implementation note:** The reliability evaluator is itself an LLM agent executing this specification autonomously (not a human). It runs after all K analysis agents have finished and has read-only access to their output directories. The same independence and reproducibility constraints apply: no internet access, deterministic query definitions, and all outputs saved to disk.

---

## Inputs (directory layout)

Agent outputs are stored at:

- `../<model>_run<NN>/`

(e.g., `../opus-4.6_run01/`, `../opus-4.6_run02/`, etc.)

Each agent directory may contain:
- `results_summary.json` (required for SUCCESS)
- `canonical_predictions.csv` (required for SUCCESS; see `/canonical_evaluation.md`)
- optional additional artifacts (logs, intermediate files)

Evaluator also has access to:
- `llm_benchmark_data.json`
- `canonical_mask.json`
- `/canonical_evaluation.md`
- `/analysis_plan.md`

---

## Step 1 — Load outputs and classify SUCCESS/FAILURE

For each `agent_<id>` directory:

Classify as **SUCCESS** iff all conditions hold:
1. `results_summary.json` parses as JSON and contains the required top-level keys:
   - `data_discovery`, `data`, `rank_analysis`, `benchmark_selection`, `prediction`, `methodology_notes`
2. `canonical_predictions.csv` exists and has required columns:
   - `model_id`, `model_name`, `benchmark_id`, `benchmark_name`, `y_pred`
3. `y_pred` is numeric for at least 95% of canonical held-out entries (coverage rule)

Otherwise classify as **FAILURE** and record a one-line failure mode, e.g.:
- missing file
- JSON parse error
- schema mismatch
- too-low canonical coverage
- non-numeric predictions

Report:
- `N_total`, `N_success`, `N_failure`
- Per-agent table: agent id → SUCCESS/FAILURE + failure mode

---

## Step 2 — Compute canonical metrics (deterministic)

Using `llm_benchmark_data.json` and `canonical_mask.json`, compute for each SUCCESS agent:

- `canonical_overall_mae` on the canonical normalized 0–100 scale
- `canonical_per_benchmark_mae`
- `canonical_coverage`

(Exact definitions in `/canonical_evaluation.md`.)

Save:
- `canonical_metrics.csv` (one row per agent)
- `canonical_metrics.json` (same data, structured)

---

## Step 3 — Design binary queries (adaptive, constrained)

Examine AT MOST 10 SUCCESS agents (randomly selected, with a fixed seed specified in `/analysis_plan.md`). Design exactly **Q = 20** binary (yes/no) queries.

**Core constraint:** Each query must be answerable deterministically from:
- `results_summary.json` fields, and/or
- computed canonical metrics from Step 2, and/or
- simple string matching on `methodology_notes`

**No free-form judging.** No case-by-case exceptions. No agent-id-specific queries.

**Variance requirement:** Each query must have at least one YES and one NO across SUCCESS agents; aim for ≥20% minority rate.

### Query tiers (exactly 4 tiers of 5)

Tier 1 — Outcomes (5)
- rank bins, canonical MAE bins, selected subset size bins, etc.

Tier 2 — Methodology forks (5)
- filtered vs full preprocessing
- normalization choice family
- decomposition family
- prediction family
- eval protocol family

Tier 3 — Specific claims surfaced by outputs (5)
- e.g., “dominant rank-1 factor” mentioned
- “scale mismatch” addressed
- “missingness as main bottleneck” claimed
(Still must be deterministic via string/field checks.)

Tier 4 — Benchmark selection structure (5)
- membership queries like “selected set includes SimpleQA”
- “includes at least one coding benchmark”
- “overlaps with benchmark category X”
(Define category membership using the benchmark metadata in `llm_benchmark_data.json`.)

**Important:** numeric thresholds used in queries must come from the allowed bin set in `/analysis_plan.md` (to avoid post-hoc threshold hacking).

Save:
- `queries.json` containing query definitions and how they are computed.

---

## Step 4 — Build response matrix

Apply all Q queries to all SUCCESS agents to produce:

- Response matrix `R ∈ {0,1}^{Q × N_success}`

Report:
- full matrix with row labels (queries) and column labels (agent ids)
- per-query agreement rate (majority fraction)
- flag any query with 100% agreement (must be replaced and rerun)

Save:
- `response_matrix.csv`
- `response_matrix.json`

---

## Step 5 — Compute pairwise TVD-MI between agents

For each pair of agents (i, j):

Let `r_i`, `r_j` be their response vectors (length Q).

1) Empirical joint distribution over `{0,1}×{0,1}` across queries:
\[
\hat{P}(x,y) = \frac{1}{Q}\sum_{q=1}^Q 1[r_{q,i}=x]1[r_{q,j}=y]
\]

2) Marginals:
\[
\hat{P}_i(x)=\sum_y \hat{P}(x,y),\quad \hat{P}_j(y)=\sum_x \hat{P}(x,y)
\]

3) TVD-MI:
\[
I_{\mathrm{TVD}}(i;j) = \frac{1}{2}\sum_{x\in\{0,1\}}\sum_{y\in\{0,1\}} \left|\hat{P}(x,y)-\hat{P}_i(x)\hat{P}_j(y)\right|
\]

Compute:
- pairwise TVD-MI matrix (N_success × N_success)
- agent welfare:
\[
w_i = \frac{1}{N_{\text{success}}-1}\sum_{j\neq i} I_{\mathrm{TVD}}(i;j)
\]
- overall welfare:
\[
W = \frac{2}{N_{\text{success}}(N_{\text{success}}-1)}\sum_{i<j} I_{\mathrm{TVD}}(i;j)
\]

Save:
- `tvdmi_matrix.csv`
- `welfare.csv`

---

## Step 6 — Fork detection via leave-one-query-out contribution

Define:
- \( W^{(-q)} \): overall welfare recomputed with query q removed
- contribution:
\[
\Delta W_q = W - W^{(-q)}
\]

Rank queries by `ΔW_q`. The top query is the **primary fork**.

Save:
- `fork_contributions.csv`

---

## Step 7 — Clustering (if structure exists)

Using the agent×agent TVD-MI matrix, run hierarchical clustering.

Report:
- whether ≥2 clusters are stable (criterion in `/analysis_plan.md`)
- cluster summaries: modal answers per query, canonical MAE distribution, typical methodology

Save:
- `clusters.json`

---

## Step 8 — Final report artifacts

Produce:
- `reliability_report.md` (human-readable)
- `reliability_summary.json` (machine-readable)

Include:
- SUCCESS/FAILURE breakdown
- canonical metrics summary
- queries and diagnostics
- welfare + fork results
- cluster characterization
- caveats (noise due to finite Q)

---

## Calibration note

With Q = 20 binary observations per pair, TVD-MI estimates are noisy. Focus on:
- large forks (ΔW clearly separated)
- coherent clusters
- robustness checks from `/analysis_plan.md`