# /analysis_plan.md
# BenchPress Reliability Audit: Analysis Plan (v1.0)

**Author:** Zachary Robertson  
**Date:** 2026-02-26

This plan defines (i) hypotheses, (ii) canonical endpoints, (iii) mutual-evaluation measurement, and (iv) robustness checks.

---

## 1. Notation

- K = number of agents run (target 50)
- S = number of SUCCESS agents (defined in `/experiment_protocol.md`)
- Q = number of binary queries (fixed at 20; see `/reliability_specification.md`)
- `canonical_overall_mae` = canonical MAE@k=5 on normalized 0–100 scale (defined in `/canonical_evaluation.md`)

---

## 2. Hypotheses (primary)

### H1 — Rank convergence
**At least 75% of SUCCESS agents report effective rank ≤ 3.**

Operationalization: use `results_summary.json.rank_analysis.effective_rank`, capped at 5+ for reporting.

---

### H2 — Benchmark subset partial convergence
Let each agent’s selected set be $S_i$ (from `selected_benchmarks` in `results_summary.json`).

Compute pairwise Jaccard similarity over the full selected sets:

$$J(i,j) = \frac{|S_i \cap S_j|}{|S_i \cup S_j|}$$

**Prediction:** mean pairwise Jaccard satisfies $0.2 < \text{mean}(J) < 0.6$.

---

### H3 — Prediction feasibility (canonical, k=5)
**At least 80% of SUCCESS agents achieve `canonical_overall_mae < 10`.**

`canonical_overall_mae` is computed from `canonical_predictions.csv` by the evaluator (not self-reported).

---

### H4 — Preprocessing as primary fork
Using mutual-evaluation fork contributions `ΔW_q` (leave-one-query-out, see `/reliability_specification.md`):

**Prediction:** the top-ranked fork query pertains to preprocessing scope, e.g. filtered vs full matrix, or normalization handling.

This is evaluated as:  
- identify `q* = argmax_q ΔW_q`
- interpret `q*` using the query definition (tier 2 expected)

---

### H5 — Core qualitative robustness
**At least 90% of SUCCESS agents support the qualitative claim:**
> “The matrix is strongly low-rank and benchmark performance is predictably structured.”

Operationalization:
- This is measured by a Tier-3 query designed by the evaluator (adaptive), but constrained to be deterministic.
- If the evaluator cannot design such a query with adequate variance (≥1 yes and ≥1 no), H5 is reported as “not testable under prereg constraints” rather than being redefined.

---

## 3. Canonical endpoints and bins

### 3.1 Canonical endpoints (computed)
For each SUCCESS agent:
- `canonical_overall_mae`
- `canonical_coverage`
- `canonical_per_benchmark_mae` (for reporting, not hypothesis-tested)

### 3.2 Pre-registered bins (allowed for numeric queries)
To avoid post-hoc thresholding in query design, numeric thresholds used in evaluator queries must be chosen from:

- MAE bins: `<5`, `5–10`, `10–20`, `≥20`
- effective-rank bins: `1`, `2`, `3`, `4`, `5+`
- subset-size bins: `1–5`, `6–10`, `11+`
- missingness bins (fraction): `<0.4`, `0.4–0.6`, `>0.6`

---

## 4. Seeds and constants

| Constant | Value | Used in |
|:---------|:------|:--------|
| `CANONICAL_SEED` | 20260226 | Canonical reveal-k mask generation |
| `RELIABILITY_SEED` | 20260227 | Reliability evaluator's random agent subsampling (Step 3 of `/reliability_specification.md`) |

---

## 5. Mutual evaluation and TVD-MI reporting

The evaluator constructs:
- response matrix `R ∈ {0,1}^{Q×S}`
- pairwise TVD-MI matrix among agents
- welfare per agent and overall welfare W

Primary mutual-eval outputs:
- overall welfare W (descriptive)
- fork contributions `ΔW_q` (primary for H4)
- clustering structure (descriptive)

### 5.1 Null test (permutation)
A permutation baseline is computed by shuffling agent labels independently within each query row of R (preserving per-query marginals), recomputing W for each shuffle, and reporting:
- null mean/std
- empirical p-value for observed W

---

## 6. Secondary analyses (descriptive + effect sizes)

### 6.1 Success/failure structure
Report SUCCESS rate and failure-mode distribution.

### 6.2 Fork-conditioned outcomes
For the primary fork query `q*`, split agents into YES vs NO groups and report:
- difference in `canonical_overall_mae` (Cohen’s d)
- difference in reported `effective_rank` (Cohen’s d or ordinal shift)

### 6.3 Cross-evaluator robustness (optional but planned)
Run **two independent reliability evaluators**, each designing Q=20 queries under the same constraints, and report:
- correlation of agent welfare vectors
- agreement on top-3 fork queries (by ΔW)

If only one evaluator is run, report this as a limitation.

---

## 7. Reporting format

The final public writeup should include:
- prereg artifacts (frozen)
- SUCCESS/FAILURE breakdown
- hypothesis table with pass/fail and observed values
- canonical MAE distribution summary
- fork + clustering narrative (with deterministic query definitions)
- limitations (e.g., finite Q noise, single-model-family inductive bias)

No post-hoc hypothesis edits; any extra findings must be labeled exploratory.