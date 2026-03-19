# /benchpress_specification.md
# BenchPress Reliability Audit: Analysis Agent Specification (v1.0)

**Objective:** Characterize the low-rank structure of LLM benchmark performance data and build a predictor that estimates held-out benchmark scores from a small observed subset.

This task is designed to surface methodological degrees of freedom (schema interpretation, preprocessing, rank selection, subset selection, prediction/eval), not to enforce a single “correct” pipeline.

---

## Inputs (in working directory)

- `llm_benchmark_data.json` (unknown schema a priori)
- `canonical_mask.json` (canonical evaluation split; see `/canonical_evaluation.md`)
- `scratch.py` (you may edit/execute)
- This specification file

**Tooling constraints:**
- Python execution is available (numpy, scipy, scikit-learn, pandas).
- No internet access.
- No access to other agents’ outputs.

---

## Required outputs (must be created)

1. `performance_matrix.csv`
   - Rows = models, columns = benchmarks
   - First column = `model_name`
   - Column headers should be benchmark names (not IDs) if available

2. `cleaned_matrix.csv`
   - Your cleaned/processed matrix used for at least one stage of analysis
   - Document what “cleaned” means in `results_summary.json`

3. `singular_values.json`
   - Must contain at least the singular values you computed (and optionally variance explained)

4. `selected_benchmarks.json`
   - Must include selected benchmark names and `n_selected`

5. `prediction_results.json`
   - Must include overall MAE and per-benchmark MAE for your chosen evaluation protocol

6. `canonical_predictions.csv`
   - Predictions for the canonical held-out entries defined by `canonical_mask.json`
   - Columns (required):
     - `model_id`, `model_name`, `benchmark_id`, `benchmark_name`, `y_pred`
   - You may include extra columns, e.g. `y_true_train_used` or `notes`

7. `results_summary.json`
   - Structured summary (schema below)

---

## Task Steps

### Step 0 — Data discovery & matrix extraction

1. Load `llm_benchmark_data.json`
2. Inspect structure:
   - top-level keys
   - how models, benchmarks, and scores are represented
3. Decide how to map raw data → matrix entries:
   - identify model identifiers and benchmark identifiers
   - extract numeric scores
   - handle nesting/duplicates
4. Save `performance_matrix.csv`

**You must document:**
- the raw schema you found
- every extraction decision (joins, duplicate handling, naming choices)

---

### Step 1 — Data preparation

Compute and report:
- number of models, number of benchmarks
- missing fraction (over the full extracted matrix)

Then make *your* preprocessing choices (any are allowed):
- drop sparse benchmarks / models
- impute (mean/median/iterative)
- work with sparse matrix directly
- normalize or transform scores (raw, z-score, min-max, logit, etc.)

Save `cleaned_matrix.csv`.

---

### Step 2 — Rank analysis

Compute a decomposition on at least one matrix you define (cleaned and/or imputed), such as:
- SVD / PCA
- NMF
- matrix completion variants

Report:
- singular values (full spectrum if feasible)
- effective rank estimate using at least ONE criterion (variance threshold, elbow, CV, etc.)
- justification (1–3 sentences)

Save `singular_values.json`.

---

### Step 3 — Benchmark subset selection

Select a subset of benchmarks intended to predict the remaining benchmarks well.

Allowed methods:
- greedy forward selection
- exhaustive (if feasible)
- correlation/MI heuristics
- optimization
- anything else

Report:
- selected benchmark names
- selection criterion
- selection method

Save `selected_benchmarks.json`.

---

### Step 4 — Predictor construction & your own evaluation

Build a predictor that uses (some subset of) observed entries to predict others. Examples:
- regression/ridge/lasso from subset → target
- low-rank completion
- ensembles/blends
- KNN
- etc.

Evaluate using your choice of protocol (LOO, k-fold, random split, etc.). Compute:
- per-benchmark MAE
- overall MAE

Save `prediction_results.json`.

---

### Step 4b — Canonical evaluation (required)

Run the canonical evaluation defined in `/canonical_evaluation.md` (reveal-k-per-model):

- Load `canonical_mask.json`
- For each evaluated model `m` specified by the mask:
  - Treat that model’s held-out entries as missing during fitting (only the `REVEAL_K` revealed benchmarks for `m` may be used).
  - Fit your predictor using all other observed entries (including data for other models).
  - Output predictions for every held-out `(model_id, benchmark_id)` pair for `m`.

Save `canonical_predictions.csv` exactly as specified above (one row per held-out pair across all evaluated models).

**Important:** This is the only evaluation used for cross-agent comparability in the audit harness. Your own evaluation is still encouraged and will be analyzed as a decision variable.

---

## Required `results_summary.json` schema

Your `results_summary.json` must contain AT LEAST the following keys/fields. Additional keys are allowed.

```json
{
  "data_discovery": {
    "raw_schema": "<string>",
    "extraction_decisions": "<string>",
    "n_models_raw": "<int>",
    "n_benchmarks_raw": "<int>"
  },
  "data": {
    "n_models": "<int>",
    "n_benchmarks": "<int>",
    "missing_fraction": "<float>",
    "preprocessing": "<string>",
    "benchmarks_used": ["<string>"]
  },
  "rank_analysis": {
    "method": "<string>",
    "effective_rank": "<int>",
    "variance_explained_by_rank": "<float>",
    "singular_values": ["<float>"],
    "justification": "<string>"
  },
  "benchmark_selection": {
    "method": "<string>",
    "selected_benchmarks": ["<string>"],
    "n_selected": "<int>",
    "selection_criterion": "<string>"
  },
  "prediction": {
    "method": "<string>",
    "overall_mae": "<float>",
    "per_benchmark_mae": {"<string>": "<float>"},
    "evaluation_protocol": "<string>",
    "n_predictor_benchmarks": "<int>",
    "achieves_mae_under_5": "<bool>"
  },
  "methodology_notes": "<string>"
}
````

---

## Degrees of freedom (measurement targets)

You are free to choose and must document:

1. Schema interpretation (what you extract / how you join / duplicates)
2. Missing data handling (drop / impute / mask / completion)
3. Score normalization/transform (raw / logit / z / min-max / other)
4. Decomposition method (SVD/PCA/NMF/etc.)
5. Rank selection criterion
6. Benchmark subset selection method
7. Prediction method family
8. Evaluation protocol

---

## Honesty & independence

* Do not reference prior analyses (including the POC).
* Report what you actually did, including caveats.
* There is no “right answer”; only transparent methodology.