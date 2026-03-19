# /canonical_evaluation.md
# BenchPress Reliability Audit: Canonical Evaluation Protocol (v1.0)

**Purpose:** Provide one standardized, leakage-minimized evaluation (under agent compliance) that all agents can be scored on, independent of their self-chosen evaluation protocol.

This protocol defines:
1) how to construct ground truth targets,
2) a deterministic reveal-k holdout mask,
3) how agents must produce canonical predictions,
4) how MAE is computed.

---

## 1. Ground truth matrix construction

From `llm_benchmark_data.json`:

- Extract models, benchmarks, and score entries.
- If multiple scores exist for the same `(model_id, benchmark_id)`, resolve duplicates by **simple average**.
- All other missing entries remain missing (not scored).

Define the set of observed cells:

$$\Omega = \{(m,b): y_{m,b}\ \text{is observed}\}$$

---

## 2. Canonical normalization (0–100 per benchmark)

To compare across mixed metrics (percentages + rating scales), evaluation is done on a per-benchmark normalized scale:

For each benchmark b:
- compute `min_b = min_{(m,b)∈Ω} y_{m,b}`
- compute `max_b = max_{(m,b)∈Ω} y_{m,b}`
- define `range_b = max(max_b - min_b, 1e-9)`

Normalize:

$$\tilde{y}_{m,b} = 100 \cdot \frac{y_{m,b} - \min_b}{\text{range}_b}$$

Apply the same transform to predictions $\hat{y}$.

(Do not clip by default; report if large out-of-range predictions occur.)

---

## 3. Canonical holdout mask (reveal-k per model)

### 3.1 Constants
Let:
- `CANONICAL_SEED = 20260226`
- `REVEAL_K = 5` (benchmarks revealed per evaluated model)
- `N_EVAL_MODELS = 12` (number of evaluated models)
- `MIN_CELLS_PER_MODEL_TO_EVAL = 15` (eligibility threshold)

### 3.2 Deterministic mask generation
We construct a held-out test set that matches the “reveal k benchmarks for a model, predict the rest” protocol.

1) For each model m, define its observed benchmark set:
$$B(m) = \\{b : (m,b) \\in \\Omega\\}$$

2) Eligible models:
$$M_{\\text{eligible}} = \\{m : |B(m)| \\ge \\texttt{MIN_CELLS_PER_MODEL_TO_EVAL}\\}$$

3) Evaluated models:
- Sort `M_eligible` lexicographically.
- Sample `N_EVAL_MODELS` without replacement using an RNG seeded by `CANONICAL_SEED`.
- If `|M_eligible| < N_EVAL_MODELS`, use all eligible models.

4) For each evaluated model m:
- Sample `REVEAL_K` benchmarks uniformly from `B(m)` using a deterministic RNG seeded by `(CANONICAL_SEED, m)` (e.g., SHA-256 hash → integer seed).
- Call this revealed set `R(m)`.

5) Define per-model heldout set:
$$\\Omega_{\\text{test}}(m) = \\{(m,b) \\in \\Omega : b \\notin R(m)\\}$$

6) Canonical heldout set:
$$\\Omega_{\\text{test}} = \\bigcup_{m \\in M_{\\text{eval}}} \\Omega_{\\text{test}}(m)$$

### 3.3 File format
The harness provides `canonical_mask.json` with:
```json
{
  "seed": 20260226,
  "reveal_k": 5,
  "n_eval_models": 12,
  "min_cells_per_model_to_eval": 15,
  "eval_models": ["..."],
  "revealed": [
    {"model_id": "...", "benchmark_ids": ["...", "...", "...", "...", "..."]},
    ...
  ],
  "pairs": [
    {"model_id": "...", "benchmark_id": "..."},
    ...
  ]
}
````

* `pairs` is the full list of held-out pairs (the union of all `Ω_test(m)`).

---

## 4. What analysis agents must do

Agents must produce predictions under the **reveal-k-per-model** rule:

For each evaluated model `m`:

* Treat all pairs in `Ω_test(m)` as **missing during fitting**.
* Fit your predictor using all other observed entries (including any held-out pairs for other models).
* Output predictions for every held-out pair in `Ω_test(m)`.

You may refit per model or reuse computation, but **must not** use any held-out entries for the model being predicted.

### Required output: `canonical_predictions.csv`

Must contain one row per held-out pair with columns:

* `model_id`
* `model_name`
* `benchmark_id`
* `benchmark_name`
* `y_pred`  (prediction in *raw* units; normalization happens in scoring)

Coverage requirement (for SUCCESS): predictions for ≥95% of held-out pairs.

---

## 5. Scoring

For each held-out pair $(m,b) \in \Omega_{\text{test}}$ where a prediction is present:

Compute normalized absolute error:

$$e_{m,b} = |\tilde{y}_{m,b} - \tilde{\hat{y}}_{m,b}|$$

### Overall MAE

$$\mathrm{MAE}_{\text{canonical}} = \frac{1}{|\Omega_{\text{scored}}|}\sum_{(m,b)\in \Omega_{\text{scored}}} e_{m,b}$$

### Per-benchmark MAE

For each benchmark $b$ with at least 1 scored pair:

$$\mathrm{MAE}_b = \frac{1}{|\Omega_{b,\text{scored}}|}\sum_{(m,b)\in \Omega_{b,\text{scored}}} e_{m,b}$$

### Coverage

$$\mathrm{coverage} = \frac{|\Omega_{\text{scored}}|}{|\Omega_{\text{test}}|}$$

Report out-of-range predictions (where `y_pred` is far outside observed min/max) as diagnostics, but do not automatically clip unless explicitly stated in the final report.

---

## 6. Rationale

* Reveal-k-per-model directly matches the practical question: “given k benchmark scores for a *new* model, how well can you predict the rest?”
* Per-benchmark min-max normalization yields a common 0–100 scale even when raw metrics differ (e.g., ratings vs % correct).
* Deterministic seeding fixes the evaluated models and revealed benchmarks, enabling cross-agent comparability.
