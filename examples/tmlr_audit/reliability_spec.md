# Specification: Replication Reliability Analysis

## Background

You are analyzing the outputs of K independent LLM agents that each attempted
the same data analysis task (TMLR editorial timeline audit). Each agent had
access to the OpenReview API and was instructed to compute decision timelines.
Some agents succeeded, some failed.

Your job: assess how reliable the replication is.

## Inputs

Conversation traces are located at `../conversations/`. Each is a JSON file.
Select conversations where BOTH conditions hold:

1. `start_entry_id` equals `"17708315495184560"`
2. `conversation_name` contains both "tmlr" and "audit" (case-insensitive)

Exclude any file whose path contains "backups".

The conversation JSON schema uses a tree structure:

```json
{
  "conversation_name": "TMLR Audit: opus-4.5 run 01",
  "start_entry_id": "17708315495184560",
  "current_path": [0, 1, ...],
  "root": {
    "children": [
      {
        "user": "...",
        "assistant": "...",
        "children": [...]
      }
    ]
  }
}
```

To extract the final assistant message:

1. Start at `root`
2. For each index in `current_path`, descend into `children[index]`
3. Take `assistant` from the terminal node
4. Fields can be `None` — always use `(node.get("assistant") or "")`

## Task

### Step 1: Extract outputs

Load each matching conversation file. Extract the final assistant message.
Classify each trace as:

- **SUCCESS**: Final message contains quantitative audit results (percentiles,
  compliance rates, sample sizes)
- **FAILURE**: Agent errored out, timed out, produced no analysis, or final
  message is an error string (e.g., starts with "Error:")

Report:

- `N_total`, `N_success`, `N_failure`
- For each trace: conversation filename, name, SUCCESS/FAILURE
- For each failure: one-line failure mode (e.g., "credit exhaustion", "timeout",
  "API error in final output")

### Step 2: Design binary queries

Examine the text of AT MOST 10 successful outputs. Design exactly 15 binary (yes/no)
queries that can be answered deterministically from each output's text.

**Design principle:** Each query should probe whether an agent's output
preserves specific information about the underlying data source. Two agents
that independently extracted the same finding from the data should both answer
"yes" — their agreement reflects shared information about the source, not
coincidence. An agent that missed or miscomputed a finding answers "no."

Higher TVD-MI between a pair means the pair's joint response pattern is
further from statistical independence — their answers are more informative
about each other than the marginals alone would predict. Queries where nearly
all agents agree contribute little because the agreement is predictable from
marginals (if 8/9 agents say "yes", knowing one agent's answer tells you
almost nothing about the other). The goal is queries with genuine variance:
findings that some agents extracted and others did not.

Organize queries into three tiers of 5:

**Tier 1 — Core statistics** (5 queries): Whether the agent successfully
extracted specific quantitative findings from the data. Examples of the
*kind* of query (design your own based on what you observe):

- "Does the output report [specific statistic] within [tolerance]?"
- "Does the output report [specific threshold] compliance rate?"

**Tier 2 — Methodology** (5 queries): Whether the agent made specific
analytical choices that affect what information is preserved.

- "Did the agent measure from [event A] rather than [event B]?"
- "Did the agent account for [specific confounder]?"

**Tier 3 — Presentation** (5 queries): Whether the agent preserved
specific interpretive information beyond raw numbers.

- "Does the output contextualize findings against [specific benchmark]?"
- "Does the output identify [specific structural pattern] in the data?"

Requirements:

- Queries must be answerable by string/number matching on the output text
- Do NOT use queries provided elsewhere — design them from the data
- Each query must have at least one "yes" and one "no" across agents
  (unanimous queries carry zero information)
- Aim for queries where at least 20% of agents answer differently from the
  majority (near-unanimous queries waste bits because TVD-MI cannot
  distinguish agreement-from-shared-information vs agreement-from-marginals
  when the marginals are near-degenerate)

### Step 3: Build response matrix

Apply each query to each successful agent. Produce the response matrix:

$$R \in \{0, 1\}^{Q \times N_{\text{success}}}$$

where $Q = 15$ is the number of queries and $N_{\text{success}}$ is the number
of successful agents. Print the full matrix with labeled rows (queries) and
columns (agent identifiers).

Also print per-query agreement rate: fraction of agents giving the majority
answer. Flag any query where agreement is 100% (these are uninformative and
should be replaced).

### Step 4: Compute pairwise TVD-MI

For each pair of agents $(i, j)$, estimate their mutual information under the
total variation distance.

**Estimation procedure for a single pair $(i, j)$:**

Let $\mathbf{r}_i, \mathbf{r}_j \in \{0,1\}^Q$ be their response vectors
(columns of $R$).

1. Compute the empirical joint distribution over $\{0,1\} \times \{0,1\}$:

$$\hat{P}(x, y) = \frac{1}{Q} \sum_{q=1}^{Q} \mathbf{1}[r_{q,i} = x] \cdot \mathbf{1}[r_{q,j} = y]$$

for each $(x, y) \in \{(0,0), (0,1), (1,0), (1,1)\}$.

2. Compute marginals:

$$\hat{P}_i(x) = \hat{P}(x, 0) + \hat{P}(x, 1), \quad \hat{P}_j(y) = \hat{P}(0, y) + \hat{P}(1, y)$$

3. Compute TVD-MI:

$$I_{\text{TVD}}(i; j) = \frac{1}{2} \sum_{x \in \{0,1\}} \sum_{y \in \{0,1\}} \left| \hat{P}(x, y) - \hat{P}_i(x) \cdot \hat{P}_j(y) \right|$$

Note: this is the total variation distance between the joint and product of
marginals, not the KL-based mutual information.

### Step 5: Report

Produce a structured summary containing:

1. **Success/failure breakdown** — counts, per-trace classification, failure modes
2. **Query design** — all 15 queries, grouped by tier, with brief rationale
3. **Response matrix** $R$ — full matrix with row/column labels
4. **Per-query diagnostics** — agreement rates, flag any uninformative queries
5. **Pairwise TVD-MI matrix** — $N_{\text{success}} \times N_{\text{success}}$
   symmetric matrix with zeros on the diagonal
6. **Agent welfare scores** — $w_i = \frac{1}{N_{\text{success}} - 1} \sum_{j \neq i} I_{\text{TVD}}(i; j)$
   (mean TVD-MI of agent $i$ with all others). Interpretation: $w_i$
   measures how much *shared information* agent $i$'s output has with the
   other agents. Higher $w_i$ means agent $i$'s output creates more
   informative joint patterns with peers — its responses are neither
   trivially predictable from marginals (saturation) nor uncorrelated
   (independence). In the peer prediction framework, higher welfare
   corresponds to higher payment.
7. **Overall welfare** — $W = \frac{2}{N_{\text{success}}(N_{\text{success}} - 1)} \sum_{i < j} I_{\text{TVD}}(i; j)$
   (mean of all pairwise TVD-MI values). Higher $W$ indicates the agent
   population shares more information overall.
8. **Interpretation** — natural groupings among agents, what drives variation,
   whether the successful agents converged on the same findings. Note: a
   cluster of agents that all agree on every query will have LOW within-cluster
   TVD-MI (their agreement is predictable from marginals). High between-cluster
   TVD-MI indicates the clusters disagree in structured ways. This is the
   expected pattern when a methodological fork splits agents into coherent
   subgroups.

## Environment

- Working directory is `scripts/`, conversations at `../conversations/`
- Use `scratch.py` for all code execution
- Up to 10 auto-continuation turns, 90s timeout per execution, 25k char output limit
- Dependencies available: standard library, `json`, `glob`, `os`
- Do NOT install packages or use `pandas`/`numpy` — use plain Python
- Do not hardcode any expected answers — derive everything from the data

## Calibration note

With $Q = 15$ binary observations per pair, TVD-MI estimates are noisy.
A pair of agents with identical response vectors yields $I_{\text{TVD}} = 0$.
A pair with maximally different responses yields $I_{\text{TVD}}$ up to 0.5.
Report raw numbers; do not over-interpret small differences. Focus on whether
agents cluster into qualitatively distinct groups versus forming a single
convergent cluster.