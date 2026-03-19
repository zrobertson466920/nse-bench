#!/usr/bin/env python3
"""
evaluate.py — Measure cross-agent structure from reliability evaluator outputs.

Usage:
    python evaluate.py --results-dir examples/benchpress/

Reads response_matrix.csv and welfare.csv from reliability evaluator runs,
computes TVD-MI welfare, runs a permutation test, and reports fork contributions.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_response_matrix(path):
    """Load binary response matrix from CSV.
    
    Format: rows = queries, columns = agents.
    First columns are query_id, tier, description; remaining columns are agent run IDs.
    Returns matrix with shape (n_agents, n_queries) — transposed for welfare computation.
    """
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None, [], []
    
    # Identify metadata vs agent columns
    meta_cols = {'query_id', 'tier', 'description'}
    all_cols = reader.fieldnames
    agent_ids = [c for c in all_cols if c not in meta_cols]
    query_names = [r['query_id'] for r in rows]
    
    # Build matrix: rows = queries, columns = agents, then transpose
    raw = np.zeros((len(rows), len(agent_ids)), dtype=int)
    for i, row in enumerate(rows):
        for j, aid in enumerate(agent_ids):
            val = row[aid].strip()
            raw[i, j] = 1 if val == '1' else 0
    
    # Transpose so matrix is (n_agents, n_queries)
    matrix = raw.T
    
    return matrix, agent_ids, query_names


def pairwise_tvd_mi(ri, rj, Q):
    """Compute TVD-MI between two agents' binary response vectors.
    
    TVD-MI = 0.5 * sum |P(ri,rj) - P(ri)*P(rj)| over all (ri,rj) outcomes.
    """
    joint = np.zeros((2, 2))
    for q in range(Q):
        joint[int(ri[q]), int(rj[q])] += 1
    joint /= Q
    
    pi = joint.sum(axis=1)  # marginal for i
    pj = joint.sum(axis=0)  # marginal for j
    
    indep = np.outer(pi, pj)
    return 0.5 * np.abs(joint - indep).sum()


def compute_welfare_from_responses(matrix):
    """
    Compute TVD-MI welfare from binary response matrix.
    
    matrix: (n_agents, n_queries) binary array.
    
    For each pair (i, j), compute tvd_mi(response_i, response_j).
    Agent welfare W_i = mean tvd_mi(i, j) over all j != i.
    Overall W = mean of all pairwise tvd_mi values.
    
    Returns: (overall_W, agent_welfare, tvdmi_matrix)
    """
    n_agents, n_queries = matrix.shape
    
    # Compute pairwise TVD-MI matrix
    tvdmi = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            val = pairwise_tvd_mi(matrix[i], matrix[j], n_queries)
            tvdmi[i, j] = val
            tvdmi[j, i] = val
    
    # Agent welfare: mean TVD-MI with all other agents
    agent_welfare = np.zeros(n_agents)
    for i in range(n_agents):
        agent_welfare[i] = tvdmi[i, :].sum() / (n_agents - 1) if n_agents > 1 else 0
    
    # Overall welfare: mean of upper triangle
    overall = 2 * np.triu(tvdmi, k=1).sum() / (n_agents * (n_agents - 1)) if n_agents > 1 else 0
    
    return float(overall), agent_welfare, tvdmi


def permutation_test(matrix, n_perms=5000, seed=42):
    """Run permutation test on TVD-MI welfare.
    
    Null: independently permute each query column (preserving marginals).
    """
    rng = np.random.RandomState(seed)
    
    observed_W, _, _ = compute_welfare_from_responses(matrix)
    
    null_dist = np.zeros(n_perms)
    for p in range(n_perms):
        # Permute each query column independently (axis=1 of agents x queries)
        perm_matrix = matrix.copy()
        for q in range(matrix.shape[1]):
            rng.shuffle(perm_matrix[:, q])
        null_W, _, _ = compute_welfare_from_responses(perm_matrix)
        null_dist[p] = null_W
    
    p_value = (null_dist >= observed_W).mean()
    z_score = (observed_W - null_dist.mean()) / (null_dist.std() + 1e-10)
    
    return {
        'W_observed': float(observed_W),
        'null_mean': float(null_dist.mean()),
        'null_std': float(null_dist.std()),
        'p_value': float(p_value),
        'z_score': float(z_score),
        'n_permutations': n_perms,
    }


def fork_contributions(matrix, query_names):
    """Compute ΔW for each query (how much overall welfare changes when that query is removed)."""
    overall_W, _, _ = compute_welfare_from_responses(matrix)
    
    contributions = []
    for q_idx, q_name in enumerate(query_names):
        # Remove this query and recompute welfare
        reduced = np.delete(matrix, q_idx, axis=1)
        reduced_W, _, _ = compute_welfare_from_responses(reduced)
        delta_W = overall_W - reduced_W
        
        yes_count = int(matrix[:, q_idx].sum())
        no_count = int(matrix.shape[0] - yes_count)
        
        contributions.append({
            'query': q_name,
            'delta_W': float(delta_W),
            'yes_count': yes_count,
            'no_count': no_count,
        })
    
    contributions.sort(key=lambda x: abs(x['delta_W']), reverse=True)
    return contributions


def find_evaluator_dirs(results_dir):
    """Find reliability evaluator output directories."""
    results_path = Path(results_dir)
    eval_dirs = []
    
    # Look for directories containing response_matrix.csv
    for d in sorted(results_path.rglob('*')):
        if d.is_dir():
            rm = d / 'response_matrix.csv'
            if rm.exists():
                eval_dirs.append(d)
    
    # Also check outputs/ subdirectory
    outputs = results_path / 'outputs'
    if outputs.exists():
        for d in sorted(outputs.iterdir()):
            if d.is_dir() and (d / 'response_matrix.csv').exists():
                if d not in eval_dirs:
                    eval_dirs.append(d)
    
    return eval_dirs


def main():
    parser = argparse.ArgumentParser(description='Evaluate cross-agent reliability structure')
    parser.add_argument('--results-dir', required=True, help='Directory containing evaluator outputs')
    parser.add_argument('--permutations', type=int, default=5000, help='Number of permutations for null test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default=None, help='Output JSON path (default: stdout)')
    args = parser.parse_args()
    
    eval_dirs = find_evaluator_dirs(args.results_dir)
    
    if not eval_dirs:
        print(f"No evaluator outputs found in {args.results_dir}", file=sys.stderr)
        print("Expected: directories containing response_matrix.csv", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(eval_dirs)} evaluator(s):", file=sys.stderr)
    for d in eval_dirs:
        print(f"  {d}", file=sys.stderr)
    
    results = []
    welfare_vectors = []
    
    for eval_dir in eval_dirs:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Evaluator: {eval_dir.name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        
        rm_path = eval_dir / 'response_matrix.csv'
        matrix, agent_ids, query_names = load_response_matrix(rm_path)
        
        if matrix is None:
            print(f"  Empty response matrix, skipping", file=sys.stderr)
            continue
        
        print(f"  Agents: {len(agent_ids)}, Queries: {len(query_names)}", file=sys.stderr)
        
        # Compute welfare
        overall_W, agent_W, query_tvds = compute_welfare_from_responses(matrix)
        print(f"  Overall welfare W = {overall_W:.6f}", file=sys.stderr)
        
        # Permutation test
        print(f"  Running permutation test ({args.permutations} perms)...", file=sys.stderr)
        perm_results = permutation_test(matrix, n_perms=args.permutations, seed=args.seed)
        print(f"  p-value = {perm_results['p_value']:.4f}, z = {perm_results['z_score']:.2f}", file=sys.stderr)
        
        # Fork contributions
        forks = fork_contributions(matrix, query_names)
        print(f"\n  Top 5 forks by |ΔW|:", file=sys.stderr)
        for f in forks[:5]:
            print(f"    {f['query']}: ΔW={f['delta_W']:.6f} (YES={f['yes_count']}, NO={f['no_count']})", file=sys.stderr)
        
        welfare_vectors.append(agent_W)
        
        results.append({
            'evaluator': eval_dir.name,
            'n_agents': len(agent_ids),
            'n_queries': len(query_names),
            'welfare': {
                'overall': overall_W,
                'per_agent': {aid: float(w) for aid, w in zip(agent_ids, agent_W)},
            },
            'permutation_test': perm_results,
            'top_forks': forks[:10],
        })
    
    # Cross-evaluator comparison
    if len(welfare_vectors) >= 2:
        from scipy.stats import pearsonr
        # Align welfare vectors (assume same agent ordering)
        min_len = min(len(v) for v in welfare_vectors)
        r, p = pearsonr(welfare_vectors[0][:min_len], welfare_vectors[1][:min_len])
        cross = {'welfare_correlation': float(r), 'correlation_p': float(p), 'n_common': min_len}
        print(f"\nCross-evaluator welfare correlation: r={r:.3f}, p={p:.4f}", file=sys.stderr)
        results.append({'cross_evaluator': cross})
    
    # Output
    output = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\nResults written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()