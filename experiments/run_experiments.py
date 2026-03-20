"""
Experiment Runner — Automates all experiments for the paper.

Experiments:
  1. Pruning ratio sweep (accuracy/memory/time vs prune ratio)
  2. Method comparison at fixed prune ratio
  3. Adaptive pruning demonstration
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.data_loader import load_dataset, adj_to_networkx, networkx_to_adj
from utils.memory import get_system_info
from prune import spectral_prune_from_adj, adaptive_spectral_prune
from baselines import random_prune, degree_prune, no_prune
from train import train_and_evaluate


def run_single_experiment(adj_pruned, original_adj, features, labels, train_mask, val_mask, test_mask,
                          prune_method, prune_ratio, seed, dataset_name,
                          verbose=False):
    """
    Run a single experiment: train → evaluate on a provided pre-pruned graph.
    
    Returns dict of results.
    """
    # Count edges
    original_edges = original_adj.nnz // 2
    pruned_edges = adj_pruned.nnz // 2
    actual_ratio = 1.0 - (pruned_edges / max(original_edges, 1))
    
    # Train and evaluate
    results = train_and_evaluate(
        adj_pruned, features, labels, train_mask, val_mask, test_mask,
        hidden_dim=config.HIDDEN_DIM,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        dropout=config.DROPOUT,
        seed=seed,
        verbose=verbose
    )
    
    results.update({
        "dataset": dataset_name,
        "prune_method": prune_method,
        "target_prune_ratio": prune_ratio,
        "actual_prune_ratio": actual_ratio,
        "original_edges": original_edges,
        "pruned_edges": pruned_edges,
        "seed": seed,
    })
    
    return results


def experiment_pruning_ratio_sweep(dataset_name, verbose=True):
    """
    Experiment 1: Sweep prune ratios for spectral method.
    Shows accuracy vs. pruning ratio trade-off.
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1: Pruning Ratio Sweep — {dataset_name.upper()}")
    print(f"{'='*70}")
    
    adj, features, labels, train_mask, val_mask, test_mask = load_dataset(dataset_name)
    
    all_results = []
    
    for ratio in config.PRUNE_RATIOS:
        print(f"\n  [Spectral] Computing pruned graph for ratio={ratio:.1f}...")
        if ratio == 0.0:
            adj_pruned = no_prune(adj)
        else:
            adj_pruned = spectral_prune_from_adj(adj, k=config.SPECTRAL_K, prune_ratio=ratio)
            
        seed_results = []
        for seed in config.SEEDS:
            print(f"    [Spectral] ratio={ratio:.1f}, seed={seed}")
            result = run_single_experiment(
                adj_pruned, adj, features, labels, train_mask, val_mask, test_mask,
                prune_method="spectral",
                prune_ratio=ratio,
                seed=seed,
                dataset_name=dataset_name,
                verbose=False
            )
            seed_results.append(result)
            print(f"    → acc={result['test_acc']:.4f}, "
                  f"mem={result['peak_memory_mb']:.1f}MB, "
                  f"time={result['training_time_s']:.2f}s, "
                  f"edges={result['pruned_edges']}")
        
        # Aggregate
        accs = [r["test_acc"] for r in seed_results]
        mems = [r["peak_memory_mb"] for r in seed_results]
        times = [r["training_time_s"] for r in seed_results]
        
        summary = {
            "dataset": dataset_name,
            "prune_method": "spectral",
            "prune_ratio": ratio,
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "mem_mean": np.mean(mems),
            "mem_std": np.std(mems),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "edges_remaining": seed_results[0]["pruned_edges"],
        }
        all_results.append(summary)
        
        print(f"\n  Summary ratio={ratio:.1f}: "
              f"acc={summary['acc_mean']:.4f}±{summary['acc_std']:.4f}, "
              f"mem={summary['mem_mean']:.1f}MB, "
              f"time={summary['time_mean']:.2f}s")
    
    return all_results


def experiment_method_comparison(dataset_name, prune_ratio=0.5, verbose=True):
    """
    Experiment 2: Compare all methods at a fixed prune ratio.
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: Method Comparison — {dataset_name.upper()} "
          f"(ratio={prune_ratio})")
    print(f"{'='*70}")
    
    adj, features, labels, train_mask, val_mask, test_mask = load_dataset(dataset_name)
    
    methods = ["none", "random", "degree", "spectral"]
    all_results = []
    
    for method in methods:
        ratio = 0.0 if method == "none" else prune_ratio
        
        # Precompute parsed graph if method is deterministic
        adj_pruned = None
        if method == "none":
            adj_pruned = no_prune(adj)
        elif method == "degree":
            adj_pruned = degree_prune(adj, prune_ratio=ratio)
        elif method == "spectral":
            print(f"\n  [{method.upper()}] Computing pruned graph for ratio={ratio:.1f}...")
            adj_pruned = spectral_prune_from_adj(adj, k=config.SPECTRAL_K, prune_ratio=ratio)
            
        seed_results = []
        for seed in config.SEEDS:
            # Random prune depends on seed
            if method == "random":
                current_adj_pruned = random_prune(adj, prune_ratio=ratio, seed=seed)
            else:
                current_adj_pruned = adj_pruned
                
            print(f"  [{method.upper()}] seed={seed}")
            result = run_single_experiment(
                current_adj_pruned, adj, features, labels, train_mask, val_mask, test_mask,
                prune_method=method,
                prune_ratio=ratio,
                seed=seed,
                dataset_name=dataset_name,
                verbose=False
            )
            seed_results.append(result)
            print(f"    → acc={result['test_acc']:.4f}, "
                  f"mem={result['peak_memory_mb']:.1f}MB, "
                  f"time={result['training_time_s']:.2f}s")
        
        # Aggregate
        accs = [r["test_acc"] for r in seed_results]
        mems = [r["peak_memory_mb"] for r in seed_results]
        times = [r["training_time_s"] for r in seed_results]
        
        summary = {
            "dataset": dataset_name,
            "prune_method": method,
            "prune_ratio": ratio,
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "mem_mean": np.mean(mems),
            "mem_std": np.std(mems),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "edges_remaining": seed_results[0]["pruned_edges"],
            "original_edges": seed_results[0]["original_edges"],
        }
        all_results.append(summary)
        
        print(f"\n  Summary [{method.upper()}]: "
              f"acc={summary['acc_mean']:.4f}±{summary['acc_std']:.4f}, "
              f"mem={summary['mem_mean']:.1f}MB")
    
    return all_results


def save_results(results, filename):
    """Save results as both JSON and CSV."""
    # JSON
    json_path = os.path.join(config.RESULTS_DIR, f"{filename}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved JSON: {json_path}")
    
    # CSV
    if results:
        csv_path = os.path.join(config.RESULTS_DIR, f"{filename}.csv")
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved CSV: {csv_path}")


def print_comparison_table(results):
    """Print a publication-ready table."""
    print(f"\n{'='*80}")
    print(f"RESULTS TABLE")
    print(f"{'='*80}")
    print(f"{'Method':<12} {'Ratio':<8} {'Accuracy':<18} {'Memory (MB)':<15} "
          f"{'Time (s)':<12} {'Edges':<10}")
    print(f"{'-'*80}")
    
    for r in results:
        acc_str = f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}"
        mem_str = f"{r['mem_mean']:.1f}±{r['mem_std']:.1f}"
        time_str = f"{r['time_mean']:.2f}±{r['time_std']:.2f}"
        print(f"{r['prune_method']:<12} {r['prune_ratio']:<8.1f} {acc_str:<18} "
              f"{mem_str:<15} {time_str:<12} {r.get('edges_remaining', 'N/A'):<10}")


def run_all_experiments(datasets=None):
    """Run all experiments and save results."""
    if datasets is None:
        datasets = config.DATASETS
    
    # System info
    sys_info = get_system_info()
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    for k, v in sys_info.items():
        print(f"  {k}: {v}")
    
    all_sweep_results = []
    all_comparison_results = []
    
    for dataset_name in datasets:
        # Experiment 1: Pruning ratio sweep
        sweep_results = experiment_pruning_ratio_sweep(dataset_name)
        all_sweep_results.extend(sweep_results)
        save_results(sweep_results, f"sweep_{dataset_name}")
        print_comparison_table(sweep_results)
        
        # Experiment 2: Method comparison
        comparison_results = experiment_method_comparison(dataset_name, prune_ratio=0.5)
        all_comparison_results.extend(comparison_results)
        save_results(comparison_results, f"comparison_{dataset_name}")
        print_comparison_table(comparison_results)
    
    # Save combined results
    save_results(all_sweep_results, "all_sweep_results")
    save_results(all_comparison_results, "all_comparison_results")
    
    # Save system info
    sys_info_path = os.path.join(config.RESULTS_DIR, "system_info.json")
    with open(sys_info_path, "w") as f:
        json.dump(sys_info, f, indent=2)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    
    return all_sweep_results, all_comparison_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GNN sparsification experiments")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to use (cora, citeseer, or 'all')")
    args = parser.parse_args()
    
    if args.dataset and args.dataset != "all":
        datasets = [args.dataset]
    else:
        datasets = config.DATASETS
    
    run_all_experiments(datasets)
