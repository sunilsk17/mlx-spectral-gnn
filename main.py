"""
Main entry point for Hardware-Aware Spectral Sparsification experiments.

Usage:
  python main.py                          # Run all experiments on all datasets
  python main.py --dataset cora           # Run on Cora only
  python main.py --dataset karate --quick # Quick sanity check
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.data_loader import load_dataset, adj_to_networkx
from utils.memory import get_system_info, Timer
from prune import spectral_prune, spectral_prune_from_adj
from baselines import random_prune, degree_prune
from train import train_and_evaluate
from experiments.run_experiments import run_all_experiments
from experiments.plot import generate_all_plots


def sanity_check():
    """
    Quick sanity check on Karate Club dataset.
    Verifies that all components work end-to-end.
    """
    print("\n" + "="*70)
    print("SANITY CHECK — Karate Club")
    print("="*70)
    
    # Load data
    adj, features, labels, train_mask, val_mask, test_mask = load_dataset("karate")
    G = adj_to_networkx(adj)
    
    print(f"\n  Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test spectral pruning
    from prune import spectral_prune
    G_pruned = spectral_prune(G, k=5, prune_ratio=0.3)
    print(f"  Pruned (30%): {G_pruned.number_of_nodes()} nodes, "
          f"{G_pruned.number_of_edges()} edges")
    
    # Test training
    print("\n  Training on full graph...")
    results_full = train_and_evaluate(
        adj, features, labels, train_mask, val_mask, test_mask,
        hidden_dim=32, epochs=100, verbose=True
    )
    print(f"\n  Full graph accuracy: {results_full['test_acc']:.4f}")
    print(f"  Memory: {results_full['peak_memory_mb']:.1f} MB")
    print(f"  Time: {results_full['training_time_s']:.2f}s")
    
    # Test training on pruned graph
    from utils.data_loader import networkx_to_adj
    adj_pruned = networkx_to_adj(G_pruned, adj.shape[0])
    
    print("\n  Training on pruned graph (30%)...")
    results_pruned = train_and_evaluate(
        adj_pruned, features, labels, train_mask, val_mask, test_mask,
        hidden_dim=32, epochs=100, verbose=True
    )
    print(f"\n  Pruned graph accuracy: {results_pruned['test_acc']:.4f}")
    print(f"  Memory: {results_pruned['peak_memory_mb']:.1f} MB")
    
    print("\n" + "="*70)
    print("SANITY CHECK PASSED")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Hardware-Aware Spectral Sparsification for GNN Training"
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset: cora, citeseer, karate, or 'all'")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check on Karate Club")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing results")
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("  Hardware-Aware Spectral Sparsification for GNN Training")
    print("  on Apple Silicon (MLX)")
    print("="*70)
    
    # System info
    sys_info = get_system_info()
    for k, v in sys_info.items():
        print(f"  {k}: {v}")
    
    if args.quick:
        sanity_check()
        return
    
    if args.plot_only:
        datasets = [args.dataset] if args.dataset and args.dataset != "all" else None
        generate_all_plots(datasets)
        return
    
    # Run experiments
    datasets = None
    if args.dataset and args.dataset != "all":
        datasets = [args.dataset]
    
    run_all_experiments(datasets)
    
    # Generate plots
    generate_all_plots(datasets)
    
    print("\n  Experimentation pipeline complete. Artifacts saved to results/.")


if __name__ == "__main__":
    main()
