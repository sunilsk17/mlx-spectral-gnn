"""
Generate publication-quality plots from experiment results.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_results(filename):
    """Load results from JSON file."""
    path = os.path.join(config.RESULTS_DIR, f"{filename}.json")
    with open(path, "r") as f:
        return json.load(f)


def plot_accuracy_vs_pruning(sweep_results, dataset_name, save=True):
    """
    Plot accuracy vs pruning ratio for the spectral method.
    """
    data = [r for r in sweep_results if r["dataset"] == dataset_name]
    
    ratios = [r["prune_ratio"] for r in data]
    accs = [r["acc_mean"] for r in data]
    stds = [r["acc_std"] for r in data]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ratios, accs, yerr=stds, marker='o', capsize=5,
                linewidth=2, markersize=8, color='#2196F3',
                label='Spectral Pruning')
    
    # Reference line for baseline (ratio=0)
    if data:
        baseline_acc = data[0]["acc_mean"]
        ax.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.7,
                   label=f'Full Graph ({baseline_acc:.4f})')
    
    ax.set_xlabel("Pruning Ratio")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Accuracy vs. Pruning Ratio — {dataset_name.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    if save:
        path = os.path.join(config.RESULTS_DIR,
                           f"accuracy_vs_pruning_{dataset_name}.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    
    plt.close(fig)
    return fig


def plot_memory_vs_pruning(sweep_results, dataset_name, save=True):
    """
    Plot memory usage vs pruning ratio.
    """
    data = [r for r in sweep_results if r["dataset"] == dataset_name]
    
    ratios = [r["prune_ratio"] for r in data]
    mems = [r["mem_mean"] for r in data]
    stds = [r["mem_std"] for r in data]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ratios, mems, yerr=stds, marker='s', capsize=5,
                linewidth=2, markersize=8, color='#4CAF50',
                label='Peak Memory')
    
    ax.set_xlabel("Pruning Ratio")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title(f"Memory Usage vs. Pruning Ratio — {dataset_name.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save:
        path = os.path.join(config.RESULTS_DIR,
                           f"memory_vs_pruning_{dataset_name}.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    
    plt.close(fig)
    return fig


def plot_method_comparison(comparison_results, dataset_name, save=True):
    """
    Bar chart comparing all methods at fixed pruning ratio.
    """
    data = [r for r in comparison_results if r["dataset"] == dataset_name]
    
    methods = [r["prune_method"] for r in data]
    accs = [r["acc_mean"] for r in data]
    acc_stds = [r["acc_std"] for r in data]
    
    method_labels = {
        "none": "Full Graph",
        "random": "Random\n(DropEdge)",
        "degree": "Degree\nBased",
        "spectral": "Spectral\n(Ours)"
    }
    
    colors = ['#9E9E9E', '#FF9800', '#F44336', '#2196F3']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, accs, yerr=acc_stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in methods])
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Method Comparison — {dataset_name.upper()} (50% Pruning)")
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Add value labels on bars
    for bar, acc, std in zip(bars, accs, acc_stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    
    if save:
        path = os.path.join(config.RESULTS_DIR,
                           f"method_comparison_{dataset_name}.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    
    plt.close(fig)
    return fig


def plot_accuracy_memory_tradeoff(sweep_results, dataset_name, save=True):
    """
    Scatter plot: accuracy vs memory (Pareto frontier style).
    """
    data = [r for r in sweep_results if r["dataset"] == dataset_name]
    
    mems = [r["mem_mean"] for r in data]
    accs = [r["acc_mean"] for r in data]
    ratios = [r["prune_ratio"] for r in data]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(mems, accs, c=ratios, cmap='coolwarm',
                        s=100, edgecolors='black', linewidth=0.5, zorder=5)
    
    # Connect points with line
    ax.plot(mems, accs, '--', color='gray', alpha=0.5, zorder=1)
    
    # Annotate points with ratio
    for mem, acc, ratio in zip(mems, accs, ratios):
        ax.annotate(f'{ratio:.1f}', (mem, acc),
                   textcoords="offset points", xytext=(8, 5),
                   fontsize=9)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pruning Ratio')
    
    ax.set_xlabel("Peak Memory (MB)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Accuracy–Memory Tradeoff — {dataset_name.upper()}")
    ax.grid(True, alpha=0.3)
    
    if save:
        path = os.path.join(config.RESULTS_DIR,
                           f"acc_mem_tradeoff_{dataset_name}.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    
    plt.close(fig)
    return fig


def generate_all_plots(datasets=None):
    """Generate all plots from saved results."""
    if datasets is None:
        datasets = config.DATASETS
    
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    for dataset_name in datasets:
        print(f"\n  Dataset: {dataset_name.upper()}")
        
        # Load results
        try:
            sweep_results = load_results(f"sweep_{dataset_name}")
            comparison_results = load_results(f"comparison_{dataset_name}")
        except FileNotFoundError as e:
            print(f"  ⚠ Results not found for {dataset_name}: {e}")
            continue
        
        # Generate plots
        plot_accuracy_vs_pruning(sweep_results, dataset_name)
        plot_memory_vs_pruning(sweep_results, dataset_name)
        plot_method_comparison(comparison_results, dataset_name)
        plot_accuracy_memory_tradeoff(sweep_results, dataset_name)
    
    print("\n  All plots generated!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate experiment plots")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()
    
    if args.dataset:
        generate_all_plots([args.dataset])
    else:
        generate_all_plots()
