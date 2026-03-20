# Hardware-Aware Spectral Sparsification for Apple Silicon (MLX)

A purely deterministic, hardware-aware structural graph sparsification pipeline natively designed to maximize computation on Apple Silicon's Unified Memory Architecture using the [Apple MLX framework](https://github.com/ml-explore/mlx). This code accompanies the research paper: *"Hardware-Aware Spectral Sparsification for Efficient GNN Training under Memory Constraints."*

## Overview
Traditional topological scaling algorithms (e.g., DropEdge) randomly drop edges blindly. This framework structurally evaluates the squared Euclidean distances within a low-dimensional spectral space derived directly from the graph Laplacian ($L = D - A$). It intelligently drops strictly redundant intra-community multi-edges while preserving crucial structural bridges between communities.

Because the sparse ARPACK solver executes directly on the CPU while MLX relies on Unified Memory Architecture, the dynamically calculated boundaries are instantly accessible to the GPU via **zero-copy memory transfer**—eliminating the severe PCIe-bottleneck penalty associated with classical spectral methods on discrete hardware.

## Installation & Requirements

Ensure you have a modern Python 3.11+ environment configured.
```bash
pip install -r requirements.txt
```

## Reproducing the Paper

You can automatically reproduce all tables, figures, and metrics mapping the memory-accuracy scaling frontier directly on Apple Silicon.

```bash
# Run all experiments for Cora and Citeseer natively
python main.py

# Run a single dataset
python main.py --dataset citeseer

# Run an ultra-fast sanity check
python main.py --quick
```

All experimental outputs (raw `.csv`/`.json` data and `.png` plots) are automatically pushed into the `results/` folder. 

## Code Structure
*   `prune.py`: The core Laplacian eigen-decomposition framework utilizing sparse `eigsh` shift-invert modes.
*   `train.py`: The native MLX node-classification loop actively demonstrating GPU synchronization.
*   `models/gcn.py`: A native 2-layer MLX-based Graph Convolutional Network built for Sparse-Dense Matrix Multiplication (SpMM).

## License
Provided under the [MIT License](LICENSE).
