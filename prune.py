"""
Spectral Graph Pruning — Core Algorithm

Implements eigenvector-guided edge sparsification:
  1. Compute graph Laplacian L = D - A
  2. Get k smallest non-trivial eigenvectors
  3. Score each edge by spectral distance: score(u,v) = ||x_u - x_v||^2
  4. Prune low-score edges (spectrally redundant)
"""

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse


def compute_spectral_embedding(adj, k=10):
    """
    Compute a k-dimensional spectral embedding for all nodes.
    
    Args:
        adj: scipy sparse adjacency matrix (or NetworkX graph)
        k: number of eigenvectors (excluding trivial)
    
    Returns:
        X: np.ndarray of shape [num_nodes, k] — spectral node embeddings
        eigvals: the corresponding eigenvalues
    """
    if isinstance(adj, nx.Graph):
        L = nx.laplacian_matrix(adj).astype(float)
        num_nodes = adj.number_of_nodes()
    else:
        # Compute Laplacian from adjacency: L = D - A
        if issparse(adj):
            degrees = np.array(adj.sum(axis=1)).flatten()
        else:
            degrees = adj.sum(axis=1)
        from scipy.sparse import diags
        D = diags(degrees)
        L = D - adj
        L = L.astype(float)
        num_nodes = adj.shape[0]
        
    # SuperLU (used in shift-invert mode by ARPACK) requires 32-bit indices
    if issparse(L):
        L = L.tocsr()
        L.indices = L.indices.astype(np.int32)
        L.indptr = L.indptr.astype(np.int32)
    
    # Clamp k to be at most num_nodes - 2
    k = min(k, num_nodes - 2)
    
    # Get k+1 smallest eigenvalues/eigenvectors using shift-invert mode
    # Shift-invert (sigma=-1e-5) is required for convergence on disconnected graphs
    eigvals, eigvecs = eigsh(L, k=k + 1, sigma=-1e-5, tol=1e-6)
    
    # Sort by eigenvalue (should already be sorted, but be safe)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Skip the trivial eigenvector (eigenvalue ≈ 0, constant vector)
    X = eigvecs[:, 1:]  # shape: [num_nodes, k]
    eigvals = eigvals[1:]
    
    return X, eigvals


def score_edges(G, X):
    """
    Score each edge by squared spectral distance between endpoints.
    
    Args:
        G: NetworkX graph
        X: spectral embedding [num_nodes, k]
    
    Returns:
        list of ((u, v), score) tuples, sorted by score ascending
    """
    node_list = list(G.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}
    
    edge_scores = []
    for u, v in G.edges():
        i = node_index[u]
        j = node_index[v]
        score = np.sum((X[i] - X[j]) ** 2)
        edge_scores.append(((u, v), float(score)))
    
    # Sort by score (ascending: low score = redundant edges)
    edge_scores.sort(key=lambda x: x[1])
    
    return edge_scores


def spectral_prune(G, k=10, prune_ratio=0.5):
    """
    Prune edges from graph G using spectral embedding distances.
    
    Edges with low spectral distance (connecting spectrally similar nodes)
    are considered redundant and removed first.
    
    Args:
        G: NetworkX graph (undirected)
        k: number of spectral dimensions
        prune_ratio: fraction of edges to remove (0.0 to 1.0)
    
    Returns:
        G_pruned: new NetworkX graph with edges removed
    """
    if prune_ratio <= 0.0:
        return G.copy()
    
    if prune_ratio >= 1.0:
        G_empty = nx.Graph()
        G_empty.add_nodes_from(G.nodes())
        return G_empty
    
    # Step 1: Compute spectral embedding
    X, eigvals = compute_spectral_embedding(G, k=k)
    
    # Step 2: Score edges
    edge_scores = score_edges(G, X)
    
    # Step 3: Remove bottom `prune_ratio` edges
    num_remove = int(prune_ratio * len(edge_scores))
    keep_edges = [e for e, _ in edge_scores[num_remove:]]
    
    # Step 4: Build pruned graph
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    G_pruned.add_edges_from(keep_edges)
    
    return G_pruned


def spectral_prune_from_adj(adj, k=10, prune_ratio=0.5):
    """
    Convenience wrapper: prune from scipy sparse adjacency matrix.
    
    Args:
        adj: scipy sparse adjacency matrix
        k: number of spectral dimensions
        prune_ratio: fraction of edges to remove
    
    Returns:
        adj_pruned: scipy sparse adjacency matrix (pruned)
    """
    G = nx.from_scipy_sparse_array(adj)
    G_pruned = spectral_prune(G, k=k, prune_ratio=prune_ratio)
    adj_pruned = nx.adjacency_matrix(G_pruned).astype(float)
    return adj_pruned


def adaptive_spectral_prune(adj, features, k=10, memory_target_mb=None,
                             max_prune_ratio=0.8, step=0.05):
    """
    Adaptively prune edges until estimated memory usage drops below target.
    
    Estimates memory as proportional to number of non-zero entries in 
    the normalized adjacency matrix + features.
    
    Args:
        adj: scipy sparse adjacency matrix
        features: np.ndarray node features
        k: number of spectral dimensions
        memory_target_mb: target memory in MB (if None, uses 50% of full)
        max_prune_ratio: maximum prune ratio to try
        step: prune ratio increment per step
    
    Returns:
        adj_pruned: pruned adjacency matrix
        final_ratio: the prune ratio used
    """
    # Estimate full graph memory
    full_nnz = adj.nnz
    feature_mem = features.nbytes / (1024 ** 2)  # MB
    adj_mem_per_edge = 8 / (1024 ** 2)  # 8 bytes per float64 entry
    full_adj_mem = full_nnz * adj_mem_per_edge
    full_mem = full_adj_mem + feature_mem
    
    if memory_target_mb is None:
        memory_target_mb = full_mem * 0.5  # 50% reduction target
    
    # Try increasing prune ratios
    G = nx.from_scipy_sparse_array(adj)
    X, eigvals = compute_spectral_embedding(adj, k=k)
    edge_scores = score_edges(G, X)
    
    best_ratio = 0.0
    ratio = step
    
    while ratio <= max_prune_ratio:
        num_remove = int(ratio * len(edge_scores))
        remaining_edges = len(edge_scores) - num_remove
        estimated_mem = remaining_edges * 2 * adj_mem_per_edge + feature_mem
        
        if estimated_mem <= memory_target_mb:
            best_ratio = ratio
            break
        
        best_ratio = ratio
        ratio += step
    
    # Apply the found ratio
    num_remove = int(best_ratio * len(edge_scores))
    keep_edges = [e for e, _ in edge_scores[num_remove:]]
    
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    G_pruned.add_edges_from(keep_edges)
    
    adj_pruned = nx.adjacency_matrix(G_pruned).astype(float)
    
    return adj_pruned, best_ratio
