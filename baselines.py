"""
Baseline pruning methods for comparison:
  1. Random edge drop (DropEdge)
  2. Degree-based pruning
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix


def random_prune(adj, prune_ratio=0.5, seed=None):
    """
    Randomly remove a fraction of edges (DropEdge baseline).
    
    Args:
        adj: scipy sparse adjacency matrix
        prune_ratio: fraction of edges to remove
        seed: random seed for reproducibility
    
    Returns:
        adj_pruned: pruned adjacency matrix
    """
    if prune_ratio <= 0.0:
        return adj.copy()
    
    rng = np.random.RandomState(seed)
    
    G = nx.from_scipy_sparse_array(adj)
    edges = list(G.edges())
    num_edges = len(edges)
    num_remove = int(prune_ratio * num_edges)
    
    # Randomly select edges to remove
    remove_indices = rng.choice(num_edges, size=num_remove, replace=False)
    remove_set = set(remove_indices)
    
    keep_edges = [e for i, e in enumerate(edges) if i not in remove_set]
    
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    G_pruned.add_edges_from(keep_edges)
    
    return nx.adjacency_matrix(G_pruned).astype(float)


def degree_prune(adj, prune_ratio=0.5):
    """
    Degree-based pruning: remove edges connecting high-degree node pairs first.
    
    The intuition is that high-degree nodes have redundant connections,
    so removing edges between them causes less information loss.
    
    Edge score = min(deg(u), deg(v))  — higher score = more redundant
    
    Args:
        adj: scipy sparse adjacency matrix
        prune_ratio: fraction of edges to remove
    
    Returns:
        adj_pruned: pruned adjacency matrix
    """
    if prune_ratio <= 0.0:
        return adj.copy()
    
    G = nx.from_scipy_sparse_array(adj)
    degrees = dict(G.degree())
    
    # Score each edge by minimum degree of endpoints
    edge_scores = []
    for u, v in G.edges():
        score = min(degrees[u], degrees[v])
        edge_scores.append(((u, v), score))
    
    # Sort by score DESCENDING (remove highest-score edges first)
    edge_scores.sort(key=lambda x: -x[1])
    
    num_remove = int(prune_ratio * len(edge_scores))
    keep_edges = [e for e, _ in edge_scores[num_remove:]]
    
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes())
    G_pruned.add_edges_from(keep_edges)
    
    return nx.adjacency_matrix(G_pruned).astype(float)


def no_prune(adj, **kwargs):
    """No pruning baseline — returns original adjacency."""
    return adj.copy()
