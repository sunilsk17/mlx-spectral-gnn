"""
Data loading utilities for Cora, Citeseer, and Karate Club datasets.
Loads from the standard Planetoid pickle format (ind.dataset.*).
"""

import os
import pickle
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.graph_utils import normalize_features


def _parse_index_file(filename):
    """Parse an index file (e.g., ind.cora.test.index)."""
    index = []
    with open(filename, "r") as f:
        for line in f:
            index.append(int(line.strip()))
    return index


def _load_pickle(filename):
    """Load a pickle file, handling both Python 2 and 3 formats."""
    with open(filename, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def load_planetoid(dataset_name):
    """
    Load a Planetoid dataset (Cora, Citeseer, PubMed) from the local data dir.
    
    Returns:
        adj: scipy sparse CSR adjacency matrix
        features: np.ndarray of shape [num_nodes, num_features]
        labels: np.ndarray of shape [num_nodes]
        train_mask: np.ndarray boolean mask
        val_mask: np.ndarray boolean mask
        test_mask: np.ndarray boolean mask
    """
    dataset_name = dataset_name.lower()
    data_dir = config.PLANETOID_DATA_DIR
    
    prefix = os.path.join(data_dir, f"ind.{dataset_name}")
    
    # Load the 6 main files
    x = _load_pickle(f"{prefix}.x")         # training features (sparse)
    y = _load_pickle(f"{prefix}.y")         # training labels (one-hot)
    allx = _load_pickle(f"{prefix}.allx")   # all features except test (sparse)
    ally = _load_pickle(f"{prefix}.ally")   # all labels except test (one-hot)
    tx = _load_pickle(f"{prefix}.tx")       # test features (sparse)
    ty = _load_pickle(f"{prefix}.ty")       # test labels (one-hot)
    graph = _load_pickle(f"{prefix}.graph") # adjacency dict
    test_index = _parse_index_file(f"{prefix}.test.index")
    
    # Convert sparse matrices to dense numpy arrays
    if hasattr(x, 'toarray'):
        x = x.toarray()
    if hasattr(allx, 'toarray'):
        allx = allx.toarray()
    if hasattr(tx, 'toarray'):
        tx = tx.toarray()
    
    # Calculate total number of nodes (some might be missing between allx and test nodes)
    num_nodes = max(max(test_index) + 1, len(allx) + len(tx))
    
    # Initialize full feature and label matrices
    features = np.zeros((num_nodes, allx.shape[1]))
    labels_onehot = np.zeros((num_nodes, ally.shape[1]))
    
    # Nodes 0..len(allx)-1 are already in place
    features[:len(allx)] = allx
    labels_onehot[:len(ally)] = ally
    
    # Place test nodes at their exact indices
    for i, idx in enumerate(test_index):
        features[idx] = tx[i]
        labels_onehot[idx] = ty[i]
    
    # Convert one-hot labels to class indices
    labels = np.argmax(labels_onehot, axis=1)
    
    # Row-normalize features
    features = normalize_features(features)
    
    num_nodes = features.shape[0]
    num_train = x.shape[0]       # 140 for Cora
    num_val = 500                 # Standard split
    num_test = len(test_index)
    
    # Build adjacency matrix from graph dict
    edges_src = []
    edges_dst = []
    for src, neighbors in graph.items():
        for dst in neighbors:
            if src < num_nodes and dst < num_nodes:
                edges_src.append(src)
                edges_dst.append(dst)
    
    adj = csr_matrix(
        (np.ones(len(edges_src)), (edges_src, edges_dst)),
        shape=(num_nodes, num_nodes)
    )
    # Make symmetric
    adj = adj + adj.T
    adj[adj > 1] = 1
    
    # Create masks
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    
    train_mask[:num_train] = True
    val_mask[num_train:num_train + num_val] = True
    for idx in test_index:
        if idx < num_nodes:
            test_mask[idx] = True
    
    print(f"[{dataset_name.upper()}] Loaded: {num_nodes} nodes, "
          f"{adj.nnz // 2} edges, {features.shape[1]} features, "
          f"{labels.max() + 1} classes")
    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, "
          f"Test: {test_mask.sum()}")
    
    return adj, features, labels, train_mask, val_mask, test_mask


def load_karate():
    """
    Load the Zachary Karate Club graph.
    
    Returns same format as load_planetoid but with identity features
    and a simple random train/val/test split.
    """
    G = nx.karate_club_graph()
    num_nodes = G.number_of_nodes()
    
    adj = nx.adjacency_matrix(G).astype(float)
    
    # Use identity features (no real features available)
    features = np.eye(num_nodes)
    
    # Labels: community membership
    labels = np.array([
        G.nodes[i].get("club", "Mr. Hi") == "Officer"
        for i in range(num_nodes)
    ], dtype=int)
    
    # Simple split: 50% train, 25% val, 25% test
    np.random.seed(42)
    perm = np.random.permutation(num_nodes)
    n_train = num_nodes // 2
    n_val = num_nodes // 4
    
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    
    print(f"[KARATE] Loaded: {num_nodes} nodes, {G.number_of_edges()} edges, "
          f"{features.shape[1]} features, {len(set(labels))} classes")
    
    return adj, features, labels, train_mask, val_mask, test_mask


def load_dataset(name):
    """Load a dataset by name."""
    name = name.lower()
    if name == "karate":
        return load_karate()
    elif name in ("cora", "citeseer", "pubmed"):
        return load_planetoid(name)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def adj_to_networkx(adj):
    """Convert a scipy sparse adjacency matrix to a NetworkX graph."""
    G = nx.from_scipy_sparse_array(adj)
    return G


def networkx_to_adj(G, num_nodes=None):
    """Convert a NetworkX graph back to scipy sparse adjacency matrix."""
    if num_nodes is None:
        num_nodes = G.number_of_nodes()
    adj = nx.adjacency_matrix(G)
    # Ensure correct size if some nodes were dropped
    if adj.shape[0] < num_nodes:
        adj.resize((num_nodes, num_nodes))
    return adj.astype(float)
