"""
Graph utility functions: normalization and format conversion.
"""

import numpy as np
from scipy.sparse import csr_matrix, eye as speye, diags


def normalize_adjacency(adj):
    """
    Symmetric normalization of adjacency matrix with self-loops.
    
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    
    Args:
        adj: scipy sparse adjacency matrix
        
    Returns:
        A_hat: normalized adjacency as dense numpy array
    """
    # Add self-loops
    adj_hat = adj + speye(adj.shape[0])
    
    # Degree matrix
    degrees = np.array(adj_hat.sum(axis=1)).flatten()
    
    # D^{-1/2}
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = diags(d_inv_sqrt)
    
    # Normalized adjacency
    A_norm = D_inv_sqrt @ adj_hat @ D_inv_sqrt
    
    return np.array(A_norm.todense())


def sparse_to_edge_index(adj):
    """Convert scipy sparse matrix to edge index format [2, num_edges]."""
    adj_coo = adj.tocoo()
    edge_index = np.vstack([adj_coo.row, adj_coo.col])
    return edge_index


def edge_index_to_sparse(edge_index, num_nodes):
    """Convert edge index [2, num_edges] to scipy sparse adjacency."""
    src, dst = edge_index[0], edge_index[1]
    data = np.ones(len(src))
    adj = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    return adj


def normalize_features(features):
    """
    Row-normalize feature matrix.
    
    Args:
        features: numpy array of shape [num_nodes, num_features]
        
    Returns:
        normalized_features: numpy array of shape [num_nodes, num_features]
    """
    rowsum = np.array(features.sum(axis=1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    
    if hasattr(features, 'tocsr'):
        features = r_mat_inv.dot(features)
        return features.toarray()
    else:
        features = r_mat_inv.dot(features)
        return features
