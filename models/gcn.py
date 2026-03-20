"""
Two-layer Graph Convolutional Network (GCN) using Apple MLX.

Implements the standard GCN from Kipf & Welling (2017):
  H^{(l+1)} = σ( Â · H^{(l)} · W^{(l)} )
where Â = D^{-1/2}(A + I)D^{-1/2}
"""

import mlx.core as mx
import mlx.nn as nn


class GCN(nn.Module):
    """
    Two-layer GCN for node classification.
    
    Architecture:
        Input -> GraphConv -> ReLU -> Dropout -> GraphConv -> Output
    where GraphConv(H) = Â · H · W + b
    """
    
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout_rate = dropout
    
    def __call__(self, X, A_norm, training=True):
        """
        Forward pass.
        
        Args:
            X: node features [num_nodes, in_features] — mx.array
            A_norm: normalized adjacency [num_nodes, num_nodes] — mx.array
            training: whether to apply dropout
            
        Returns:
            logits: [num_nodes, num_classes]
        """
        # Layer 1: Aggregate neighbors, then project
        # H = σ(Â · X · W1 + b1)
        H = A_norm @ X           # Message passing: aggregate neighbor features
        H = self.fc1(H)          # Linear projection
        H = nn.relu(H)           # Non-linearity
        
        # Dropout
        if training and self.dropout_rate > 0:
            # Inverted dropout: scale by 1/(1-p) during training
            mask = mx.random.uniform(shape=H.shape) > self.dropout_rate
            H = H * mask / (1.0 - self.dropout_rate)
        
        # Layer 2: Aggregate neighbors, then project
        # Out = Â · H · W2 + b2
        H = A_norm @ H           # Message passing
        H = self.fc2(H)          # Linear projection → logits
        
        return H
