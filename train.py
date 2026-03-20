"""
Training loop for GCN using Apple MLX.
Handles forward/backward pass, evaluation, and metric collection.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gcn import GCN
from utils.graph_utils import normalize_adjacency
from utils.memory import MemoryTracker, Timer


def cross_entropy_loss(logits, labels, mask):
    """
    Masked cross-entropy loss for node classification.
    
    Args:
        logits: [num_nodes, num_classes]
        labels: [num_nodes] integer class labels
        mask: [num_nodes] boolean mask for which nodes to compute loss
    """
    # Softmax + cross entropy
    log_probs = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    
    num_classes = logits.shape[1]
    
    # One-hot encode labels using identity matrix indexing
    eye = mx.eye(num_classes)
    labels_onehot = eye[labels]
    
    # Compute per-node loss
    loss_per_node = -mx.sum(labels_onehot * log_probs, axis=1)
    
    # Apply mask
    mask_float = mask.astype(mx.float32)
    loss = mx.sum(loss_per_node * mask_float) / mx.maximum(mx.sum(mask_float), mx.array(1.0))
    
    return loss


def compute_accuracy(logits, labels, mask):
    """Compute classification accuracy on masked nodes."""
    predictions = mx.argmax(logits, axis=1)
    correct = (predictions == labels).astype(mx.float32)
    mask_float = mask.astype(mx.float32)
    accuracy = mx.sum(correct * mask_float) / mx.maximum(mx.sum(mask_float), mx.array(1.0))
    return accuracy.item()


def train_and_evaluate(adj, features, labels, train_mask, val_mask, test_mask,
                       hidden_dim=64, epochs=200, lr=0.01, weight_decay=5e-4,
                       dropout=0.5, seed=42, verbose=True):
    """
    Executes training and evaluation synchronization.
    
    Args:
        adj: scipy sparse adjacency matrix
        features: np.ndarray [num_nodes, num_features]
        labels: np.ndarray [num_nodes] integer labels
        train_mask, val_mask, test_mask: np.ndarray boolean masks
        hidden_dim: GCN hidden layer dimension
        epochs: number of training epochs
        lr: learning rate
        weight_decay: L2 regularization
        dropout: dropout rate
        seed: random seed
        verbose: print training progress
    
    Returns:
        dict with: test_acc, val_acc, train_acc, peak_memory_mb, 
                   training_time_s, loss_history
    """
    # Set seed
    np.random.seed(seed)
    mx.random.seed(seed)
    
    # Memory and time tracking
    mem_tracker = MemoryTracker()
    timer = Timer()
    
    mem_tracker.start()
    timer.start()
    
    # Normalize adjacency
    A_norm_np = normalize_adjacency(adj)
    
    # Convert to MLX arrays
    X = mx.array(features.astype(np.float32))
    A_norm = mx.array(A_norm_np.astype(np.float32))
    y = mx.array(labels.astype(np.int32))
    mask_train = mx.array(train_mask)
    mask_val = mx.array(val_mask)
    mask_test = mx.array(test_mask)
    
    # Model
    num_features = features.shape[1]
    num_classes = int(labels.max()) + 1
    model = GCN(num_features, hidden_dim, num_classes, dropout=dropout)
    mx.eval(model.parameters())
    
    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)
    
    # Loss function with weight decay
    def loss_fn(model, X, A_norm, y, mask):
        logits = model(X, A_norm, training=True)
        ce_loss = cross_entropy_loss(logits, y, mask)
        
        # L2 regularization on weights (standard GCN only applies to first layer)
        l2_reg = mx.sum(model.fc1.weight ** 2)
        
        return ce_loss + weight_decay * l2_reg
    
    # Training function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    loss_history = []
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        # Forward + backward
        loss, grads = loss_and_grad_fn(model, X, A_norm, y, mask_train)
        optimizer.update(model, grads)
        
        # Synchronize MLX lazy evaluation queue
        mx.eval(model.parameters(), optimizer.state)
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # Track memory
        mem_tracker.update()
        
        # Evaluate periodically
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logits = model(X, A_norm, training=False)
            mx.eval(logits)
            
            train_acc = compute_accuracy(logits, y, mask_train)
            val_acc = compute_accuracy(logits, y, mask_val)
            test_acc = compute_accuracy(logits, y, mask_test)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            
            if verbose and ((epoch + 1) % 50 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch+1:3d}: loss={loss_val:.4f}  "
                      f"train={train_acc:.4f}  val={val_acc:.4f}  "
                      f"test={test_acc:.4f}")
    
    # Final evaluation
    training_time = timer.stop()
    peak_memory = mem_tracker.get_peak_mb()
    
    logits = model(X, A_norm, training=False)
    mx.eval(logits)
    
    final_train_acc = compute_accuracy(logits, y, mask_train)
    final_val_acc = compute_accuracy(logits, y, mask_val)
    final_test_acc = compute_accuracy(logits, y, mask_test)
    
    results = {
        "test_acc": best_test_acc,
        "val_acc": best_val_acc,
        "train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "peak_memory_mb": peak_memory,
        "training_time_s": training_time,
        "loss_history": loss_history,
        "num_edges": adj.nnz // 2,
    }
    
    return results
