import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np
import math
import os

class GraphConvolution(nn.Module):
    """Graph Convolution Layer for spatial dependencies with optimized operations"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 graph_type: str = 'dense', sparsity_threshold: float = 0.1):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias
            graph_type: Type of graph ('dense' or 'sparse')
            sparsity_threshold: Threshold for converting dense to sparse matrix
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_type = graph_type
        self.sparsity_threshold = sparsity_threshold
        
        # Initialize weights
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights using Kaiming initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def _to_sparse(self, adj: torch.Tensor) -> torch.Tensor:
        """Convert dense adjacency matrix to sparse format if needed"""
        if self.graph_type == 'sparse' or (self.graph_type == 'auto' and 
            (adj < self.sparsity_threshold).float().mean() > 0.5):
            return adj.to_sparse()
        return adj
        
    def _validate_inputs(self, x: torch.Tensor, adj: torch.Tensor) -> None:
        """Validate input shapes and types"""
        if x.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D input tensor, got {x.dim()}D")
        if adj.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D adjacency matrix, got {adj.dim()}D")
            
        if x.dim() == 2:
            if x.shape[0] != adj.shape[0]:
                raise ValueError(f"Node dimension mismatch: x has {x.shape[0]} nodes but adj has {adj.shape[0]} nodes")
        else:
            if x.shape[1] != adj.shape[0]:
                raise ValueError(f"Node dimension mismatch: x has {x.shape[1]} nodes but adj has {adj.shape[0]} nodes")
                
    def forward(self, x: torch.Tensor, adj: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input features of shape [batch_size, num_nodes, in_features] or [num_nodes, in_features]
            adj: Adjacency matrix of shape [num_nodes, num_nodes] or [batch_size, num_nodes, num_nodes]
            return_attention: Whether to return attention weights
            
        Returns:
            Output features of shape [batch_size, num_nodes, out_features]
            Attention weights if return_attention is True
        """
        # Validate inputs
        self._validate_inputs(x, adj)
        
        # Ensure input tensor has correct shape
        if x.dim() == 2:  # [num_nodes, in_features]
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Apply graph convolution
        support = torch.matmul(x, self.weight)  # [batch_size, num_nodes, out_features]
        
        # Handle adjacency matrix
        if adj.dim() == 2:  # Single adjacency matrix [num_nodes, num_nodes]
            if adj.is_sparse:
                # For sparse matrices, we need to handle each batch separately
                output = []
                for i in range(support.shape[0]):
                    # Convert support to 2D for sparse multiplication
                    support_2d = support[i].view(-1, support.shape[-1])
                    # Perform sparse multiplication
                    out = torch.sparse.mm(adj, support_2d)
                    output.append(out)
                output = torch.stack(output)
            else:
                # For dense matrices, we can use batch multiplication
                adj = adj.unsqueeze(0)  # [1, num_nodes, num_nodes]
                adj = adj.expand(support.shape[0], -1, -1)  # [batch_size, num_nodes, num_nodes]
                output = torch.bmm(adj, support)
        else:  # Batched adjacency matrices
            if adj.is_sparse:
                raise ValueError("Batched sparse adjacency matrices are not supported")
            if adj.shape[0] == 1:  # Single adjacency matrix in batch dimension
                adj = adj.expand(support.shape[0], -1, -1)
            elif adj.shape[0] != support.shape[0]:
                raise ValueError(f"Batch size mismatch: adj has {adj.shape[0]} batches but support has {support.shape[0]} batches")
            output = torch.bmm(adj, support)
            
        # Add bias if present
        if self.bias is not None:
            output += self.bias
            
        if return_attention:
            # Return both output and adjacency matrix as attention weights
            # The adjacency matrix serves as the attention weights for graph convolution
            return output, adj
        return output

class TemporalConvolution(nn.Module):
    """Temporal Convolution Layer for temporal dependencies"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        # Calculate padding to maintain sequence length
        padding = ((kernel_size - 1) // 2) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, num_nodes, seq_len, in_channels]
            return_attention: Whether to return activations as attention proxy
        Returns:
            Output tensor of shape [batch_size, num_nodes, seq_len, out_channels]
            If return_attention is True, also returns normalized attention weights
        """
        batch_size, num_nodes, seq_len, in_channels = x.shape
        x = x.permute(0, 1, 3, 2)  # [batch_size, num_nodes, in_channels, seq_len]
        x = x.reshape(-1, in_channels, seq_len)  # [batch_size * num_nodes, in_channels, seq_len]
        
        # Apply temporal convolution
        x = self.conv(x)  # [batch_size * num_nodes, out_channels, seq_len]
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Reshape back to original format
        x = x.reshape(batch_size, num_nodes, -1, seq_len)  # [batch_size, num_nodes, out_channels, seq_len]
        x = x.permute(0, 1, 3, 2)  # [batch_size, num_nodes, seq_len, out_channels]
        
        if return_attention:
            # Calculate attention weights by taking mean across feature dimension
            attention = x.mean(dim=-1)  # [batch_size, num_nodes, seq_len]
            # Normalize attention weights using softmax
            attention = F.softmax(attention, dim=-1)  # Normalize across sequence length
            return x, attention
        return x

class STGNNModel(nn.Module):
    """Spatio-Temporal Graph Neural Network for time series prediction"""
    def __init__(self, 
                 num_nodes: int,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 kernel_size: int = 3):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Spatial layers
        self.spatial_layers = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Temporal layers
        self.temporal_layers = nn.ModuleList([
            TemporalConvolution(hidden_dim, hidden_dim, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape [batch_size, num_nodes, seq_len, input_dim]
            adj: Adjacency matrix of shape [num_nodes, num_nodes]
            
        Returns:
            Tuple of (output, attention_dict)
            - output: Model predictions of shape [batch_size, num_nodes, output_dim]
            - attention_dict: Dictionary containing attention weights for visualization
        """
        batch_size, num_nodes, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, num_nodes, seq_len, hidden_dim]
        
        attention_dict = {}
        
        for i in range(self.num_layers):
            # Spatial processing
            # Process each time step separately to maintain node dimensions
            x_spatial = []
            for t in range(seq_len):
                # Extract features for current time step
                x_t = x[:, :, t, :]  # [batch_size, num_nodes, hidden_dim]
                # Apply graph convolution
                x_t, attn = self.spatial_layers[i](x_t, adj, return_attention=True)  # [batch_size, num_nodes, hidden_dim]
                x_spatial.append(x_t)
            # Stack time steps back together
            x_spatial = torch.stack(x_spatial, dim=2)  # [batch_size, num_nodes, seq_len, hidden_dim]
            
            # Temporal processing
            x_temporal, attn = self.temporal_layers[i](x_spatial, return_attention=True)  # [batch_size, num_nodes, seq_len, hidden_dim]
            
            # Residual connection and layer normalization
            if x_temporal.shape[-1] != x.shape[-1]:
                x_temporal = x_temporal[..., :x.shape[-1]]
            x = x + x_temporal
            x = self.layer_norm(x)
            
            attention_dict[f'layer_{i}_temporal'] = attn
        
        # Take last time step and make prediction
        x = x[:, :, -1, :]  # [batch_size, num_nodes, hidden_dim]
        output = self.output_proj(x)  # [batch_size, num_nodes, output_dim]
        
        return output, attention_dict

    def get_attention_weights(self, features, adj_matrix):
        """
        Get attention weights for model explanation
        
        Args:
            features: Input features tensor
            adj_matrix: Adjacency matrix tensor
            
        Returns:
            Dictionary containing attention weights for each layer
        """
        attention_weights = {}
        x = self.input_proj(features)
        original_shape = x.shape  # Save for reshaping
        
        # Forward pass through spatial layers
        for i, layer in enumerate(self.spatial_layers):
            if x.dim() == 4:
                batch_size, num_nodes, seq_len, hidden_dim = x.shape
                x_reshaped = x.reshape(-1, num_nodes, hidden_dim)
            else:
                # If already 3D, infer batch_size and seq_len from original shape
                batch_size, num_nodes, seq_len, hidden_dim = original_shape
                x_reshaped = x
            print(f"Spatial layer {i} input shape: {x_reshaped.shape}")
            x, attn = layer(x_reshaped, adj_matrix, return_attention=True)
            attention_weights[f'layer_{i}_spatial'] = attn
        # Reshape x back to 4D for temporal layers
        x = x.reshape(batch_size, num_nodes, seq_len, hidden_dim)
        # Forward pass through temporal layers
        for i, layer in enumerate(self.temporal_layers):
            x, attn = layer(x, return_attention=True)
            attention_weights[f'layer_{i}_temporal'] = attn
        
        return attention_weights

def train_stgnn(model: STGNNModel,
                dataloader: torch.utils.data.DataLoader,
                adj: torch.Tensor,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                num_epochs: int = 10,
                early_stopping_patience: int = 5) -> STGNNModel:
    """
    Train the STGNN model
    
    Args:
        model: STGNN model to train
        dataloader: DataLoader containing training data
        adj: Adjacency matrix (shared for all batches)
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        Trained STGNN model
    """
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            adj_batch = adj.to(device)
            
            # Do NOT permute X_batch; keep as [batch, num_nodes, seq_len, input_dim]
            # X_batch = X_batch.permute(0, 2, 1, 3)
            
            # Forward pass
            optimizer.zero_grad()
            output, _ = model(X_batch, adj_batch)
            output = output.squeeze(-1)  # Remove last dimension to match target shape
            loss = criterion(output, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Early stopping check
        avg_epoch_loss = epoch_loss / len(dataloader)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
    
    return model

def predict_stgnn(model: STGNNModel,
                 X: torch.Tensor,
                 adj: torch.Tensor,
                 device: torch.device) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Make predictions using the trained STGNN model
    
    Args:
        model: Trained STGNN model
        X: Input tensor of shape [batch_size, num_nodes, seq_len, input_dim]
        adj: Adjacency matrix of shape [num_nodes, num_nodes]
        device: Device to run inference on
        
    Returns:
        Tuple of (predictions, attention_dict)
        - predictions: Model predictions of shape [batch_size, num_nodes, output_dim]
        - attention_dict: Dictionary containing attention weights for visualization
    """
    model.eval()
    with torch.no_grad():
        # Move tensors to device
        X = X.to(device)
        adj = adj.to(device)
        
        # Ensure input tensor has correct shape
        if X.dim() == 3:  # [num_nodes, seq_len, input_dim]
            X = X.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        output, attention_dict = model(X, adj)
        
        # Convert attention weights to numpy
        attention_dict = {k: v.cpu().numpy() for k, v in attention_dict.items()}
        
    return output.cpu().numpy(), attention_dict

def save_stgnn(model: STGNNModel, path: str) -> None:
    """
    Save the STGNN model to disk
    
    Args:
        model: STGNN model to save
        path: Path to save the model to
    """
    if not path:
        raise ValueError("Save path cannot be empty")
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_nodes': model.num_nodes,
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'output_dim': model.output_dim,
        'num_layers': model.num_layers,
        'kernel_size': model.kernel_size
    }, path)

def load_stgnn(path: str) -> STGNNModel:
    """
    Load the STGNN model from disk
    
    Args:
        path: Path to load the model from
        
    Returns:
        Loaded STGNN model
    """
    # Load checkpoint
    checkpoint = torch.load(path)
    
    # Create model
    model = STGNNModel(
        num_nodes=checkpoint['num_nodes'],
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim'],
        num_layers=checkpoint['num_layers'],
        kernel_size=checkpoint['kernel_size']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model 