import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np
import math
import os
import gc
import psutil
from torch.utils.data import Dataset
import pandas as pd

class EnhancedPositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable parameters and relative positions"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Learnable parameters for positional encoding
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x * math.sqrt(self.d_model)  # Scale embeddings
        # Get positional encoding for the sequence length and expand to batch size
        pe = self.pe[:x.size(1)].transpose(0, 1)  # [1, seq_len, d_model]
        pe = pe.expand(x.size(0), -1, -1)  # [batch_size, seq_len, d_model]
        x = x + self.alpha * pe + self.beta  # Add scaled positional encoding
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int = 3,  # -1, 0, 1
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = EnhancedPositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection for classification
        self.output_proj = nn.Linear(d_model, num_classes)
        
        # Store sequence length for dataset creation
        self.sequence_length = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer model
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            
        Returns:
            Tensor of shape [batch_size, num_classes] containing logits
        """
        # Ensure input is 3D
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input tensor [batch_size, sequence_length, input_size], got shape {x.shape}")
            
        # Input projection
        x = self.input_proj(x)  # [batch_size, sequence_length, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)  # [batch_size, sequence_length, d_model]
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, sequence_length, d_model]
        
        # Take the last sequence output
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # Output projection (return logits, no softmax)
        x = self.output_proj(x)  # [batch_size, num_classes]
        
        return x

class MemoryEfficientDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset for transformer training"""
    def __init__(self, X: torch.Tensor, y: torch.Tensor, sequence_length: int, device: torch.device):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.device = device
        self.num_samples = len(X)  # Changed: no need to subtract sequence_length since X is already in sequence form
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence of features and its target
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - X: Shape [sequence_length, input_size]
                - y: Shape [1]
        """
        # Get sequence of features - X is already in sequence form
        X = self.X[idx]  # Shape: [sequence_length, input_size]
        y = self.y[idx:idx + 1]  # Shape: [1]
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        return X, y

def custom_collate_fn(batch):
    """Custom collate function to ensure correct tensor shapes
    
    Args:
        batch: List of tuples (X, y) where:
            - X: Shape [sequence_length, input_size]
            - y: Shape [1]
            
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - X: Shape [batch_size, sequence_length, input_size]
            - y: Shape [batch_size]
    """
    # Stack sequences into [batch_size, sequence_length, input_size]
    X = torch.stack([item[0] for item in batch])
    # Concatenate targets into [batch_size]
    y = torch.cat([item[1] for item in batch])
    return X, y

def get_device() -> torch.device:
    """Get the appropriate device for training"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def cleanup_memory(device: torch.device) -> None:
    """Clean up memory based on the device being used"""
    gc.collect()
    if device.type == "mps":
        # MPS doesn't have a direct cache clearing method like CUDA
        # We can force garbage collection and empty cache
        gc.collect()
        torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    # For CPU, just do garbage collection

def calculate_precision(predictions: torch.Tensor, targets: torch.Tensor, class_idx: int) -> float:
    """Calculate precision for a specific class
    
    Args:
        predictions: Model predictions (logits)
        targets: Target values (already shifted to [0, 1, 2])
        class_idx: Class index to calculate precision for
        
    Returns:
        Precision score for the specified class
    """
    # Get predictions for this class
    pred_class = (predictions.argmax(dim=1) == class_idx)
    true_class = (targets == class_idx)
    
    # If no predictions for this class, return 0
    if pred_class.sum() == 0:
        return 0.0
        
    # Calculate precision
    true_positives = (pred_class & true_class).sum().float()
    total_predictions = pred_class.sum().float()
    
    return true_positives / total_predictions

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def train_transformer(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    early_stopping_patience: int,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    memory_optimization: bool = True,
    class_weights: Optional[torch.Tensor] = None,
    progress_callback: Optional[callable] = None,
    focal_loss_gamma: float = 2.0
) -> Dict[str, List[float]]:
    """Train transformer model with early stopping and batch processing"""
    # Set device if not provided
    if device is None:
        device = get_device()
    
    print(f"Starting training on device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Use Focal Loss for imbalanced classes
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = FocalLoss(gamma=focal_loss_gamma, weight=class_weights)
    
    # Create memory-efficient datasets
    print("Creating datasets...")
    train_dataset = MemoryEfficientDataset(X_train, y_train, model.sequence_length, device)
    val_dataset = MemoryEfficientDataset(X_val, y_val, model.sequence_length, device)
    
    # Create dataloaders with memory-efficient settings
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single worker to avoid memory issues
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"Training with {len(train_loader)} batches per epoch")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_precision_up': [],    # Track precision for upward class
        'train_precision_down': [],  # Track precision for downward class
        'val_precision_up': [],      # Track precision for upward class
        'val_precision_down': []     # Track precision for downward class
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        all_train_preds = []
        all_train_targets = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"Training batch {batch_idx + 1}/{len(train_loader)}")
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            # Shift targets from [-1, 0, 1] to [0, 1, 2] for CrossEntropyLoss
            y_batch_shifted = y_batch + 1
            loss = criterion(outputs, y_batch_shifted.long())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Store predictions and targets for precision calculation
            all_train_preds.append(outputs.detach())
            all_train_targets.append(y_batch_shifted.detach())
            
            # Clear memory after each batch
            del X_batch, y_batch, outputs, loss
            if memory_optimization:
                cleanup_memory(device)
        
        avg_train_loss = train_loss / num_batches
        
        # Calculate training precision for minority classes
        train_preds = torch.cat(all_train_preds)
        train_targets = torch.cat(all_train_targets)
        train_precision_up = calculate_precision(train_preds, train_targets, 2)    # Upward class (shifted from 1)
        train_precision_down = calculate_precision(train_preds, train_targets, 0)  # Downward class (shifted from -1)
        
        # Validation
        print("Running validation...")
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                if batch_idx % 10 == 0:
                    print(f"Validation batch {batch_idx + 1}/{len(val_loader)}")
                
                val_outputs = model(X_batch)
                # Shift targets from [-1, 0, 1] to [0, 1, 2] for CrossEntropyLoss
                y_batch_shifted = y_batch + 1
                batch_val_loss = criterion(val_outputs, y_batch_shifted.long())
                
                val_loss += batch_val_loss.item()
                num_val_batches += 1
                
                # Store predictions and targets for precision calculation
                all_val_preds.append(val_outputs)
                all_val_targets.append(y_batch_shifted)
                
                # Clear memory after each batch
                del X_batch, y_batch, val_outputs, batch_val_loss
                if memory_optimization:
                    cleanup_memory(device)
        
        avg_val_loss = val_loss / num_val_batches
        
        # Calculate validation precision for minority classes
        val_preds = torch.cat(all_val_preds)
        val_targets = torch.cat(all_val_targets)
        val_precision_up = calculate_precision(val_preds, val_targets, 2)    # Upward class (shifted from 1)
        val_precision_down = calculate_precision(val_preds, val_targets, 0)  # Downward class (shifted from -1)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_precision_up'].append(train_precision_up)
        history['train_precision_down'].append(train_precision_down)
        history['val_precision_up'].append(val_precision_up)
        history['val_precision_down'].append(val_precision_down)
        
        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(
                epoch,
                avg_train_loss,
                avg_val_loss,
                train_precision_up,
                train_precision_down,
                val_precision_up,
                val_precision_down
            )
        
        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_transformer_model.pt')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        print(f"  Train Precision (Up) = {train_precision_up:.4f}")
        print(f"  Train Precision (Down) = {train_precision_down:.4f}")
        print(f"  Val Precision (Up) = {val_precision_up:.4f}")
        print(f"  Val Precision (Down) = {val_precision_down:.4f}")
        if memory_optimization:
            print(f"  Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Load best model
    print("Loading best model...")
    model.load_state_dict(torch.load('best_transformer_model.pt'))
    
    # Final cleanup
    if memory_optimization:
        cleanup_memory(device)
    
    return history

def predict_transformer(
    model: nn.Module,
    X: torch.Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    memory_optimization: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions using trained transformer model with batch processing"""
    # Set device if not provided
    if device is None:
        device = get_device()
    
    model.eval()
    
    # Create memory-efficient dataset
    dataset = MemoryEfficientDataset(X, torch.zeros(len(X)), model.sequence_length, device)
    
    # Create dataloader with memory-efficient settings
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for X_batch, _ in dataloader:
            # Make predictions
            probabilities = model(X_batch)
            predictions = probabilities.argmax(dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            
            # Clear memory after each batch
            del X_batch, probabilities, predictions
            if memory_optimization:
                cleanup_memory(device)
    
    # Final cleanup
    if memory_optimization:
        cleanup_memory(device)
    
    return np.concatenate(all_predictions), np.concatenate(all_probabilities)

def save_transformer(model: nn.Module, path: str) -> None:
    """Save the transformer model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_transformer(model: nn.Module, path: str) -> nn.Module:
    """Load a saved transformer model"""
    model.load_state_dict(torch.load(path))
    return model

def visualize_attention(
    attention_dict: Dict[str, np.ndarray],
    sequence_length: int,
    feature_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Process attention weights for visualization
    
    Args:
        attention_dict: Dictionary of attention weights from model
        sequence_length: Length of input sequences
        feature_names: Optional list of feature names
        
    Returns:
        Dictionary of processed attention weights for visualization
    """
    processed_attention = {}
    
    for layer_name, attention in attention_dict.items():
        # Average attention across heads
        avg_attention = attention.mean(axis=1)
        
        # Create time step labels
        time_steps = [f't-{i}' for i in range(sequence_length-1, -1, -1)]
        
        # Store processed attention
        processed_attention[layer_name] = {
            'attention_matrix': avg_attention,
            'time_steps': time_steps,
            'feature_names': feature_names
        }
    
    return processed_attention

def prepare_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare data for transformer model with memory optimization"""
    # Ensure features are numeric
    features = features.select_dtypes(include=[np.number])
    
    # Convert to numpy arrays
    X = features.values
    y = target.values
    
    # Create sequences in batches to reduce memory usage
    if self.config.memory_optimization:
        X_sequences = []
        y_sequences = []
        batch_size = 1000  # Process 1000 sequences at a time
        
        for i in range(0, len(X) - self.config.sequence_length, batch_size):
            end_idx = min(i + batch_size, len(X) - self.config.sequence_length)
            # Create sequences with correct shape [batch_size, sequence_length, features]
            X_batch = np.array([X[j:j + self.config.sequence_length] for j in range(i, end_idx)])
            y_batch = np.array([y[j + self.config.sequence_length - 1] for j in range(i, end_idx)])
            
            X_sequences.append(X_batch)
            y_sequences.append(y_batch)
            
            # Force garbage collection after each batch
            if i % (batch_size * 10) == 0:
                import gc
                gc.collect()
        
        # Concatenate all batches
        X_tensor = torch.FloatTensor(np.concatenate(X_sequences, axis=0))
        y_tensor = torch.LongTensor(np.concatenate(y_sequences, axis=0))
    else:
        # Original implementation for smaller datasets
        X_sequences = np.array([X[i:i + self.config.sequence_length] for i in range(len(X) - self.config.sequence_length)])
        y_sequences = np.array([y[i + self.config.sequence_length - 1] for i in range(len(X) - self.config.sequence_length)])
        
        X_tensor = torch.FloatTensor(X_sequences)
        y_tensor = torch.LongTensor(y_sequences)
    
    return X_tensor, y_tensor 