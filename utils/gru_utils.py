import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import numpy as np
import os

class WeightedFocalCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
        if class_weights is None:
            # Default: make minority classes 100x more important
            self.class_weights = torch.tensor([100.0, 1.0, 100.0])  # [-1, 0, 1] -> [0, 1, 2]
        else:
            self.class_weights = torch.tensor(class_weights)
    
    def forward(self, inputs, targets):
        # Shift targets from [-1, 0, 1] to [0, 1, 2]
        targets_shifted = targets + 1
        targets_shifted = targets_shifted.long()
        
        # Move weights to same device as inputs
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # Calculate weighted cross entropy with focal loss
        ce_loss = nn.functional.cross_entropy(inputs, targets_shifted, 
                                             weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class GRUAttentionModel(nn.Module):
    """GRU model with attention mechanism for time series prediction"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layer
        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim,
            output_dim
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GRU model with attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (output, attention_weights)
            - output: Model predictions of shape (batch_size, output_dim)
            - attention_weights: Attention weights of shape (batch_size, seq_len, 1)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim * directions)
        
        # Calculate attention weights
        attention_weights = self.attention(gru_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * gru_out, dim=1)  # (batch_size, hidden_dim * directions)
        
        # Final prediction
        output = self.fc(context)  # (batch_size, output_dim)
        
        return output, attention_weights

def train_gru(model: GRUAttentionModel, 
             dataloader: torch.utils.data.DataLoader,
             criterion: nn.Module,
             optimizer: torch.optim.Optimizer,
             device: torch.device,
             num_epochs: int = 10,
             early_stopping_patience: int = 5) -> GRUAttentionModel:
    """
    Train the GRU model
    
    Args:
        model: GRU model to train
        dataloader: DataLoader containing training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        Trained GRU model
    """
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            # Move batch to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output, _ = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            
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

def predict_gru(model: GRUAttentionModel,
               X: torch.Tensor,
               device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the trained GRU model
    
    Args:
        model: Trained GRU model
        X: Input tensor
        device: Device to run inference on
        
    Returns:
        Tuple of (predictions, attention_weights)
        - predictions: Model predictions
        - attention_weights: Attention weights for each prediction
    """
    model.eval()
    with torch.no_grad():
        # Move input to device
        X = X.to(device)
        output, attention_weights = model(X)
        # Move outputs back to CPU for numpy conversion
        output = output.cpu().numpy()
        attention_weights = attention_weights.cpu().numpy()
    return output, attention_weights

def save_gru(model: GRUAttentionModel, path: str) -> None:
    """
    Save the GRU model to disk
    
    Args:
        model: GRU model to save
        path: Path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_gru(model_class: type,
            path: str,
            *args,
            **kwargs) -> GRUAttentionModel:
    """
    Load a saved GRU model
    
    Args:
        model_class: GRU model class
        path: Path to saved model
        *args: Positional arguments for model initialization
        **kwargs: Keyword arguments for model initialization
        
    Returns:
        Loaded GRU model
    """
    model = model_class(*args, **kwargs)
    # Load model to CPU first, then move to appropriate device
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model 