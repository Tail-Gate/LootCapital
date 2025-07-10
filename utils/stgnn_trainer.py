import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.stgnn_config import STGNNConfig
from utils.stgnn_utils import STGNNModel
from utils.stgnn_data import STGNNDataProcessor

class STGNNTrainer:
    """Trainer for STGNN model"""
    
    def __init__(self, config: STGNNConfig, data_processor: STGNNDataProcessor):
        """
        Initialize trainer
        
        Args:
            config: STGNN configuration
            data_processor: Data processor instance
        """
        self.config = config
        self.data_processor = data_processor
        self.device = torch.device('cpu')  # Force CPU for reliability
        
        # Initialize model
        self.model = STGNNModel(
            num_nodes=config.num_nodes,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            kernel_size=config.kernel_size
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Store adjacency matrix
        _, self.adj, _ = self.data_processor.prepare_data()
        self.adj = self.adj.to(self.device)
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            # Move data to device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            y_pred, _ = self.model(X_batch, self.adj)
            y_pred = y_pred.squeeze(-1)  # Ensure shape matches target
            loss = self.criterion(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
        
    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Validate model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Move data to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                y_pred, _ = self.model(X_batch, self.adj)
                y_pred = y_pred.squeeze(-1)  # Ensure shape matches target
                loss = self.criterion(y_pred, y_batch)
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
        
    def train(self) -> Dict[str, List[float]]:
        """
        Train model with early stopping
        
        Returns:
            Dictionary containing training and validation losses
        """
        # Prepare data
        X, _, y = self.data_processor.prepare_data()
        X_train, y_train, X_val, y_val = self.data_processor.split_data(X, y)
        
        # Create dataloaders
        train_loader = self.data_processor.create_dataloader(X_train, y_train)
        val_loader = self.data_processor.create_dataloader(X_val, y_val)
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{self.config.num_epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        self.model.eval()
        with torch.set_grad_enabled(X.requires_grad):  # Enable gradients if input requires them
            X = X.to(self.device)
            predictions, _ = self.model(X, self.adj)
            return predictions.squeeze(-1)  # Ensure shape matches target
            
    def save_model(self, path: str):
        """
        Save model and training history
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)
        
    def load_model(self, path: str):
        """
        Load model and training history
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss'] 