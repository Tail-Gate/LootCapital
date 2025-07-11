#!/usr/bin/env python3
"""
Improved STGNN Model Training Script

This script trains the STGNN model for ETH/USD futures direction prediction.
- Uses classification (up/down/no direction) instead of regression
- Detects 2%+ price movements
- Handles class imbalance with weighted loss, SMOTE, and Focal Loss
- Focuses on precision and recall metrics
- Uses comprehensive features from FeatureGenerator
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support
from collections import Counter
from imblearn.over_sampling import SMOTE

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from utils.feature_generator import FeatureGenerator
from utils.stgnn_trainer import STGNNTrainer
from utils.stgnn_utils import save_stgnn
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stgnn_improved_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    
    Focal Loss reduces the relative loss for well-classified examples (pt > 0.5)
    and puts more focus on hard, misclassified examples.
    
    Args:
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Reduction method ('none', 'mean', 'sum')
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Forward pass of Focal Loss
        
        Args:
            inputs: Logits from model [N, C] where C is number of classes
            targets: Ground truth labels [N]
            
        Returns:
            Focal loss value
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(inputs, dim=1)
        
        # Get probability of the correct class
        batch_size = inputs.size(0)
        probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal loss
        focal_weight = (1 - probs) ** self.gamma
        focal_loss = -self.alpha * focal_weight * torch.log(probs + 1e-7)
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss that combines class weights with focal loss.
    
    Args:
        class_weights (torch.Tensor): Class weights for handling imbalance
        alpha (float): Focal loss alpha parameter (default: 1.0)
        gamma (float): Focal loss gamma parameter (default: 2.0)
        reduction (str): Reduction method ('none', 'mean', 'sum')
    """
    
    def __init__(self, class_weights, alpha=1.0, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Forward pass of Weighted Focal Loss
        
        Args:
            inputs: Logits from model [N, C] where C is number of classes
            targets: Ground truth labels [N]
            
        Returns:
            Weighted focal loss value
        """
        # CRITICAL FIX: Validate class weights
        if torch.any(self.class_weights <= 0):
            logger.error(f"Invalid class weights detected: {self.class_weights}")
            # Use uniform weights as fallback
            self.class_weights = torch.ones_like(self.class_weights)
        
        # CRITICAL FIX: Validate inputs and targets
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            logger.error("Invalid inputs detected (NaN or Inf)")
            return torch.tensor(float('inf'), device=inputs.device)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(inputs, dim=1)
        
        # Get probability of the correct class
        batch_size = inputs.size(0)
        probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Get class weights for the targets
        target_weights = self.class_weights[targets]
        
        # CRITICAL FIX: Add epsilon to prevent log(0)
        epsilon = 1e-7
        probs = torch.clamp(probs, epsilon, 1.0 - epsilon)
        
        # Calculate weighted focal loss
        focal_weight = (1 - probs) ** self.gamma
        weighted_focal_loss = -target_weights * self.alpha * focal_weight * torch.log(probs)
        
        # CRITICAL FIX: Check for invalid loss values
        if torch.isnan(weighted_focal_loss).any() or torch.isinf(weighted_focal_loss).any():
            logger.error("Invalid focal loss detected (NaN or Inf)")
            return torch.tensor(float('inf'), device=inputs.device)
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_focal_loss.mean()
        elif self.reduction == 'sum':
            return weighted_focal_loss.sum()
        else:
            return weighted_focal_loss

class EnhancedSTGNNDataProcessor(STGNNDataProcessor):
    """Enhanced STGNN data processor that uses FeatureGenerator for comprehensive features"""
    
    def __init__(self, config: STGNNConfig, market_data: MarketData, technical_indicators: TechnicalIndicators):
        super().__init__(config, market_data, technical_indicators)
        self.feature_generator = FeatureGenerator()
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features using FeatureGenerator for comprehensive feature engineering
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with prepared features
        """
        logger.info("Using FeatureGenerator for comprehensive feature engineering...")
        
        # Store original price data for event-based analysis (from parent class)
        self._original_prices = data['close'].copy()
        
        # Use FeatureGenerator to generate comprehensive features
        features = self.feature_generator.generate_features(data)
        
        # Store returns separately for target calculation
        self._returns = features['returns'] if 'returns' in features.columns else pd.Series(0, index=data.index)
        
        # Ensure all required features from config are present
        for feat in self.config.features:
            if feat not in features.columns:
                logger.warning(f"Feature '{feat}' not found in FeatureGenerator output, using 0")
                features[feat] = 0
        
        # Select only the features specified in config
        features = features[self.config.features]
        
        # Handle missing/infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill()
        features = features.fillna(0)
        
        logger.info(f"Generated {len(features.columns)} features using FeatureGenerator")
        logger.info(f"Feature columns: {list(features.columns)}")
        
        return features
        
    def set_scaler(self, scaler_type: str = 'minmax'):
        """Set scaler type for feature normalization"""
        super().set_scaler(scaler_type)
        logger.info(f"Enhanced data processor scaler set to: {scaler_type}")
        
    def fit_scaler(self, features: pd.DataFrame):
        """Fit scaler on training data"""
        super().fit_scaler(features)
        logger.info("Enhanced data processor scaler fitted successfully")
        
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler"""
        return super().transform_features(features)

class STGNNClassificationModel(nn.Module):
    """STGNN model modified for classification"""
    
    def __init__(self, num_nodes, input_dim, hidden_dim, num_classes=3, num_layers=2, dropout=0.2, kernel_size=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        # Import the original STGNN model
        from utils.stgnn_utils import STGNNModel
        self.stgnn = STGNNModel(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Use hidden_dim as intermediate output
            num_layers=num_layers,
            dropout=dropout,
            kernel_size=kernel_size
        )
        
        # Classification head - Fixed for proper tensor shapes
        # Use LayerNorm instead of BatchNorm1d to handle variable batch sizes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Use LayerNorm instead of BatchNorm1d
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),  # Use LayerNorm instead of BatchNorm1d
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, adj):
        # Get STGNN features
        features, attention_dict = self.stgnn(x, adj)
        
        # Reshape features from [batch_size, num_nodes, hidden_dim]
        # to [batch_size * num_nodes, hidden_dim] for the classifier
        reshaped_features = features.view(-1, self.hidden_dim)
        
        # Apply classification head
        logits = self.classifier(reshaped_features)
        
        # The trainer's `train_epoch` and `validate` methods expect
        # [batch_size * num_nodes, num_classes] directly from the classifier,
        # so no need to reshape back here for the return value.
        return logits, attention_dict

class ClassificationSTGNNTrainer:
    """Trainer for STGNN classification model"""
    
    def __init__(self, config, data_processor, price_threshold=0.018, focal_alpha=1.0, focal_gamma=2.0, class_weights=None, start_time=None, end_time=None, device=None):
        """
        Initialize trainer
        
        Args:
            config: STGNN configuration
            data_processor: Data processor
            price_threshold: Threshold for price movement classification
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            class_weights: Pre-calculated class weights
            start_time: Optional start time for data range
            end_time: Optional end time for data range
            device: Device to use for training (cuda/cpu)
        """
        self.config = config
        self.data_processor = data_processor
        self.price_threshold = price_threshold
        self.start_time = start_time
        self.end_time = end_time
        
        # Always use CPU device
        self.device = torch.device('cpu')
        logger.info("Using CPU device for training")
        
        # Initialize model
        num_nodes = len(config.assets)
        input_dim = len(config.features)
        self.model = STGNNClassificationModel(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_classes=3,  # Down, No Direction, Up
            num_layers=config.num_layers,
            dropout=config.dropout,
            kernel_size=config.kernel_size
        ).to(self.device)
        
        # Log device and model info
        logger.info("CPU training mode - using all available cores for parallel processing")
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Set up loss function
        if class_weights is not None:
            # Use pre-calculated class weights
            class_weights = class_weights.to(self.device)
            self.criterion = WeightedFocalLoss(
                class_weights=class_weights,
                alpha=focal_alpha,
                gamma=focal_gamma
            )
            logger.info(f"Using pre-calculated class weights: {class_weights}")
        else:
            # Calculate class weights from data
            calculated_weights = self._calculate_class_weights()
            calculated_weights = calculated_weights.to(self.device)
            self.criterion = WeightedFocalLoss(
                class_weights=calculated_weights,
                alpha=focal_alpha,
                gamma=focal_gamma
            )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Store adjacency matrix
        self.adj = None
        
    def _calculate_class_weights(self):
        """Calculate class weights to handle imbalance"""
        # Prepare data to get target distribution
        X, adj, y = self.data_processor.prepare_data()
        y_flat = y.flatten().numpy()
        
        # Convert to classes
        classes = self._returns_to_classes(y_flat)
        class_counts = Counter(classes)
        
        # Calculate weights (inverse frequency)
        total_samples = len(classes)
        class_weights = torch.FloatTensor([
            total_samples / (len(class_counts) * class_counts[i]) 
            for i in range(3)
        ])
        
        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Class weights: {class_weights}")
        
        return class_weights
    
    def _returns_to_classes(self, returns):
        """Convert returns to classes (0=down, 1=no direction, 2=up)"""
        classes = np.ones(len(returns), dtype=int)  # Default to no direction
        classes[returns > self.price_threshold] = 2   # Up
        classes[returns < -self.price_threshold] = 0  # Down
        return classes
    
    def prepare_classification_data(self):
        """Prepare data for classification"""
        # Pass time window parameters to data processor
        X, adj, y = self.data_processor.prepare_data(self.start_time, self.end_time)
        
        # Store adjacency matrix for later use
        self.adj = adj.to(self.device)
        
        # Convert returns to classes
        y_flat = y.flatten().numpy()
        y_classes = self._returns_to_classes(y_flat)
        
        # Reshape back to [batch_size, num_nodes]
        y_classes = y_classes.reshape(y.shape)
        
        # Convert to tensors and move to device
        y_classes = torch.LongTensor(y_classes).to(self.device)
        
        return X, adj, y_classes
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            # Move data to device
            X_batch = X_batch.to(self.device, non_blocking=False)
            y_batch = y_batch.to(self.device, non_blocking=False)

            # Ensure adjacency matrix is on correct device
            if self.adj is not None:
                if self.adj.device != self.device:
                    self.adj = self.adj.to(self.device, non_blocking=False)
            else:
                logger.error("self.adj is None right before model forward pass!")
                return float('inf')  # Fail fast if adj is still None

            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(X_batch, self.adj)

            # Model now returns logits in shape [batch_size * num_nodes, num_classes]
            # and y_batch needs to be flattened to [batch_size * num_nodes]
            y_batch = y_batch.view(-1)  # [batch_size * num_nodes]

            # Validate shapes before loss calculation
            if logits.shape[0] != y_batch.shape[0]:
                raise ValueError(f"Shape mismatch: logits {logits.shape} vs y_batch {y_batch.shape}")

            loss = self.criterion(logits, y_batch)

            # Check for infinite or NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss detected: {loss.item()}")
                logger.error(f"Logits stats: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
                logger.error(f"Y batch stats: min={y_batch.min().item()}, max={y_batch.max().item()}")
                continue

            # Backward pass
            loss.backward()
            
            # CRITICAL FIX: Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Move data to device
                X_batch = X_batch.to(self.device, non_blocking=False)
                y_batch = y_batch.to(self.device, non_blocking=False)

                # Ensure adjacency matrix is on correct device
                if self.adj is not None:
                    if self.adj.device != self.device:
                        self.adj = self.adj.to(self.device, non_blocking=False)
                else:
                    logger.error("self.adj is None right before model forward pass!")
                    return float('inf')

                # Forward pass
                logits, _ = self.model(X_batch, self.adj)

                # Model now returns logits in shape [batch_size * num_nodes, num_classes]
                # and y_batch needs to be flattened to [batch_size * num_nodes]
                y_batch = y_batch.view(-1)  # [batch_size * num_nodes]

                # Validate shapes before loss calculation
                if logits.shape[0] != y_batch.shape[0]:
                    raise ValueError(f"Shape mismatch: logits {logits.shape} vs y_batch {y_batch.shape}")

                loss = self.criterion(logits, y_batch)

                # Check for infinite or NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Invalid loss detected: {loss.item()}")
                    logger.error(f"Logits stats: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
                    logger.error(f"Y batch stats: min={y_batch.min().item()}, max={y_batch.max().item()}")
                    continue

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        return total_loss / len(val_loader), correct / total
    
    def train(self):
        """Train model with early stopping and SMOTE for class balance"""
        # Prepare data
        X, adj, y_classes = self.prepare_classification_data()
        X_train, y_train, X_val, y_val = self.data_processor.split_data(X, y_classes)
        
        # Apply SMOTE to training data only
        logger.info("Applying SMOTE to training data for class balance...")
        
        # Get original shapes
        batch_size_train, num_nodes, seq_len, input_dim = X_train.shape
        
        # Log class distribution before SMOTE
        y_train_flat = y_train.reshape(-1)
        unique_before, counts_before = np.unique(y_train_flat.numpy(), return_counts=True)
        class_dist_before = dict(zip(unique_before, counts_before))
        logger.info(f"Training data class distribution before SMOTE: {class_dist_before}")
        
        # Apply SMOTE at the node level (batch_size * num_nodes level)
        # This preserves the temporal structure while balancing classes
        X_train_node_level = X_train.reshape(batch_size_train * num_nodes, seq_len * input_dim)
        y_train_node_level = y_train.reshape(-1)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(class_dist_before.values()) - 1))
        X_train_balanced_node, y_train_balanced_node = smote.fit_resample(X_train_node_level.numpy(), y_train_node_level.numpy())
        
        # Log class distribution after SMOTE
        unique_after, counts_after = np.unique(y_train_balanced_node, return_counts=True)
        class_dist_after = dict(zip(unique_after, counts_after))
        logger.info(f"Training data class distribution after SMOTE: {class_dist_after}")
        
        # Reshape back to original format
        # Calculate new batch size based on balanced data
        new_batch_size = len(y_train_balanced_node) // num_nodes
        if len(y_train_balanced_node) % num_nodes != 0:
            # Pad if necessary to make it divisible by num_nodes
            padding_needed = num_nodes - (len(y_train_balanced_node) % num_nodes)
            X_train_balanced_node = np.vstack([X_train_balanced_node, X_train_balanced_node[:padding_needed]])
            y_train_balanced_node = np.hstack([y_train_balanced_node, y_train_balanced_node[:padding_needed]])
            new_batch_size = len(y_train_balanced_node) // num_nodes
        
        # Reshape X: [new_batch_size * num_nodes, seq_len * input_dim] -> [new_batch_size, num_nodes, seq_len, input_dim]
        X_train_balanced = X_train_balanced_node.reshape(new_batch_size, num_nodes, seq_len, input_dim)
        
        # Reshape y: [new_batch_size * num_nodes] -> [new_batch_size, num_nodes]
        y_train_balanced = y_train_balanced_node.reshape(new_batch_size, num_nodes)
        
        # Convert back to tensors
        X_train_balanced = torch.FloatTensor(X_train_balanced)
        y_train_balanced = torch.LongTensor(y_train_balanced)
        
        logger.info(f"Training data shapes after SMOTE - X: {X_train_balanced.shape}, y: {y_train_balanced.shape}")
        
        # Create dataloaders with balanced training data
        train_loader = self.data_processor.create_dataloader(X_train_balanced, y_train_balanced, drop_last=True)
        val_loader = self.data_processor.create_dataloader(X_val, y_val, drop_last=False)
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f'Early stopping at epoch {epoch + 1}')
                break
                
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch + 1}/{self.config.num_epochs}:')
                logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'smote_info': {
                'class_distribution_before': class_dist_before,
                'class_distribution_after': class_dist_after,
                'original_batch_size': batch_size_train,
                'balanced_batch_size': new_batch_size
            }
        }
    
    def evaluate(self, X, y_classes):
        """Evaluate model with precision/recall metrics and predicted probabilities"""
        self.model.eval()
        
        with torch.no_grad():
            X = X.to(self.device)
            logits, _ = self.model(X, self.adj)
            
            # Model now returns logits in shape [batch_size * num_nodes, num_classes]
            # and y_classes needs to be flattened to [batch_size * num_nodes]
            y_classes = y_classes.view(-1)  # [batch_size * num_nodes]
            
            # Get predicted probabilities
            probabilities = torch.softmax(logits, dim=1)
            
            # Get predictions
            _, predicted = torch.max(logits, 1)
            
            # Convert to numpy for sklearn metrics
            predicted = predicted.cpu().numpy()
            y_true = y_classes.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()
            
            # Calculate metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, predicted, average=None, labels=[0, 1, 2]
            )
            
            # Create classification report
            class_names = ['Down', 'No Direction', 'Up']
            report = classification_report(
                y_true, predicted, 
                target_names=class_names,
                output_dict=True
            )
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'classification_report': report,
                'predictions': predicted,
                'true_labels': y_true,
                'probabilities': probabilities_np  # Add predicted probabilities
            }

def create_improved_config():
    """Create configuration for improved STGNN training with comprehensive features"""
    
    # Focus on ETH/USD
    assets = ['ETH/USD']
    
    # Engineered features only - NO raw OHLCV data
    # These features are derived from price/volume data and are suitable for ML training
    features = [
        # Price-derived features (safe for ML)
        'returns', 'log_returns',
        
        # Technical indicators (derived from price)
        'rsi', 'atr', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'adx', 'swing_rsi',
        
        # Volume-derived features (safe for ML)
        'volume_ma', 'volume_std', 'volume_surge', 'volume_ratio',
        
        # Moving averages and momentum
        'ma_crossover', 'price_momentum', 'volatility_regime',
        
        # Support/Resistance and breakout
        'support', 'resistance', 'breakout_intensity',
        
        # VWAP and cumulative delta
        'vwap_ratio', 'cumulative_delta'
    ]
    
    # Create configuration
    config = STGNNConfig(
        num_nodes=len(assets),
        input_dim=len(features),
        hidden_dim=128,  # Increased for more features
        output_dim=3,  # 3 classes: down/no direction/up
        num_layers=3,  # Increased for more complex patterns
        dropout=0.3,  # Slightly increased for regularization
        kernel_size=3,
        learning_rate=0.0005,  # Reduced for stability with more features
        batch_size=16,  # Reduced for memory management
        num_epochs=100,
        early_stopping_patience=10,
        seq_len=200,
        prediction_horizon=15,  
        features=features,
        assets=assets,
        confidence_threshold=0.51,
        buy_threshold=0.6,
        sell_threshold=0.4,
        retrain_interval=24,
        focal_alpha=1.0,  # Keep this as 1.0 given you have class_weights
        focal_gamma=3.0   # Experiment with 1.0, 2.0 (current), 3.0, 5.0
    )
    
    return config

def main():
    """Main training function"""
    
    logger.info("Starting improved STGNN classification training with Focal Loss...")
    
    try:
        # Create configuration
        config = create_improved_config()
        logger.info("Configuration created successfully")
        
        # Initialize components
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        
        # Create enhanced data processor
        data_processor = EnhancedSTGNNDataProcessor(config, market_data, technical_indicators)
        
        # Set scaler type (can be 'minmax' or 'standard')
        data_processor.set_scaler('minmax')  # or 'standard' for StandardScaler
        logger.info("Feature scaling enabled")
        
        # Create trainer with Focal Loss parameters from config
        trainer = ClassificationSTGNNTrainer(
            config, 
            data_processor, 
            price_threshold=0.018,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
        )
        
        # Train model
        training_history = trainer.train()
        
        # Evaluate model
        X, adj, y_classes = trainer.prepare_classification_data()
        X_train, y_train, X_val, y_val = data_processor.split_data(X, y_classes)
        
        evaluation_results = trainer.evaluate(X_val, y_val)
        
        # Print results
        print("\n" + "="*60)
        print("IMPROVED STGNN CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Asset: {config.assets[0]}")
        print(f"Prediction horizon: 15 candlesticks in the future")
        print(f"Price threshold: 1.8%")
        print(f"Features: {len(config.features)} comprehensive features")
        print(f"Feature scaling: MinMaxScaler (-1 to 1)")
        print(f"Loss function: Weighted Focal Loss (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
        print(f"Training epochs: {len(training_history['train_losses'])}")
        print(f"Final training accuracy: {training_history['train_accuracies'][-1]:.4f}")
        print(f"Final validation accuracy: {training_history['val_accuracies'][-1]:.4f}")
        print(f"Best validation loss: {trainer.best_val_loss:.6f}")
        
        # Display SMOTE information
        if 'smote_info' in training_history:
            smote_info = training_history['smote_info']
            print(f"\nSMOTE Class Balancing:")
            print(f"  Original batch size: {smote_info['original_batch_size']}")
            print(f"  Balanced batch size: {smote_info['balanced_batch_size']}")
            print(f"  Class distribution before SMOTE: {smote_info['class_distribution_before']}")
            print(f"  Class distribution after SMOTE: {smote_info['class_distribution_after']}")
        
        print("\nClassification Metrics:")
        class_names = ['Down', 'No Direction', 'Up']
        for i, name in enumerate(class_names):
            print(f"{name:>12}: Precision={evaluation_results['precision'][i]:.4f}, "
                  f"Recall={evaluation_results['recall'][i]:.4f}, "
                  f"F1={evaluation_results['f1'][i]:.4f}")
        
        print("\nDetailed Classification Report:")
        print(evaluation_results['classification_report'])
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = f'models/stgnn_classification_horizon15_{timestamp}.pt'
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'config': config,
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'scaler_fitted': data_processor.scaler_fitted,
            'focal_loss_params': {
                'alpha': config.focal_alpha,
                'gamma': config.focal_gamma
            }
        }, model_path)
        
        # Save configuration
        config_path = f'models/stgnn_classification_horizon15_config_{timestamp}.json'
        config.save(config_path)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Configuration saved to: {config_path}")
        
        print(f"\nModel saved to: {model_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 