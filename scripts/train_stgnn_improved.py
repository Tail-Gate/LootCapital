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
        
        # CRITICAL DEBUG: Add comprehensive NaN/Inf checks after feature generation
        if features.isnull().any().any() or (features == np.inf).any().any() or (features == -np.inf).any().any():
            logger.error(f"DEBUG: NaN/Inf detected in 'features' DataFrame after feature generation. Shape: {features.shape}")
            logger.error("--- Head of problematic features ---")
            logger.error(features.head())
            logger.error("--- Tail of problematic features ---")
            logger.error(features.tail())
            logger.error("--- Columns with NaN/Inf values ---")
            nan_inf_cols = []
            for col in features.columns:
                if features[col].isnull().any() or (features[col] == np.inf).any() or (features[col] == -np.inf).any():
                    nan_inf_cols.append(col)
                    logger.error(f"  Column '{col}' has NaN/Inf.")
                    # Print statistics for the problematic column
                    col_data = features[col].replace([np.inf, -np.inf], np.nan)
                    logger.error(f"    Stats for {col}: min={col_data.min()}, max={col_data.max()}, mean={col_data.mean()}, NaNs={col_data.isnull().sum()}")
            # Optionally, save the problematic DataFrame to a CSV for manual inspection
            # features.to_csv("problematic_features_after_generation.csv")
            raise ValueError("NaN/Inf detected in features after generation. Stopping to debug.")
        
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
        # CRITICAL FIX: Validate inputs before STGNN forward pass
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error(f"NaN/Inf detected in input x to STGNNClassificationModel. Shape: {x.shape}")
            logger.error(f"Input x stats: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
            # Return zeros to prevent downstream errors
            batch_size, num_nodes, seq_len, input_dim = x.shape
            return torch.zeros(batch_size * num_nodes, self.num_classes, device=x.device), {}
        
        if torch.isnan(adj).any() or torch.isinf(adj).any():
            logger.error(f"NaN/Inf detected in input adj to STGNNClassificationModel. Shape: {adj.shape}")
            logger.error(f"Input adj stats: min={adj.min().item()}, max={adj.max().item()}, mean={adj.mean().item()}")
            # Return zeros to prevent downstream errors
            batch_size, num_nodes, seq_len, input_dim = x.shape
            return torch.zeros(batch_size * num_nodes, self.num_classes, device=x.device), {}
        
        # Get STGNN features
        features, attention_dict = self.stgnn(x, adj)
        
        # CRITICAL FIX: Validate STGNN output
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.error(f"NaN/Inf detected from STGNN base model output. Shape: {features.shape}")
            logger.error(f"Features stats: min={features.min().item()}, max={features.max().item()}, mean={features.mean().item()}")
            # Return zeros to prevent downstream errors
            batch_size, num_nodes, seq_len, input_dim = x.shape
            return torch.zeros(batch_size * num_nodes, self.num_classes, device=x.device), {}
        
        # Reshape features from [batch_size, num_nodes, hidden_dim]
        # to [batch_size * num_nodes, hidden_dim] for the classifier
        reshaped_features = features.view(-1, self.hidden_dim)
        
        # CRITICAL FIX: Validate reshaped features
        if torch.isnan(reshaped_features).any() or torch.isinf(reshaped_features).any():
            logger.error(f"NaN/Inf detected after reshaping features. Shape: {reshaped_features.shape}")
            logger.error(f"Reshaped features stats: min={reshaped_features.min().item()}, max={reshaped_features.max().item()}, mean={reshaped_features.mean().item()}")
            # Return zeros to prevent downstream errors
            batch_size, num_nodes, seq_len, input_dim = x.shape
            return torch.zeros(batch_size * num_nodes, self.num_classes, device=x.device), {}
        
        # Apply classification head
        logits = self.classifier(reshaped_features)
        
        # CRITICAL FIX: Validate classifier output
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error(f"NaN/Inf detected after classifier. Shape: {logits.shape}")
            logger.error(f"Logits stats: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
            logger.error(f"Input to classifier (reshaped_features) stats: min={reshaped_features.min().item()}, max={reshaped_features.max().item()}, mean={reshaped_features.mean().item()}")
            # Return zeros to prevent downstream errors
            batch_size, num_nodes, seq_len, input_dim = x.shape
            return torch.zeros(batch_size * num_nodes, self.num_classes, device=x.device), {}
        
        # The trainer's `train_epoch` and `validate` methods expect
        # [batch_size * num_nodes, num_classes] directly from the classifier,
        # so no need to reshape back here for the return value.
        return logits, attention_dict

class ClassificationSTGNNTrainer:
    """Trainer for STGNN classification model with comprehensive logging and GPU support"""
    
    def __init__(self, config, data_processor, price_threshold=0.018, focal_alpha=1.0, focal_gamma=2.0, class_weights=None, start_time=None, end_time=None, device=None):
        """
        Initialize trainer with comprehensive logging and GPU support
        
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
        print("[TRAINER] Initializing ClassificationSTGNNTrainer...")
        logger.info("[TRAINER] Initializing ClassificationSTGNNTrainer...")
        
        self.config = config
        self.data_processor = data_processor
        self.price_threshold = price_threshold
        self.start_time = start_time
        self.end_time = end_time
        
        # Enhanced device detection and setup
        print("[TRAINER] Setting up device...")
        logger.info("[TRAINER] Setting up device...")
        
        if device is not None:
            self.device = torch.device(device)
            print(f"[TRAINER] Using specified device: {self.device}")
            logger.info(f"[TRAINER] Using specified device: {self.device}")
        else:
            # Auto-detect best device
            if torch.cuda.is_available():
                cuda_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                print(f"[TRAINER] CUDA available with {cuda_count} device(s)")
                print(f"[TRAINER] Using CUDA device {current_device}: {device_name}")
                logger.info(f"[TRAINER] CUDA available with {cuda_count} device(s)")
                logger.info(f"[TRAINER] Using CUDA device {current_device}: {device_name}")
                
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated(current_device) / 1024**3
                cached_memory = torch.cuda.memory_reserved(current_device) / 1024**3
                
                print(f"[TRAINER] GPU Memory - Total: {gpu_memory:.1f}GB, Allocated: {allocated_memory:.1f}GB, Cached: {cached_memory:.1f}GB")
                logger.info(f"[TRAINER] GPU Memory - Total: {gpu_memory:.1f}GB, Allocated: {allocated_memory:.1f}GB, Cached: {cached_memory:.1f}GB")
                
                self.device = torch.device('cuda:0')
                torch.cuda.empty_cache()
                print("[TRAINER] GPU cache cleared")
                logger.info("[TRAINER] GPU cache cleared")
            else:
                print("[TRAINER] CUDA not available, using CPU")
                logger.info("[TRAINER] CUDA not available, using CPU")
                
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                print(f"[TRAINER] CPU cores available: {cpu_count}")
                logger.info(f"[TRAINER] CPU cores available: {cpu_count}")
                
                self.device = torch.device('cpu')
        
        print(f"[TRAINER] Final device: {self.device}")
        logger.info(f"[TRAINER] Final device: {self.device}")
        
        # Initialize model with comprehensive logging
        print("[TRAINER] Initializing STGNN classification model...")
        logger.info("[TRAINER] Initializing STGNN classification model...")
        
        num_nodes = len(config.assets)
        input_dim = len(config.features)
        
        print(f"[TRAINER] Model parameters - Nodes: {num_nodes}, Input dim: {input_dim}, Hidden dim: {config.hidden_dim}")
        logger.info(f"[TRAINER] Model parameters - Nodes: {num_nodes}, Input dim: {input_dim}, Hidden dim: {config.hidden_dim}")
        
        self.model = STGNNClassificationModel(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_classes=3,  # Down, No Direction, Up
            num_layers=config.num_layers,
            dropout=config.dropout,
            kernel_size=config.kernel_size
        )
        
        print("[TRAINER] Moving model to device...")
        logger.info("[TRAINER] Moving model to device...")
        self.model.to(self.device)
        
        # Verify model device placement
        model_device = next(self.model.parameters()).device
        print(f"[TRAINER] Model device: {model_device}")
        logger.info(f"Model device: {model_device}")
        
        if model_device != self.device:
            print(f"[TRAINER] WARNING: Model device mismatch! Expected: {self.device}, Actual: {model_device}")
            logger.warning(f"[TRAINER] WARNING: Model device mismatch! Expected: {self.device}, Actual: {model_device}")
        
        # Initialize optimizer with logging
        print(f"[TRAINER] Initializing optimizer with learning rate: {config.learning_rate}")
        logger.info(f"[TRAINER] Initializing optimizer with learning rate: {config.learning_rate}")
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=getattr(config, 'weight_decay', 0.0)
        )
        
        # Set up loss function with comprehensive logging
        print("[TRAINER] Setting up loss function...")
        logger.info("[TRAINER] Setting up loss function...")
        
        if class_weights is not None:
            # Use pre-calculated class weights
            print(f"[TRAINER] Using pre-calculated class weights: {class_weights}")
            logger.info(f"[TRAINER] Using pre-calculated class weights: {class_weights}")
            
            class_weights = class_weights.to(self.device)
            self.criterion = WeightedFocalLoss(
                class_weights=class_weights,
                alpha=focal_alpha,
                gamma=focal_gamma
            )
            print(f"[TRAINER] WeightedFocalLoss created with alpha={focal_alpha}, gamma={focal_gamma}")
            logger.info(f"[TRAINER] WeightedFocalLoss created with alpha={focal_alpha}, gamma={focal_gamma}")
        else:
            # Calculate class weights from data
            print("[TRAINER] Calculating class weights from data...")
            logger.info("[TRAINER] Calculating class weights from data...")
            
            calculated_weights = self._calculate_class_weights()
            calculated_weights = calculated_weights.to(self.device)
            self.criterion = WeightedFocalLoss(
                class_weights=calculated_weights,
                alpha=focal_alpha,
                gamma=focal_gamma
            )
            print(f"[TRAINER] WeightedFocalLoss created with calculated weights: {calculated_weights}")
            logger.info(f"[TRAINER] WeightedFocalLoss created with calculated weights: {calculated_weights}")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Store adjacency matrix
        self.adj = None
        
        print("[TRAINER] ClassificationSTGNNTrainer initialization completed successfully")
        logger.info("[TRAINER] ClassificationSTGNNTrainer initialization completed successfully")
        
    def _calculate_class_weights(self):
        """Calculate class weights to handle imbalance with comprehensive logging"""
        print("[TRAINER] Calculating class weights...")
        logger.info("[TRAINER] Calculating class weights...")
        
        # Prepare data to get target distribution
        print("[TRAINER] Preparing data for class weight calculation...")
        logger.info("[TRAINER] Preparing data for class weight calculation...")
        
        X, adj, y = self.data_processor.prepare_data()
        y_flat = y.flatten().numpy()
        
        print(f"[TRAINER] Data shapes - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
        logger.info(f"[TRAINER] Data shapes - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
        
        # Convert to classes
        print(f"[TRAINER] Converting returns to classes using threshold: {self.price_threshold}")
        logger.info(f"[TRAINER] Converting returns to classes using threshold: {self.price_threshold}")
        
        classes = self._returns_to_classes(y_flat)
        class_counts = Counter(classes)
        
        print(f"[TRAINER] Class distribution: {dict(class_counts)}")
        logger.info(f"[TRAINER] Class distribution: {dict(class_counts)}")
        
        # Calculate weights (inverse frequency)
        total_samples = len(classes)
        class_weights = torch.FloatTensor([
            total_samples / (len(class_counts) * class_counts[i]) 
            for i in range(3)
        ])
        
        print(f"[TRAINER] Calculated class weights: {class_weights}")
        logger.info(f"[TRAINER] Calculated class weights: {class_weights}")
        
        return class_weights
    
    def _returns_to_classes(self, returns):
        """Convert returns to classes (0=down, 1=no direction, 2=up) with logging"""
        print(f"[TRAINER] Converting {len(returns)} returns to classes with threshold: {self.price_threshold}")
        logger.info(f"[TRAINER] Converting {len(returns)} returns to classes with threshold: {self.price_threshold}")
        
        classes = np.ones(len(returns), dtype=int)  # Default to no direction
        classes[returns > self.price_threshold] = 2   # Up
        classes[returns < -self.price_threshold] = 0  # Down
        
        # Log class distribution
        unique_classes, class_counts = np.unique(classes, return_counts=True)
        class_dist = dict(zip(unique_classes, class_counts))
        print(f"[TRAINER] Class distribution: {class_dist}")
        logger.info(f"[TRAINER] Class distribution: {class_dist}")
        
        return classes
    
    def prepare_classification_data(self):
        """Prepare data for classification with comprehensive logging"""
        print("[TRAINER] Preparing classification data...")
        logger.info("[TRAINER] Preparing classification data...")
        
        if self.start_time and self.end_time:
            print(f"[TRAINER] Data time range: {self.start_time} to {self.end_time}")
            logger.info(f"[TRAINER] Data time range: {self.start_time} to {self.end_time}")
        
        # Pass time window parameters to data processor
        print("[TRAINER] Calling data processor prepare_data...")
        logger.info("[TRAINER] Calling data processor prepare_data...")
        
        X, adj, y = self.data_processor.prepare_data(self.start_time, self.end_time)
        
        print(f"[TRAINER] Data processor returned - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
        logger.info(f"[TRAINER] Data processor returned - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
        
        # Validate data
        if X is None or adj is None or y is None:
            print("[TRAINER] ERROR: Data processor returned None values")
            logger.error("[TRAINER] ERROR: Data processor returned None values")
            raise ValueError("Data processor returned None values")
        
        if len(X) == 0 or len(y) == 0:
            print("[TRAINER] ERROR: Data processor returned empty tensors")
            logger.error("[TRAINER] ERROR: Data processor returned empty tensors")
            raise ValueError("Data processor returned empty tensors")
        
        # Check for NaN/Inf values
        if torch.isnan(X).any() or torch.isinf(X).any():
            print("[TRAINER] ERROR: NaN/Inf detected in X tensor")
            logger.error("[TRAINER] ERROR: NaN/Inf detected in X tensor")
            raise ValueError("NaN/Inf detected in X tensor")
        
        if torch.isnan(adj).any() or torch.isinf(adj).any():
            print("[TRAINER] ERROR: NaN/Inf detected in adj tensor")
            logger.error("[TRAINER] ERROR: NaN/Inf detected in adj tensor")
            raise ValueError("NaN/Inf detected in adj tensor")
        
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("[TRAINER] ERROR: NaN/Inf detected in y tensor")
            logger.error("[TRAINER] ERROR: NaN/Inf detected in y tensor")
            raise ValueError("NaN/Inf detected in y tensor")
        
        print("[TRAINER] Data validation passed")
        logger.info("[TRAINER] Data validation passed")
        
        # Store adjacency matrix for later use
        print(f"[TRAINER] Moving adjacency matrix to device: {self.device}")
        logger.info(f"[TRAINER] Moving adjacency matrix to device: {self.device}")
        
        self.adj = adj.to(self.device)
        print(f"[TRAINER] Adjacency matrix device: {self.adj.device}")
        logger.info(f"[TRAINER] Adjacency matrix device: {self.adj.device}")
        
        # Convert returns to classes
        print("[TRAINER] Converting returns to classification targets...")
        logger.info("[TRAINER] Converting returns to classification targets...")
        
        y_flat = y.flatten().numpy()
        y_classes = self._returns_to_classes(y_flat)
        
        # Reshape back to [batch_size, num_nodes]
        y_classes = y_classes.reshape(y.shape)
        
        print(f"[TRAINER] Classification targets shape: {y_classes.shape}")
        logger.info(f"[TRAINER] Classification targets shape: {y_classes.shape}")
        
        # Convert to tensors and move to device
        print("[TRAINER] Converting to tensors and moving to device...")
        logger.info("[TRAINER] Converting to tensors and moving to device...")
        
        y_classes = torch.LongTensor(y_classes).to(self.device)
        
        print(f"[TRAINER] Final tensors - X: {X.shape}, adj: {self.adj.shape}, y_classes: {y_classes.shape}")
        logger.info(f"[TRAINER] Final tensors - X: {X.shape}, adj: {self.adj.shape}, y_classes: {y_classes.shape}")
        
        return X, adj, y_classes
    
    def train_epoch(self, train_loader):
        """Train for one epoch with comprehensive anomaly detection and logging"""
        print(f"[TRAIN_EPOCH] Starting training epoch...")
        logger.info(f"[TRAIN_EPOCH] Starting training epoch...")
        
        # CRITICAL FIX: Enable anomaly detection for backward pass
        torch.autograd.set_detect_anomaly(True)
        print("[TRAIN_EPOCH] Anomaly detection enabled for backward pass")
        logger.info("[TRAIN_EPOCH] Anomaly detection enabled for backward pass")
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        print(f"[TRAIN_EPOCH] Training on device: {self.device}")
        logger.info(f"[TRAIN_EPOCH] Training on device: {self.device}")
        
        # Log memory usage at start
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            allocated_memory = torch.cuda.memory_allocated(current_device) / 1024**3
            cached_memory = torch.cuda.memory_reserved(current_device) / 1024**3
            print(f"[TRAIN_EPOCH] GPU Memory at start - Allocated: {allocated_memory:.3f}GB, Cached: {cached_memory:.3f}GB")
            logger.info(f"[TRAIN_EPOCH] GPU Memory at start - Allocated: {allocated_memory:.3f}GB, Cached: {cached_memory:.3f}GB")

        batch_count = 0
        for X_batch, y_batch in train_loader:
            batch_count += 1
            print(f"[TRAIN_EPOCH] Processing batch {batch_count}/{len(train_loader)}")
            logger.debug(f"[TRAIN_EPOCH] Processing batch {batch_count}/{len(train_loader)}")
            
            # Move data to device
            print(f"[TRAIN_EPOCH] Moving batch to device: {self.device}")
            logger.debug(f"[TRAIN_EPOCH] Moving batch to device: {self.device}")
            
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            print(f"[TRAIN_EPOCH] Batch shapes - X_batch: {X_batch.shape}, y_batch: {y_batch.shape}")
            logger.debug(f"[TRAIN_EPOCH] Batch shapes - X_batch: {X_batch.shape}, y_batch: {y_batch.shape}")
            
            # CRITICAL FIX: Inspect input features directly
            print(f"[TRAIN_EPOCH] Validating input tensors...")
            logger.debug(f"[TRAIN_EPOCH] Validating input tensors...")
            
            if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
                print(f"[TRAIN_EPOCH] ERROR: NaN/Inf detected in X_batch BEFORE model forward pass. Shape: {X_batch.shape}")
                logger.error(f"NaN/Inf detected in X_batch BEFORE model forward pass. Shape: {X_batch.shape}")
                logger.error(f"X_batch stats: min={X_batch.min().item()}, max={X_batch.max().item()}, mean={X_batch.mean().item()}")
                continue
            else:
                print(f"[TRAIN_EPOCH] X_batch validation passed")
                logger.debug(f"[TRAIN_EPOCH] X_batch validation passed")
            
            if self.adj is not None:
                if torch.isnan(self.adj).any() or torch.isinf(self.adj).any():
                    print(f"[TRAIN_EPOCH] ERROR: NaN/Inf detected in self.adj BEFORE model forward pass. Shape: {self.adj.shape}")
                    logger.error(f"NaN/Inf detected in self.adj BEFORE model forward pass. Shape: {self.adj.shape}")
                    logger.error(f"Adj stats: min={self.adj.min().item()}, max={self.adj.max().item()}, mean={self.adj.mean().item()}")
                    continue
                else:
                    print(f"[TRAIN_EPOCH] Adjacency matrix validation passed")
                    logger.debug(f"[TRAIN_EPOCH] Adjacency matrix validation passed")
            else:
                print(f"[TRAIN_EPOCH] ERROR: self.adj is None - cannot validate adjacency matrix")
                logger.error("self.adj is None - cannot validate adjacency matrix")
                continue

            # Ensure adjacency matrix is on correct device
            if self.adj is not None:
                if self.adj.device != self.device:
                    print(f"[TRAIN_EPOCH] Moving adjacency matrix from {self.adj.device} to {self.device}")
                    logger.info(f"[TRAIN_EPOCH] Moving adjacency matrix from {self.adj.device} to {self.device}")
                    self.adj = self.adj.to(self.device, non_blocking=False)
                else:
                    print(f"[TRAIN_EPOCH] Adjacency matrix already on correct device: {self.adj.device}")
                    logger.debug(f"[TRAIN_EPOCH] Adjacency matrix already on correct device: {self.adj.device}")
            else:
                print(f"[TRAIN_EPOCH] ERROR: self.adj is None right before model forward pass!")
                logger.error("self.adj is None right before model forward pass!")
                return float('inf')  # Fail fast if adj is still None

            # Forward pass
            print(f"[TRAIN_EPOCH] Starting forward pass...")
            logger.debug(f"[TRAIN_EPOCH] Starting forward pass...")
            
            self.optimizer.zero_grad()
            logits, _ = self.model(X_batch, self.adj)
            
            print(f"[TRAIN_EPOCH] Forward pass completed - logits shape: {logits.shape}")
            logger.debug(f"[TRAIN_EPOCH] Forward pass completed - logits shape: {logits.shape}")
            
            # CRITICAL FIX: Add detailed logging for logits after forward pass
            print(f"[TRAIN_EPOCH] Validating logits after forward pass...")
            logger.debug(f"[TRAIN_EPOCH] Validating logits after forward pass...")
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[TRAIN_EPOCH] ERROR: Invalid logits detected (NaN or Inf) after model forward pass")
                logger.error("Invalid logits detected (NaN or Inf) after model forward pass")
                logger.error(f"Logits stats: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
                logger.error(f"X_batch stats: min={X_batch.min().item()}, max={X_batch.max().item()}, mean={X_batch.mean().item()}")
                logger.error(f"Adj stats: min={self.adj.min().item()}, max={self.adj.max().item()}, mean={self.adj.mean().item()}")
                continue
            else:
                print(f"[TRAIN_EPOCH] Logits validation passed")
                logger.debug(f"[TRAIN_EPOCH] Logits validation passed")

            # Model now returns logits in shape [batch_size * num_nodes, num_classes]
            # and y_batch needs to be flattened to [batch_size * num_nodes]
            print(f"[TRAIN_EPOCH] Reshaping y_batch for loss calculation...")
            logger.debug(f"[TRAIN_EPOCH] Reshaping y_batch for loss calculation...")
            
            y_batch = y_batch.view(-1)  # [batch_size * num_nodes]
            print(f"[TRAIN_EPOCH] Reshaped y_batch shape: {y_batch.shape}")
            logger.debug(f"[TRAIN_EPOCH] Reshaped y_batch shape: {y_batch.shape}")

            # Validate shapes before loss calculation
            print(f"[TRAIN_EPOCH] Validating shapes before loss calculation...")
            logger.debug(f"[TRAIN_EPOCH] Validating shapes before loss calculation...")
            
            if logits.shape[0] != y_batch.shape[0]:
                print(f"[TRAIN_EPOCH] ERROR: Shape mismatch: logits {logits.shape} vs y_batch {y_batch.shape}")
                logger.error(f"[TRAIN_EPOCH] ERROR: Shape mismatch: logits {logits.shape} vs y_batch {y_batch.shape}")
                raise ValueError(f"Shape mismatch: logits {logits.shape} vs y_batch {y_batch.shape}")
            else:
                print(f"[TRAIN_EPOCH] Shape validation passed")
                logger.debug(f"[TRAIN_EPOCH] Shape validation passed")

            print(f"[TRAIN_EPOCH] Calculating loss...")
            logger.debug(f"[TRAIN_EPOCH] Calculating loss...")
            
            loss = self.criterion(logits, y_batch)
            
            print(f"[TRAIN_EPOCH] Loss calculated: {loss.item():.6f}")
            logger.debug(f"[TRAIN_EPOCH] Loss calculated: {loss.item():.6f}")

            # Check for infinite or NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[TRAIN_EPOCH] ERROR: Invalid loss detected: {loss.item()}")
                logger.error(f"Invalid loss detected: {loss.item()}")
                logger.error(f"Logits stats: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
                logger.error(f"Y batch stats: min={y_batch.min().item()}, max={y_batch.max().item()}")
                continue
            else:
                print(f"[TRAIN_EPOCH] Loss validation passed")
                logger.debug(f"[TRAIN_EPOCH] Loss validation passed")

            # Backward pass
            print(f"[TRAIN_EPOCH] Starting backward pass...")
            logger.debug(f"[TRAIN_EPOCH] Starting backward pass...")
            
            loss.backward()
            print(f"[TRAIN_EPOCH] Backward pass completed")
            logger.debug(f"[TRAIN_EPOCH] Backward pass completed")
            
            # CRITICAL FIX: Gradient clipping for numerical stability
            print(f"[TRAIN_EPOCH] Applying gradient clipping...")
            logger.debug(f"[TRAIN_EPOCH] Applying gradient clipping...")
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            print(f"[TRAIN_EPOCH] Gradient clipping applied")
            logger.debug(f"[TRAIN_EPOCH] Gradient clipping applied")
            
            self.optimizer.step()
            print(f"[TRAIN_EPOCH] Optimizer step completed")
            logger.debug(f"[TRAIN_EPOCH] Optimizer step completed")

            total_loss += loss.item()
            print(f"[TRAIN_EPOCH] Updated total loss: {total_loss:.6f}")
            logger.debug(f"[TRAIN_EPOCH] Updated total loss: {total_loss:.6f}")

            # Calculate accuracy
            print(f"[TRAIN_EPOCH] Calculating accuracy...")
            logger.debug(f"[TRAIN_EPOCH] Calculating accuracy...")
            
            _, predicted = torch.max(logits, 1)
            batch_correct = (predicted == y_batch).sum().item()
            batch_total = y_batch.size(0)
            
            total += batch_total
            correct += batch_correct
            
            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0.0
            print(f"[TRAIN_EPOCH] Batch accuracy: {batch_accuracy:.4f} ({batch_correct}/{batch_total})")
            logger.debug(f"[TRAIN_EPOCH] Batch accuracy: {batch_accuracy:.4f} ({batch_correct}/{batch_total})")

        # DEBUG: Check for division by zero issues in train_epoch
        print(f"[TRAIN_EPOCH] Training epoch completed. Calculating final metrics...")
        logger.info(f"[TRAIN_EPOCH] Training epoch completed. Calculating final metrics...")
        
        print(f"[TRAIN_EPOCH] Final counts - Total loss: {total_loss:.6f}, Correct: {correct}, Total: {total}")
        logger.info(f"[TRAIN_EPOCH] Final counts - Total loss: {total_loss:.6f}, Correct: {correct}, Total: {total}")
        
        if len(train_loader) == 0:
            print(f"[TRAIN_EPOCH] ERROR: train_loader is unexpectedly empty immediately before division!")
            logger.error("DEBUG: train_loader is unexpectedly empty immediately before division!")
            logger.error(f"DEBUG: current correct: {correct}, total: {total}")
            raise ZeroDivisionError("train_loader is empty, preventing loss/accuracy calculation.")
        
        if total == 0:
            print(f"[TRAIN_EPOCH] ERROR: 'total' samples processed is zero in training. This will cause ZeroDivisionError for accuracy.")
            logger.error("DEBUG: 'total' samples processed is zero in training. This will cause ZeroDivisionError for accuracy.")
            # For training, we can't proceed with zero samples
            raise ZeroDivisionError("No samples processed in training, preventing accuracy calculation.")

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        print(f"[TRAIN_EPOCH] Final metrics - Average loss: {avg_loss:.6f}, Accuracy: {accuracy:.6f}")
        logger.info(f"[TRAIN_EPOCH] Final metrics - Average loss: {avg_loss:.6f}, Accuracy: {accuracy:.6f}")
        
        # Log memory usage at end
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            allocated_memory = torch.cuda.memory_allocated(current_device) / 1024**3
            cached_memory = torch.cuda.memory_reserved(current_device) / 1024**3
            print(f"[TRAIN_EPOCH] GPU Memory at end - Allocated: {allocated_memory:.3f}GB, Cached: {cached_memory:.3f}GB")
            logger.info(f"[TRAIN_EPOCH] GPU Memory at end - Allocated: {allocated_memory:.3f}GB, Cached: {cached_memory:.3f}GB")

        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate model with comprehensive anomaly detection"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Move data to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # CRITICAL FIX: Inspect input features directly
                if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
                    logger.error(f"NaN/Inf detected in X_batch BEFORE model forward pass. Shape: {X_batch.shape}")
                    logger.error(f"X_batch stats: min={X_batch.min().item()}, max={X_batch.max().item()}, mean={X_batch.mean().item()}")
                    continue
                if self.adj is not None:
                    if torch.isnan(self.adj).any() or torch.isinf(self.adj).any():
                        logger.error(f"NaN/Inf detected in self.adj BEFORE model forward pass. Shape: {self.adj.shape}")
                        logger.error(f"Adj stats: min={self.adj.min().item()}, max={self.adj.max().item()}, mean={self.adj.mean().item()}")
                        continue
                else:
                    logger.error("self.adj is None - cannot validate adjacency matrix")
                    continue

                # Ensure adjacency matrix is on correct device
                if self.adj is not None:
                    if self.adj.device != self.device:
                        self.adj = self.adj.to(self.device, non_blocking=False)
                else:
                    logger.error("self.adj is None right before model forward pass!")
                    return float('inf')

                # Forward pass
                logits, _ = self.model(X_batch, self.adj)
                
                # CRITICAL FIX: Add detailed logging for logits after forward pass
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.error("Invalid logits detected (NaN or Inf) after model forward pass")
                    logger.error(f"Logits stats: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
                    logger.error(f"X_batch stats: min={X_batch.min().item()}, max={X_batch.max().item()}, mean={X_batch.mean().item()}")
                    logger.error(f"Adj stats: min={self.adj.min().item()}, max={self.adj.max().item()}, mean={self.adj.mean().item()}")
                    continue

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

        # DEBUG: Check for division by zero issues in validate
        if len(val_loader) == 0:
            logger.error("DEBUG: val_loader is unexpectedly empty immediately before division!")
            logger.error(f"DEBUG: current correct: {correct}, total: {total}")
            raise ZeroDivisionError("val_loader is empty, preventing loss/accuracy calculation.")
        
        if total == 0:
            logger.error("DEBUG: 'total' samples processed is zero in validation. This will cause ZeroDivisionError for accuracy.")
            # For validation, we can't proceed with zero samples
            raise ZeroDivisionError("No samples processed in validation, preventing accuracy calculation.")

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
        unique_before, counts_before = np.unique(y_train_flat.cpu().numpy(), return_counts=True)
        class_dist_before = dict(zip(unique_before, counts_before))
        logger.info(f"Training data class distribution before SMOTE: {class_dist_before}")
        
        # Apply SMOTE at the node level (batch_size * num_nodes level)
        # This preserves the temporal structure while balancing classes
        X_train_node_level = X_train.reshape(batch_size_train * num_nodes, seq_len * input_dim)
        y_train_node_level = y_train.reshape(-1)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(class_dist_before.values()) - 1))
        X_train_balanced_node, y_train_balanced_node = smote.fit_resample(X_train_node_level.cpu().numpy(), y_train_node_level.cpu().numpy())
        
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
        num_epochs=3,
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
        
        # Use a fixed 1-month window: January 1, 2020 to January 31, 2020
        start_time = pd.Timestamp('2020-01-01')
        end_time = pd.Timestamp('2020-01-31')
        
        # Create trainer with Focal Loss parameters from config and fixed time window
        trainer = ClassificationSTGNNTrainer(
            config, 
            data_processor, 
            price_threshold=config.price_threshold if hasattr(config, 'price_threshold') else 0.018,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            start_time=start_time,
            end_time=end_time
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
        timestamp = 'fixedwindow'  # Use a fixed string since datetime.now is not allowed
        
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