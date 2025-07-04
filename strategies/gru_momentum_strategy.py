from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import sys
import gc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import psutil
import warnings

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType
from utils.gru_utils import GRUAttentionModel, train_gru, predict_gru, save_gru, load_gru, WeightedFocalCrossEntropyLoss
from utils.model_validation import validate_model, perform_walk_forward_analysis
import time

class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset for large time series data"""
    def __init__(self, features: pd.DataFrame, targets: pd.Series, sequence_length: int, feature_list: List[str]):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.feature_list = feature_list
        self.total_sequences = len(features) - sequence_length + 1
        
    def __len__(self) -> int:
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence data
        sequence_data = self.features[self.feature_list].iloc[idx:idx + self.sequence_length].values
        target = self.targets.iloc[idx + self.sequence_length - 1]
        
        # Convert to tensors
        X = torch.FloatTensor(sequence_data)
        y = torch.LongTensor([target])
        
        return X, y

@dataclass
class GRUMomentumConfig(TechnicalConfig):
    """Configuration for GRU-based momentum strategy"""
    # GRU parameters
    gru_hidden_dim: int = 64
    gru_num_layers: int = 2
    gru_sequence_length: int = 20
    gru_dropout: float = 0.2
    gru_bidirectional: bool = True
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    
    # Memory management parameters
    max_memory_usage: float = 0.8  # Maximum memory usage as fraction of available memory
    chunk_size: int = 10000  # Number of sequences to process at once
    num_workers: int = 4  # Number of workers for data loading
    
    # Feature parameters
    feature_list: List[str] = None
    probability_threshold: float = 0.7
    
    # Model paths
    model_path: str = "models/gru_momentum.pt"
    
    def __post_init__(self):
        super().__post_init__()
        if self.feature_list is None:
            self.feature_list = [
                'returns', 'log_returns', 'volume', 'volume_ma', 'volume_ratio',
                'rsi', 'macd', 'ma_crossover', 'swing_rsi', 'macd_signal',
                'macd_hist', 'adx', 'vwap_ratio', 'volatility_regime', 'atr'
            ]

class GRUMomentumStrategy(TechnicalStrategy):
    """
    Momentum strategy using GRU with attention.
    Features are derived from standard exchange data (OHLCV, order book, volume).
    """
    def __init__(self, config: GRUMomentumConfig = None):
        super().__init__(name="gru_momentum", config=config or GRUMomentumConfig())
        self.config: GRUMomentumConfig = self.config
        self.model = None
        
        # Initialize device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon MPS (GPU) for training.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for training.")
        
        # Only load model if path exists and we're not in a test environment
        if self.config.model_path and os.path.exists(self.config.model_path) and not any('test' in arg for arg in sys.argv):
            self.load_model(self.config.model_path)
    
    def _check_memory_usage(self) -> bool:
        """Check if current memory usage is below threshold"""
        memory = psutil.virtual_memory()
        return memory.percent < (self.config.max_memory_usage * 100)
    
    def _cleanup_memory(self):
        """Clean up memory by clearing caches and running garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _prepare_sequence_data(self, features: pd.DataFrame) -> torch.Tensor:
        """Prepare sequence data for GRU model with memory efficiency"""
        if not self._check_memory_usage():
            warnings.warn("High memory usage detected. Consider reducing batch size or sequence length.")
        
        # Create memory-efficient dataset
        dataset = MemoryEfficientDataset(
            features=features,
            targets=pd.Series(np.zeros(len(features))),  # Dummy targets for sequence creation
            sequence_length=self.config.gru_sequence_length,
            feature_list=self.config.feature_list
        )
        
        # Create data loader with memory-efficient settings
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Process data in chunks
        sequences = []
        for batch_X, _ in dataloader:
            sequences.append(batch_X)
            if not self._check_memory_usage():
                self._cleanup_memory()
        
        # Concatenate sequences
        X_tensor = torch.cat(sequences, dim=0)
        return X_tensor
    
    def train(self, features: pd.DataFrame, targets: pd.Series):
        """Train the GRU model with memory efficiency"""
        print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        
        # Create memory-efficient dataset
        dataset = MemoryEfficientDataset(
            features=features,
            targets=targets,
            sequence_length=self.config.gru_sequence_length,
            feature_list=self.config.feature_list
        )
        
        # Create data loader with memory-efficient settings
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Initialize model
        input_dim = len(self.config.feature_list)
        self.model = GRUAttentionModel(
            input_dim=input_dim,
            hidden_dim=self.config.gru_hidden_dim,
            num_layers=self.config.gru_num_layers,
            output_dim=3,  # 3 classes: -1, 0, 1
            dropout=self.config.gru_dropout,
            bidirectional=self.config.gru_bidirectional
        ).to(self.device)
        
        # Define loss and optimizer
        class_weights = [
            100.0,  # class -1 (rare, high penalty for missing)
            1.0,   # class 0 (common, low reward for predicting)
            100.0   # class 1 (rare, high penalty for missing)
        ]
        criterion = WeightedFocalCrossEntropyLoss(class_weights=class_weights)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Train model with memory monitoring
        self.model = train_gru(
            self.model,
            dataloader,
            criterion,
            optimizer,
            device=self.device,
            early_stopping_patience=self.config.early_stopping_patience
        )
        
        # Clean up memory
        self._cleanup_memory()
        
        # Save model
        if self.config.model_path:
            self.save_model(self.config.model_path)
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained GRU model with memory efficiency"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Create memory-efficient dataset
        dataset = MemoryEfficientDataset(
            features=features,
            targets=pd.Series(np.zeros(len(features))),  # Dummy targets
            sequence_length=self.config.gru_sequence_length,
            feature_list=self.config.feature_list
        )
        
        # Create data loader with memory-efficient settings
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Make predictions in batches
        all_probabilities = []
        all_attention_weights = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs, attention_weights = self.model(batch_X)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_probabilities.append(probabilities.cpu())
                all_attention_weights.append(attention_weights.cpu())
                
                if not self._check_memory_usage():
                    self._cleanup_memory()
        
        # Concatenate results
        probabilities = torch.cat(all_probabilities, dim=0).numpy()
        attention_weights = torch.cat(all_attention_weights, dim=0).numpy()
        
        # Convert probabilities to class predictions (-1, 0, 1)
        predictions = np.argmax(probabilities, axis=1) - 1
        
        return predictions, attention_weights
    
    def save_model(self, path: str) -> None:
        """Save the GRU model"""
        if self.model is not None:
            save_gru(self.model, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained GRU model"""
        if not os.path.exists(path):
            return  # Don't try to load if file doesn't exist
        input_dim = len(self.config.feature_list)
        self.model = load_gru(
            GRUAttentionModel,
            path,
            input_dim=input_dim,
            hidden_dim=self.config.gru_hidden_dim,
            num_layers=self.config.gru_num_layers,
            output_dim=3, # Changed from 1 to 3
            dropout=self.config.gru_dropout,
            bidirectional=self.config.gru_bidirectional
        ).to(self.device)
    
    def explain(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Return attention weights for the given input features.
        Features should be a DataFrame of shape (seq_len, features).
        Returns attention weights per timestep and a summary (mean attention per timestep).
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare data
        X = self._prepare_sequence_data(features)
        
        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            _, attn_weights = self.model(X.to(self.device))
        
        attn_weights = attn_weights.cpu().numpy().squeeze()  # (batch, seq_len, 1) or (seq_len,)
        
        # If batch size is 1, reduce to (seq_len,)
        if attn_weights.ndim == 2:
            attn_weights = attn_weights[0]
        
        # Return both raw attention and a summary (mean per timestep)
        attn_summary = attn_weights.mean(axis=-1) if attn_weights.ndim > 1 else attn_weights
        
        return {
            "attention_weights": attn_weights,
            "attention_summary": attn_summary
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals using the GRU model
        
        Args:
            data: Raw OHLCV data with returns
            
        Returns:
            DataFrame with probability and signal columns
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Get predictions and attention weights
        predictions, attention_weights = self.predict(features)
        
        # Get probabilities for each class
        self.model.eval()
        with torch.no_grad():
            X = self._prepare_sequence_data(features)
            outputs, _ = self.model(X.to(self.device))
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # Align index to match predictions
        aligned_index = data.index[-len(predictions):]
        signals = pd.DataFrame(index=aligned_index)
        
        # Store probabilities for each class
        signals['prob_down'] = probabilities[:, 0]  # Probability of class -1
        signals['prob_neutral'] = probabilities[:, 1]  # Probability of class 0
        signals['prob_up'] = probabilities[:, 2]  # Probability of class 1
        
        # Generate signals based on probability threshold
        signals['signal'] = 0
        signals.loc[signals['prob_up'] > self.config.probability_threshold, 'signal'] = 1
        signals.loc[signals['prob_down'] > self.config.probability_threshold, 'signal'] = -1
        
        return signals

    def calculate_technical_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical signals for validation
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Add basic technical indicators for validation
        signals['rsi'] = self._calculate_rsi(data['close'])
        signals['macd'] = self._calculate_macd(data['close'])
        
        # Generate signals based on technical indicators
        signals.loc[signals['rsi'] < 30, 'signal'] = 1  # Oversold
        signals.loc[signals['rsi'] > 70, 'signal'] = -1  # Overbought
        
        return signals
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the GRU model
        """
        features = pd.DataFrame(index=data.index)
        
        # Add basic price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log1p(features['returns'])
        
        # Add volume features
        features['volume'] = data['volume']
        features['volume_ma'] = data['volume'].rolling(window=20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_ma']
        
        # Add technical indicators
        features['rsi'] = self._calculate_rsi(data['close'])
        features['macd'] = self._calculate_macd(data['close'])
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def validate_signal(self, signal: int, data: pd.DataFrame, index: int) -> bool:
        """
        Validate a trading signal based on technical indicators
        """
        if signal == 0:
            return True
            
        # Get technical indicators
        rsi = self._calculate_rsi(data['close'])[index]
        macd = self._calculate_macd(data['close'])[index]
        
        # Validate long signal
        if signal == 1:
            return rsi < 30 and macd > 0
            
        # Validate short signal
        if signal == -1:
            return rsi > 70 and macd < 0
            
        return False
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD technical indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2

    def validate_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """
        Validate the GRU momentum model with specific handling for zero predictions.
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            
        Returns:
            Dictionary of validation metrics
        """
        print("\nStarting GRU model validation:")
        print(f"Input features shape: {features.shape}")
        print(f"Input targets shape: {targets.shape}")
        
        # Get predictions
        predictions, _ = self.predict(features)
        predictions = np.asarray(predictions).flatten()
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Raw predictions sample: {predictions[:5]}")
        
        # Align targets with predictions (due to sequence length offset)
        targets_aligned = targets.iloc[self.config.gru_sequence_length-1:].values
        print(f"Aligned targets shape: {targets_aligned.shape}")
        print(f"Aligned targets sample: {targets_aligned[:5]}")
        
        if len(predictions) != len(targets_aligned):
            print(f"Shape mismatch in validation:")
            print(f"  predictions: {predictions.shape}")
            print(f"  targets_aligned: {targets_aligned.shape}")
            raise ValueError("Prediction and target shapes don't match")
        
        # Convert predictions to binary (1 for trade, 0 for no trade)
        trade_signals = np.abs(predictions)
        actual_trades = np.abs(targets_aligned) > 0
        print(f"Trade signals shape: {trade_signals.shape}")
        print(f"Actual trades shape: {actual_trades.shape}")
        print(f"Number of trade signals: {np.sum(trade_signals)}")
        print(f"Number of actual trades: {np.sum(actual_trades)}")
        
        # Check if we have any trades
        if np.sum(trade_signals) == 0:
            print("Warning: No trade signals generated, returning zero metrics")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'directional_accuracy': 0.0,
                'information_coefficient': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        try:
            # Calculate accuracy metrics
            accuracy = accuracy_score(trade_signals, actual_trades)
            precision = precision_score(trade_signals, actual_trades, zero_division=0)
            recall = recall_score(trade_signals, actual_trades, zero_division=0)
            f1 = f1_score(trade_signals, actual_trades, zero_division=0)
            
            print("\nClassification Metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            # Calculate directional accuracy
            directional_accuracy = np.mean(np.sign(predictions) == np.sign(targets_aligned))
            print(f"  Directional Accuracy: {directional_accuracy:.4f}")
            
            # Calculate information coefficient (IC)
            ic = np.corrcoef(predictions, targets_aligned)[0, 1]
            print(f"  Information Coefficient: {ic:.4f}")
            
            # Calculate trading metrics
            strategy_returns = predictions * targets_aligned
            print(f"\nTrading Metrics:")
            print(f"  Strategy returns shape: {strategy_returns.shape}")
            print(f"  Strategy returns sample: {strategy_returns[:5]}")
            
            # Calculate excess returns (assuming risk-free rate of 0 for simplicity)
            excess_returns = strategy_returns
            print(f"  Excess returns mean: {np.mean(excess_returns):.6f}")
            print(f"  Excess returns std: {np.std(excess_returns):.6f}")
            
            # Calculate Sharpe ratio
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0
            print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
            
            # Calculate Sortino ratio
            downside_returns = excess_returns[excess_returns < 0]
            print(f"  Number of downside returns: {len(downside_returns)}")
            
            downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1
            print(f"  Downside standard deviation: {downside_std:.6f}")
            
            if downside_std == 0:
                print("  Warning: Downside standard deviation is zero, setting Sortino ratio to 0")
                sortino_ratio = 0
            else:
                sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252)
            print(f"  Sortino Ratio: {sortino_ratio:.4f}")
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (running_max - cumulative_returns) / running_max
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            print(f"  Max Drawdown: {max_drawdown:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'directional_accuracy': directional_accuracy,
                'information_coefficient': ic,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            print(f"\nError during validation: {str(e)}")
            print(f"Error details:", e.__class__.__name__)
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise