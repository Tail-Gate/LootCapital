from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from utils.lstm_utils import WeightedFocalCrossEntropyLoss
from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType
from utils.lstm_utils import LSTMAttentionModel, train_lstm, predict_lstm, save_lstm, load_lstm
from utils.model_validation import validate_model, perform_walk_forward_analysis
import time
import sys
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    """Dataset for time series data"""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

@dataclass
class LSTMMomentumConfig(TechnicalConfig):
    """Configuration for LSTM-based momentum strategy"""
    # LSTM parameters
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_sequence_length: int = 20
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = False  # Changed to False for live trading
    lstm_attention_heads: int = 4
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    clip_grad_norm: float = 1.0  # Added clip_grad_norm parameter
    num_epochs: int = 100  # Added num_epochs parameter
    
    # Feature parameters
    feature_list: List[str] = None
    probability_threshold: float = 0.51  # Changed to 51% threshold
    
    # Model paths
    model_path: str = "models/lstm_momentum.pt"
    features_path: str = "models/lstm_momentum_features.parquet"
    target_path: str = "models/lstm_momentum_target.parquet"
    
    def __post_init__(self):
        super().validate()
        if self.feature_list is None:
            self.feature_list = [
                # Price-based features
                'returns', 'log_returns',
                
                # Technical indicators
                'rsi', 'atr',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                
                # Volume features
                'volume_ma', 'volume_std', 'volume_surge',
                
                # Momentum features
                'price_momentum', 'volatility_regime',
                
                # Support/Resistance
                'support', 'resistance',
                
                # Breakout detection
                'breakout_intensity',
                
                # Trend strength
                'adx',
                
                # Cumulative delta
                'cumulative_delta'
            ]

class LSTMMomentumStrategy(TechnicalStrategy):
    """
    Momentum strategy using Bidirectional LSTM with attention.
    Features are derived from standard exchange data (OHLCV, order book, volume).
    """
    def __init__(self, config: LSTMMomentumConfig = None, market_data=None, technical_indicators=None):
        super().__init__(
            name="lstm_momentum",
            config=config or LSTMMomentumConfig(),
            market_data=market_data,
            technical_indicators=technical_indicators
        )
        self.config: LSTMMomentumConfig = self.config
        self.model = None
        self.scaler = StandardScaler()  # Initialize scaler
        
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
    
    def _prepare_sequence_data(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequence data for LSTM training"""
        # Convert features to numpy array
        X = features.values
        
        # Shift targets from [-1, 0, 1] to [0, 1, 2]
        y = targets.values + 1
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.config.lstm_sequence_length):
            X_sequences.append(X[i:(i + self.config.lstm_sequence_length)])
            y_sequences.append(y[i + self.config.lstm_sequence_length - 1])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(np.array(X_sequences))
        y_tensor = torch.LongTensor(np.array(y_sequences))
        
        return X_tensor, y_tensor
    
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """
        Train the LSTM model.
        
        Args:
            features: DataFrame with features
            targets: Series with target values
        """
        print("\nStarting training process:")
        print(f"Input features shape: {features.shape}")
        print(f"Input targets shape: {targets.shape}")
        print(f"Unique target values in input: {targets.unique()}")
        
        # Validate target values
        invalid_targets = targets[~targets.isin([-1, 0, 1])]
        if not invalid_targets.empty:
            print(f"\nFound {len(invalid_targets)} invalid target values")
            print("Invalid target values:", invalid_targets.unique())
            print("Replacing invalid targets with 0 (no movement)")
            targets = targets.replace(invalid_targets, 0)
            print(f"Unique target values after replacement: {targets.unique()}")
        
        # Scale features
        print("\nScaling features...")
        scaled_features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Prepare sequence data
        X_sequences, y_sequences = self._prepare_sequence_data(scaled_features, targets)
        
        # Validate target values
        unique_targets = torch.unique(y_sequences)
        print(f"\nValidating target values in sequences:")
        print(f"Unique target values in sequences: {unique_targets}")
        if not all(t in [0, 1, 2] for t in unique_targets):
            print(f"Warning: Invalid target values found: {unique_targets}")
            # Force all invalid values to 1 (no movement)
            y_sequences = torch.where(
                (y_sequences < 0) | (y_sequences > 2),
                torch.tensor(1, dtype=torch.long),
                y_sequences
            )
            print(f"Target values after correction: {torch.unique(y_sequences)}")
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(X_sequences, y_sequences)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        # Initialize model
        print("\nInitializing LSTM model...")
        self.model = LSTMAttentionModel(
            input_dim=len(features.columns),
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            output_dim=3,  # 3 classes: -1, 0, 1
            bidirectional=self.config.lstm_bidirectional
        )
        
        # Move model to CPU for training
        self.model = self.model.to('cpu')
        
        # Initialize loss function and optimizer
        criterion = WeightedFocalCrossEntropyLoss(class_weights=torch.tensor([100.0, 1.0, 100.0]))
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = Path(self.config.model_path).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique checkpoint path based on hyperparameters
        checkpoint_name = f"lstm_momentum_checkpoint_h{self.config.lstm_hidden_dim}_l{self.config.lstm_num_layers}_b{self.config.batch_size}.pt"
        checkpoint_path = str(checkpoint_dir / checkpoint_name)
        
        # Train model with checkpointing
        print("\nTraining LSTM model...")
        try:
            self.model = train_lstm(
                model=self.model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=torch.device('cpu'),  # Use CPU for training
                num_epochs=self.config.num_epochs,  # Use num_epochs from config
                early_stopping_patience=self.config.early_stopping_patience,
                clip_grad_norm=self.config.clip_grad_norm,
                gradient_accumulation_steps=4,  # Accumulate gradients for 4 steps
                model_path=self.config.model_path,  # Changed from checkpoint_path
                periodic_checkpoint_base_path=checkpoint_path,  # Added periodic checkpoint path
                checkpoint_frequency=5  # Save checkpoint every 5 epochs
            )
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print("Starting training from scratch...")
            # If there's an error loading the checkpoint, start fresh
            self.model = train_lstm(
                model=self.model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=torch.device('cpu'),
                num_epochs=self.config.num_epochs,  # Use num_epochs from config
                early_stopping_patience=self.config.early_stopping_patience,
                clip_grad_norm=self.config.clip_grad_norm,
                gradient_accumulation_steps=4,
                model_path=self.config.model_path,  # Changed from checkpoint_path
                periodic_checkpoint_base_path=checkpoint_path,  # Added periodic checkpoint path
                checkpoint_frequency=5
            )
        
        # Save the final model
        print("\nSaving trained model...")
        torch.save(self.model.state_dict(), self.config.model_path)
        print(f"Model saved to: {self.config.model_path}")
        
        # Save the scaler
        print("\nSaving feature scaler...")
        scaler_path = str(Path(self.config.model_path).parent / "lstm_momentum_feature_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {scaler_path}")
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained LSTM model.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        print("\nLSTM Prediction Process:")
        print(f"Input features shape: {features.shape}")
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Scale features
        scaled_features = pd.DataFrame(
            self.scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Prepare sequence data
        print("\nPreparing sequence data:")
        print(f"Input features shape: {scaled_features.shape}")
        
        # Convert features to numpy array
        feature_array = scaled_features.values
        print(f"Feature array shape: {feature_array.shape}")
        
        # Get sequence length from config
        sequence_length = self.config.lstm_sequence_length
        print(f"Sequence length: {sequence_length}")
        
        # Create sequences
        sequences = []
        for i in range(len(feature_array) - sequence_length + 1):
            sequences.append(feature_array[i:i + sequence_length])
        sequences = np.array(sequences)
        print(f"Created sequences shape: {sequences.shape}")
        
        # Convert to PyTorch tensor
        sequences = torch.FloatTensor(sequences)
        
        # Use CPU for prediction to avoid memory issues
        device = 'cpu'
        print("Using CPU for prediction to avoid memory issues")
        sequences = sequences.to(device)
        self.model = self.model.to(device)
        
        # Process in batches
        batch_size = 32  # Smaller batch size to prevent memory issues
        all_probabilities = []
        all_attention_weights = []
        
        # Make predictions in batches
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                probabilities, attention_weights = self.model(batch)
                
                # Move to CPU and convert to numpy
                probabilities = probabilities.cpu().numpy()
                attention_weights = attention_weights.cpu().numpy()
                
                all_probabilities.append(probabilities)
                all_attention_weights.append(attention_weights)
        
        # Concatenate results
        probabilities = np.concatenate(all_probabilities, axis=0)
        attention_weights = np.concatenate(all_attention_weights, axis=0)
        
        print("Final output shapes:")
        print(f"  probabilities: {probabilities.shape}")
        print(f"  attention_weights: {attention_weights.shape}")
        
        return probabilities, attention_weights
    
    def save_model(self, path: str) -> None:
        """Save the trained LSTM model"""
        if self.model is not None:
            save_lstm(self.model, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained LSTM model"""
        input_dim = len(self.config.feature_list)
        self.model, checkpoint_info = load_lstm(
            LSTMAttentionModel,
            path,
            input_dim=input_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            output_dim=3,
            bidirectional=self.config.lstm_bidirectional
        )
        self.model = self.model.to(self.device)
        
        # If we have checkpoint info, print it
        if checkpoint_info:
            print(f"Loaded model from epoch {checkpoint_info['epoch']}")
            print(f"Best loss: {checkpoint_info['best_loss']:.4f}")
    
    def calculate_signals(self, features: pd.DataFrame) -> Tuple[float, float, TradeType]:
        """Generate trading signals using the LSTM model"""
        if self.model is None:
            return self.calculate_technical_signals(features)
        
        # Get model prediction
        probabilities, attention_weights = self.predict(features)
        
        if len(probabilities) == 0:
            return self.calculate_technical_signals(features)
            
        # Get probabilities for each class
        prob_last = probabilities[-1]  # Shape: (3,) for classes [-1, 0, 1]
        
        # Find predicted class and its probability
        predicted_class = np.argmax(prob_last) - 1  # Shift back from [0,1,2] to [-1,0,1]
        predicted_prob = prob_last[np.argmax(prob_last)]
        
        # Only generate signal if probability exceeds threshold
        if predicted_prob >= self.config.probability_threshold:
            signal = float(predicted_class)
            confidence = predicted_prob
        else:
            signal = 0.0  # No trade if confidence is too low
            confidence = predicted_prob
        
        # Determine trade type
        trade_type = TradeType.DAY_TRADE if self.is_intraday_data(features) else TradeType.SWING_TRADE
        
        return signal, confidence, trade_type
    
    def explain(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate model explanations using attention weights"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Get predictions and attention weights
        _, attention_weights = self.predict(features)
        
        # Calculate feature importance based on attention weights
        feature_importance = np.mean(attention_weights, axis=0)
        
        return {
            "attention_weights": attention_weights,
            "feature_importance": feature_importance
        }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the LSTM model"""
        df = self.prepare_base_features(data)
        
        # Calculate lagged returns
        df['returns_1'] = df['close'].pct_change(periods=1)
        df['returns_2'] = df['close'].pct_change(periods=2)
        df['returns_3'] = df['close'].pct_change(periods=3)
        
        # Calculate rolling statistics
        df['rolling_mean'] = df['returns_1'].rolling(window=20).mean()
        df['rolling_std'] = df['returns_1'].rolling(window=20).std()
        
        # Add ROC for multiple periods
        for period in [1, 3, 5, 10]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period)
            df[f'norm_roc_{period}'] = df[f'roc_{period}'] / df['rolling_std']
        
        # Add Volume ROC
        df['volume_roc_1'] = df['volume'].pct_change(periods=1)
        df['volume_roc_3'] = df['volume'].pct_change(periods=3)
        df['volume_roc_5'] = df['volume'].pct_change(periods=5)
        
        # Normalize Volume ROC
        volume_ma = df['volume'].rolling(window=20).mean()
        df['norm_volume_roc_1'] = df['volume_roc_1'] / volume_ma
        df['norm_volume_roc_3'] = df['volume_roc_3'] / volume_ma
        df['norm_volume_roc_5'] = df['volume_roc_5'] / volume_ma
        
        # Order Book Features
        if all(col in df.columns for col in ['bid_volume', 'ask_volume']):
            df['order_book_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
            df['market_depth'] = df['bid_volume'] + df['ask_volume']
            df['depth_imbalance'] = (df['bid_volume'] - df['ask_volume']) / df['market_depth']
            df['volume_delta'] = df['bid_volume'] - df['ask_volume']
            df['cumulative_delta'] = df['volume_delta'].cumsum()
            df['volume_pressure'] = df['volume_delta'] * df['returns_1']
            df['significant_imbalance'] = abs(df['order_book_imbalance']) > 0.2
        else:
            # Placeholder values if order book data isn't available
            df['order_book_imbalance'] = 0
            df['market_depth'] = df['volume']
            df['depth_imbalance'] = 0
            df['volume_delta'] = 0
            df['cumulative_delta'] = 0
            df['volume_pressure'] = 0
            df['significant_imbalance'] = False
        
        # Technical Indicators
        df['rsi'] = self.ti.calculate_rsi(df['close'], period=14)
        macd, signal, hist = self.ti.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        df['adx'] = self.ti.calculate_adx(df['high'], df['low'], df['close'], period=14)
        df['atr'] = self.ti.calculate_atr(df['high'], df['low'], df['close'], period=14)
        
        # Volatility Features
        df['historical_vol'] = df['returns_1'].rolling(window=20).std() * np.sqrt(252)
        df['realized_vol'] = np.sqrt(df['returns_1'].rolling(window=20).apply(lambda x: np.sum(x**2))) * np.sqrt(252)
        df['vol_ratio'] = df['realized_vol'] / df['historical_vol']
        
        # Time-based Features
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df 
    
    def validate_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """
        Validate the LSTM momentum model with specific handling for zero predictions.
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            
        Returns:
            Dictionary of validation metrics
        """
        print("\nStarting LSTM model validation:")
        print(f"Input features shape: {features.shape}")
        print(f"Input targets shape: {targets.shape}")
        print(f"Features index range: {features.index[0]} to {features.index[-1]}")
        print(f"Targets index range: {targets.index[0]} to {targets.index[-1]}")
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        try:
            # Get predictions
            probabilities, _ = self.predict(features)
            predictions = np.argmax(probabilities, axis=1) - 1  # Convert to -1, 0, 1
            
            print(f"\nValidation Process:")
            print(f"Raw probabilities shape: {probabilities.shape}")
            print(f"Raw predictions shape: {predictions.shape}")
            print(f"Raw predictions sample: {predictions[:5]}")
            print(f"Raw predictions distribution: {np.unique(predictions, return_counts=True)}")
            
            # Align targets with predictions (due to sequence length offset)
            sequence_length = self.config.lstm_sequence_length
            targets_aligned = targets.iloc[sequence_length-1:].values
            print(f"Aligned targets shape: {targets_aligned.shape}")
            print(f"Aligned targets sample: {targets_aligned[:5]}")
            print(f"Aligned targets distribution: {np.unique(targets_aligned, return_counts=True)}")
            
            if len(predictions) != len(targets_aligned):
                print(f"\nShape mismatch in validation:")
                print(f"  predictions: {predictions.shape}")
                print(f"  targets_aligned: {targets_aligned.shape}")
                print(f"  sequence_length: {sequence_length}")
                print(f"  features length: {len(features)}")
                print(f"  targets length: {len(targets)}")
                
                # Log the sequence preparation details
                print("\nSequence preparation details:")
                print(f"  Sequence length from config: {sequence_length}")
                print(f"  Expected prediction length: {len(features) - sequence_length + 1}")
                print(f"  Actual prediction length: {len(predictions)}")
                
                raise ValueError(
                    f"Prediction and target shapes don't match. "
                    f"Expected predictions length: {len(features) - sequence_length + 1}, "
                    f"Got: {len(predictions)}"
                )
            
            # Convert predictions to binary (1 for trade, 0 for no trade)
            trade_signals = np.abs(predictions)
            actual_trades = np.abs(targets_aligned) > 0
            print(f"\nTrade Analysis:")
            print(f"Trade signals shape: {trade_signals.shape}")
            print(f"Actual trades shape: {actual_trades.shape}")
            print(f"Number of trade signals: {np.sum(trade_signals)}")
            print(f"Number of actual trades: {np.sum(actual_trades)}")
            print(f"Trade signal distribution: {np.unique(trade_signals, return_counts=True)}")
            print(f"Actual trade distribution: {np.unique(actual_trades, return_counts=True)}")
            
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
            print(f"\nReturns Analysis:")
            print(f"Strategy returns shape: {strategy_returns.shape}")
            print(f"Strategy returns sample: {strategy_returns[:5]}")
            print(f"Returns distribution: min={np.min(strategy_returns):.4f}, max={np.max(strategy_returns):.4f}, mean={np.mean(strategy_returns):.4f}, std={np.std(strategy_returns):.4f}")
            
            # Calculate Sharpe ratio
            returns_mean = np.mean(strategy_returns)
            returns_std = np.std(strategy_returns)
            sharpe_ratio = returns_mean / returns_std if returns_std > 0 else 0
            
            # Calculate Sortino ratio
            downside_returns = strategy_returns[strategy_returns < 0]
            print(f"Downside returns: count={len(downside_returns)}, mean={np.mean(downside_returns):.4f}, std={np.std(downside_returns):.4f}")
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
            sortino_ratio = returns_mean / downside_std if downside_std > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumsum(strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            
            print("\nTrading Metrics:")
            print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"  Sortino Ratio: {sortino_ratio:.4f}")
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

    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate momentum signals using technical indicators"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            # Day Trading Signals
            signals = {
                'momentum': np.sign(current['norm_roc_10']),
                'rsi': -1 if current['rsi'] > self.config.rsi_overbought else 
                        1 if current['rsi'] < self.config.rsi_oversold else 0,
                'macd': np.sign(current['macd_hist']),
                'vwap': np.sign(current['close'] - current['vwap']),
                'volume': np.sign(current['volume_ratio'] - 1)
            }
            
            weights = {
                'momentum': 0.3,
                'rsi': 0.2,
                'macd': 0.2,
                'vwap': 0.2,
                'volume': 0.1
            }
            trade_type = TradeType.DAY_TRADE
            
        else:
            # Swing Trading Signals
            signals = {
                'momentum': np.sign(current['norm_roc_10']),
                'rsi': -1 if current['rsi'] > self.config.rsi_overbought else 
                        1 if current['rsi'] < self.config.rsi_oversold else 0,
                'macd': np.sign(current['macd_hist']),
                'adx': 1 if current['adx'] > self.config.adx_strong_trend else 
                      -1 if current['adx'] < self.config.adx_weak_trend else 0,
                'volume': 1 if current['volume_ratio'] > self.config.min_volume_ratio else -1
            }
            
            weights = {
                'momentum': 0.3,
                'rsi': 0.2,
                'macd': 0.2,
                'adx': 0.2,
                'volume': 0.1
            }
            trade_type = TradeType.SWING_TRADE
        
        # Calculate weighted signal
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            confidence = agreement_ratio * abs(total_signal)
        else:
            confidence = 0
        
        return total_signal, confidence, trade_type

    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Validate trading signal with additional checks."""
        if not self.validate_technical_signal(signal, features):
            return False
        
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            validations = [
                # Sufficient intraday volume
                current['volume_ratio'] > 1.0,
                
                # Not near session VWAP during low-volatility periods
                not (abs(current['vwap_ratio'] - 1) < 0.001 and 
                     current['volatility_regime'] < 0.5),
                
                # Strong enough momentum
                abs(current['norm_roc_10']) > current['atr'] / current['close']
            ]
        else:
            validations = [
                # Strong enough trend
                current['adx'] > self.config.adx_weak_trend,
                
                # Volume confirms trend
                current['volume_ratio'] > self.config.min_volume_ratio,
                
                # Not overextended
                abs(current['ma_crossover']) < current['atr'] * 2
            ]
        
        return all(validations) 