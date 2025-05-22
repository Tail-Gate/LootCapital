from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType
from utils.lstm_utils import LSTMAttentionModel, train_lstm, predict_lstm, save_lstm, load_lstm
from utils.model_validation import validate_model, perform_walk_forward_analysis

@dataclass
class LSTMMomentumConfig(TechnicalConfig):
    """Configuration for LSTM-based momentum strategy"""
    # LSTM parameters
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_sequence_length: int = 20
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = True
    lstm_attention_heads: int = 4
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    
    # Feature parameters
    feature_list: List[str] = None
    probability_threshold: float = 0.7
    
    # Model paths
    model_path: str = "models/lstm_momentum.pt"
    
    def __post_init__(self):
        super().validate()
        if self.feature_list is None:
            self.feature_list = [
                # Price-based features
                'returns_1', 'returns_2', 'returns_3',
                'rolling_mean', 'rolling_std',
                'roc_1', 'roc_3', 'roc_5', 'roc_10',
                'norm_roc_1', 'norm_roc_3', 'norm_roc_5', 'norm_roc_10',
                
                # Volume-based features
                'volume_roc_1', 'volume_roc_3', 'volume_roc_5',
                'norm_volume_roc_1', 'norm_volume_roc_3', 'norm_volume_roc_5',
                'volume_spike', 'volume_pressure',
                
                # Order book features
                'order_book_imbalance', 'market_depth',
                'depth_imbalance', 'volume_delta', 'cumulative_delta',
                'significant_imbalance',
                
                # Technical indicators
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'adx', 'atr',
                
                # Volatility features
                'historical_vol', 'realized_vol', 'vol_ratio',
                'volatility_percentile', 'volatility_divergence',
                
                # Time-based features
                'hour_of_day', 'day_of_week', 'is_weekend'
            ]

class LSTMMomentumStrategy(TechnicalStrategy):
    """
    Momentum strategy using Bidirectional LSTM with attention.
    Features are derived from standard exchange data (OHLCV, order book, volume).
    """
    def __init__(self, config: LSTMMomentumConfig = None):
        super().__init__(name="lstm_momentum", config=config or LSTMMomentumConfig())
        self.config: LSTMMomentumConfig = self.config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model if path exists
        if self.config.model_path:
            self.load_model(self.config.model_path)
    
    def _prepare_sequence_data(self, features: pd.DataFrame) -> torch.Tensor:
        """Prepare sequence data for LSTM model"""
        # Get relevant features
        X = features[self.config.feature_list].values
        
        # Create sequences
        sequences = []
        for i in range(len(X) - self.config.lstm_sequence_length + 1):
            sequences.append(X[i:i + self.config.lstm_sequence_length])
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(sequences)
        return X_tensor
    
    def train(self, features: pd.DataFrame, targets: pd.Series):
        """Train the LSTM model"""
        # Prepare data
        X = self._prepare_sequence_data(features)
        y = torch.FloatTensor(targets.values[self.config.lstm_sequence_length-1:])
        
        # Create data loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Initialize model
        input_dim = len(self.config.feature_list)
        self.model = LSTMAttentionModel(
            input_dim=input_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            output_dim=1,
            bidirectional=self.config.lstm_bidirectional
        ).to(self.device)
        
        # Define loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Train model
        self.model = train_lstm(
            self.model,
            dataloader,
            criterion,
            optimizer,
            device=self.device,
            early_stopping_patience=self.config.early_stopping_patience
        )
        
        # Save model
        if self.config.model_path:
            self.save_model(self.config.model_path)
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained LSTM model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare data
        X = self._prepare_sequence_data(features)
        
        # Make predictions
        probabilities, attention_weights = predict_lstm(
            self.model,
            X,
            device=self.device
        )
        
        return probabilities, attention_weights
    
    def save_model(self, path: str) -> None:
        """Save the trained LSTM model"""
        if self.model is not None:
            save_lstm(self.model, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained LSTM model"""
        input_dim = len(self.config.feature_list)
        self.model = load_lstm(
            LSTMAttentionModel,
            path,
            input_dim=input_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            output_dim=1,
            bidirectional=self.config.lstm_bidirectional
        ).to(self.device)
    
    def calculate_signals(self, features: pd.DataFrame) -> Tuple[float, float, TradeType]:
        """Generate trading signals using the LSTM model"""
        if self.model is None:
            # Fall back to traditional signals if model is not available
            return self.calculate_technical_signals(features)
        
        # Get model prediction
        probabilities, attention_weights = self.predict(features)
        prob = probabilities[-1]  # Get last prediction
        
        # Generate signal based on probability threshold
        signal = 1 if prob > self.config.probability_threshold else -1 if prob < (1 - self.config.probability_threshold) else 0
        
        # Calculate confidence based on probability distance from threshold
        confidence = abs(prob - 0.5) * 2  # Scale to [0,1]
        
        # Determine trade type based on data frequency
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
    
    def validate(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        n_splits: int = 5,
        test_size: float = 0.2,
        walk_forward: bool = True,
        initial_train_size: int = None,
        step_size: int = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Validate the LSTM model using time series cross-validation and walk-forward analysis.
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            n_splits: Number of time series splits for cross-validation
            test_size: Proportion of data to use for testing
            walk_forward: Whether to perform walk-forward analysis
            initial_train_size: Initial training set size for walk-forward analysis
            step_size: Number of samples to move forward in each step
            
        Returns:
            Dictionary of validation results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Perform time series cross-validation
        cv_results = validate_model(
            self,
            features,
            targets,
            n_splits=n_splits,
            test_size=test_size
        )
        
        # Perform walk-forward analysis if requested
        if walk_forward:
            if initial_train_size is None:
                initial_train_size = int(len(features) * 0.5)
            if step_size is None:
                step_size = int(len(features) * 0.1)
            
            walk_forward_results = perform_walk_forward_analysis(
                self,
                features,
                targets,
                initial_train_size=initial_train_size,
                step_size=step_size,
                n_splits=n_splits
            )
            
            cv_results['walk_forward'] = walk_forward_results
        
        return cv_results 

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