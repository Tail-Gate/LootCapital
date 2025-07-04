import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging

from utils.stgnn_utils import STGNNModel, train_stgnn, predict_stgnn, save_stgnn, load_stgnn
from strategies.base_strategy import BaseStrategy, TradeType
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators
from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from utils.stgnn_trainer import STGNNTrainer

class STGNNStrategy(BaseStrategy):
    """
    Spatio-Temporal Graph Neural Network based trading strategy.
    This strategy uses a graph neural network to model both spatial (cross-asset) and
    temporal (time-series) dependencies in the market data.
    """
    
    def __init__(self,
                 config: STGNNConfig,
                 market_data: MarketData,
                 technical_indicators: TechnicalIndicators):
        """
        Initialize STGNN strategy
        
        Args:
            config: Strategy configuration
            market_data: Market data provider
            technical_indicators: Technical indicators calculator
        """
        super().__init__(config, market_data, technical_indicators)
        
        # Set attributes from config
        self.assets = config.assets
        self.features = config.features
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.kernel_size = config.kernel_size
        self.seq_len = config.seq_len
        self.prediction_horizon = config.prediction_horizon  # Number of candlesticks ahead to predict
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.early_stopping_patience = config.early_stopping_patience
        self.confidence_threshold = config.confidence_threshold
        self.buy_threshold = config.buy_threshold
        self.sell_threshold = config.sell_threshold
        self.retrain_interval = config.retrain_interval
        
        # Initialize model
        self.num_nodes = len(self.assets)
        self.input_dim = len(self.features)
        self.model = STGNNModel(
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,  # Predict single value (price movement)
            num_layers=self.num_layers,
            dropout=self.dropout,
            kernel_size=self.kernel_size
        )
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize signals
        self.signals = {asset: 'HOLD' for asset in self.assets}
        
        # Set model path
        self.model_path = f'models/stgnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        
        # Initialize components
        self.data_processor = STGNNDataProcessor(config, market_data, technical_indicators)
        self.trainer = STGNNTrainer(config, self.data_processor)
        
        # Trading state
        self.positions = {asset: 0 for asset in config.assets}
        self.last_trade_time = None
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for strategy analysis
        
        Args:
            data: Raw market data
            
        Returns:
            DataFrame with technical features
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
        for feature in self.features:
            if feature not in ['close', 'volume']:
                indicator_func = getattr(self.technical_indicators, f'calculate_{feature}')
                features[feature] = indicator_func(data)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
        
    def calculate_signals(self, features: pd.DataFrame) -> Tuple[float, float, TradeType]:
        """
        Calculate trading signals based on features
        
        Args:
            features: DataFrame with technical features
            
        Returns:
            Tuple of (signal_strength, confidence, trade_type)
            - signal_strength: -1.0 to 1.0 (short to long)
            - confidence: 0.0 to 1.0
            - trade_type: TradeType enum
        """
        # Make prediction
        prediction, confidence = self.predict(features)
        
        # Convert prediction to signal strength
        signal_strength = float(prediction)
        
        # Determine trade type based on signal strength
        if abs(signal_strength) < self.buy_threshold:
            trade_type = TradeType.NONE
        elif signal_strength > 0:
            trade_type = TradeType.LONG
        else:
            trade_type = TradeType.SHORT
            
        # Use average confidence across assets
        avg_confidence = float(np.mean(list(confidence.values())))
        
        return signal_strength, avg_confidence, trade_type
        
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """
        Validate if a signal should be acted upon
        
        Args:
            signal: Signal strength (-1.0 to 1.0)
            features: DataFrame with technical features
            
        Returns:
            Boolean indicating if signal is valid
        """
        if abs(signal) < self.buy_threshold:
            return False
            
        # Check if we have enough recent data
        if len(features) < self.seq_len:
            return False
            
        # Check if volatility is too high
        returns = features['returns'].iloc[-20:]
        if returns.std() > 0.05:
            return False
            
        return True
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and prediction
        
        Returns:
            Tuple of (X, adj, y)
            - X: Input features of shape [batch_size, num_nodes, seq_len, input_dim]
            - adj: Adjacency matrix of shape [num_nodes, num_nodes]
            - y: Target values of shape [batch_size, num_nodes, output_dim]
        """
        return self.data_processor.prepare_data()
        
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model
        
        Returns:
            Dictionary containing training and validation losses
        """
        return self.trainer.train()
        
    def predict(self, data: pd.DataFrame) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Make predictions using the trained model
        Returns:
            Tuple of (predictions, confidence)
            - predictions: Tensor of predicted price movements for each asset
            - confidence: Dictionary containing confidence scores for each asset
        """
        # Prepare input data
        X, adj, _ = self.prepare_data()
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        X_last = X[-1:, :, :, :]  # shape: [1, num_nodes, seq_len, input_dim]
        predictions, attention_dict = predict_stgnn(
            model=self.model,
            X=X_last,
            adj=adj,
            device=self.device
        )
        # predictions: [1, num_nodes, 1] -> [num_nodes]
        predictions_tensor = torch.tensor(predictions[0, :, 0])
        confidence = {}
        for i, asset in enumerate(self.assets):
            attention_weights = np.mean([v[0, i, -1] for v in attention_dict.values() if v.shape[1] > i])
            confidence[asset] = float(attention_weights)
        return predictions_tensor, confidence
        
    def generate_signals(self, predictions) -> Dict[str, int]:
        """
        Generate trading signals from predictions
        Args:
            predictions: Model predictions (tensor, array, or float)
        Returns:
            Dictionary mapping asset symbols to signals (-1: sell, 0: hold, 1: buy)
        """
        logger = logging.getLogger("STGNNStrategy.generate_signals")
        signals = {}
        # Convert to numpy array if needed
        if isinstance(predictions, torch.Tensor):
            preds = predictions.detach().cpu().numpy().flatten()
        elif isinstance(predictions, (list, np.ndarray)):
            preds = np.array(predictions).flatten()
        else:
            preds = np.array([predictions])
        for i, asset in enumerate(self.assets):
            pred = preds[i] if i < len(preds) else 0.0
            if pred >= self.buy_threshold:
                signal = 1
            elif pred <= self.sell_threshold:
                signal = -1
            else:
                signal = 0
            signals[asset] = signal
            logger.info(f"Asset: {asset}, Prediction: {pred}, Buy Threshold: {self.buy_threshold}, Sell Threshold: {self.sell_threshold}, Signal: {signal}")
        return signals
        
    def update_positions(self, signals: Dict[str, int], current_prices: Dict[str, float]):
        """
        Update positions based on signals
        
        Args:
            signals: Dictionary mapping asset symbols to signals
            current_prices: Dictionary mapping asset symbols to current prices
        """
        for asset, signal in signals.items():
            current_position = self.positions[asset]
            
            if signal == 1 and current_position <= 0:
                # Buy signal
                self.positions[asset] = 1
            elif signal == -1 and current_position >= 0:
                # Sell signal
                self.positions[asset] = -1
                
    def should_retrain(self, current_time: pd.Timestamp) -> bool:
        """
        Check if model should be retrained
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if model should be retrained
        """
        if self.last_trade_time is None:
            return True
            
        time_since_last_trade = current_time - self.last_trade_time
        return time_since_last_trade.total_seconds() >= self.retrain_interval
        
    def execute_trades(self, current_time: pd.Timestamp) -> Dict[str, Dict[str, float]]:
        """
        Execute trading strategy
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Dictionary containing trade information
        """
        # Check if retraining is needed
        if self.should_retrain(current_time):
            print('Retraining model...')
            self.train()
            self.last_trade_time = current_time
            
        # Get latest data
        X, adj, _ = self.prepare_data()
        
        # Make predictions
        predictions = self.predict(X[-1:])  # Use last sequence
        
        # Generate signals
        signals = self.generate_signals(predictions[0])
        
        # Get current prices
        current_prices = {
            asset: self.market_data.get_latest_price(asset)
            for asset in self.assets
        }
        
        # Update positions
        self.update_positions(signals, current_prices)
        
        # Prepare trade information
        trades = {}
        for asset in self.assets:
            trades[asset] = {
                'signal': signals[asset],
                'position': self.positions[asset],
                'price': current_prices[asset],
                'prediction': predictions[0][self.assets.index(asset)].item()
            }
            
        return trades
        
    def save_state(self, path: str):
        """Save strategy state"""
        state = {
            'positions': self.positions,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'model_state': self.model.state_dict(),
            'config': self.config.to_dict()
        }
        torch.save(state, path)
        
    def load_state(self, path: str):
        """Load strategy state"""
        state = torch.load(path, weights_only=False)  # Allow loading non-tensor objects
        
        # Load positions
        self.positions = state['positions']
        
        # Load last trade time
        if state['last_trade_time']:
            self.last_trade_time = pd.Timestamp(state['last_trade_time'])
        else:
            self.last_trade_time = None
            
        # Load model state
        self.model.load_state_dict(state['model_state'])
        
        # Load config
        self.config = STGNNConfig.from_dict(state['config'])
        
    def update(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Update the strategy with new market data
        
        Args:
            data: Market data for the current time step
            
        Returns:
            Dictionary mapping asset symbols to trading signals (-1: sell, 0: hold, 1: buy)
        """
        # Retrain model periodically
        if self.should_retrain(pd.Timestamp(datetime.now())):
            self.train()
            
        # Generate new signals
        signals = {}
        for asset in self.assets:
            # Get latest data
            asset_data = data[asset] if isinstance(data, dict) else data
            
            # Make prediction
            prediction, confidence = self.predict(asset_data)
            
            # Generate signal
            signals[asset] = self.generate_signals(prediction)[asset]
            
        return signals
        
    def explain(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate model explanations using attention weights and feature importance
        
        Args:
            features: DataFrame with technical features
            
        Returns:
            Dictionary containing:
            - attention_weights: Dictionary of attention weights per layer
            - feature_importance: Array of feature importance scores
            - temporal_importance: Array of temporal importance scores
            - spatial_importance: Array of spatial importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        # Prepare data
        X, adj, _ = self.prepare_data()
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        
        # Select the last sequence for prediction
        # X shape: [batch_size, num_nodes, seq_len, input_dim]
        X_last = X[-1:, :, :, :]  # shape: [1, num_nodes, seq_len, input_dim]
        
        # Get predictions and attention weights
        predictions, attention_dict = predict_stgnn(
            model=self.model,
            X=X_last,
            adj=adj,
            device=self.device
        )
        
        # Calculate feature importance
        feature_importance = np.zeros(len(self.features))
        temporal_importance = np.zeros(self.seq_len)
        spatial_importance = np.zeros(self.num_nodes)
        
        # Aggregate attention weights across layers
        for layer_name, attn_weights in attention_dict.items():
            # Average attention weights across batch dimension
            attn_weights = attn_weights.mean(axis=0)  # [num_nodes, seq_len]
            
            # Update temporal importance
            temporal_importance += attn_weights.mean(axis=0)  # Average across nodes
            
            # Update spatial importance
            spatial_importance += attn_weights.mean(axis=1)  # Average across time steps
            
            # Update feature importance (using last layer's attention)
            if layer_name == f'layer_{self.num_layers-1}_temporal':
                feature_importance = attn_weights.mean(axis=(0, 1))  # Average across nodes and time
        
        # Normalize importance scores
        feature_importance = feature_importance / feature_importance.sum()
        temporal_importance = temporal_importance / temporal_importance.sum()
        spatial_importance = spatial_importance / spatial_importance.sum()
        
        return {
            "attention_weights": attention_dict,
            "feature_importance": feature_importance,
            "temporal_importance": temporal_importance,
            "spatial_importance": spatial_importance
        } 