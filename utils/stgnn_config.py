from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import os

@dataclass
class STGNNConfig:
    """Configuration for STGNN model and training"""
    # Model architecture
    num_nodes: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int = 2
    dropout: float = 0.2
    kernel_size: int = 3
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4  # L2 regularization
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Data parameters
    seq_len: int = 100  # Increased for better capture of technical indicator patterns
    prediction_horizon: int = 15
    features: List[str] = None
    assets: List[str] = None
    
    # Strategy parameters
    confidence_threshold: float = 0.51
    buy_threshold: float = 0.5
    sell_threshold: float = 0.5
    retrain_interval: int = 24
    
    # Focal Loss parameters for handling class imbalance
    focal_alpha: float = 1.0  # Weighting factor for rare class
    focal_gamma: float = 3.0  # Focusing parameter (1.0, 2.0, 3.0, 5.0)
    
    # Feature Engineering Hyperparameters (NEW)
    # RSI parameters
    rsi_period: int = 14
    
    # MACD parameters
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    
    # Bollinger Bands parameters
    bb_period: int = 20
    bb_num_std_dev: float = 2.0
    
    # ATR parameters
    atr_period: int = 14
    
    # ADX parameters
    adx_period: int = 14
    
    # Volume parameters
    volume_ma_period: int = 20
    
    # Momentum parameters
    price_momentum_lookback: int = 5
    
    # Price threshold for classification (fixed for 0.5% movements)
    price_threshold: float = 0.005  # 0.5% threshold
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'STGNNConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'num_nodes': self.num_nodes,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'kernel_size': self.kernel_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'seq_len': self.seq_len,
            'prediction_horizon': self.prediction_horizon,
            'features': self.features,
            'assets': self.assets,
            'confidence_threshold': self.confidence_threshold,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'retrain_interval': self.retrain_interval,
            'focal_alpha': self.focal_alpha,
            'focal_gamma': self.focal_gamma,
            # Feature engineering parameters
            'rsi_period': self.rsi_period,
            'macd_fast_period': self.macd_fast_period,
            'macd_slow_period': self.macd_slow_period,
            'macd_signal_period': self.macd_signal_period,
            'bb_period': self.bb_period,
            'bb_num_std_dev': self.bb_num_std_dev,
            'atr_period': self.atr_period,
            'adx_period': self.adx_period,
            'volume_ma_period': self.volume_ma_period,
            'price_momentum_lookback': self.price_momentum_lookback,
            'price_threshold': self.price_threshold
        }
    
    def save(self, path: str) -> None:
        """Save config to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'STGNNConfig':
        """Load config from file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.num_nodes > 0, "Number of nodes must be positive"
        assert self.input_dim > 0, "Input dimension must be positive"
        assert self.hidden_dim > 0, "Hidden dimension must be positive"
        assert self.output_dim > 0, "Output dimension must be positive"
        assert 0 < self.num_layers <= 10, "Number of layers must be between 1 and 10"
        assert 0 <= self.dropout < 1, "Dropout must be between 0 and 1"
        assert self.kernel_size > 0, "Kernel size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.early_stopping_patience > 0, "Early stopping patience must be positive"
        assert self.seq_len > 0, "Sequence length must be positive"
        assert self.prediction_horizon > 0, "Prediction horizon must be positive"
        assert self.features is not None and len(self.features) > 0, "Features list cannot be empty"
        assert self.assets is not None and len(self.assets) > 0, "Assets list cannot be empty"
        assert 0 <= self.confidence_threshold <= 1, "Confidence threshold must be between 0 and 1"
        assert 0 <= self.buy_threshold <= 1, "Buy threshold must be between 0 and 1"
        assert 0 <= self.sell_threshold <= 1, "Sell threshold must be between 0 and 1"
        assert self.retrain_interval > 0, "Retrain interval must be positive"
        assert 0 <= self.focal_alpha <= 1, "Focal alpha must be between 0 and 1"
        assert self.focal_gamma > 0, "Focal gamma must be positive"
        # Feature engineering parameter validation
        assert 5 <= self.rsi_period <= 50, "RSI period must be between 5 and 50"
        assert 5 <= self.macd_fast_period <= 30, "MACD fast period must be between 5 and 30"
        assert 20 <= self.macd_slow_period <= 50, "MACD slow period must be between 20 and 50"
        assert 5 <= self.macd_signal_period <= 20, "MACD signal period must be between 5 and 20"
        assert 10 <= self.bb_period <= 50, "Bollinger Bands period must be between 10 and 50"
        assert 1.0 <= self.bb_num_std_dev <= 3.0, "Bollinger Bands std dev must be between 1.0 and 3.0"
        assert 5 <= self.atr_period <= 30, "ATR period must be between 5 and 30"
        assert 5 <= self.adx_period <= 30, "ADX period must be between 5 and 30"
        assert 10 <= self.volume_ma_period <= 50, "Volume MA period must be between 10 and 50"
        assert 3 <= self.price_momentum_lookback <= 20, "Price momentum lookback must be between 3 and 20"
        assert 0.001 <= self.price_threshold <= 0.05, "Price threshold must be between 0.1% and 5%" 