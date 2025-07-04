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
    seq_len: int = 10
    prediction_horizon: int = 15
    features: List[str] = None
    assets: List[str] = None
    
    # Strategy parameters
    confidence_threshold: float = 0.51
    buy_threshold: float = 0.5
    sell_threshold: float = 0.5
    retrain_interval: int = 24
    
    # Focal Loss parameters for handling class imbalance
    focal_alpha: float = 2.0  # Weighting factor for rare class
    focal_gamma: float = 3.0  # Focusing parameter (1.0, 2.0, 3.0, 5.0)
    
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
            'focal_gamma': self.focal_gamma
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