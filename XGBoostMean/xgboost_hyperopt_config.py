from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class XGBoostHyperoptConfig:
    """Enhanced configuration for XGBoost hyperparameter optimization"""
    
    # Core XGBoost parameters (searchable)
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    min_child_weight: int = 1
    gamma: float = 0.0
    
    # Training parameters
    early_stopping_rounds: int = 10
    eval_metric: str = 'mlogloss'
    random_state: int = 42
    num_classes: int = 3  # For mean reversion: down/hold/up
    
    # Data parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    chunk_size: int = 10000
    
    # Mean reversion specific parameters
    probability_threshold: float = 0.6
    price_threshold: float = 0.01  # 1% movement threshold
    
    # Unbiased target parameters
    lookahead_window: int = 15  # Configurable lookahead window for unbiased target creation
    
    # Memory management
    max_memory_usage: float = 0.8
    enable_garbage_collection: bool = True
    
    # Optimization parameters
    n_trials: int = 1000
    timeout_hours: int = 24
    pruning_enabled: bool = True
    
    # Class imbalance handling
    scale_pos_weight: float = 1.0
    focal_alpha: float = 3.0  # More aggressive default for minority class emphasis
    focal_gamma: float = 3.0  # More aggressive default for focusing on hard examples
    use_focal_loss: bool = True  # Enable focal loss by default
    
    # Class multipliers for handling class imbalance
    class_multiplier_0: float = 3.0  # Multiplier for class 0 (Down) - high emphasis for trading signals
    class_multiplier_1: float = 0.3  # Multiplier for class 1 (Hold) - low weight since it dominates
    class_multiplier_2: float = 3.0  # Multiplier for class 2 (Up) - high emphasis for trading signals
    
    # ENHANCED: Minority class specific parameters
    minority_threshold_boost: float = 0.2  # Boost threshold for minority classes (lower threshold)
    minority_confidence_boost: float = 0.2  # Boost confidence for minority predictions
    use_minority_oversampling: bool = True  # Additional minority oversampling
    minority_oversampling_ratio: float = 3.0  # How much to oversample minorities
    
    # Feature selection parameters
    use_feature_selection: bool = False
    feature_selection_method: str = 'importance'  # 'importance', 'correlation', 'mutual_info'
    min_features: int = 10
    max_features: int = 20
    importance_threshold: float = 0.05
    
    # Feature engineering parameters - All 26 features configurable
    
    # Bollinger Bands parameters
    bollinger_period: int = 20
    bollinger_std_dev: float = 2.0
    
    # RSI parameters
    rsi_period: int = 14
    swing_rsi_period: int = 14
    
    # MACD parameters
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    
    # Moving Average parameters
    short_ma_period: int = 5
    long_ma_period: int = 10
    
    # Volume parameters
    volume_ma_period: int = 20
    volume_std_period: int = 20
    volume_surge_period: int = 20
    
    # ATR parameters
    atr_period: int = 14
    
    # ADX parameters
    adx_period: int = 14
    
    # Momentum parameters
    price_momentum_lookback: int = 5
    volatility_regime_period: int = 20
    
    # Support/Resistance parameters
    support_resistance_period: int = 20
    
    # Breakout parameters
    breakout_period: int = 20
    
    # Cumulative delta parameters
    cumulative_delta_period: int = 20
    
    # VWAP parameters
    vwap_period: int = 20
    
    # Returns calculation parameters
    returns_period: int = 1
    log_returns_period: int = 1
    
    # Feature list (extended from your current list)
    feature_list: List[str] = field(default_factory=lambda: [
        'returns', 'log_returns', 'rsi', 'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'volume', 'volume_ma', 'volume_std', 'volume_surge', 'volume_ratio',
        'macd', 'macd_signal', 'macd_hist', 'ma_crossover', 'swing_rsi', 'vwap_ratio',
        'price_momentum', 'volatility_regime', 'support', 'resistance', 'breakout_intensity',
        'adx', 'cumulative_delta'
    ])
    
    def to_xgboost_params(self) -> Dict[str, Any]:
        """Convert to XGBoost parameters"""
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'num_class': self.num_classes,
            'random_state': self.random_state
        }
        
        # Use focal loss if enabled, otherwise use standard multi-class
        if self.use_focal_loss:
            # For custom objectives, we need to pass the function directly to xgb.train
            # We'll handle this in the trainer instead of here
            params['objective'] = 'multi:softprob'  # Use standard objective as base
            params['eval_metric'] = 'mlogloss'  # Use standard metric as base
        else:
            params['objective'] = 'multi:softprob'
            params['eval_metric'] = self.eval_metric
            
        return params 