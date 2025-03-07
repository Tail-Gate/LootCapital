from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Type
import pandas as pd
import numpy as np

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType
from utils.volatility_analyzer import VolatilityAnalyzer

@dataclass
class VolatilityEnhancedConfig(TechnicalConfig):
    """Configuration for volatility enhancement layer"""
    # Volatility thresholds
    vol_overbought: float = 80.0  # Percentile threshold for high volatility
    vol_oversold: float = 20.0    # Percentile threshold for low volatility
    
    # Signal adjustment parameters
    vol_signal_weight: float = 0.3  # Weight given to volatility signal
    base_signal_weight: float = 0.7  # Weight given to base strategy signal
    
    # Position sizing adjustments
    reduce_size_high_vol: float = 0.5  # Reduce position by this factor in high volatility
    increase_size_low_vol: float = 1.2  # Increase position by this factor in low volatility
    
    # Stop-loss adjustments
    high_vol_sl_multiplier: float = 1.5  # Widen stops in high volatility
    low_vol_sl_multiplier: float = 0.8   # Tighten stops in low volatility
    
    # Profit target adjustments
    high_vol_tp_multiplier: float = 1.2  # Widen targets in high volatility
    low_vol_tp_multiplier: float = 0.9   # Tighten targets in low volatility
    
    # Feature weights for volatility regimes
    vol_regime_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        super().validate()
        if not self.vol_regime_weights:
            self.vol_regime_weights = {
                'high_vol': {
                    'vol_percentile': 0.3,
                    'atr_ratio': 0.2,
                    'vol_zscore': 0.2,
                    'vol_divergence': 0.2,
                    'atr_rsi': 0.1
                },
                'low_vol': {
                    'vol_percentile': 0.25,
                    'atr_ratio': 0.2,
                    'vol_zscore': 0.2,
                    'vol_divergence': 0.15,
                    'atr_rsi': 0.2
                }
            }

class VolatilityEnhancedStrategy:
    """
    A decorator pattern that enhances existing technical strategies with
    volatility mean reversion signals and risk management adjustments.
    
    This is not a standalone strategy, but rather a layer that can be applied
    to any technical strategy to make it volatility-aware.
    """
    
    def __init__(
        self, 
        base_strategy: TechnicalStrategy,
        config: VolatilityEnhancedConfig = None
    ):
        """
        Initialize volatility enhancement layer
        
        Args:
            base_strategy: The underlying technical strategy to enhance
            config: Configuration for volatility enhancement
        """
        self.base_strategy = base_strategy
        self.config = config or VolatilityEnhancedConfig()
        self.name = f"vol_enhanced_{base_strategy.name}"
        self.vol_analyzer = VolatilityAnalyzer()
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for both base strategy and volatility analysis
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with combined features
        """
        # Get base strategy features
        base_features = self.base_strategy.prepare_features(data)
        
        # Check if intraday data
        is_intraday = self._is_intraday_data(base_features)
        
        # Add volatility features
        enhanced_features = self.vol_analyzer.prepare_volatility_features(
            base_features, 
            is_intraday
        )
        
        return enhanced_features
    
    def calculate_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """
        Calculate combined signals from base strategy and volatility analysis
        
        Args:
            features: DataFrame with technical and volatility features
            
        Returns:
            Tuple of (signal_strength, confidence, trade_type)
        """
        # Get base strategy signal
        base_signal, base_confidence, trade_type = self.base_strategy.calculate_signals(features)
        
        # Get volatility signal
        vol_signal, vol_confidence = self.vol_analyzer.get_volatility_signal(
            features,
            vol_threshold=self.config.vol_overbought,
            vol_oversold=self.config.vol_oversold
        )
        
        # Determine if we're in high or low volatility regime
        current = features.iloc[-1]
        high_vol = current.get('vol_percentile', 50) > self.config.vol_overbought
        low_vol = current.get('vol_percentile', 50) < self.config.vol_oversold
        
        # Adjust weights based on volatility regime
        vol_weight = self.config.vol_signal_weight
        base_weight = self.config.base_signal_weight
        
        if high_vol:
            # In high volatility, give more weight to volatility signal
            vol_weight = min(vol_weight * 1.5, 0.5)
            base_weight = 1 - vol_weight
        elif low_vol:
            # In low volatility, trust base strategy more
            vol_weight = max(vol_weight * 0.7, 0.1)
            base_weight = 1 - vol_weight
        
        # Combine signals
        combined_signal = (base_signal * base_weight) + (vol_signal * vol_weight)
        
        # Adjust confidence based on agreement
        if np.sign(base_signal) == np.sign(vol_signal) and np.sign(base_signal) != 0:
            # Signals agree - boost confidence
            combined_confidence = max(base_confidence, vol_confidence) * 1.2
        elif np.sign(vol_signal) == 0:
            # Neutral volatility signal - use base confidence
            combined_confidence = base_confidence
        else:
            # Signals disagree - reduce confidence
            combined_confidence = base_confidence * 0.8
            
        # Cap confidence at 1.0
        combined_confidence = min(combined_confidence, 1.0)
        
        return combined_signal, combined_confidence, trade_type
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """
        Validate signal with both base strategy and volatility considerations
        
        Args:
            signal: The calculated signal strength
            features: Features DataFrame
            
        Returns:
            Boolean indicating if signal passes validation
        """
        # First, check base strategy validation
        if not self.base_strategy.validate_signal(signal, features):
            return False
            
        # Additional volatility-based validations
        current = features.iloc[-1]
        
        # Skip trades during extremely high volatility unless it's a mean reversion signal
        extremely_high_vol = current.get('vol_percentile', 50) > 95
        if extremely_high_vol and current.get('vol_zscore', 0) < 3:
            return False
            
        # Verify that volatility isn't rapidly accelerating/climaxing
        vol_acceleration = features['atr'].pct_change(5).iloc[-1]
        if abs(vol_acceleration) > 0.5:  # Volatility changing too quickly
            return False
            
        return True
    
    def adjust_position_size(self, base_size: float, features: pd.DataFrame) -> float:
        """
        Adjust position size based on volatility regime
        
        Args:
            base_size: Original position size
            features: Features DataFrame
            
        Returns:
            Adjusted position size
        """
        current = features.iloc[-1]
        
        # Determine volatility regime
        high_vol = current.get('vol_percentile', 50) > self.config.vol_overbought
        low_vol = current.get('vol_percentile', 50) < self.config.vol_oversold
        
        # Adjust size based on regime
        if high_vol:
            return base_size * self.config.reduce_size_high_vol
        elif low_vol:
            return base_size * self.config.increase_size_low_vol
        else:
            return base_size
            
    def adjust_stop_loss(self, base_stop: float, features: pd.DataFrame) -> float:
        """
        Adjust stop-loss distance based on volatility regime
        
        Args:
            base_stop: Original stop-loss distance (e.g., in ATR units)
            features: Features DataFrame
            
        Returns:
            Adjusted stop-loss distance
        """
        current = features.iloc[-1]
        
        # Determine volatility regime
        high_vol = current.get('vol_percentile', 50) > self.config.vol_overbought
        low_vol = current.get('vol_percentile', 50) < self.config.vol_oversold
        
        # Adjust stop based on regime
        if high_vol:
            return base_stop * self.config.high_vol_sl_multiplier
        elif low_vol:
            return base_stop * self.config.low_vol_sl_multiplier
        else:
            return base_stop
            
    def adjust_take_profit(self, base_tp: float, features: pd.DataFrame) -> float:
        """
        Adjust take-profit distance based on volatility regime
        
        Args:
            base_tp: Original take-profit distance
            features: Features DataFrame
            
        Returns:
            Adjusted take-profit distance
        """
        current = features.iloc[-1]
        
        # Determine volatility regime
        high_vol = current.get('vol_percentile', 50) > self.config.vol_overbought
        low_vol = current.get('vol_percentile', 50) < self.config.vol_oversold
        
        # Adjust take-profit based on regime
        if high_vol:
            return base_tp * self.config.high_vol_tp_multiplier
        elif low_vol:
            return base_tp * self.config.low_vol_tp_multiplier
        else:
            return base_tp
    
    def _is_intraday_data(self, df: pd.DataFrame) -> bool:
        """Check if data is intraday"""
        if len(df) < 2:
            return False
        time_diff = df.index[1] - df.index[0]
        return time_diff.total_seconds() < 24 * 60 * 60