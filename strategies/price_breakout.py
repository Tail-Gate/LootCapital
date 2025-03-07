from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class BreakoutConfig(TechnicalConfig):
    """Configuration for breakout strategy"""
    # ATR Multipliers
    day_atr_multiplier: float = 1.5
    swing_atr_multiplier: float = 2.0
    
    # Timeframes
    day_range_period: int = 20  # for 20-period ATR
    swing_range_period: int = 30  # for 30-day historical ranges
    
    # Volume thresholds
    min_volume_surge_ratio: float = 1.5  # Minimum volume increase for confirmation
    sustained_volume_periods: int = 3  # Number of periods for sustained volume
    
    # Bollinger Band parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Support/Resistance
    support_resistance_periods: int = 20
    support_resistance_threshold: float = 0.02  # 2% threshold for level identification
    
    # Volatility thresholds
    min_volatility_percentile: float = 20  # Minimum volatility for valid breakout
    max_volatility_percentile: float = 80  # Maximum volatility for valid breakout
    
    # Feature weights for different conditions
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'low_volatility': {
                    'price_break': 0.4,
                    'volume_surge': 0.3,
                    'volatility_expansion': 0.3
                },
                'high_volatility': {
                    'price_break': 0.3,
                    'volume_surge': 0.4,
                    'volatility_expansion': 0.3
                }
            }

class BreakoutStrategy(TechnicalStrategy):
    def __init__(self, config: BreakoutConfig = None):
        super().__init__(name="breakout", config=config or BreakoutConfig())
        self.config: BreakoutConfig = self.config  
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare breakout specific features"""
        df = self.prepare_base_features(data)
        
        # Calculate support and resistance levels
        self._calculate_support_resistance(df)
        
        # Calculate volatility features
        self._calculate_volatility_features(df)
        
        # Calculate Bollinger Bands
        self._calculate_bollinger_bands(df)
        
        # VWAP analysis
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        
        return df
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> None:
        """Calculate support and resistance levels"""
        for trade_type in ['day', 'swing']:
            period = getattr(self.config, f'{trade_type}_range_period')
            atr_mult = getattr(self.config, f'{trade_type}_atr_multiplier')
            
            df[f'{trade_type}_high'] = df['high'].rolling(window=period).max()
            df[f'{trade_type}_low'] = df['low'].rolling(window=period).min()
            df[f'{trade_type}_threshold'] = df['atr'] * atr_mult
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> None:
        """Calculate volatility features"""
        for trade_type in ['day', 'swing']:
            df[f'{trade_type}_volatility_expansion'] = (
                df['volatility'] / df['volatility'].rolling(window=self.config.day_range_period).mean()
            )
            df[f'{trade_type}_volume_surge'] = (
                df['volume'] / df['volume'].rolling(window=self.config.day_range_period).mean()
            )
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> None:
        """Calculate Bollinger Bands"""
        upper, middle, lower = self.ti.calculate_bollinger_bands(
            df['close'],
            period=self.config.bb_period,
            num_std=self.config.bb_std_dev
        )
        df['bb_width'] = (upper - lower) / middle
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=20).quantile(0.2)
    
    def _identify_support_resistance(self, data: pd.DataFrame) -> None:
        """Identify key support and resistance levels"""
        window = self.config.support_resistance_periods
        threshold = self.config.support_resistance_threshold
        
        # Find price clusters
        price_clusters = pd.concat([
            data['high'].rolling(window=window).max(),
            data['low'].rolling(window=window).min()
        ]).value_counts(normalize=True)
        
        # Identify significant levels
        significant_levels = price_clusters[price_clusters > threshold].index
        
        # Separate into support and resistance
        current_price = data['close'].iloc[-1]
        self.support_levels = sorted([p for p in significant_levels if p < current_price])
        self.resistance_levels = sorted([p for p in significant_levels if p > current_price])
    
    def _calculate_breakout_score(
        self, 
        price: float, 
        level: float, 
        atr: float,
        volume_surge: float,
        volatility_expansion: float,
        weights: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate breakout score and confidence"""
        # Price break score
        price_score = abs(price - level) / atr
        
        # Volume confirmation
        volume_score = volume_surge / self.config.min_volume_surge_ratio
        
        # Volatility confirmation
        volatility_score = volatility_expansion
        
        # Combined score
        total_score = (
            weights['price_break'] * price_score +
            weights['volume_surge'] * volume_score +
            weights['volatility_expansion'] * volatility_score
        )
        
        # Confidence based on confirmation factors
        confidence = min(
            (volume_score * 0.4 + 
             volatility_score * 0.3 + 
             (price_score / 2) * 0.3),
            1.0
        )
        
        return total_score, confidence
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate breakout signals"""
        current = features.iloc[-1]
        
        # Update support/resistance levels
        self._identify_support_resistance(features)
        
        # Get appropriate feature weights
        weights = self.config.feature_weights[
            'high_volatility' if self.market_regime.startswith('high_volatility')
            else 'low_volatility'
        ]
        
        # Check for breakouts in both timeframes
        signals = {}
        confidences = {}
        
        for trade_type in ['day', 'swing']:
            atr = current[f'{trade_type}_threshold']
            volume_surge = current[f'{trade_type}_volume_surge']
            volatility_expansion = current[f'{trade_type}_volatility_expansion']
            
            # Check resistance breakout (long signal)
            if self.resistance_levels:
                nearest_resistance = min(self.resistance_levels)
                if current['close'] > nearest_resistance:
                    score, conf = self._calculate_breakout_score(
                        current['close'],
                        nearest_resistance,
                        atr,
                        volume_surge,
                        volatility_expansion,
                        weights
                    )
                    signals[trade_type] = score
                    confidences[trade_type] = conf
                    
            # Check support breakout (short signal)
            elif self.support_levels:
                nearest_support = max(self.support_levels)
                if current['close'] < nearest_support:
                    score, conf = self._calculate_breakout_score(
                        current['close'],
                        nearest_support,
                        atr,
                        volume_surge,
                        volatility_expansion,
                        weights
                    )
                    signals[trade_type] = -score
                    confidences[trade_type] = conf
            
            # No breakout
            else:
                signals[trade_type] = 0
                confidences[trade_type] = 0
        
        # Choose between day and swing signals
        day_signal, day_conf = signals.get('day', 0), confidences.get('day', 0)
        swing_signal, swing_conf = signals.get('swing', 0), confidences.get('swing', 0)
        
        # Determine trade type based on signal strength and confirmation
        if abs(swing_signal) > abs(day_signal) and swing_conf > self.confidence_threshold:
            return swing_signal, swing_conf, TradeType.SWING_TRADE
        elif day_conf > self.confidence_threshold:
            return day_signal, day_conf, TradeType.DAY_TRADE
        else:
            return 0, 0, None
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional breakout specific validation"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        
        # Additional breakout-specific validations
        validations = [
            # Volume confirmation
            current['volume_ratio'] > self.config.min_volume_surge_ratio,
            
            # Not in squeeze
            not current['bb_squeeze'],
            
            # Reasonable volatility level
            self.config.min_volatility_percentile <= 
            current['volatility_regime'] <= 
            self.config.max_volatility_percentile,
            
            # Strong directional move
            abs(current['returns']) > current['atr'] / current['close']
        ]
        
        return all(validations)

    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update strategy parameters based on recent performance"""
        # Update support/resistance levels
        self._identify_support_resistance(features)
        
        # Could add more sophisticated updates here, such as:
        # - Adjusting ATR multipliers based on success rate
        # - Updating volume thresholds
        # - Modifying feature weights
        pass