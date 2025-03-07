from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class MeanReversionConfig(TechnicalConfig):
    """Configuration specific to mean reversion strategy"""
    # Z-Score parameters
    day_zscore_period: int = 60  # 1 hour
    swing_zscore_period: int = 20  # 20 days
    zscore_entry_threshold: float = 2.0
    
    # RSI thresholds
    day_rsi_overbought: float = 80.0
    day_rsi_oversold: float = 20.0
    swing_rsi_overbought: float = 70.0
    swing_rsi_oversold: float = 30.0
    
    # Half-life thresholds
    max_half_life_day: float = 4.0  # hours
    max_half_life_swing: float = 3.0  # days
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'high_volatility': {
                    'zscore': 0.3,
                    'rsi': 0.2,
                    'price_to_vwap': 0.3,
                    'volume': 0.2
                },
                'low_volatility': {
                    'zscore': 0.4,
                    'rsi': 0.3,
                    'price_to_vwap': 0.2,
                    'volume': 0.1
                }
            }

class MeanReversionStrategy(TechnicalStrategy):
    def __init__(self, config: MeanReversionConfig = None):
        super().__init__(name="mean_reversion", config=config or MeanReversionConfig())
        self.config: MeanReversionConfig = self.config  # Type hint for IDE support
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare mean reversion specific features"""
        # Get base technical features first
        df = self.prepare_base_features(data)
        
        # Add mean reversion specific features
        for trade_type in ['day', 'swing']:
            # Z-Score calculations
            period = getattr(self.config, f'{trade_type}_zscore_period')
            df[f'{trade_type}_zscore'] = self.ti.calculate_zscore(
                df['close'], 
                period
            )
            
            # Distance from moving averages
            df[f'{trade_type}_ma'] = df['close'].rolling(window=period).mean()
            df[f'{trade_type}_ma_dist'] = (
                (df['close'] - df[f'{trade_type}_ma']) / 
                df[f'{trade_type}_ma']
            )
            
            # Bollinger Bands
            upper, middle, lower = self.ti.calculate_bollinger_bands(
                df['close'], 
                period=period
            )
            df[f'{trade_type}_bb_position'] = (
                (df['close'] - lower) / (upper - lower)
            )
        
        return df

    def _calculate_half_life(self, prices: pd.Series) -> float:
        """Calculate mean reversion half-life"""
        try:
            delta_p = prices.diff().dropna()
            lag_p = prices.shift().dropna()
            
            # Regression of price changes against price levels
            X = sm.add_constant(lag_p[1:])
            model = sm.OLS(delta_p[1:], X).fit()
            
            # Calculate half-life
            lambda_param = -model.params[1]
            half_life = np.log(2) / lambda_param if lambda_param > 0 else np.inf
            
            return half_life
        except Exception as e:  # Improved error handling
            print(f"Error calculating half-life: {e}")
            return np.inf

    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate mean reversion signals"""
        current = features.iloc[-1]
        weights = self.config.feature_weights['high_volatility' if self.market_regime.startswith('high_volatility') else 'low_volatility']
        
        # Calculate signals for both timeframes
        signals = {}
        confidence_scores = {}
        half_lives = {}
        
        for trade_type in ['day', 'swing']:
            # Get relevant thresholds
            rsi_ob = getattr(self.config, f'{trade_type}_rsi_overbought')
            rsi_os = getattr(self.config, f'{trade_type}_rsi_oversold')
            max_half_life = getattr(self.config, f'max_half_life_{trade_type}')
            
            # Calculate half-life for timeframe
            period = getattr(self.config, f'{trade_type}_zscore_period')
            prices = features['close'].tail(period * 2)
            half_life = self._calculate_half_life(prices)
            half_lives[trade_type] = half_life
            
            # Individual signals
            zscore_signal = -np.clip(
                current[f'{trade_type}_zscore'] / 
                self.config.zscore_entry_threshold, 
                -1, 
                1
            )
            
            rsi = current['rsi']
            if rsi > rsi_ob:
                rsi_signal = -1
            elif rsi < rsi_os:
                rsi_signal = 1
            else:
                rsi_signal = 0
            
            vwap_signal = -np.clip(
                (current['close'] - current['vwap']) / 
                (current['atr'] * 2), 
                -1, 
                1
            )
            
            volume_signal = np.clip(
                current['volume_ratio'] - 1, 
                -1, 
                1
            )
            
            # Combine signals using weights
            total_signal = (
                weights['zscore'] * zscore_signal +
                weights['rsi'] * rsi_signal +
                weights['price_to_vwap'] * vwap_signal +
                weights['volume'] * volume_signal
            )
            
            # Calculate confidence
            signal_agreement = (
                abs(zscore_signal) > 0.5 and
                abs(rsi_signal) > 0.5 and
                abs(vwap_signal) > 0.5
            )
            
            confidence = (
                abs(total_signal) * 
                (0.7 + 0.3 * signal_agreement) *
                (1.0 if half_life < max_half_life else 0.5)
            )
            
            signals[trade_type] = total_signal
            confidence_scores[trade_type] = min(confidence, 1.0)
        
        # Choose between day and swing trade
        day_signal, day_conf = signals['day'], confidence_scores['day']
        swing_signal, swing_conf = signals['swing'], confidence_scores['swing']
        
        if abs(swing_signal) > abs(day_signal) and half_lives['swing'] < self.config.max_half_life_swing:
            return swing_signal, swing_conf, TradeType.SWING_TRADE
        else:
            return day_signal, day_conf, TradeType.DAY_TRADE

    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional mean reversion specific validation"""
        # First check basic technical validations
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        
        # Additional mean reversion specific validations
        validations = [
            self._is_within_price_gap(current),
            self._is_volume_significant(current),
            self._is_not_in_extreme_volatility(),
            self._is_price_near_vwap(current)
        ]
        
        return all(validations)

    def _is_within_price_gap(self, current) -> bool:
        return abs(current['returns']) < self.config.price_gap_threshold

    def _is_volume_significant(self, current) -> bool:
        return 0.5 < current['volume_ratio'] < 5.0

    def _is_not_in_extreme_volatility(self) -> bool:
        return self.market_regime not in ['high_volatility_trending', 'high_volatility_choppy']

    def _is_price_near_vwap(self, current) -> bool:
        return abs(current['vwap_ratio'] - 1) < 0.1

    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update strategy parameters based on recent performance"""
        # Calculate prediction accuracy for different features
        accuracy_scores = {}
        
        # Update feature weights based on accuracy
        # This is a simplified version - in practice, you'd want more sophisticated weight updating
        for regime in ['high_volatility', 'low_volatility']:
            for feature in self.config.feature_weights[regime]:
                # Here you would calculate how well each feature predicted the direction
                # For now, we'll keep the weights static
                pass