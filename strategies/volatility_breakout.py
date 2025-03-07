from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class VolatilityBreakoutConfig(TechnicalConfig):
    """Configuration for volatility breakout strategy"""
    # Bollinger Band parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_expansion_threshold: float = 1.5  # Expansion ratio threshold for swing
    bb_day_expansion_threshold: float = 2.0  # Expansion ratio threshold for day

    # ATR parameters
    atr_period: int = 14
    atr_donchian_multiplier: float = 1.5  # For ATR-adjusted Donchian channels
    
    # Keltner Channel parameters
    kc_period: int = 20
    kc_atr_multiplier: float = 2.0
    kc_squeeze_threshold: float = 0.3  # Width threshold for squeeze
    
    # Donchian Channel parameters
    donchian_period: int = 20
    
    # Momentum parameters
    rsi_period: int = 14
    rsi_atr_threshold: float = 60  # RSI-ATR score threshold
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_slope_period: int = 3
    
    # Volume confirmation
    min_volume_ratio: float = 1.2  # Minimum volume surge for confirmation
    
    # Trade execution parameters
    day_entry_atr_fraction: float = 0.5  # Entry margin as fraction of ATR
    swing_entry_atr_fraction: float = 0.7
    
    # Stop loss parameters
    day_stop_loss_atr: float = 1.0  # Stop loss in ATR units for day trading
    swing_stop_loss_atr: float = 2.0  # Stop loss in ATR units for swing trading
    
    # Take profit parameters
    day_take_profit_ratio: float = 2.0  # Reward-to-risk ratio for day trading
    swing_take_profit_ratio: float = 3.0  # Reward-to-risk ratio for swing trading
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'day': {
                    'bb_breakout': 0.25,
                    'kc_breakout': 0.20,
                    'rsi_atr_score': 0.20,
                    'macd_slope': 0.15,
                    'volume_surge': 0.20
                },
                'swing': {
                    'bb_breakout': 0.20,
                    'donchian_breakout': 0.25,
                    'range_compression': 0.20,
                    'macd_slope': 0.15,
                    'volume_surge': 0.20
                }
            }


class VolatilityBreakoutStrategy(TechnicalStrategy):
    def __init__(self, config: VolatilityBreakoutConfig = None):
        super().__init__(name="volatility_breakout", config=config or VolatilityBreakoutConfig())
        self.config: VolatilityBreakoutConfig = self.config
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare volatility breakout specific features"""
        df = self.prepare_base_features(data)
        
        # Calculate Bollinger Bands and expansion ratio
        df = self._calculate_bollinger_features(df)
        
        # Calculate Keltner Channels and squeeze
        df = self._calculate_keltner_features(df)
        
        # Calculate ATR-adjusted Donchian Channels
        df = self._calculate_donchian_features(df)
        
        # Calculate Range Compression Ratio
        df = self._calculate_range_compression(df)
        
        # Calculate Momentum features
        df = self._calculate_momentum_features(df)
        
        # Calculate Volume features
        df = self._calculate_volume_features(df)
        
        return df
    
    def _calculate_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Band features"""
        # Calculate standard Bollinger Bands
        upper, middle, lower = self.ti.calculate_bollinger_bands(
            df['close'],
            period=self.config.bb_period,
            num_std=self.config.bb_std_dev
        )
        
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # Calculate Bollinger Band width
        df['bb_width'] = (upper - lower) / middle
        
        # Calculate Bollinger Band expansion ratio
        df['bb_expansion_ratio'] = df['bb_width'] / df['bb_width'].rolling(window=self.config.bb_period).mean()
        
        # Calculate Bollinger Band breakout signals
        df['bb_upper_breakout'] = df['close'] > df['bb_upper']
        df['bb_lower_breakout'] = df['close'] < df['bb_lower']
        
        return df
    
    def _calculate_keltner_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channel features"""
        # Calculate Keltner Channels
        ema = df['close'].ewm(span=self.config.kc_period).mean()
        atr = self.ti.calculate_atr(
            df['high'], df['low'], df['close'], 
            period=self.config.atr_period
        )
        
        df['kc_middle'] = ema
        df['kc_upper'] = ema + (atr * self.config.kc_atr_multiplier)
        df['kc_lower'] = ema - (atr * self.config.kc_atr_multiplier)
        
        # Calculate Keltner Channel width
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
        
        # Identify Keltner Channel squeeze
        df['kc_squeeze'] = df['kc_width'] < self.config.kc_squeeze_threshold
        
        # Calculate Keltner Channel breakout signals
        df['kc_upper_breakout'] = df['close'] > df['kc_upper']
        df['kc_lower_breakout'] = df['close'] < df['kc_lower']
        
        return df
    
    def _calculate_donchian_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR-adjusted Donchian Channel features"""
        period = self.config.donchian_period
        atr = df['atr']
        atr_mult = self.config.atr_donchian_multiplier
        
        # Calculate standard Donchian Channels
        df['donchian_high'] = df['high'].rolling(window=period).max()
        df['donchian_low'] = df['low'].rolling(window=period).min()
        
        # Calculate ATR-adjusted Donchian Channels
        df['donchian_atr_upper'] = df['donchian_high'] + (atr * atr_mult)
        df['donchian_atr_lower'] = df['donchian_low'] - (atr * atr_mult)
        
        # Calculate breakout signals
        df['donchian_upper_breakout'] = df['close'] > df['donchian_high']
        df['donchian_atr_upper_breakout'] = df['close'] > df['donchian_atr_upper']
        df['donchian_lower_breakout'] = df['close'] < df['donchian_low']
        df['donchian_atr_lower_breakout'] = df['close'] < df['donchian_atr_lower']
        
        return df
    
    def _calculate_range_compression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Range Compression Ratio"""
        # Current high-low range
        df['daily_range'] = df['high'] - df['low']
        
        # Average range over lookback period
        df['avg_range'] = df['daily_range'].rolling(window=self.config.bb_period).mean()
        
        # Range compression ratio
        df['range_compression_ratio'] = df['daily_range'] / df['avg_range']
        
        # Flag compression periods (ratio < 0.7 indicates compression)
        df['range_compressed'] = df['range_compression_ratio'] < 0.7
        
        return df
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators for confirmation"""
        # RSI
        if 'rsi' not in df.columns:
            df['rsi'] = self.ti.calculate_rsi(df['close'], self.config.rsi_period)
        
        # ATR percent change
        df['atr_pct_change'] = df['atr'].pct_change(5) * 100
        
        # RSI-ATR Score (combines momentum with volatility change)
        df['rsi_atr_score'] = df['rsi'] + (0.5 * df['atr_pct_change'])
        
        # MACD
        ema_fast = df['close'].ewm(span=self.config.macd_fast).mean()
        ema_slow = df['close'].ewm(span=self.config.macd_slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # MACD Histogram Slope (rate of change)
        df['macd_hist_slope'] = df['macd_hist'].diff(self.config.macd_slope_period)
        
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based confirmation features"""
        # Relative volume (compared to moving average)
        if 'volume_ratio' not in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=self.config.bb_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volume surge flags
        df['volume_surge'] = df['volume_ratio'] > self.config.min_volume_ratio
        
        # Volume-price alignment
        df['volume_price_aligned'] = (
            (df['close'] > df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1)) |
            (df['close'] < df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1))
        )
        
        return df
    
    def is_intraday_data(self, df: pd.DataFrame) -> bool:
        """Check if working with intraday data"""
        if len(df) < 2:
            return False
        time_diff = df.index[1] - df.index[0]
        return time_diff.total_seconds() < 24 * 60 * 60
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate volatility breakout signals"""
        if len(features) < self.config.bb_period:
            return 0, 0, None
            
        current = features.iloc[-1]
        previous = features.iloc[-2]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            return self._calculate_day_trading_signals(current, previous, features)
        else:
            return self._calculate_swing_trading_signals(current, previous, features)
    
    def _calculate_day_trading_signals(
        self, 
        current: pd.Series,
        previous: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate day trading volatility breakout signals"""
        # Initialize signal components
        signals = {}
        
        # Bollinger Band breakout signal
        if current['bb_upper_breakout'] and not previous['bb_upper_breakout']:
            # Fresh upside breakout
            signals['bb_breakout'] = 1
        elif current['bb_lower_breakout'] and not previous['bb_lower_breakout']:
            # Fresh downside breakout
            signals['bb_breakout'] = -1
        else:
            signals['bb_breakout'] = 0
        
        # Keltner Channel breakout signal
        if current['kc_upper_breakout'] and not previous['kc_upper_breakout']:
            signals['kc_breakout'] = 1
        elif current['kc_lower_breakout'] and not previous['kc_lower_breakout']:
            signals['kc_breakout'] = -1
        else:
            signals['kc_breakout'] = 0
        
        # RSI-ATR Score signal
        if current['rsi_atr_score'] > self.config.rsi_atr_threshold:
            signals['rsi_atr_score'] = 1
        elif current['rsi_atr_score'] < (100 - self.config.rsi_atr_threshold):
            signals['rsi_atr_score'] = -1
        else:
            signals['rsi_atr_score'] = 0
        
        # MACD Histogram Slope signal
        signals['macd_slope'] = np.sign(current['macd_hist_slope'])
        
        # Volume confirmation
        if current['volume_surge']:
            signals['volume_surge'] = np.sign(current['close'] - previous['close'])
        else:
            signals['volume_surge'] = 0
        
        # Combine signals with weights
        weights = self.config.feature_weights['day']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement and volatility expansion
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            # Calculate agreement ratio
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Adjust confidence based on volatility expansion
            expansion_boost = min(current['bb_expansion_ratio'] / self.config.bb_day_expansion_threshold, 1.0)
            confidence = agreement_ratio * 0.7 + expansion_boost * 0.3
        else:
            confidence = 0
        
        return total_signal, min(confidence, 1.0), TradeType.DAY_TRADE
    
    def _calculate_swing_trading_signals(
        self, 
        current: pd.Series,
        previous: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate swing trading volatility breakout signals"""
        # Initialize signal components
        signals = {}
        
        # Bollinger Band breakout signal
        if current['bb_upper_breakout'] and not previous['bb_upper_breakout']:
            # Fresh upside breakout
            signals['bb_breakout'] = 1
        elif current['bb_lower_breakout'] and not previous['bb_lower_breakout']:
            # Fresh downside breakout
            signals['bb_breakout'] = -1
        else:
            signals['bb_breakout'] = 0
        
        # Donchian Channel breakout signal (especially relevant for swing)
        if current['donchian_atr_upper_breakout'] and not previous['donchian_atr_upper_breakout']:
            signals['donchian_breakout'] = 1
        elif current['donchian_atr_lower_breakout'] and not previous['donchian_atr_lower_breakout']:
            signals['donchian_breakout'] = -1
        else:
            signals['donchian_breakout'] = 0
        
        # Range compression signal (breakout from tight range)
        # Check if we had compression recently and now breaking out
        had_compression = features['range_compressed'].iloc[-5:-1].any()
        if had_compression:
            if current['close'] > previous['close'] * 1.01:  # 1% up move
                signals['range_compression'] = 1
            elif current['close'] < previous['close'] * 0.99:  # 1% down move
                signals['range_compression'] = -1
            else:
                signals['range_compression'] = 0
        else:
            signals['range_compression'] = 0
        
        # MACD Histogram Slope signal
        signals['macd_slope'] = np.sign(current['macd_hist_slope'])
        
        # Volume confirmation
        if current['volume_surge']:
            signals['volume_surge'] = np.sign(current['close'] - previous['close'])
        else:
            signals['volume_surge'] = 0
        
        # Combine signals with weights
        weights = self.config.feature_weights['swing']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement and volatility expansion
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            # Calculate agreement ratio
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Adjust confidence based on volatility expansion and range compression
            expansion_boost = min(current['bb_expansion_ratio'] / self.config.bb_expansion_threshold, 1.0)
            compression_factor = 1.2 if had_compression else 1.0
            
            confidence = (agreement_ratio * 0.6 + expansion_boost * 0.4) * compression_factor
        else:
            confidence = 0
        
        return total_signal, min(confidence, 1.0), TradeType.SWING_TRADE
    
    def calculate_stop_loss(self, entry_price: float, signal: float, features: pd.DataFrame) -> float:
        """Calculate dynamic stop loss level based on ATR"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Get appropriate ATR multiplier
        atr_multiplier = self.config.day_stop_loss_atr if is_intraday else self.config.swing_stop_loss_atr
        
        # Calculate stop distance
        stop_distance = current['atr'] * atr_multiplier
        
        # Set stop level based on direction (signal)
        if signal > 0:  # Long position
            return entry_price - stop_distance
        else:  # Short position
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, signal: float, features: pd.DataFrame) -> float:
        """Calculate take profit level based on reward-to-risk ratio"""
        is_intraday = self.is_intraday_data(features)
        
        # Get appropriate reward-to-risk ratio
        rr_ratio = self.config.day_take_profit_ratio if is_intraday else self.config.swing_take_profit_ratio
        
        # Calculate risk
        risk = abs(entry_price - stop_loss)
        
        # Calculate target
        if signal > 0:  # Long position
            return entry_price + (risk * rr_ratio)
        else:  # Short position
            return entry_price - (risk * rr_ratio)
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional volatility breakout specific validation"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            validations = [
                # Confirm we have sufficient volatility expansion
                current['bb_expansion_ratio'] >= self.config.bb_day_expansion_threshold * 0.8,
                
                # Volume confirmation
                current['volume_ratio'] >= self.config.min_volume_ratio,
                
                # Ensure momentum alignment
                (signal > 0 and current['macd'] > 0) or (signal < 0 and current['macd'] < 0)
            ]
        else:
            validations = [
                # Confirm we have sufficient volatility expansion for swing
                current['bb_expansion_ratio'] >= self.config.bb_expansion_threshold * 0.8,
                
                # Volume confirmation is stronger for swing trades
                current['volume_ratio'] >= self.config.min_volume_ratio * 1.2,
                
                # Make sure we're not entering after an extended move
                abs(current['rsi'] - 50) < 20  # Not already overbought/oversold
            ]
        
        return all(validations)

    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update strategy parameters based on recent performance"""
        # Dynamic parameter adjustments could be implemented here
        pass