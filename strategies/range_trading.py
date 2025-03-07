from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class RangeTradingConfig(TechnicalConfig):
    """Configuration for range trading strategy"""
    # Range boundaries parameters
    swing_range_period: int = 50  # 50 candles for swing trading range calculation
    day_range_period: int = 20    # 20 candles for day trading range calculation
    
    # ATR parameters for stop loss and take profit
    swing_atr_multiplier: float = 1.5
    day_atr_multiplier: float = 0.75
    
    # Bollinger Bands parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Z-score parameters
    zscore_period: int = 20
    zscore_threshold: float = 1.5
    
    # Volume and order flow parameters
    min_volume_ratio: float = 1.2
    obi_threshold: float = 0.2  # Order Book Imbalance threshold
    
    # ADX parameters (trend strength)
    adx_period: int = 14
    adx_non_trend_threshold: float = 25.0  # Below this value indicates ranging market
    
    # Hurst exponent parameters
    hurst_lookback: int = 100
    hurst_mean_reversion_threshold: float = 0.4  # Below 0.5 indicates mean-reversion
    
    # Profit targets
    day_take_profit: List[float] = None  # [0.03, 0.06, 0.09]  # 3%, 6%, 9% targets
    swing_take_profit: float = 0.10  # 10% target
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.day_take_profit is None:
            self.day_take_profit = [0.03, 0.06, 0.09]
            
        if self.feature_weights is None:
            self.feature_weights = {
                'swing': {
                    'range_position': 0.35,
                    'zscore': 0.25,
                    'bollinger': 0.20,
                    'volume': 0.10,
                    'adx': 0.10
                },
                'day': {
                    'range_position': 0.30,
                    'zscore': 0.20,
                    'vwap_deviation': 0.20,
                    'order_flow': 0.20,
                    'adx': 0.10
                }
            }

class RangeTradingStrategy(TechnicalStrategy):
    def __init__(self, config: RangeTradingConfig = None):
        super().__init__(name="range_trading", config=config or RangeTradingConfig())
        self.config: RangeTradingConfig = self.config
        self.current_range = {
            'swing': {'support': None, 'resistance': None, 'valid': False},
            'day': {'support': None, 'resistance': None, 'valid': False}
        }
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare range trading specific features"""
        df = self.prepare_base_features(data)
        
        # Calculate range boundaries for both timeframes
        df = self._calculate_range_boundaries(df)
        
        # Calculate Bollinger Bands
        df = self._calculate_bollinger_features(df)
        
        # Calculate Z-scores
        df = self._calculate_zscore_features(df)
        
        # Calculate ADX (trend strength)
        df['adx'] = self.ti.calculate_directional_movement(
            df['high'], df['low'], df['close'], 
            self.config.adx_period
        )
        
        # Calculate VWAP-based features
        df = self._calculate_vwap_features(df)
        
        # Calculate volume and order flow features
        df = self._calculate_volume_features(df)
        
        # Calculate fractal patterns
        df = self._calculate_fractal_patterns(df)
        
        # Calculate Hurst exponent (for mean-reversion tendency)
        df['hurst'] = self._calculate_rolling_hurst(df['close'])
        
        return df
    
    def _calculate_range_boundaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels for different timeframes"""
        for trade_type in ['day', 'swing']:
            period = getattr(self.config, f'{trade_type}_range_period')
            
            # Calculate rolling max/min
            df[f'{trade_type}_resistance'] = df['high'].rolling(window=period).max()
            df[f'{trade_type}_support'] = df['low'].rolling(window=period).min()
            
            # Calculate range width
            df[f'{trade_type}_range_width'] = df[f'{trade_type}_resistance'] - df[f'{trade_type}_support']
            
            # Calculate relative position within range (0 = at support, 1 = at resistance)
            df[f'{trade_type}_range_position'] = (df['close'] - df[f'{trade_type}_support']) / (
                df[f'{trade_type}_range_width']
            )
            
            # Mark if we're near range boundaries
            df[f'{trade_type}_near_support'] = df[f'{trade_type}_range_position'] < 0.2
            df[f'{trade_type}_near_resistance'] = df[f'{trade_type}_range_position'] > 0.8
            
            # Calculate range stability (consistency over time)
            df[f'{trade_type}_range_stability'] = 1 - (
                df[f'{trade_type}_range_width'].pct_change().rolling(window=10).std()
            )
            
            # Update current range boundaries
            if len(df) > 0:
                last_row = df.iloc[-1]
                self.current_range[trade_type]['support'] = last_row[f'{trade_type}_support']
                self.current_range[trade_type]['resistance'] = last_row[f'{trade_type}_resistance']
                
                # Range is valid if ADX is below threshold and range is stable
                self.current_range[trade_type]['valid'] = (
                    last_row['adx'] < self.config.adx_non_trend_threshold and
                    not pd.isna(last_row[f'{trade_type}_range_stability']) and
                    last_row[f'{trade_type}_range_stability'] > 0.7
                )
        
        return df
    
    def _calculate_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands features"""
        upper, middle, lower = self.ti.calculate_bollinger_bands(
            df['close'],
            period=self.config.bb_period,
            num_std=self.config.bb_std_dev
        )
        
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        
        # Calculate BB position (0 = at lower band, 1 = at upper band)
        df['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # BB squeeze (narrow bands) indicates potential breakout
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=20).quantile(0.2)
        
        return df
    
    def _calculate_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Z-score features"""
        df['price_zscore'] = self.ti.calculate_zscore(
            df['close'], 
            self.config.zscore_period
        )
        
        # Z-score based signals
        df['zscore_overbought'] = df['price_zscore'] > self.config.zscore_threshold
        df['zscore_oversold'] = df['price_zscore'] < -self.config.zscore_threshold
        
        return df
    
    def _calculate_vwap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP-based features"""
        if 'vwap' not in df.columns:
            df['vwap'] = self.ti.calculate_vwap(df)
            
        # Normalized deviation from VWAP
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['atr']
        
        # Classify positions relative to VWAP
        df['above_vwap'] = df['close'] > df['vwap']
        df['vwap_crossover'] = (df['close'] > df['vwap']) & (df['close'].shift(1) <= df['vwap'].shift(1))
        df['vwap_crossunder'] = (df['close'] < df['vwap']) & (df['close'].shift(1) >= df['vwap'].shift(1))
        
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume and order flow features"""
        # Basic volume ratio
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Order book imbalance (if available)
        if all(col in df.columns for col in ['bid_volume', 'ask_volume']):
            df['obi'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
            
            # Cumulative Volume Delta
            df['volume_delta'] = df['bid_volume'] - df['ask_volume']
            df['cvd'] = df['volume_delta'].cumsum()
            
            # Volume pressure at range boundaries
            for boundary in ['support', 'resistance']:
                for trade_type in ['day', 'swing']:
                    col = f'{trade_type}_near_{boundary}'
                    if col in df.columns:
                        df[f'{col}_vol_pressure'] = df[col] * df['volume_ratio'] * np.sign(df['obi'])
        
        return df
    
    def _calculate_fractal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fractal patterns for support/resistance identification"""
        # A simple implementation of fractals (could be expanded)
        if len(df) >= 5:
            # Bullish fractals (for support)
            df['fractal_support'] = (
                (df['low'].shift(2) > df['low'].shift(1)) &
                (df['low'].shift(1) > df['low']) &
                (df['low'] < df['low'].shift(-1)) &
                (df['low'].shift(-1) < df['low'].shift(-2))
            )
            
            # Bearish fractals (for resistance)
            df['fractal_resistance'] = (
                (df['high'].shift(2) < df['high'].shift(1)) &
                (df['high'].shift(1) < df['high']) &
                (df['high'] > df['high'].shift(-1)) &
                (df['high'].shift(-1) > df['high'].shift(-2))
            )
            
            # Fill NaN values
            df['fractal_support'] = df['fractal_support'].fillna(False)
            df['fractal_resistance'] = df['fractal_resistance'].fillna(False)
        else:
            df['fractal_support'] = False
            df['fractal_resistance'] = False
            
        return df
    
    def _calculate_rolling_hurst(self, series: pd.Series, min_periods: int = 50) -> pd.Series:
        """Calculate rolling Hurst exponent to identify mean-reversion tendency"""
        hurst_values = []
        
        for i in range(len(series)):
            if i < min_periods:
                hurst_values.append(np.nan)
                continue
                
            # Use at most config.hurst_lookback periods
            start_idx = max(0, i - self.config.hurst_lookback)
            window = series.iloc[start_idx:i+1]
            
            try:
                hurst = self._hurst_exponent(window)
                hurst_values.append(hurst)
            except:
                # Use previous value or NaN if calculation fails
                prev_value = hurst_values[-1] if hurst_values else np.nan
                hurst_values.append(prev_value)
                
        return pd.Series(hurst_values, index=series.index)
    
    def _hurst_exponent(self, series: pd.Series) -> float:
        """Calculate Hurst exponent for a time series
        H < 0.5 indicates mean-reversion (good for range trading)
        H = 0.5 indicates random walk
        H > 0.5 indicates momentum/trend
        """
        # Convert price to returns
        returns = np.log(series / series.shift(1)).dropna()
        if len(returns) < 10:
            return 0.5  # Default to random walk for small samples
            
        # Calculate variance of return differences at different lags
        lags = range(2, min(20, len(returns) // 4))
        tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
        
        # Linear regression on log-log scale
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Hurst exponent is the slope
        return m[0]
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate range trading signals"""
        if len(features) < self.config.day_range_period:
            return 0, 0, None
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Check if we're in a valid range-bound market
        trade_type = 'day' if is_intraday else 'swing'
        if not self.current_range[trade_type]['valid'] or current['adx'] > self.config.adx_non_trend_threshold:
            return 0, 0, None
            
        # Calculate signals based on timeframe
        if is_intraday:
            return self._calculate_day_trading_signals(current, features)
        else:
            return self._calculate_swing_trading_signals(current, features)
    
    def _calculate_swing_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate swing trading signals for range trading"""
        # Individual signals
        signals = {}
        
        # 1. Range position signal (buy near support, sell near resistance)
        range_pos = current.get('swing_range_position', 0.5)
        if range_pos < 0.2:  # Near support - buy signal
            signals['range_position'] = 1
        elif range_pos > 0.8:  # Near resistance - sell signal
            signals['range_position'] = -1
        else:
            signals['range_position'] = 0
            
        # 2. Z-score signal
        zscore = current.get('price_zscore', 0)
        if zscore < -self.config.zscore_threshold:  # Oversold - buy signal
            signals['zscore'] = 1
        elif zscore > self.config.zscore_threshold:  # Overbought - sell signal
            signals['zscore'] = -1
        else:
            signals['zscore'] = 0
            
        # 3. Bollinger Bands signal
        bb_pos = current.get('bb_position', 0.5)
        if bb_pos < 0.1:  # Near lower band - buy signal
            signals['bollinger'] = 1
        elif bb_pos > 0.9:  # Near upper band - sell signal
            signals['bollinger'] = -1
        else:
            signals['bollinger'] = 0
            
        # 4. Volume signal - look for volume confirmation at boundaries
        vol_ratio = current.get('volume_ratio', 1)
        if signals['range_position'] != 0 and vol_ratio > self.config.min_volume_ratio:
            signals['volume'] = signals['range_position']  # Same direction as range signal
        else:
            signals['volume'] = 0
            
        # 5. ADX signal - stronger signal in established ranges
        adx = current.get('adx', 30)
        # Lower ADX means stronger range trading signal
        adx_signal = max(0, (self.config.adx_non_trend_threshold - adx) / self.config.adx_non_trend_threshold)
        # Direction from range position
        if signals['range_position'] != 0:
            signals['adx'] = adx_signal * np.sign(signals['range_position'])
        else:
            signals['adx'] = 0
            
        # Combine signals with weights
        weights = self.config.feature_weights['swing']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on agreement between signals
        # and consideration of Hurst exponent (mean-reversion tendency)
        hurst = current.get('hurst', 0.5)
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            # Boost confidence if market shows mean-reversion tendency
            mean_reversion_boost = 1 + max(0, (0.5 - hurst) * 2)
            confidence = agreement_ratio * mean_reversion_boost * min(1, abs(total_signal))
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.SWING_TRADE
    
    def _calculate_day_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate day trading signals for range trading"""
        # Individual signals
        signals = {}
        
        # 1. Range position signal
        range_pos = current.get('day_range_position', 0.5)
        if range_pos < 0.2:  # Near support - buy signal
            signals['range_position'] = 1
        elif range_pos > 0.8:  # Near resistance - sell signal
            signals['range_position'] = -1
        else:
            signals['range_position'] = 0
            
        # 2. Z-score signal (similar to swing trading)
        zscore = current.get('price_zscore', 0)
        if zscore < -self.config.zscore_threshold:
            signals['zscore'] = 1
        elif zscore > self.config.zscore_threshold:
            signals['zscore'] = -1
        else:
            signals['zscore'] = 0
            
        # 3. VWAP deviation signal - more important for day trading
        vwap_dev = current.get('vwap_deviation', 0)
        if vwap_dev < -1.5:  # Significantly below VWAP - buy signal
            signals['vwap_deviation'] = 1
        elif vwap_dev > 1.5:  # Significantly above VWAP - sell signal
            signals['vwap_deviation'] = -1
        else:
            signals['vwap_deviation'] = 0
            
        # 4. Order flow signal
        if 'obi' in current:
            obi = current['obi']
            if abs(obi) > self.config.obi_threshold:
                signals['order_flow'] = np.sign(obi)
            else:
                signals['order_flow'] = 0
        else:
            # Fallback to volume if OBI not available
            vol_spike = current.get('volume_ratio', 1) > 1.5
            if vol_spike and signals['range_position'] != 0:
                signals['order_flow'] = signals['range_position']
            else:
                signals['order_flow'] = 0
                
        # 5. ADX signal (similar to swing trading)
        adx = current.get('adx', 30)
        adx_signal = max(0, (self.config.adx_non_trend_threshold - adx) / self.config.adx_non_trend_threshold)
        if signals['range_position'] != 0:
            signals['adx'] = adx_signal * np.sign(signals['range_position'])
        else:
            signals['adx'] = 0
            
        # Combine signals with weights
        weights = self.config.feature_weights['day']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            # Day trading confidence also considers time of day
            time_factor = 1.0  # Default
            if hasattr(current.index, 'hour'):
                hour = current.index.hour
                # Avoid trading during low liquidity periods
                if hour < 10 or hour > 15:
                    time_factor = 0.7
                    
            confidence = agreement_ratio * time_factor * min(1, abs(total_signal))
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.DAY_TRADE
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional range trading specific validation"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        trade_type = 'day' if is_intraday else 'swing'
        
        # 1. Validate we're in a range-bound market
        if not self.current_range[trade_type]['valid']:
            return False
            
        # 2. ADX should indicate non-trending market
        if current['adx'] > self.config.adx_non_trend_threshold:
            return False
            
        # 3. Validate based on trading timeframe
        if is_intraday:
            validations = [
                # Check for volume confirmation
                current['volume_ratio'] > 1.0,
                
                # Make sure we're not in a Bollinger squeeze (potential breakout)
                not current.get('bb_squeeze', False),
                
                # For day trading, we prefer mean-reverting conditions
                current.get('hurst', 0.5) < 0.5,
                
                # Avoid trading in extreme volatility
                0.3 < current.get('day_range_position', 0.5) < 0.7 or
                abs(signal) > 0.7  # Unless signal is very strong
            ]
        else:
            # Swing trading validations
            validations = [
                # For swing trades, we want price to be closer to boundaries
                current.get('swing_range_position', 0.5) < 0.3 or 
                current.get('swing_range_position', 0.5) > 0.7,
                
                # Range should be stable
                current.get('swing_range_stability', 0) > 0.7,
                
                # Confirm with volume
                current['volume_ratio'] > self.config.min_volume_ratio,
                
                # Avoid potentially imminent breakouts
                not current.get('bb_squeeze', False)
            ]
        
        return all(validations)
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        features: pd.DataFrame, 
        trade_type: TradeType, 
        signal: float
    ) -> float:
        """Calculate adaptive stop loss for range trading"""
        current_atr = features['atr'].iloc[-1]
        
        if trade_type == TradeType.DAY_TRADE:
            # For day trading, use tighter stops based on ATR
            stop_distance = current_atr * self.config.day_atr_multiplier
        else:
            # For swing trading, use wider stops
            stop_distance = current_atr * self.config.swing_atr_multiplier
            
        # Long position: stop below entry
        if signal > 0:
            return entry_price - stop_distance
        # Short position: stop above entry
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(
        self, 
        entry_price: float, 
        features: pd.DataFrame, 
        trade_type: TradeType, 
        signal: float
    ) -> List[float]:
        """Calculate take profit levels for range trading"""
        if trade_type == TradeType.DAY_TRADE:
            # For day trading, use staged profit targets
            if signal > 0:  # Long position
                return [entry_price * (1 + target) for target in self.config.day_take_profit]
            else:  # Short position
                return [entry_price * (1 - target) for target in self.config.day_take_profit]
        else:
            # For swing trading, target the opposite boundary
            trade_type_str = 'swing'
            if signal > 0:  # Long position, target resistance
                resistance = self.current_range[trade_type_str]['resistance']
                return [resistance]
            else:  # Short position, target support
                support = self.current_range[trade_type_str]['support']
                return [support]
    
    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update strategy parameters based on recent performance"""
        # Recalculate range boundaries
        self._calculate_range_boundaries(features)
        
        # Potential future enhancements:
        # 1. Dynamically adjust ATR multipliers based on recent volatility
        # 2. Update feature weights based on performance metrics
        # 3. Adapt range period based on market conditions
        pass