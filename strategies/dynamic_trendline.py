from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class DynamicTrendlineConfig(TechnicalConfig):
    """Configuration for dynamic trendline detection strategy"""
    # Pivot detection parameters
    pivot_lookback: int = 30  # Bars to look back for pivot points
    pivot_threshold: float = 0.618  # Fibonacci ratio threshold
    
    # Trendline parameters
    min_pivots_required: int = 3  # Minimum pivot points needed for valid trendline
    regression_lookback: int = 60  # Period for linear regression
    
    # Deviation parameters
    deviation_threshold_atr: float = 1.0  # Deviation threshold in ATR units
    breakout_threshold_atr: float = 1.2  # Breakout threshold in ATR units
    
    # Confirmation parameters
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Risk parameters
    day_atr_multiplier: float = 1.0  # ATR multiplier for day trading stops
    swing_atr_multiplier: float = 2.5  # ATR multiplier for swing trading stops
    
    # Volume parameters
    volume_breakout_threshold: float = 2.0  # Volume increase for breakout confirmation
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'swing': {
                    'trendline_deviation': 0.35,
                    'slope_strength': 0.25,
                    'momentum': 0.20,
                    'volatility': 0.10,
                    'volume': 0.10
                },
                'day': {
                    'trendline_deviation': 0.30,
                    'slope_strength': 0.20,
                    'momentum': 0.20,
                    'volatility': 0.15,
                    'time_of_day': 0.15
                }
            }

class DynamicTrendlineStrategy(TechnicalStrategy):
    def __init__(self, config: DynamicTrendlineConfig = None):
        super().__init__(name="dynamic_trendline", config=config or DynamicTrendlineConfig())
        self.config: DynamicTrendlineConfig = self.config
        self.current_trendlines: Dict[str, dict] = {
            'uptrend': {'slope': 0, 'intercept': 0, 'pivots': [], 'strength': 0},
            'downtrend': {'slope': 0, 'intercept': 0, 'pivots': [], 'strength': 0}
        }
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare trendline specific features"""
        df = self.prepare_base_features(data)
        
        # Calculate pivots and trendlines
        df = self._calculate_pivot_points(df)
        df = self._calculate_trendlines(df)
        
        # Calculate price deviation from trendlines
        df = self._calculate_deviations(df)
        
        # Calculate momentum indicators
        df = self._calculate_momentum_features(df)
        
        # Calculate volume features
        df = self._calculate_volume_features(df)
        
        # Calculate time-based features if intraday
        if self.is_intraday_data(df):
            df = self._calculate_time_features(df)
            
        return df
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify pivot points (swing highs and lows)"""
        lookback = self.config.pivot_lookback
        
        # Initialize pivot columns
        df['pivot_high'] = False
        df['pivot_low'] = False
        
        # Skip the first and last lookback/2 periods
        half_lookback = lookback // 2
        
        for i in range(half_lookback, len(df) - half_lookback):
            # Check if current point is a local maximum
            window_high = df['high'].iloc[i-half_lookback:i+half_lookback+1]
            if df['high'].iloc[i] == window_high.max():
                df.loc[df.index[i], 'pivot_high'] = True
                
            # Check if current point is a local minimum
            window_low = df['low'].iloc[i-half_lookback:i+half_lookback+1]
            if df['low'].iloc[i] == window_low.min():
                df.loc[df.index[i], 'pivot_low'] = True
        
        # Filter pivots based on significance
        avg_range = df['atr'].rolling(window=lookback).mean()
        threshold = avg_range * self.config.pivot_threshold
        
        # Only keep significant pivots
        for i in range(1, len(df)):
            if df['pivot_high'].iloc[i]:
                # Check if the high is significantly higher than previous pivot high
                prev_highs = df[df['pivot_high']].iloc[:i]['high']
                if len(prev_highs) > 0 and (df['high'].iloc[i] - prev_highs.iloc[-1]) < threshold.iloc[i]:
                    df.loc[df.index[i], 'pivot_high'] = False
                    
            if df['pivot_low'].iloc[i]:
                # Check if the low is significantly lower than previous pivot low
                prev_lows = df[df['pivot_low']].iloc[:i]['low']
                if len(prev_lows) > 0 and (prev_lows.iloc[-1] - df['low'].iloc[i]) < threshold.iloc[i]:
                    df.loc[df.index[i], 'pivot_low'] = False
        
        return df
    
    def _calculate_trendlines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dynamic trendlines using pivot points"""
        # Get indices of pivot highs and lows
        high_idx = df.index[df['pivot_high']]
        low_idx = df.index[df['pivot_low']]
        
        # Calculate uptrend line (connecting lows)
        if len(low_idx) >= self.config.min_pivots_required:
            # Get recent pivot lows
            recent_lows = df.loc[low_idx[-self.config.min_pivots_required:]]
            
            # Use linear regression to find the trendline
            x = np.array(range(len(recent_lows)))
            y = recent_lows['low'].values
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            
            # Store trendline information
            self.current_trendlines['uptrend'] = {
                'slope': slope,
                'intercept': intercept,
                'pivots': recent_lows.index.tolist(),
                'strength': abs(r_value)
            }
            
            # Calculate trendline values for each point
            # We need to map each index to its relative position for the regression
            idx_map = {idx: i for i, idx in enumerate(recent_lows.index)}
            
            # Calculate uptrend line for all points after the first pivot
            first_pivot_idx = min(idx_map.keys())
            uptrend_values = []
            
            for i, idx in enumerate(df.index):
                if idx >= first_pivot_idx:
                    # Find relative x-position
                    days_since_first = (idx - first_pivot_idx).total_seconds() / (24 * 60 * 60)
                    # Calculate y value using slope and intercept
                    uptrend_values.append(intercept + slope * days_since_first)
                else:
                    uptrend_values.append(np.nan)
                    
            df['uptrend_line'] = uptrend_values
        else:
            df['uptrend_line'] = np.nan
            
        # Calculate downtrend line (connecting highs)
        if len(high_idx) >= self.config.min_pivots_required:
            # Get recent pivot highs
            recent_highs = df.loc[high_idx[-self.config.min_pivots_required:]]
            
            # Use linear regression to find the trendline
            x = np.array(range(len(recent_highs)))
            y = recent_highs['high'].values
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            
            # Store trendline information
            self.current_trendlines['downtrend'] = {
                'slope': slope,
                'intercept': intercept,
                'pivots': recent_highs.index.tolist(),
                'strength': abs(r_value)
            }
            
            # Calculate trendline values similar to uptrend
            idx_map = {idx: i for i, idx in enumerate(recent_highs.index)}
            first_pivot_idx = min(idx_map.keys())
            downtrend_values = []
            
            for i, idx in enumerate(df.index):
                if idx >= first_pivot_idx:
                    days_since_first = (idx - first_pivot_idx).total_seconds() / (24 * 60 * 60)
                    downtrend_values.append(intercept + slope * days_since_first)
                else:
                    downtrend_values.append(np.nan)
                    
            df['downtrend_line'] = downtrend_values
        else:
            df['downtrend_line'] = np.nan
            
        return df
    
    def _calculate_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price deviations from trendlines"""
        # Absolute deviation
        if 'uptrend_line' in df.columns:
            df['uptrend_deviation'] = df['close'] - df['uptrend_line']
            # Normalized deviation by ATR
            df['uptrend_deviation_norm'] = df['uptrend_deviation'] / df['atr']
        
        if 'downtrend_line' in df.columns:
            df['downtrend_deviation'] = df['close'] - df['downtrend_line']
            # Normalized deviation by ATR
            df['downtrend_deviation_norm'] = df['downtrend_deviation'] / df['atr']
            
        # Rate of change of deviation
        for trend in ['uptrend', 'downtrend']:
            col = f'{trend}_deviation_norm'
            if col in df.columns:
                df[f'{col}_roc'] = df[col].pct_change(5)
        
        return df
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        # RSI
        df['rsi'] = self.ti.calculate_rsi(df['close'], self.config.rsi_period)
        
        # MACD
        ema_fast = df['close'].ewm(span=self.config.macd_fast).mean()
        ema_slow = df['close'].ewm(span=self.config.macd_slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Rate of Change
        df['price_roc'] = df['close'].pct_change(5)
        
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volume-price divergence
        df['close_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Check for volume-price divergence
        df['vol_price_divergence'] = (
            (df['close_change'] > 0) & (df['volume_change'] < 0) |
            (df['close_change'] < 0) & (df['volume_change'] > 0)
        )
        
        return df
    
    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features for intraday data"""
        # Extract hour
        df['hour'] = df.index.hour
        
        # Market session flags (example for US market)
        df['market_open'] = (df['hour'] >= 9) & (df['hour'] < 10)
        df['lunch_hour'] = (df['hour'] >= 12) & (df['hour'] < 13)
        df['market_close'] = (df['hour'] >= 15) & (df['hour'] < 16)
        
        # Day of week
        df['day_of_week'] = df.index.dayofweek
        
        return df
    
    def is_intraday_data(self, df: pd.DataFrame) -> bool:
        """Check if we're working with intraday data"""
        if len(df) < 2:
            return False
        time_diff = df.index[1] - df.index[0]
        return time_diff.total_seconds() < 24 * 60 * 60
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate trendline signals"""
        if len(features) < self.config.pivot_lookback:
            return 0, 0, None
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            return self._calculate_day_trading_signals(current, features)
        else:
            return self._calculate_swing_trading_signals(current, features)
    
    def _calculate_swing_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate swing trading signals"""
        signals = {}
        
        # Trendline deviation signal
        up_dev = current.get('uptrend_deviation_norm', 0)
        down_dev = current.get('downtrend_deviation_norm', 0)
        
        breakout_threshold = self.config.breakout_threshold_atr
        
        # Positive value means price is above trendline
        trendline_signal = 0
        if up_dev > breakout_threshold:  # Bullish breakout of uptrend line
            trendline_signal = 1
        elif down_dev < -breakout_threshold:  # Bearish breakout of downtrend line
            trendline_signal = -1
            
        signals['trendline_deviation'] = trendline_signal
        
        # Slope strength signal
        uptrend_strength = self.current_trendlines['uptrend']['strength']
        downtrend_strength = self.current_trendlines['downtrend']['strength']
        
        if uptrend_strength > downtrend_strength:
            slope_signal = uptrend_strength  # Positive if uptrend is stronger
        else:
            slope_signal = -downtrend_strength  # Negative if downtrend is stronger
            
        signals['slope_strength'] = slope_signal
        
        # Momentum signal
        rsi_signal = 0
        if current['rsi'] < self.config.rsi_oversold:
            rsi_signal = 1  # Bullish
        elif current['rsi'] > self.config.rsi_overbought:
            rsi_signal = -1  # Bearish
            
        macd_signal = np.sign(current['macd_hist'])
        
        momentum_signal = (rsi_signal + macd_signal) / 2
        signals['momentum'] = momentum_signal
        
        # Volatility signal - lower volatility gives higher confidence
        volatility_percentile = features['atr'].rolling(window=30).rank(pct=True).iloc[-1]
        volatility_signal = 0.5 - volatility_percentile  # Higher when volatility is lower
        signals['volatility'] = volatility_signal
        
        # Volume signal
        volume_signal = 0
        if current['volume_ratio'] > self.config.volume_breakout_threshold:
            volume_signal = np.sign(current['close_change'])  # Direction of price change
        signals['volume'] = volume_signal
        
        # Combine signals with weights
        weights = self.config.feature_weights['swing']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on agreement and trendline strength
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            # Boost confidence if trendline strength is high
            trendline_strength = max(uptrend_strength, downtrend_strength)
            confidence = agreement_ratio * 0.7 + trendline_strength * 0.3
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.SWING_TRADE
    
    def _calculate_day_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate day trading signals"""
        signals = {}
        
        # Trendline deviation with faster response for day trading
        up_dev = current.get('uptrend_deviation_norm', 0)
        down_dev = current.get('downtrend_deviation_norm', 0)
        
        day_threshold = self.config.breakout_threshold_atr * 0.8  # More sensitive for day trading
        
        trendline_signal = 0
        if up_dev > day_threshold:  # Bullish breakout
            trendline_signal = 1
        elif down_dev < -day_threshold:  # Bearish breakout
            trendline_signal = -1
            
        signals['trendline_deviation'] = trendline_signal
        
        # Slope strength - similar to swing trading
        uptrend_strength = self.current_trendlines['uptrend']['strength']
        downtrend_strength = self.current_trendlines['downtrend']['strength']
        
        if uptrend_strength > downtrend_strength:
            slope_signal = uptrend_strength
        else:
            slope_signal = -downtrend_strength
            
        signals['slope_strength'] = slope_signal
        
        # Momentum - more weight on immediate momentum
        rsi_signal = 0
        if current['rsi'] < self.config.rsi_oversold:
            rsi_signal = 1
        elif current['rsi'] > self.config.rsi_overbought:
            rsi_signal = -1
            
        macd_signal = np.sign(current['macd_hist'])
        price_roc_signal = np.sign(current['price_roc'])
        
        # More emphasis on recent price movement for day trading
        momentum_signal = (rsi_signal * 0.3 + macd_signal * 0.3 + price_roc_signal * 0.4)
        signals['momentum'] = momentum_signal
        
        # Volatility signal
        recent_volatility = features['atr'].iloc[-5:].mean() / features['atr'].iloc[-20:].mean()
        # For day trading, we prefer some volatility but not extreme
        volatility_signal = -abs(recent_volatility - 1.2)  # Optimal around 1.2x average
        signals['volatility'] = volatility_signal
        
        # Time of day signal
        time_signal = 0
        if current['market_open']:
            time_signal = 0.5  # Slightly favor trading at open
        elif current['lunch_hour']:
            time_signal = -0.5  # Avoid lunch hour
        elif current['market_close']:
            time_signal = -0.3  # Slightly avoid close
            
        if np.sign(time_signal) != np.sign(trendline_signal):
            time_signal = 0  # Neutralize if contradicting trendline
            
        signals['time_of_day'] = time_signal
        
        # Combine signals
        weights = self.config.feature_weights['day']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            confidence = agreement_ratio * 0.8 + max(uptrend_strength, downtrend_strength) * 0.2
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.DAY_TRADE
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional trendline specific validation"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Get trendline strengths
        uptrend_strength = self.current_trendlines['uptrend']['strength']
        downtrend_strength = self.current_trendlines['downtrend']['strength']
        
        if is_intraday:
            validations = [
                # Require stronger trendlines for day trading
                max(uptrend_strength, downtrend_strength) > 0.6,
                
                # Avoid low volume periods
                current['volume_ratio'] > 0.8,
                
                # Avoid trading against strong divergence
                not (current['vol_price_divergence'] and 
                     abs(current['volume_change']) > 0.5),
                     
                # Avoid lunch hour for day trading
                not current['lunch_hour'],
                
                # Require reasonable ATR
                current['atr'] > features['atr'].rolling(window=20).quantile(0.3).iloc[-1]
            ]
        else:
            validations = [
                # Minimum trendline strength
                max(uptrend_strength, downtrend_strength) > 0.5,
                
                # Significant deviation from trendline
                (abs(current.get('uptrend_deviation_norm', 0)) > self.config.deviation_threshold_atr or
                 abs(current.get('downtrend_deviation_norm', 0)) > self.config.deviation_threshold_atr),
                 
                # Volume confirmation
                current['volume_ratio'] > 1.2 if signal > 0 else current['volume_ratio'] > 0.8,
                
                # Momentum alignment
                np.sign(current['macd']) == np.sign(signal)
            ]
        
        return all(validations)

    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update strategy parameters based on recent performance"""
        # Recalculate trendlines with latest data
        self._calculate_pivot_points(features)
        self._calculate_trendlines(features)
        # Additional parameter optimization could be added here
        pass