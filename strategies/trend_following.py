from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class TrendFollowingConfig(TechnicalConfig):
    """Configuration for trend following strategy"""
    # Moving Averages parameters
    swing_fast_ma_period: int = 50  # Fast MA period for swing trading
    swing_slow_ma_period: int = 200  # Slow MA period for swing trading
    day_fast_ma_period: int = 5     # Fast MA period for day trading
    day_slow_ma_period: int = 20    # Slow MA period for day trading
    
    # Trend strength parameters
    adx_period: int = 14           # Period for ADX calculation
    adx_threshold: float = 25.0    # Threshold for strong trend (typically 25)
    
    # MACD parameters
    swing_macd_fast: int = 12
    swing_macd_slow: int = 26
    swing_macd_signal: int = 9
    day_macd_fast: int = 5       # Faster for day trading
    day_macd_slow: int = 13
    day_macd_signal: int = 4
    
    # Volatility parameters
    atr_period: int = 14
    swing_atr_multiplier: float = 2.0  # Stop loss in ATR units (swing)
    day_atr_multiplier: float = 1.5    # Stop loss in ATR units (day)
    
    # Volume confirmation
    volume_threshold: float = 1.2  # Minimum volume surge for confirmation
    
    # Placeholder for crypto/Ethereum-specific event/calendar logic
    # TODO: Add funding rate, exchange event, or blockchain event parameters as needed
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'swing': {
                    'ma_crossover': 0.25,
                    'macd': 0.20,
                    'adx': 0.20,
                    'price_momentum': 0.15,
                    'volume': 0.10,
                    # 'seasonality': 0.10  # Removed gas-specific
                },
                'day': {
                    'ma_crossover': 0.20,
                    'macd': 0.20,
                    'vwap': 0.20,
                    'volume_momentum': 0.15,
                    'price_acceleration': 0.15,
                    'order_flow': 0.10
                }
            }


class TrendFollowingStrategy(TechnicalStrategy):
    """
    Trend following strategy for crypto (Ethereum-focused, asset-agnostic).
    """
    def __init__(self, config: TrendFollowingConfig = None):
        super().__init__(name="trend_following", config=config or TrendFollowingConfig())
        self.config: TrendFollowingConfig = self.config
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for trend following analysis"""
        df = self.prepare_base_features(data)
        
        # Determine if we're working with intraday data
        is_intraday = self.is_intraday_data(df)
        
        # Add common features
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        
        # Add timeframe-specific features
        if is_intraday:
            df = self._add_intraday_features(df)
        else:
            df = self._add_swing_features(df)
        
        # TODO: Add crypto/Ethereum-specific event/calendar features here
        # (e.g., funding rate events, exchange maintenance, blockchain upgrades)
        
        return df
    
    def is_intraday_data(self, df: pd.DataFrame) -> bool:
        """Check if we're working with intraday data"""
        if len(df) < 2:
            return False
        time_diff = df.index[1] - df.index[0]
        return time_diff.total_seconds() < 24 * 60 * 60
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Calculate ATR if not already present
        if 'atr' not in df.columns:
            df['atr'] = self.ti.calculate_atr(
                df['high'], df['low'], df['close'], 
                period=self.config.atr_period
            )
        
        # Normalized volatility (compare to recent history)
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(window=50).mean()
        
        # Detect volatility regimes
        df['high_vol_regime'] = df['volatility_ratio'] > 1.5  # 50% higher than average
        df['low_vol_regime'] = df['volatility_ratio'] < 0.7   # 30% lower than average
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features for trend confirmation"""
        if 'volume' not in df.columns:
            return df
            
        # Volume relative to moving average
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Volume momentum (acceleration/deceleration)
        df['volume_change'] = df['volume'].pct_change(5)
        
        # Volume-price relationship
        df['volume_price_trend'] = df['returns'].rolling(window=5).sum() * df['volume_ratio']
        
        # Detect volume surges
        df['volume_surge'] = df['volume_ratio'] > self.config.volume_threshold
        
        # Create volume confirmation flag
        df['volume_confirms_price'] = (
            (df['returns'] > 0) & (df['volume_ratio'] > 1.0) |  # Rising price with above-average volume
            (df['returns'] < 0) & (df['volume_ratio'] > 1.0)    # Falling price with above-average volume
        )
        
        return df
    
    def _add_swing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to swing trading trend following"""
        # Moving averages for trend direction
        df[f'ma_{self.config.swing_fast_ma_period}'] = df['close'].rolling(
            window=self.config.swing_fast_ma_period).mean()
        df[f'ma_{self.config.swing_slow_ma_period}'] = df['close'].rolling(
            window=self.config.swing_slow_ma_period).mean()
        
        # MA crossover
        df['swing_ma_crossover'] = (
            df[f'ma_{self.config.swing_fast_ma_period}'] - 
            df[f'ma_{self.config.swing_slow_ma_period}']
        )
        
        # Trend direction based on MA crossover
        df['ma_trend_direction'] = np.sign(df['swing_ma_crossover'])
        
        # Calculate if we just had a crossover (signal for entry)
        df['ma_crossover_signal'] = (
            (df['ma_trend_direction'] != df['ma_trend_direction'].shift(1)) & 
            (df['ma_trend_direction'] != 0)
        )
        
        # MACD for momentum
        ema_fast = df['close'].ewm(span=self.config.swing_macd_fast).mean()
        ema_slow = df['close'].ewm(span=self.config.swing_macd_slow).mean()
        df['swing_macd'] = ema_fast - ema_slow
        df['swing_macd_signal'] = df['swing_macd'].ewm(span=self.config.swing_macd_signal).mean()
        df['swing_macd_hist'] = df['swing_macd'] - df['swing_macd_signal']
        
        # Normalize MACD by ATR for volatility adjustment
        df['swing_macd_norm'] = df['swing_macd_hist'] / df['atr']
        
        # ADX for trend strength
        df['adx'] = self.ti.calculate_directional_movement(
            df['high'], df['low'], df['close'], 
            self.config.adx_period
        )
        
        # Strong trend filter
        df['strong_trend'] = df['adx'] > self.config.adx_threshold
        
        # Price momentum
        df['price_momentum_10'] = df['close'].pct_change(10)
        df['price_momentum_20'] = df['close'].pct_change(20)
        
        # Trend persistence (has trend direction been consistent?)
        df['trend_persistence'] = df['ma_trend_direction'].rolling(window=10).mean().abs()
        
        return df
    
    def _add_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to intraday trend following"""
        # Short-term moving averages
        df[f'ma_{self.config.day_fast_ma_period}'] = df['close'].rolling(
            window=self.config.day_fast_ma_period).mean()
        df[f'ma_{self.config.day_slow_ma_period}'] = df['close'].rolling(
            window=self.config.day_slow_ma_period).mean()
        
        # MA crossover for intraday
        df['day_ma_crossover'] = (
            df[f'ma_{self.config.day_fast_ma_period}'] - 
            df[f'ma_{self.config.day_slow_ma_period}']
        )
        
        # Intraday trend direction
        df['intraday_trend_direction'] = np.sign(df['day_ma_crossover'])
        
        # Detect fresh crossovers
        df['intraday_crossover_signal'] = (
            (df['intraday_trend_direction'] != df['intraday_trend_direction'].shift(1)) & 
            (df['intraday_trend_direction'] != 0)
        )
        
        # Fast MACD for intraday momentum
        ema_fast = df['close'].ewm(span=self.config.day_macd_fast).mean()
        ema_slow = df['close'].ewm(span=self.config.day_macd_slow).mean()
        df['day_macd'] = ema_fast - ema_slow
        df['day_macd_signal'] = df['day_macd'].ewm(span=self.config.day_macd_signal).mean()
        df['day_macd_hist'] = df['day_macd'] - df['day_macd_signal']
        
        # Normalize MACD by intraday volatility
        short_atr = self.ti.calculate_atr(
            df['high'], df['low'], df['close'], 
            period=5  # Shorter period for intraday
        )
        df['day_macd_norm'] = df['day_macd_hist'] / short_atr
        
        # VWAP features for intraday
        if 'vwap' not in df.columns:
            df['vwap'] = self.ti.calculate_vwap(df)
        
        # VWAP deviation in ATR units
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['atr']
        
        # VWAP slope (trend in VWAP itself)
        df['vwap_slope'] = df['vwap'].pct_change(3)
        
        # Price acceleration (momentum of momentum)
        df['price_velocity'] = df['returns'].rolling(window=3).sum()
        df['price_acceleration'] = df['price_velocity'].diff(3)
        
        # Volume momentum (short-term)
        if 'volume' in df.columns:
            df['volume_momentum'] = df['volume'].pct_change(3)
            df['volume_momentum_signal'] = np.sign(df['volume_momentum'] * df['returns'])
        
        # Order flow features (if available)
        if all(col in df.columns for col in ['bid_volume', 'ask_volume']):
            df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
            df['order_flow_signal'] = np.sign(df['order_imbalance'])
        else:
            df['order_flow_signal'] = 0
        
        # Time of day features
        if hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
            
            # Market session flags (example for US market)
            df['market_open'] = (df['hour'] == 9) | (df['hour'] == 10)  # Early session
            df['lunch_hour'] = (df['hour'] >= 12) & (df['hour'] < 13)
            df['market_close'] = (df['hour'] >= 15) & (df['hour'] < 16)
        
        return df
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate trend following signals"""
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
        """Calculate swing trading signals for trend following"""
        signals = {}
        
        # 1. Moving Average Crossover Signal
        if 'swing_ma_crossover' in current:
            # Trend direction from MA crossover
            ma_signal = np.sign(current['swing_ma_crossover'])
            
            # Stronger signal if we just had a crossover
            if current.get('ma_crossover_signal', False):
                ma_signal *= 1.5
                
            signals['ma_crossover'] = ma_signal
        else:
            signals['ma_crossover'] = 0
            
        # 2. MACD Momentum Signal
        if 'swing_macd_norm' in current:
            # Normalize to -1 to 1 range
            macd_signal = np.clip(current['swing_macd_norm'] * 2, -1, 1)
            signals['macd'] = macd_signal
        else:
            signals['macd'] = 0
            
        # 3. ADX Trend Strength
        if 'adx' in current and 'ma_trend_direction' in current:
            # Only consider ADX in direction of MA trend
            if current['adx'] > self.config.adx_threshold:
                adx_signal = current['ma_trend_direction'] * min((current['adx'] - 20) / 30, 1.0)
            else:
                adx_signal = 0
                
            signals['adx'] = adx_signal
        else:
            signals['adx'] = 0
            
        # 4. Price Momentum
        if 'price_momentum_20' in current:
            # Scale momentum to reasonable range
            momentum_signal = np.clip(current['price_momentum_20'] * 10, -1, 1)
            signals['price_momentum'] = momentum_signal
        else:
            signals['price_momentum'] = 0
            
        # 5. Volume Confirmation
        if 'volume_confirms_price' in current and 'volume_ratio' in current:
            # Volume above average in direction of trend
            if current['volume_confirms_price'] and current['volume_ratio'] > 1.0:
                volume_signal = np.sign(current['returns']) * min(current['volume_ratio'] - 0.8, 1.0)
            else:
                volume_signal = 0
                
            signals['volume'] = volume_signal
        else:
            signals['volume'] = 0
            
        # 6. Seasonality Adjustment
        signals['seasonality'] = 0
        
        # Combine signals with weights
        weights = self.config.feature_weights['swing']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement and strength
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            # Calculate agreement ratio among signals
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Adjust for trend strength
            if 'adx' in current:
                adx_factor = min(current['adx'] / 50.0, 1.0)  # Scale 0-1 based on ADX
            else:
                adx_factor = 0.5  # Default middle value
                
            # Factor in trend persistence
            persistence = current.get('trend_persistence', 0.5)
            
            # Calculate final confidence
            confidence = agreement_ratio * 0.6 + adx_factor * 0.2 + persistence * 0.2
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.SWING_TRADE
    
    def _calculate_day_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate day trading signals for trend following"""
        signals = {}
        
        # 1. Intraday MA Crossover Signal
        if 'day_ma_crossover' in current:
            # Trend direction from MA crossover
            ma_signal = np.sign(current['day_ma_crossover'])
            
            # Stronger signal if we just had a crossover
            if current.get('intraday_crossover_signal', False):
                ma_signal *= 1.5
                
            signals['ma_crossover'] = ma_signal
        else:
            signals['ma_crossover'] = 0
            
        # 2. MACD Momentum Signal
        if 'day_macd_norm' in current:
            # Normalize to -1 to 1 range
            macd_signal = np.clip(current['day_macd_norm'] * 2, -1, 1)
            signals['macd'] = macd_signal
        else:
            signals['macd'] = 0
            
        # 3. VWAP Signal
        if 'vwap_deviation' in current:
            # Price position relative to VWAP
            vwap_signal = np.clip(current['vwap_deviation'] / self.config.vwap_deviation_threshold, -1, 1)
            
            # Add VWAP slope for trend confirmation
            if 'vwap_slope' in current:
                vwap_signal *= np.sign(current['vwap_slope']) if abs(current['vwap_slope']) > 0.001 else 1
                
            signals['vwap'] = vwap_signal
        else:
            signals['vwap'] = 0
            
        # 4. Volume Momentum
        if 'volume_momentum' in current:
            vol_signal = np.clip(current['volume_momentum'] * 3, -1, 1)
            
            # Align with price direction
            if 'returns' in current:
                vol_signal *= np.sign(current['returns'])
                
            signals['volume_momentum'] = vol_signal
        else:
            signals['volume_momentum'] = 0
            
        # 5. Price Acceleration
        if 'price_acceleration' in current:
            accel_signal = np.clip(current['price_acceleration'] * 50, -1, 1)  # Scale appropriately
            signals['price_acceleration'] = accel_signal
        else:
            signals['price_acceleration'] = 0
            
        # 6. Order Flow Signal
        if 'order_flow_signal' in current:
            signals['order_flow'] = current['order_flow_signal']
        else:
            signals['order_flow'] = 0
            
        # Combine signals with weights
        weights = self.config.feature_weights['day']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            # Calculate agreement ratio
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Factor in market session
            session_factor = 1.0  # Default
            
            if current.get('market_open', False):
                session_factor = 1.2  # More confident during opening hour
            elif current.get('lunch_hour', False):
                session_factor = 0.8  # Less confident during lunch
            elif current.get('market_close', False):
                session_factor = 0.9  # Slightly less confident at close
                
            # Calculate final confidence
            confidence = agreement_ratio * session_factor
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.DAY_TRADE
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional trend following specific validation"""
        if not self.validate_technical_signal(signal, features):
            return False
        
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Common validations
        validations = [
            # Signal strength must be meaningful
            abs(signal) >= 0.2
        ]
        
        # Add timeframe-specific validations
        if is_intraday:
            # Day trading validations
            day_validations = [
                # ADX above threshold or trending VWAP
                current.get('adx', 0) > 20 or abs(current.get('vwap_slope', 0)) > 0.001,
                # Volume confirmation
                current.get('volume_ratio', 0) > 0.8
            ]
            validations.extend(day_validations)
        else:
            # Swing trading validations
            swing_validations = [
                # Strong enough trend
                current.get('adx', 0) > self.config.adx_threshold,
                # MACD aligned with signal direction
                np.sign(current.get('swing_macd', 0)) == np.sign(signal)
            ]
            validations.extend(swing_validations)
        
        return all(validations)
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        signal: float, 
        features: pd.DataFrame
    ) -> float:
        """Calculate dynamic stop loss level for trend following"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Get ATR for volatility-based stop
        atr = current.get('atr', entry_price * 0.01)  # Default to 1% if ATR not available
        
        if is_intraday:
            # Base stop distance for day trading
            stop_distance = atr * self.config.day_atr_multiplier
        else:
            # Base stop distance for swing trading
            stop_distance = atr * self.config.swing_atr_multiplier
            # Adjust based on ADX - stronger trends need wider stops
            if 'adx' in current:
                adx_factor = min(1.0 + (current['adx'] - 25) / 50.0, 1.5)  # Scale from 1.0 to 1.5
                stop_distance *= adx_factor
        
        # Apply stop based on signal direction
        if signal > 0:  # Long position
            return entry_price - stop_distance
        else:  # Short position
            return entry_price + stop_distance
    
    def calculate_take_profit(
        self, 
        entry_price: float, 
        stop_loss: float, 
        signal: float, 
        features: pd.DataFrame
    ) -> float:
        """Calculate take profit level for trend following"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Calculate risk (distance to stop)
        risk = abs(entry_price - stop_loss)
        
        if is_intraday:
            # Day trading - typically 1.5:1 to 2:1 reward:risk
            reward_risk_ratio = 2.0
            # Adjust based on time of day
            if current.get('market_open', False):
                reward_risk_ratio = 2.5
            elif current.get('market_close', False):
                reward_risk_ratio = 1.5
        else:
            # Swing trading - typically 2:1 to 3:1 reward:risk for trend following
            reward_risk_ratio = 2.5
            # Adjust based on trend strength
            if 'adx' in current:
                adx_factor = min(1.0 + (current['adx'] - 25) / 40.0, 1.5)  # Scale from 1.0 to 1.5
                reward_risk_ratio *= adx_factor
        
        # Calculate reward based on risk
        reward = risk * reward_risk_ratio
        
        # Apply reward based on signal direction
        if signal > 0:  # Long position
            return entry_price + reward
        else:  # Short position
            return entry_price - reward
    
    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update strategy parameters based on recent performance"""
        # Could implement adaptive parameters here
        # For example, adjusting the ADX threshold based on recent trend quality
        # or updating the MA periods based on market volatility
        pass