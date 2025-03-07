from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class MomentumReversalConfig(TechnicalConfig):
    """Configuration for momentum reversal strategy"""
    # Stochastic Oscillator Parameters
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0
    
    # Keltner Channel Parameters
    keltner_ema_period: int = 20
    keltner_atr_multiplier: float = 2.0
    
    # RSI Parameters
    rsi_period: int = 14
    rsi_overbought: float = 75.0
    rsi_oversold: float = 25.0
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Swing Trading Parameters
    swing_momentum_threshold: float = 0.10  # 10% move
    swing_momentum_period: int = 5  # 3-5 days
    swing_atr_multiplier: float = 2.0
    swing_take_profit: float = 0.10  # 10% move
    
    # Day Trading Parameters
    day_momentum_threshold: float = 0.03  # 3% move
    day_vwap_deviation: float = 2.5  # std deviations
    day_atr_multiplier: float = 1.5
    day_take_profit: float = 0.03  # 3% move
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'swing': {
                    'momentum_overextension': 0.3,
                    'stochastic': 0.2,
                    'keltner': 0.2,
                    'rsi_divergence': 0.2,
                    'volume_delta': 0.1
                },
                'day': {
                    'fast_momentum': 0.3,
                    'vwap_deviation': 0.2,
                    'order_flow': 0.2,
                    'bollinger': 0.2,
                    'volume_delta': 0.1
                }
            }

class MomentumReversalStrategy(TechnicalStrategy):
    def __init__(self, config: MomentumReversalConfig = None):
        super().__init__(name="momentum_reversal", config=config or MomentumReversalConfig())
        self.config: MomentumReversalConfig = self.config
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare momentum reversal specific features"""
        df = self.prepare_base_features(data)
        
        # Stochastic Oscillator
        df = self._calculate_stochastic(df)
        
        # Keltner Channels
        df = self._calculate_keltner_channels(df)
        
        # RSI with divergence
        df = self._calculate_rsi_divergence(df)
        
        # Bollinger Bands
        df = self._calculate_bollinger_features(df)
        
        # Volume Delta and CVD
        df = self._calculate_volume_features(df)
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator features"""
        # Fast Stochastic (%K)
        highest_high = df['high'].rolling(window=self.config.stoch_k_period).max()
        lowest_low = df['low'].rolling(window=self.config.stoch_k_period).min()
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        
        # Slow Stochastic (%D)
        df['stoch_d'] = df['stoch_k'].rolling(window=self.config.stoch_d_period).mean()
        
        # Crossover signals
        df['stoch_crossover'] = np.where(
            (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)),
            1,  # Bullish crossover
            np.where(
                (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)),
                -1,  # Bearish crossover
                0
            )
        )
        
        return df
    
    def _calculate_keltner_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channel features"""
        # Middle line (EMA)
        df['keltner_middle'] = df['close'].ewm(span=self.config.keltner_ema_period).mean()
        
        # ATR for channel width
        df['keltner_atr'] = self.ti.calculate_atr(
            df['high'], df['low'], df['close'], 
            self.config.keltner_ema_period
        )
        
        # Upper and lower bands
        mult = self.config.keltner_atr_multiplier
        df['keltner_upper'] = df['keltner_middle'] + (df['keltner_atr'] * mult)
        df['keltner_lower'] = df['keltner_middle'] - (df['keltner_atr'] * mult)
        
        # Deviation signals
        df['keltner_position'] = np.where(
            df['close'] > df['keltner_upper'], 1,
            np.where(df['close'] < df['keltner_lower'], -1, 0)
        )
        
        return df
    
    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and detect divergences"""
        df['rsi'] = self.ti.calculate_rsi(df['close'], self.config.rsi_period)
        
        # Regular divergence detection
        df['price_high'] = df['close'].rolling(window=5).max()
        df['rsi_high'] = df['rsi'].rolling(window=5).max()
        
        df['bearish_divergence'] = (
            (df['close'] > df['price_high'].shift(5)) & 
            (df['rsi'] < df['rsi_high'].shift(5))
        )
        
        df['bullish_divergence'] = (
            (df['close'] < df['price_high'].shift(5)) & 
            (df['rsi'] > df['rsi_high'].shift(5))
        )
        
        return df
    
    def _calculate_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Band features"""
        upper, middle, lower = self.ti.calculate_bollinger_bands(
            df['close'],
            period=self.config.bb_period,
            num_std=self.config.bb_std_dev
        )
        
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        
        # Detect mean reversion signals
        df['bb_position'] = np.where(
            df['close'] > upper, 1,
            np.where(df['close'] < lower, -1, 0)
        )
        
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            # Volume Delta
            df['volume_delta'] = df['bid_volume'] - df['ask_volume']
            
            # Cumulative Volume Delta
            df['cvd'] = df['volume_delta'].cumsum()
            
            # Volume Delta Spikes
            df['volume_delta_ma'] = df['volume_delta'].rolling(window=20).mean()
            df['volume_spike'] = df['volume_delta'].abs() > (
                df['volume_delta_ma'].abs() * 2
            )
        
        return df
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate momentum reversal signals"""
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
        signals = {
            'momentum_overextension': self._calculate_momentum_overextension(features),
            'stochastic': -current['stoch_crossover'],
            'keltner': -current['keltner_position'],
            'rsi_divergence': (
                1 if current['bullish_divergence'] 
                else -1 if current['bearish_divergence'] 
                else 0
            ),
            'volume_delta': np.sign(current.get('volume_delta', 0))
        }
        
        weights = self.config.feature_weights['swing']
        trade_type = TradeType.SWING_TRADE
        
        return self._combine_signals(signals, weights, trade_type)
    
    def _calculate_day_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate day trading signals"""
        signals = {
            'fast_momentum': -np.sign(current['returns']) if abs(current['returns']) > self.config.day_momentum_threshold else 0,
            'vwap_deviation': -np.sign(current['vwap_ratio'] - 1) if abs(current['vwap_ratio'] - 1) > self.config.day_vwap_deviation else 0,
            'order_flow': np.sign(current.get('volume_delta', 0)),
            'bollinger': -current['bb_position'],
            'volume_delta': np.sign(current.get('volume_delta', 0))
        }
        
        weights = self.config.feature_weights['day']
        trade_type = TradeType.DAY_TRADE
        
        return self._combine_signals(signals, weights, trade_type)
    
    def _calculate_momentum_overextension(self, features: pd.DataFrame) -> float:
        """Calculate momentum overextension signal"""
        returns = features['returns'].rolling(
            window=self.config.swing_momentum_period
        ).sum()
        
        if abs(returns.iloc[-1]) > self.config.swing_momentum_threshold:
            return -np.sign(returns.iloc[-1])
        return 0
    
    def _combine_signals(
        self,
        signals: Dict[str, float],
        weights: Dict[str, float],
        trade_type: TradeType
    ) -> Tuple[float, float, TradeType]:
        """Combine individual signals into final signal"""
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            confidence = agreement_ratio * abs(total_signal)
        else:
            confidence = 0
        
        return total_signal, confidence, trade_type
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional momentum reversal specific validation"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            validations = [
                # Significant price move
                abs(current['returns']) > self.config.day_momentum_threshold,
                
                # Volume confirmation
                current['volume_ratio'] > 1.5,
                
                # Volatility expansion
                current['bb_width'] > current['bb_width'].rolling(window=20).mean(),
                
                # Not in low liquidity period
                self._is_valid_trading_period(features)
            ]
        else:
            validations = [
                # Momentum overextension
                abs(features['returns'].rolling(window=self.config.swing_momentum_period).sum().iloc[-1]) > self.config.swing_momentum_threshold,
                
                # RSI extreme
                (current['rsi'] > self.config.rsi_overbought) or (current['rsi'] < self.config.rsi_oversold),
                
                # Volume confirmation
                current['volume_ratio'] > 1.5
            ]
        
        return all(validations)
    
    def _is_valid_trading_period(self, features: pd.DataFrame) -> bool:
        """Check if current time is valid for trading"""
        if not self.is_intraday_data(features):
            return True
            
        current_time = features.index[-1].time()
        # Avoid first 15 minutes after open and last 15 minutes before close
        # This is a simplified example - adjust based on market hours
        market_open = pd.Timestamp('09:30').time()
        market_close = pd.Timestamp('16:00').time()
        
        return (
            current_time > pd.Timestamp('09:45').time() and
            current_time < pd.Timestamp('15:45').time()
        )

    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update strategy parameters based on recent performance"""
        # Implementation for parameter updates based on performance
        pass