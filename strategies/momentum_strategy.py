from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class MomentumConfig(TechnicalConfig):
    """Configuration for momentum/trend following strategy"""
    # Moving Averages (Swing Trading)
    short_ma_period: int = 5  # 5-day MA
    long_ma_period: int = 10  # 10-day MA
    
    # RSI Parameters
    swing_rsi_period: int = 14  # 14-period RSI for swing trading
    day_rsi_period: int = 14    # 14-period for intraday
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # MACD Parameters
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    
    # ADX Parameters
    adx_period: int = 14
    adx_strong_trend: float = 40.0
    adx_weak_trend: float = 20.0
    
    # ATR Parameters
    swing_atr_multiplier: float = 1.5
    day_atr_multiplier: float = 1.2
    
    # Intraday Parameters
    short_ema_period: int = 15  # 15-minute EMA
    long_ema_period: int = 35   # 35-minute EMA
    
    # Volume Parameters
    min_volume_ratio: float = 1.5
    obi_threshold: float = 0.2  # Order Book Imbalance threshold
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'swing': {
                    'ma_crossover': 0.3,
                    'rsi': 0.2,
                    'macd': 0.2,
                    'adx': 0.2,
                    'volume': 0.1
                },
                'day': {
                    'ema_crossover': 0.3,
                    'rsi': 0.2,
                    'momentum': 0.2,
                    'vwap': 0.2,
                    'obi': 0.1
                }
            }

class MomentumStrategy(TechnicalStrategy):
    def __init__(self, config: MomentumConfig = None):
        super().__init__(name="momentum", config=config or MomentumConfig())
        self.config: MomentumConfig = self.config
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare momentum/trend specific features"""
        df = self.prepare_base_features(data)
        
        # Swing Trading Features
        # Moving Averages
        df['short_ma'] = df['close'].rolling(window=self.config.short_ma_period).mean()
        df['long_ma'] = df['close'].rolling(window=self.config.long_ma_period).mean()
        df['ma_crossover'] = df['short_ma'] - df['long_ma']
        
        # RSI
        df['swing_rsi'] = self.ti.calculate_rsi(df['close'], self.config.swing_rsi_period)
        
        # MACD
        ema_fast = df['close'].ewm(span=self.config.macd_fast_period).mean()
        ema_slow = df['close'].ewm(span=self.config.macd_slow_period).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal_period).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ADX
        df['adx'] = self.ti.calculate_directional_movement(
            df['high'], 
            df['low'], 
            df['close'],
            self.config.adx_period
        )
        
        # Day Trading Features
        if self.is_intraday_data(df):
            # Short-term EMAs
            df['short_ema'] = df['close'].ewm(span=self.config.short_ema_period).mean()
            df['long_ema'] = df['close'].ewm(span=self.config.long_ema_period).mean()
            df['ema_crossover'] = df['short_ema'] - df['long_ema']
            
            # Intraday RSI
            df['day_rsi'] = self.ti.calculate_rsi(df['close'], self.config.day_rsi_period)
            
            # Rate of Change normalized by volatility
            df['roc'] = df['close'].pct_change(periods=15)
            df['norm_roc'] = df['roc'] / df['volatility']
            
            # Order Book Imbalance (if available)
            if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
                df['obi'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        
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
        """Calculate momentum/trend signals"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            # Day Trading Signals
            signals = {
                'ema_crossover': np.sign(current['ema_crossover']),
                'rsi': -1 if current['day_rsi'] > self.config.rsi_overbought else 
                        1 if current['day_rsi'] < self.config.rsi_oversold else 0,
                'momentum': np.sign(current['norm_roc']),
                'vwap': np.sign(current['close'] - current['vwap']),
                'obi': np.sign(current['obi']) if 'obi' in current else 0
            }
            
            weights = self.config.feature_weights['day']
            trade_type = TradeType.DAY_TRADE
            
        else:
            # Swing Trading Signals
            signals = {
                'ma_crossover': np.sign(current['ma_crossover']),
                'rsi': -1 if current['swing_rsi'] > self.config.rsi_overbought else 
                        1 if current['swing_rsi'] < self.config.rsi_oversold else 0,
                'macd': np.sign(current['macd_hist']),
                'adx': 1 if current['adx'] > self.config.adx_strong_trend else 
                      -1 if current['adx'] < self.config.adx_weak_trend else 0,
                'volume': 1 if current['volume_ratio'] > self.config.min_volume_ratio else -1
            }
            
            weights = self.config.feature_weights['swing']
            trade_type = TradeType.SWING_TRADE
        
        # Calculate weighted signal
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
        """Additional momentum/trend specific validation"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            validations = [
                # Sufficient intraday volume
                current['volume_ratio'] > 1.0,
                
                # Not near session VWAP during low-volatility periods
                not (abs(current['vwap_ratio'] - 1) < 0.001 and 
                     current['volatility_regime'] < 0.5),
                
                # Strong enough momentum
                abs(current['norm_roc']) > current['atr'] / current['close']
            ]
        else:
            validations = [
                # Strong enough trend
                current['adx'] > self.config.adx_weak_trend,
                
                # Volume confirms trend
                current['volume_ratio'] > self.config.min_volume_ratio,
                
                # Not overextended
                abs(current['ma_crossover']) < current['atr'] * 2
            ]
        
        return all(validations)

    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update strategy parameters based on recent performance"""
        # Implementation for parameter updates based on performance
        pass