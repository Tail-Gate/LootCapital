from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import torch

from strategies.base_strategy import BaseStrategy, TradeType
from utils.technical_indicators import TechnicalIndicators

@dataclass
class TechnicalConfig:
    """Base configuration for technical strategies"""
    # Time windows
    day_window: int = 60  # 1-hour for intraday
    swing_window: int = 20  # 20 days for swing
    
    # Volume thresholds
    min_volume_percentile: float = 20.0
    volume_surge_threshold: float = 2.0
    
    # Volatility settings
    volatility_window: int = 20
    high_volatility_threshold: float = 1.5
    low_volatility_threshold: float = 0.5
    
    # Technical indicator periods
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Price action
    price_gap_threshold: float = 0.02  # 2% gap
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate configuration parameters"""
        assert 0 < self.min_volume_percentile < 100
        assert self.volume_surge_threshold > 1
        assert self.volatility_window > 0
        assert self.high_volatility_threshold > self.low_volatility_threshold

class TechnicalStrategy(BaseStrategy, ABC):
    """Abstract base class for all technical trading strategies"""
    
    def __init__(self, name: str, config: TechnicalConfig = None):
        super().__init__(name)
        self.config = config or TechnicalConfig()
        self.ti = TechnicalIndicators()
        self.market_regime: Optional[str] = None
        
    def prepare_base_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare common technical features used by all technical strategies
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with basic technical features
        """
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=self.config.day_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(
            window=self.config.volatility_window
        ).std()
        
        # Common technical indicators
        df['rsi'] = self.ti.calculate_rsi(
            df['close'], 
            period=self.config.rsi_period
        )
        
        df['atr'] = self.ti.calculate_atr(
            df['high'], 
            df['low'], 
            df['close'], 
            period=self.config.atr_period
        )
        
        # VWAP
        df['vwap'] = self.ti.calculate_vwap(df)
        df['vwap_ratio'] = df['close'] / df['vwap']
        
        # Market regime features
        df['regime_volatility'] = self.ti.calculate_volatility_regime(
            df['close'],
            self.config.volatility_window
        )
        
        return df
    
    def detect_market_regime(self, features: pd.DataFrame) -> str:
        """
        Detect current market regime based on technical features
        
        Args:
            features: DataFrame with technical features
            
        Returns:
            String indicating market regime
        """
        current = features.iloc[-1]
        
        # Volatility regime
        high_vol = current['regime_volatility'] > self.config.high_volatility_threshold
        low_vol = current['regime_volatility'] < self.config.low_volatility_threshold
        
        # Volume regime
        high_volume = current['volume_ratio'] > self.config.volume_surge_threshold
        
        # Determine regime
        if high_vol and high_volume:
            regime = 'high_volatility_trending'
        elif high_vol and not high_volume:
            regime = 'high_volatility_choppy'
        elif low_vol and high_volume:
            regime = 'low_volatility_accumulation'
        elif low_vol and not high_volume:
            regime = 'low_volatility_range'
        else:
            regime = 'normal'
            
        self.market_regime = regime
        return regime
    
    def validate_technical_signal(
        self, 
        signal: float, 
        features: pd.DataFrame
    ) -> bool:
        """
        Common technical validation rules
        
        Args:
            signal: The calculated signal strength
            features: Technical features DataFrame
            
        Returns:
            Boolean indicating if signal passes technical validation
        """
        current = features.iloc[-1]
        
        # Volume validation
        volume_sufficient = (
            current['volume_ratio'] > 
            np.percentile(features['volume_ratio'], self.config.min_volume_percentile)
        )
        
        # Volatility validation
        volatility_normal = (
            current['regime_volatility'] < 
            self.config.high_volatility_threshold * 1.5
        )
        
        # Price gap validation
        no_large_gaps = abs(features['returns'].iloc[-1]) < self.config.price_gap_threshold
        
        return all([volume_sufficient, volatility_normal, no_large_gaps])
    
    @abstractmethod
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """
        Calculate strategy-specific technical signals
        
        Args:
            features: Technical features DataFrame
            
        Returns:
            Tuple of (signal_strength, confidence_score, trade_type)
        """
        pass
    
    def calculate_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """
        Implement base strategy method with technical analysis
        """
        # Update market regime
        self.detect_market_regime(features)
        
        # Get strategy-specific signals
        return self.calculate_technical_signals(features)
    
    def get_required_features(self) -> List[str]:
        """List of required features for technical analysis"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'vwap', 'returns', 'volatility'
        ]
    
    def save_state(self, path: str) -> None:
        """Save strategy state"""
        state = {
            'config': self.config.__dict__,
            'market_regime': self.market_regime,
            'model_state': self.model.state_dict() if self.model else None
        }
        torch.save(state, path)
    
    def load_state(self, path: str) -> None:
        """Load strategy state"""
        state = torch.load(path)
        self.config = TechnicalConfig(**state['config'])
        self.market_regime = state['market_regime']
        if state['model_state'] and self.model:
            self.model.load_state_dict(state['model_state'])