import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import joblib

from market_analysis.market_regime import MarketRegimeDetector

class MarketAnalyzer:
    """
    Analyzes market conditions and provides context to trading strategies.
    Combines traditional indicators with machine learning for crypto markets (Ethereum-focused, asset-agnostic).
    """
    
    def __init__(
        self,
        regime_detector: Optional[MarketRegimeDetector] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize the market analyzer
        
        Args:
            regime_detector: Optional pre-configured market regime detector
            model_path: Path to load saved models from
        """
        # Initialize regime detector
        if regime_detector:
            self.regime_detector = regime_detector
        elif model_path and os.path.exists(os.path.join(model_path, 'regime_detector.joblib')):
            self.regime_detector = MarketRegimeDetector()
            self.regime_detector.load_model(os.path.join(model_path, 'regime_detector.joblib'))
        else:
            self.regime_detector = MarketRegimeDetector()
        
        # Current market regime
        self.current_regime = None
        self.regime_history = []
        
        # Placeholder for crypto-specific context (e.g., funding rates, exchange events)
        self.crypto_context = {}
    
    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """
        Analyze market conditions for crypto (Ethereum)
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            Dictionary with market analysis results
        """
        # Ensure we have enough data
        if len(data) < self.regime_detector.lookback_period:
            return {'regime': None, 'regime_name': 'Unknown', 'confidence': 0}
        
        # Detect market regime
        regimes = self.regime_detector.predict(data)
        self.current_regime = regimes.iloc[-1]
        
        # Calculate regime stability
        recent_regimes = regimes.iloc[-20:]
        if len(recent_regimes) > 0:
            regime_counts = recent_regimes.value_counts()
            regime_stability = regime_counts.iloc[0] / len(recent_regimes) if len(regime_counts) > 0 else 0
        else:
            regime_stability = 0
        
        # Get regime name
        regime_name = self.regime_detector.get_regime_name(self.current_regime)
        
        # Placeholder for crypto-specific context (e.g., funding rates, exchange events)
        # TODO: Add real crypto context features here
        
        # Detect volatility conditions
        volatility = data['returns'].rolling(window=20).std().iloc[-1]
        volatility_percentile = self._calculate_percentile(volatility, data['returns'].rolling(window=20).std())
        
        # Build result
        result = {
            'regime': self.current_regime,
            'regime_name': regime_name,
            'regime_stability': regime_stability,
            'confidence': min(regime_stability * 1.5, 1.0),
            'volatility': volatility,
            'volatility_percentile': volatility_percentile,
            # Add crypto context fields as needed
        }
        
        # Update regime history
        self.regime_history.append(result)
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        return result
    
    def _calculate_percentile(self, value: float, distribution: pd.Series) -> float:
        """
        Calculate percentile of a value in a distribution
        
        Args:
            value: Value to calculate percentile for
            distribution: Reference distribution
        
        Returns:
            Percentile (0-100)
        """
        distribution = distribution.dropna()
        if len(distribution) == 0:
            return 50.0
        
        return sum(distribution <= value) / len(distribution) * 100
    
    def get_strategy_adjustments(self, regime_info: Dict) -> Dict:
        """
        Get strategy adjustments based on market regime and crypto-specific context
        
        Args:
            regime_info: Market regime information from analyze_market
        
        Returns:
            Dictionary with strategy adjustments
        """
        regime_name = regime_info['regime_name']
        confidence = regime_info['confidence']
        volatility_percentile = regime_info.get('volatility_percentile', 50)
        
        # Default adjustments (can be extended for crypto context)
        adjustments = {
            'position_size_factor': 1.0,
            'stop_loss_factor': 1.0,
            'take_profit_factor': 1.0,
            'favored_strategies': [],
            'disfavored_strategies': []
        }
        
        # Adjust based on regime (generic, not gas-specific)
        if regime_name == "High Volatility Trending":
            adjustments['position_size_factor'] = 0.7
            adjustments['stop_loss_factor'] = 1.5
            adjustments['take_profit_factor'] = 1.2
            adjustments['favored_strategies'] = ['momentum', 'volatility_breakout']
            adjustments['disfavored_strategies'] = ['mean_reversion', 'range_trading']
        elif regime_name == "Low Volatility Trending":
            adjustments['position_size_factor'] = 1.2
            adjustments['stop_loss_factor'] = 0.8
            adjustments['take_profit_factor'] = 1.0
            adjustments['favored_strategies'] = ['momentum', 'dynamic_trendline']
            adjustments['disfavored_strategies'] = ['volatility_breakout']
        elif regime_name == "High Volatility Mean Reversion":
            adjustments['position_size_factor'] = 0.6
            adjustments['stop_loss_factor'] = 1.5
            adjustments['take_profit_factor'] = 0.8
            adjustments['favored_strategies'] = ['mean_reversion', 'volatility_clustering']
            adjustments['disfavored_strategies'] = ['momentum', 'breakout']
        elif regime_name == "Low Volatility Mean Reversion":
            adjustments['position_size_factor'] = 1.0
            adjustments['stop_loss_factor'] = 0.9
            adjustments['take_profit_factor'] = 1.1
            adjustments['favored_strategies'] = ['range_trading', 'mean_reversion']
            adjustments['disfavored_strategies'] = ['volatility_breakout']
        elif regime_name == "High Volatility Choppy":
            adjustments['position_size_factor'] = 0.5
            adjustments['stop_loss_factor'] = 1.7
            adjustments['take_profit_factor'] = 0.7
            adjustments['favored_strategies'] = ['volatility_clustering']
            adjustments['disfavored_strategies'] = ['momentum', 'dynamic_trendline']
        elif regime_name == "Low Volatility Consolidation":
            adjustments['position_size_factor'] = 0.8
            adjustments['stop_loss_factor'] = 1.0
            adjustments['take_profit_factor'] = 1.2
            adjustments['favored_strategies'] = ['range_trading', 'volatility_breakout']
            adjustments['disfavored_strategies'] = ['momentum']
        
        # Volatility-based adjustments (generic)
        if volatility_percentile > 80:
            adjustments['position_size_factor'] *= 0.8
            adjustments['stop_loss_factor'] *= 1.2
        elif volatility_percentile < 20:
            adjustments['position_size_factor'] *= 1.2
            adjustments['stop_loss_factor'] *= 0.8
        
        # Scale adjustments by confidence
        for key in ['position_size_factor', 'stop_loss_factor', 'take_profit_factor']:
            adjustments[key] = 1.0 + (adjustments[key] - 1.0) * confidence
        
        return adjustments
    
    def train_regime_detector(self, data: pd.DataFrame, epochs: int = 100):
        """
        Train the market regime detector
        
        Args:
            data: DataFrame with OHLCV data
            epochs: Number of training epochs
        """
        self.regime_detector.fit(data, epochs=epochs)
    
    def save_models(self, path: str):
        """
        Save trained models to disk
        
        Args:
            path: Directory path to save models to
        """
        os.makedirs(path, exist_ok=True)
        self.regime_detector.save_model(os.path.join(path, 'regime_detector.joblib'))
        
        # Save analyzer state
        analyzer_state = {
            'current_regime': self.current_regime,
            'crypto_context': self.crypto_context
        }
        joblib.dump(analyzer_state, os.path.join(path, 'analyzer_state.joblib'))
    
    def load_models(self, path: str):
        """
        Load models from disk
        
        Args:
            path: Directory path to load models from
        """
        # Load regime detector
        self.regime_detector.load_model(os.path.join(path, 'regime_detector.joblib'))
        
        # Load analyzer state
        analyzer_state_path = os.path.join(path, 'analyzer_state.joblib')
        if os.path.exists(analyzer_state_path):
            analyzer_state = joblib.load(analyzer_state_path)
            self.current_regime = analyzer_state.get('current_regime')
            self.crypto_context = analyzer_state.get('crypto_context', {})