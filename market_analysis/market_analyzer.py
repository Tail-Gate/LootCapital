import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import joblib

from market_analysis.market_regime import MarketRegimeDetector

class MarketAnalyzer:
    """
    Analyzes market conditions and provides context to trading strategies.
    Combines traditional indicators with machine learning for natural gas markets.
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
        
        # Natural gas specific seasonal factors
        self.is_heating_season = False
        self.is_injection_season = False
        self.is_report_day = False
    
    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """
        Analyze market conditions for natural gas futures
        
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
        
        # Check for natural gas specific seasonal factors
        current_date = data.index[-1]
        
        # Heating season: November through March
        self.is_heating_season = current_date.month in [11, 12, 1, 2, 3]
        
        # Injection season: April through October
        self.is_injection_season = current_date.month in [4, 5, 6, 7, 8, 9, 10]
        
        # EIA Natural Gas Storage Report day (Thursday)
        self.is_report_day = current_date.dayofweek == 3
        
        # Detect volatility conditions
        volatility = data['returns'].rolling(window=20).std().iloc[-1]
        volatility_percentile = self._calculate_percentile(volatility, data['returns'].rolling(window=20).std())
        
        # Build result
        result = {
            'regime': self.current_regime,
            'regime_name': regime_name,
            'regime_stability': regime_stability,
            'confidence': min(regime_stability * 1.5, 1.0),
            'heating_season': self.is_heating_season,
            'injection_season': self.is_injection_season,
            'report_day': self.is_report_day,
            'volatility': volatility,
            'volatility_percentile': volatility_percentile
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
        Get strategy adjustments based on market regime and natural gas specific factors
        
        Args:
            regime_info: Market regime information from analyze_market
            
        Returns:
            Dictionary with strategy adjustments
        """
        regime_name = regime_info['regime_name']
        confidence = regime_info['confidence']
        volatility_percentile = regime_info.get('volatility_percentile', 50)
        
        # Default adjustments
        adjustments = {
            'position_size_factor': 1.0,
            'stop_loss_factor': 1.0,
            'take_profit_factor': 1.0,
            'favored_strategies': [],
            'disfavored_strategies': []
        }
        
        # Adjust based on regime
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
        
        # Natural gas specific adjustments
        
        # Heating season adjustments (higher volatility, more trending)
        if self.is_heating_season:
            # Favor momentum strategies during heating season
            if 'momentum' not in adjustments['favored_strategies']:
                adjustments['favored_strategies'].append('momentum')
            
            # Increase stop loss for higher volatility
            adjustments['stop_loss_factor'] *= 1.2
            
            # Be more cautious with position sizing
            adjustments['position_size_factor'] *= 0.9
        
        # Injection season adjustments (more range-bound)
        if self.is_injection_season:
            # Favor range trading and mean reversion during injection season
            if 'range_trading' not in adjustments['favored_strategies']:
                adjustments['favored_strategies'].append('range_trading')
                
            # Tighter stop losses in more predictable market
            adjustments['stop_loss_factor'] *= 0.9
        
        # EIA report day adjustments (high volatility)
        if self.is_report_day:
            # Reduce position size on report days
            adjustments['position_size_factor'] *= 0.7
            
            # Widen stop losses for report day volatility
            adjustments['stop_loss_factor'] *= 1.3
            
            # Favor volatility strategies on report days
            if 'volatility_clustering' not in adjustments['favored_strategies']:
                adjustments['favored_strategies'].append('volatility_clustering')
                
            # Avoid range strategies on report days
            if 'range_trading' not in adjustments['disfavored_strategies']:
                adjustments['disfavored_strategies'].append('range_trading')
        
        # Volatility-based adjustments
        if volatility_percentile > 80:
            # Very high volatility - reduce size further
            adjustments['position_size_factor'] *= 0.8
            adjustments['stop_loss_factor'] *= 1.2
        elif volatility_percentile < 20:
            # Very low volatility - can increase size
            adjustments['position_size_factor'] *= 1.2
            adjustments['stop_loss_factor'] *= 0.8
        
        # Scale adjustments by confidence
        for key in ['position_size_factor', 'stop_loss_factor', 'take_profit_factor']:
            # Scale adjustment toward 1.0 based on confidence
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
            'is_heating_season': self.is_heating_season,
            'is_injection_season': self.is_injection_season,
            'is_report_day': self.is_report_day
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
            self.is_heating_season = analyzer_state.get('is_heating_season', False)
            self.is_injection_season = analyzer_state.get('is_injection_season', False)
            self.is_report_day = analyzer_state.get('is_report_day', False)