from typing import Dict, Type, Optional
from dataclasses import dataclass
from strategies.base_strategy import BaseStrategy
from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig
from strategies.momentum_strategy import MomentumStrategy, MomentumConfig
from strategies.fundamental_strategy import FundamentalStrategy
from strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from strategies.price_breakout import BreakoutStrategy, BreakoutConfig
from strategies.dynamic_trendline import DynamicTrendlineStrategy, DynamicTrendlineConfig
from strategies.range_trading import RangeTradingStrategy, RangeTradingConfig
from strategies.volatility_enhanced_strategy import VolatilityEnhancedStrategy, VolatilityEnhancedConfig
from strategies.volatility_breakout import VolatilityBreakoutStrategy, VolatilityBreakoutConfig
from strategies.momentum_reversal_strategy import MomentumReversalStrategy, MomentumReversalConfig
from strategies.volatility_clustering_strategy import VolatilityClusteringStrategy, VolatilityClusteringConfig
from strategies.time_calender import TimeCalendarStrategy, TimeCalendarConfig
from strategies.order_flow_strategy import OrderFlowStrategy, OrderFlowConfig
from strategies.time_series import TimeSeriesStrategy, TimeSeriesConfig
from strategies.stochastic_process import StochasticProcessStrategy, StochasticProcessConfig
from strategies.bayesian_inference_strategy import BayesianInferenceStrategy,BayesianInferenceConfig
from strategies.markov_chain_strategy import MarkovChainStrategy, MarkovChainConfig
from strategies.trend_following import TrendFollowingConfig,TrendFollowingStrategy
from strategies.xgboost_mean_reversion import XGBoostMeanReversionStrategy, XGBoostMeanReversionConfig
from strategies.lstm_swing_mean_reversion import LSTMSwingMeanReversionStrategy, LSTMSwingMeanReversionConfig
from strategies.advanced_time_series import AdvancedTimeSeriesStrategy, AdvancedTimeSeriesConfig

@dataclass
class StrategyConfig:
    """Configuration for strategy creation"""
    strategy_type: str
    parameters: dict = None
    volatility_enhanced: bool = False
    volatility_parameters: dict = None
    
class StrategyFactory:
    """Factory for creating trading strategies"""
    
    _strategies: Dict[str, Type[BaseStrategy]] = {
        'mean_reversion': MeanReversionStrategy,
        'breakout': BreakoutStrategy,  
        "technical": TechnicalStrategy,
        'momentum': MomentumStrategy,
        'dynamic_trendline': DynamicTrendlineStrategy,
        'range_trading': RangeTradingStrategy,
        'volatility_breakout': VolatilityBreakoutStrategy,
        'momentum_reversal': MomentumReversalStrategy,
        'volatility_clustering': VolatilityClusteringStrategy,
        'time_calendar': TimeCalendarStrategy,
        'order_flow': OrderFlowStrategy,
        'time_series': TimeSeriesStrategy,
        'stochastic_process': StochasticProcessStrategy,
        'bayesian_inference': BayesianInferenceStrategy,
        'markov_chain': MarkovChainStrategy,
        'trend_following': TrendFollowingStrategy,
        'xgboost_mean_reversion': XGBoostMeanReversionStrategy,
        'lstm_swing_mean_reversion': LSTMSwingMeanReversionStrategy,
        'advanced_time_series': AdvancedTimeSeriesStrategy
    }

    # Update the _configs dictionary
    _configs: Dict[str, Type] = {
        'mean_reversion': MeanReversionConfig,
        'breakout': BreakoutConfig,    
        'technical': TechnicalConfig,
        'momentum': MomentumConfig,
        'dynamic_trendline': DynamicTrendlineConfig,
        'range_trading': RangeTradingConfig,
        'volatility_breakout': VolatilityBreakoutConfig,
        'momentum_reversal': MomentumReversalConfig,
        'volatility_clustering': VolatilityClusteringConfig,
        'time_calendar': TimeCalendarConfig,
        'order_flow': OrderFlowConfig,
        'time_series': TimeSeriesConfig,
        'stochastic_process': StochasticProcessConfig,
        'bayesian_inference': BayesianInferenceConfig,
        'markov_chain': MarkovChainConfig,
        'trend_following': TrendFollowingConfig,
        'xgboost_mean_reversion': XGBoostMeanReversionConfig,
        'lstm_swing_mean_reversion': LSTMSwingMeanReversionConfig,
        'advanced_time_series': AdvancedTimeSeriesConfig
    }
    
    @classmethod
    def register_strategy(
        cls, 
        name: str, 
        strategy_class: Type[BaseStrategy],
        config_class: Type = None
    ) -> None:
        """
        Register a new strategy type
        
        Args:
            name: Strategy identifier
            strategy_class: Strategy class to register
            config_class: Configuration class for the strategy
        """
        cls._strategies[name] = strategy_class
        if config_class:
            cls._configs[name] = config_class
    
    @classmethod
    def get_available_strategies(cls) -> list:
        """
        Get list of available strategy types
        
        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())

    @classmethod
    def create_strategy(cls, config: StrategyConfig) -> BaseStrategy:
        """
        Create a standard strategy instance
        
        Args:
            config: Strategy configuration
            
        Returns:
            Configured strategy instance
        """
        if config.strategy_type not in cls._strategies:
            raise ValueError(
                f"Unknown strategy type: {config.strategy_type}. "
                f"Available strategies: {list(cls._strategies.keys())}"
            )
        
        # Get strategy class and config class
        strategy_class = cls._strategies[config.strategy_type]
        config_class = cls._configs.get(config.strategy_type)
        
        # Create strategy config if parameters provided
        strategy_config = None
        if config.parameters and config_class:
            strategy_config = config_class(**config.parameters)
        
        # Create and return strategy instance
        return strategy_class(config=strategy_config)

    @classmethod
    def create_enhanced_strategy(
        cls, 
        base_strategy_config: StrategyConfig,
        volatility_config: dict = None
    ) -> BaseStrategy:
        """
        Create a volatility-enhanced strategy
        
        Args:
            base_strategy_config: Configuration for the base strategy
            volatility_config: Parameters for volatility enhancement
            
        Returns:
            Volatility-enhanced strategy instance
        """
        # Create base strategy
        base_strategy = cls.create_strategy(base_strategy_config)
        
        # Create volatility enhancement config
        vol_enhancement_config = None
        if volatility_config:
            vol_enhancement_config = VolatilityEnhancedConfig(**volatility_config)
        
        # Create and return enhanced strategy
        return VolatilityEnhancedStrategy(
            base_strategy=base_strategy,
            config=vol_enhancement_config
        )
    
# Example usage:
if __name__ == "__main__":
    # Create a mean reversion strategy with custom config
    config = StrategyConfig(
        strategy_type='mean_reversion',
        parameters={
            'zscore_entry_threshold': 2.5,
            'day_rsi_oversold': 25,
            'day_rsi_overbought': 75
        }
    )
    
    strategy = StrategyFactory.create_strategy(config)
    
    # Print available strategies
    print(f"Available strategies: {StrategyFactory.get_available_strategies()}")