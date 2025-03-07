from typing import Dict, Type, Optional
from dataclasses import dataclass
from strategies.momentum_strategy import MomentumStrategy, MomentumConfig
from strategies.base_strategy import BaseStrategy
from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig
from strategies.fundamental_strategy import FundamentalStrategy
from strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from strategies.price_breakout import BreakoutStrategy,BreakoutConfig
from strategies.dynamic_trendline import DynamicTrendlineStrategy,DynamicTrendlineConfig
from strategies.range_trading import RangeTradingStrategy,RangeTradingConfig

@dataclass
class StrategyConfig:
    """Configuration for strategy creation"""
    strategy_type: str
    parameters: dict = None
    
class StrategyFactory:
    """Factory for creating trading strategies"""
    
    _strategies: Dict[str, Type[BaseStrategy]] = {
        'mean_reversion': MeanReversionStrategy,
        'breakout': BreakoutStrategy,  
        "technical":TechnicalStrategy,
        'momentum': MomentumStrategy,
        'dynamic_trendline': DynamicTrendlineStrategy,
        'range_trading': RangeTradingStrategy
    }
    
    _configs: Dict[str, Type] = {
        'mean_reversion': MeanReversionConfig,
        'breakout': BreakoutConfig,    
        'technical': TechnicalConfig,
        'momentum': MomentumConfig,
        'dynamic_trendline':DynamicTrendlineConfig,
        'range_trading': RangeTradingConfig
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
    def create_strategy(
        cls, 
        config: StrategyConfig
    ) -> BaseStrategy:
        """
        Create a strategy instance
        
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
    def get_available_strategies(cls) -> list:
        """Get list of registered strategies"""
        return list(cls._strategies.keys())

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