from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import torch
import os
import joblib
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

class TradeType(Enum):
    """Types of trades"""
    NONE = "none"
    LONG = "long"
    SHORT = "short"
    DAY_TRADE = "day"
    SWING_TRADE = "swing"
    POSITION_TRADE = "position"

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, 
                 config: Dict,
                 market_data: MarketData,
                 technical_indicators: TechnicalIndicators):
        """
        Initialize the base strategy.
        
        Args:
            config: Strategy configuration dictionary
            market_data: MarketData instance for data access
            technical_indicators: TechnicalIndicators instance for indicator calculations
        """
        self.config = config
        self.market_data = market_data
        self.technical_indicators = technical_indicators
        self.signals: Dict[str, str] = {}
        self.name = ""
        self.model = None
        self.confidence_threshold = 0.6
        
        # ML components
        self.ml_models = {}
        self.use_ml = False
        
        # Signal enhancer
        self.signal_enhancer = None
    
    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for strategy analysis
        
        Args:
            data: Raw market data
            
        Returns:
            DataFrame with technical features
        """
        pass
    
    @abstractmethod
    def calculate_signals(self, features: pd.DataFrame) -> Tuple[float, float, TradeType]:
        """
        Calculate trading signals based on features
        
        Args:
            features: DataFrame with technical features
            
        Returns:
            Tuple of (signal_strength, confidence, trade_type)
            - signal_strength: -1.0 to 1.0 (short to long)
            - confidence: 0.0 to 1.0
            - trade_type: TradeType enum
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """
        Validate if a signal should be acted upon
        
        Args:
            signal: Signal strength (-1.0 to 1.0)
            features: DataFrame with technical features
            
        Returns:
            Boolean indicating if signal is valid
        """
        pass
    
    def get_required_features(self) -> List[str]:
        """
        Get list of required features for this strategy
        
        Returns:
            List of required feature names
        """
        return ['open', 'high', 'low', 'close']
    
    def adjust_with_market_context(
        self, 
        signal: float, 
        confidence: float, 
        trade_type: TradeType,
        market_context: Dict
    ) -> Tuple[float, float, TradeType]:
        """
        Adjust signals based on broader market context
        
        Args:
            signal: Signal strength (-1.0 to 1.0)
            confidence: Signal confidence (0.0 to 1.0)
            trade_type: TradeType enum
            market_context: Dictionary with market analysis results
            
        Returns:
            Adjusted (signal_strength, confidence, trade_type)
        """
        # Default implementation - can be overridden by specific strategies
        if 'favored_strategies' in market_context and self.name in market_context['favored_strategies']:
            # Boost confidence for favored strategies
            confidence = min(confidence * 1.2, 1.0)
        
        if 'disfavored_strategies' in market_context and self.name in market_context['disfavored_strategies']:
            # Reduce confidence for disfavored strategies
            confidence = confidence * 0.8
        
        return signal, confidence, trade_type
    
    def enhance_signal_with_ml(
        self, 
        signal: float, 
        confidence: float, 
        data: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Enhance signal using machine learning
        
        Args:
            signal: Original signal strength (-1.0 to 1.0)
            confidence: Original confidence (0.0 to 1.0)
            data: Raw market data
            
        Returns:
            Enhanced (signal_strength, confidence)
        """
        if not self.use_ml or self.signal_enhancer is None:
            return signal, confidence
        
        # Create Series from signal for enhancement
        signal_series = pd.Series(signal, index=[data.index[-1]])
        
        # Enhance signal
        enhanced_signal = self.signal_enhancer.enhance_signals(data, signal_series)
        
        # Adjust confidence based on agreement
        signal_agreement = np.sign(signal) == np.sign(enhanced_signal.iloc[0])
        if signal_agreement:
            # Signals agree - boost confidence
            enhanced_confidence = min(confidence * 1.2, 1.0)
        else:
            # Signals disagree - reduce confidence
            enhanced_confidence = confidence * 0.8
        
        return enhanced_signal.iloc[0], enhanced_confidence
    
    def calculate_stop_loss(self, entry_price: float, signal: float, features: pd.DataFrame) -> float:
        """
        Calculate dynamic stop loss level
        
        Args:
            entry_price: Entry price
            signal: Signal strength (-1.0 to 1.0)
            features: DataFrame with technical features
            
        Returns:
            Stop loss price
        """
        # Default implementation - should be overridden by specific strategies
        atr = features['atr'].iloc[-1] if 'atr' in features.columns else 0.01 * entry_price
        
        if signal > 0:  # Long position
            return entry_price - atr * 2
        else:  # Short position
            return entry_price + atr * 2
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, signal: float, features: pd.DataFrame) -> float:
        """
        Calculate take profit level
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            signal: Signal strength (-1.0 to 1.0)
            features: DataFrame with technical features
            
        Returns:
            Take profit price
        """
        # Default implementation - reward:risk ratio of 2:1
        risk = abs(entry_price - stop_loss)
        
        if signal > 0:  # Long position
            return entry_price + risk * 2
        else:  # Short position
            return entry_price - risk * 2
    
    def set_signal_enhancer(self, enhancer) -> None:
        """
        Set the signal enhancer for this strategy
        
        Args:
            enhancer: Signal enhancer instance
        """
        self.signal_enhancer = enhancer
        self.use_ml = True
    
    def add_ml_model(self, model_name: str, model) -> None:
        """
        Add a machine learning model to the strategy
        
        Args:
            model_name: Name identifier for the model
            model: Trained ML model 
        """
        self.ml_models[model_name] = model
        self.use_ml = True
    
    def save_state(self, path: str) -> None:
        """
        Save strategy state and models
        
        Args:
            path: Path to save state
        """
        # Base implementation - should be extended by strategies with more state
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'name': self.name,
            'confidence_threshold': self.confidence_threshold,
            'use_ml': self.use_ml
        }
        
        # Save ML models if they exist
        if self.ml_models:
            for model_name, model in self.ml_models.items():
                if model is not None:
                    if hasattr(model, 'state_dict'):  # PyTorch model
                        torch.save(model.state_dict(), f"{path}_{model_name}.pt")
                    else:  # Other model type
                        joblib.dump(model, f"{path}_{model_name}.joblib")
                    state[f'{model_name}_path'] = f"{path}_{model_name}"
        
        # Save signal enhancer if it exists
        if self.signal_enhancer is not None:
            self.signal_enhancer.save_model(f"{path}_signal_enhancer")
            state['signal_enhancer_path'] = f"{path}_signal_enhancer"
        
        # Save main state dictionary
        joblib.dump(state, path)
    
    def load_state(self, path: str) -> None:
        """
        Load strategy state and models
        
        Args:
            path: Path to load state from
        """
        # Base implementation - should be extended by strategies with more state
        if not os.path.exists(path):
            return
            
        state = joblib.load(path)
        
        self.name = state.get('name', self.name)
        self.confidence_threshold = state.get('confidence_threshold', self.confidence_threshold)
        self.use_ml = state.get('use_ml', self.use_ml)
        
        # Load ML models if paths exist
        self.ml_models = {}
        for key, value in state.items():
            if key.endswith('_path') and not key == 'signal_enhancer_path':
                model_name = key[:-5]  # Remove '_path' suffix
                
                # Check for PyTorch model
                if os.path.exists(f"{value}.pt"):
                    # Load PyTorch model - would need to know architecture
                    pass
                # Check for joblib model
                elif os.path.exists(f"{value}.joblib"):
                    self.ml_models[model_name] = joblib.load(f"{value}.joblib")
        
        # Load signal enhancer if path exists
        if 'signal_enhancer_path' in state and os.path.exists(state['signal_enhancer_path']):
            from market_analysis.signal_enhancer import NaturalGasSignalEnhancer
            self.signal_enhancer = NaturalGasSignalEnhancer()
            self.signal_enhancer.load_model(state['signal_enhancer_path'])

    def update(self) -> None:
        """
        Update the strategy with new market data.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement update()")
        
    def get_signals(self) -> Dict[str, str]:
        """
        Get the current trading signals.
        
        Returns:
            Dictionary mapping asset symbols to trading signals
        """
        return self.signals