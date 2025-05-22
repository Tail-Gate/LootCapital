import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import joblib
import logging
from datetime import datetime

from strategies.strategy_factory import StrategyFactory, StrategyConfig
from market_analysis.market_analyzer import MarketAnalyzer
from market_analysis.signal_enhancer import CryptoSignalEnhancer
from risk_management.parameter_optimizer import DQNParameterOptimizer

class TradingSystem:
    """
    Main trading system that coordinates strategies, data, execution, and risk management.
    Integrates AI components for market analysis and optimization.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize the trading system
        
        Args:
            config_path: Path to load configuration from
            model_path: Path to load AI models from
        """
        # Set up logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.strategies = {}
        self.market_analyzer = MarketAnalyzer(model_path=model_path)
        self.signal_enhancer = None
        self.parameter_optimizer = None
        
        # Trading state
        self.active_positions = {}
        self.trade_history = []
        self.performance_metrics = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Set up AI components if enabled
        if self.config.get('use_ai', False):
            self._setup_ai_components(model_path)
        
        # Load strategies
        self._load_strategies()
    
    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TradingSystem')
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration from file or use defaults
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'use_ai': True,
            'strategies': [
                {
                    'name': 'volatility_clustering',
                    'parameters': {},
                    'weight': 1.0
                },
                {
                    'name': 'mean_reversion',
                    'parameters': {},
                    'weight': 1.0
                }
            ],
            'risk_management': {
                'max_position_size': 100000,  # $100k max position
                'max_risk_per_trade': 0.01,   # 1% risk per trade
                'max_positions': 5,           # Max 5 concurrent positions
                'portfolio_heat': 0.3         # 30% max portfolio heat
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                config = joblib.load(config_path)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                self.logger.error(f"Error loading configuration: {e}")
                return default_config
        else:
            self.logger.info("Using default configuration")
            return default_config
    
    def _setup_ai_components(self, model_path: Optional[str]):
        """
        Set up AI components
        
        Args:
            model_path: Path to load models from
        """
        # Set up signal enhancer for crypto (Ethereum-focused, asset-agnostic)
        enhancer_path = os.path.join(model_path, 'signal_enhancer') if model_path else None
        self.signal_enhancer = CryptoSignalEnhancer(model_path=enhancer_path)
        
        # Set up parameter optimizer
        # For crypto, define asset-agnostic parameter space
        parameter_space = {
            'zscore_entry_threshold': [1.5, 2.0, 2.5, 3.0],
            'volatility_window': [10, 20, 30],
            'atr_multiplier': [1.0, 1.5, 2.0, 2.5],
            'volume_surge_threshold': [1.5, 2.0, 2.5]
        }
        
        optimizer_path = os.path.join(model_path, 'parameter_optimizer') if model_path else None
        self.parameter_optimizer = DQNParameterOptimizer(
            parameter_space=parameter_space,
            model_path=optimizer_path
        )
        
        self.logger.info("AI components initialized")
    
    def _load_strategies(self):
        """Load trading strategies from configuration"""
        for strategy_config in self.config['strategies']:
            name = strategy_config['name']
            parameters = strategy_config.get('parameters', {})
            weight = strategy_config.get('weight', 1.0)
            
            # Create strategy configuration
            config = StrategyConfig(
                strategy_type=name,
                parameters=parameters
            )
            
            # Create strategy
            try:
                strategy = StrategyFactory.create_strategy(config)
                
                # Add signal enhancer if available
                if self.signal_enhancer:
                    strategy.set_signal_enhancer(self.signal_enhancer)
                
                # Store strategy with weight
                self.strategies[name] = {
                    'strategy': strategy,
                    'weight': weight
                }
                self.logger.info(f"Loaded strategy: {name}")
            except Exception as e:
                self.logger.error(f"Error loading strategy {name}: {e}")
    
    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """
        Analyze market conditions
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Market analysis results
        """
        # Call market analyzer to detect regime
        regime_info = self.market_analyzer.analyze_market(data)
        
        # Get strategy adjustments
        adjustments = self.market_analyzer.get_strategy_adjustments(regime_info)
        
        # Log regime detection
        self.logger.info(f"Detected market regime: {regime_info['regime_name']} "
                        f"with confidence {regime_info['confidence']:.2f}")
        
        return {**regime_info, **adjustments}
    
    def train_ai_components(self, data: pd.DataFrame):
        """
        Train AI components with historical data
        
        Args:
            data: Historical market data
        """
        self.logger.info("Training AI components...")
        
        # Train market regime detector
        self.logger.info("Training market regime detector...")
        self.market_analyzer.train_regime_detector(data)
        
        # Train signal enhancer
        if self.signal_enhancer:
            self.logger.info("Training signal enhancer...")
            self.signal_enhancer.train(data)
        
        self.logger.info("AI training complete")
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """
        Generate trading signals from all strategies
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of signals from each strategy
        """
        signals = {}
        
        # Analyze market
        market_context = self.analyze_market(data)
        
        # Get signals from each strategy
        for name, strategy_info in self.strategies.items():
            strategy = strategy_info['strategy']
            weight = strategy_info['weight']
            
            try:
                # Prepare features
                features = strategy.prepare_features(data)
                
                # Calculate signals
                signal, confidence, trade_type = strategy.calculate_signals(features)
                
                # Adjust signals with market context
                signal, confidence, trade_type = strategy.adjust_with_market_context(
                    signal, confidence, trade_type, market_context
                )
                
                # Enhance signals with AI if enabled
                if self.signal_enhancer and strategy.use_ml:
                    signal, confidence = strategy.enhance_signal_with_ml(
                        signal, confidence, data
                    )
                
                # Apply strategy weight
                weighted_signal = signal * weight
                
                # Validate signal
                is_valid = strategy.validate_signal(signal, features)
                
                signals[name] = {
                    'signal': signal,
                    'weighted_signal': weighted_signal,
                    'confidence': confidence,
                    'trade_type': trade_type,
                    'valid': is_valid
                }
                
                self.logger.info(f"Strategy {name} signal: {signal:.2f}, "
                                f"confidence: {confidence:.2f}, valid: {is_valid}")
                
            except Exception as e:
                self.logger.error(f"Error generating signals for {name}: {e}")
                signals[name] = {
                    'signal': 0,
                    'weighted_signal': 0,
                    'confidence': 0,
                    'trade_type': None,
                    'valid': False
                }
        
        return signals
    
    def combine_signals(self, signals: Dict) -> Dict:
        """
        Combine signals from multiple strategies
        
        Args:
            signals: Dictionary of signals from each strategy
            
        Returns:
            Combined signal information
        """
        valid_signals = [
            s for name, s in signals.items() 
            if s['valid'] and s['confidence'] > 0.5
        ]
        
        if not valid_signals:
            return {
                'signal': 0,
                'confidence': 0,
                'trade_type': None,
                'valid': False
            }
        
        # Calculate weighted average of signals
        total_weight = sum(s['confidence'] for s in valid_signals)
        weighted_signal = sum(s['weighted_signal'] * s['confidence'] for s in valid_signals) / total_weight
        
        # Determine trade type (use most confident signal's trade type)
        most_confident = max(valid_signals, key=lambda s: s['confidence'])
        trade_type = most_confident['trade_type']
        
        # Calculate average confidence
        avg_confidence = sum(s['confidence'] for s in valid_signals) / len(valid_signals)
        
        return {
            'signal': weighted_signal,
            'confidence': avg_confidence,
            'trade_type': trade_type,
            'valid': True
        }
    
    def execute_trade(
        self, 
        data: pd.DataFrame, 
        signal_info: Dict
    ) -> Dict:
        """
        Execute trade based on signal
        
        Args:
            data: Market data DataFrame
            signal_info: Combined signal information
            
        Returns:
            Trade information dictionary
        """
        # Check if signal is valid
        if not signal_info['valid'] or abs(signal_info['signal']) < 0.2:
            return None
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Use parameter optimizer to get optimal parameters for this market regime
        if self.parameter_optimizer:
            optimal_params = self.parameter_optimizer.select_parameters(data)
            
            # Log optimized parameters
            self.logger.info(f"Using optimized parameters: {optimal_params}")
        else:
            optimal_params = {}
        
        # Calculate position size based on risk management
        position_size = self._calculate_position_size(
            data, signal_info, optimal_params
        )
        
        # Calculate stop loss and take profit levels
        # For simplicity, using the most confident strategy's method
        most_confident_strategy = max(
            self.strategies.items(),
            key=lambda s: signals.get(s[0], {}).get('confidence', 0)
        )[1]['strategy']
        
        features = most_confident_strategy.prepare_features(data)
        
        stop_loss = most_confident_strategy.calculate_stop_loss(
            current_price, signal_info['signal'], features
        )
        
        take_profit = most_confident_strategy.calculate_take_profit(
            current_price, stop_loss, signal_info['signal'], features
        )
        
        # Create trade information
        trade_info = {
            'entry_time': data.index[-1],
            'entry_price': current_price,
            'signal': signal_info['signal'],
            'confidence': signal_info['confidence'],
            'trade_type': signal_info['trade_type'],
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': 'open',
            'exit_time': None,
            'exit_price': None,
            'pnl': 0,
            'params': optimal_params
        }
        
        # Log trade execution
        direction = "LONG" if signal_info['signal'] > 0 else "SHORT"
        self.logger.info(f"Executing {direction} trade: Entry: ${current_price:.2f}, "
                        f"Size: {position_size}, Stop: ${stop_loss:.2f}, "
                        f"Target: ${take_profit:.2f}")
        
        # Add to active positions
        trade_id = f"trade_{len(self.trade_history) + 1}"
        self.active_positions[trade_id] = trade_info
        
        return trade_info
    
    def _calculate_position_size(
        self, 
        data: pd.DataFrame, 
        signal_info: Dict,
        optimal_params: Dict
    ) -> float:
        """
        Calculate position size based on risk management rules for crypto (Ethereum-focused, asset-agnostic)
        
        Args:
            data: Market data DataFrame
            signal_info: Signal information
            optimal_params: Optimized parameters
        
        Returns:
            Position size (in units/coins/contracts)
        """
        current_price = data['close'].iloc[-1]
        
        # Calculate volatility (ATR or similar)
        if 'atr' in data.columns:
            volatility = data['atr'].iloc[-1]
        else:
            # Simple volatility measure if ATR not available
            volatility = data['close'].pct_change().rolling(window=20).std().iloc[-1] * current_price
        
        # Get risk parameters
        max_position_size = self.config['risk_management']['max_position_size']
        max_risk_per_trade = self.config['risk_management']['max_risk_per_trade']
        
        # Calculate stop distance
        if 'atr_multiplier' in optimal_params:
            stop_distance = volatility * optimal_params['atr_multiplier']
        else:
            stop_distance = volatility * 2  # Default 2 ATR stop
        
        # Calculate position size based on fixed risk
        account_size = 1000000  # Example account size
        risk_amount = account_size * max_risk_per_trade
        position_size = risk_amount / stop_distance
        
        # Adjust position size based on confidence
        position_size = position_size * signal_info['confidence']
        
        # Apply market regime adjustment
        market_context = self.analyze_market(data)
        if 'position_size_factor' in market_context:
            position_size = position_size * market_context['position_size_factor']
        
        # Volatility adjustment (generic)
        vol_ratio = data['returns'].rolling(window=5).std().iloc[-1] / \
                    data['returns'].rolling(window=20).std().iloc[-1]
        if vol_ratio > 1.2:  # Higher short-term volatility
            position_size = position_size * 0.8
        
        # Ensure we don't exceed maximum position size
        position_size = min(position_size, max_position_size)
        
        # TODO: Implement crypto/Ethereum-specific lot size, leverage, and exchange rules here
        # For now, return the calculated position size in units/coins
        return position_size
    
    def update_positions(self, data: pd.DataFrame):
        """
        Update and manage active positions
        
        Args:
            data: Latest market data
        """
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]
        
        for trade_id, trade in list(self.active_positions.items()):
            # Check for stop loss or take profit
            is_long = trade['signal'] > 0
            
            # Check stop loss
            stop_triggered = (is_long and current_price <= trade['stop_loss']) or \
                             (not is_long and current_price >= trade['stop_loss'])
            
            # Check take profit
            take_profit_triggered = (is_long and current_price >= trade['take_profit']) or \
                                   (not is_long and current_price <= trade['take_profit'])
            
            # Exit if either is triggered
            if stop_triggered or take_profit_triggered:
                # Calculate P&L
                if is_long:
                    pnl = (current_price - trade['entry_price']) * trade['position_size']
                else:
                    pnl = (trade['entry_price'] - current_price) * trade['position_size']
                
                # Update trade info
                trade['exit_time'] = current_time
                trade['exit_price'] = current_price
                trade['pnl'] = pnl
                trade['status'] = 'closed'
                
                # Log trade exit
                exit_reason = "STOP LOSS" if stop_triggered else "TAKE PROFIT"
                self.logger.info(f"Exiting trade {trade_id} due to {exit_reason}: "
                                f"Entry: ${trade['entry_price']:.2f}, "
                                f"Exit: ${current_price:.2f}, "
                                f"P&L: ${pnl:.2f}")
                
                # Move to trade history
                self.trade_history.append(trade)
                
                # Remove from active positions
                del self.active_positions[trade_id]
                
                # Update parameter optimizer if we're using it
                if self.parameter_optimizer:
                    # Calculate market volatility
                    market_volatility = data['returns'].rolling(window=20).std().iloc[-1]
                    
                    # Calculate maximum drawdown during trade
                    if is_long:
                        lowest_price = data['low'].loc[trade['entry_time']:current_time].min()
                        drawdown = (trade['entry_price'] - lowest_price) / trade['entry_price']
                    else:
                        highest_price = data['high'].loc[trade['entry_time']:current_time].max()
                        drawdown = (highest_price - trade['entry_price']) / trade['entry_price']
                    
                    # Update optimizer
                    self.parameter_optimizer.update_from_trade_result(
                        data, pnl, market_volatility, drawdown
                    )
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trade_history:
            return self.performance_metrics
        
        # Extract PnLs
        pnls = [trade['pnl'] for trade in self.trade_history]
        
        # Total P&L
        total_pnl = sum(pnls)
        
        # Win rate
        wins = sum(1 for pnl in pnls if pnl > 0)
        win_rate = wins / len(pnls) if pnls else 0
        
        # Calculate returns
        returns = pd.Series(pnls) / 1000000  # Assuming $1M account size
        
        # Sharpe ratio (annualized)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Update performance metrics
        self.performance_metrics = {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return self.performance_metrics
    
    def process_data(self, data: pd.DataFrame):
        """
        Process new market data
        
        Args:
            data: Market data DataFrame
        """
        # Update active positions
        self.update_positions(data)
        
        # Generate trading signals
        signals = self.generate_signals(data)
        
        # Combine signals
        combined_signal = self.combine_signals(signals)
        
        # Execute trade if applicable
        if combined_signal['valid']:
            self.execute_trade(data, combined_signal)
        
        # Update performance metrics
        self.calculate_performance_metrics()
    
    def save_state(self, path: str):
        """
        Save trading system state
        
        Args:
            path: Directory path to save state
        """
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        joblib.dump(self.config, os.path.join(path, 'config.joblib'))
        
        # Save strategies
        for name, strategy_info in self.strategies.items():
            strategy = strategy_info['strategy']
            strategy.save_state(os.path.join(path, f'strategy_{name}.joblib'))
        
        # Save AI components
        if self.market_analyzer:
            self.market_analyzer.save_models(os.path.join(path, 'market_analyzer'))
        
        if self.signal_enhancer:
            self.signal_enhancer.save_model(os.path.join(path, 'signal_enhancer'))
        
        if self.parameter_optimizer:
            self.parameter_optimizer.save_model(os.path.join(path, 'parameter_optimizer'))
        
        # Save trading state
        state = {
            'active_positions': self.active_positions,
            'trade_history': self.trade_history,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(state, os.path.join(path, 'trading_state.joblib'))
        
        self.logger.info(f"Trading system state saved to {path}")
    
    def load_state(self, path: str):
        """
        Load trading system state
        
        Args:
            path: Directory path to load state from
        """
        if not os.path.exists(path):
            self.logger.error(f"State path {path} does not exist")
            return
        
        try:
            # Load configuration
            config_path = os.path.join(path, 'config.joblib')
            if os.path.exists(config_path):
                self.config = joblib.load(config_path)
            
            # Load strategies
            self._load_strategies()
            for name, strategy_info in self.strategies.items():
                strategy = strategy_info['strategy']
                strategy_path = os.path.join(path, f'strategy_{name}.joblib')
                if os.path.exists(strategy_path):
                    strategy.load_state(strategy_path)
            
            # Load AI components
            if self.market_analyzer:
                self.market_analyzer.load_models(os.path.join(path, 'market_analyzer'))
            
            if self.signal_enhancer:
                self.signal_enhancer.load_model(os.path.join(path, 'signal_enhancer'))
            
            if self.parameter_optimizer:
                self.parameter_optimizer.load_model(os.path.join(path, 'parameter_optimizer'))
            
            # Load trading state
            state_path = os.path.join(path, 'trading_state.joblib')
            if os.path.exists(state_path):
                state = joblib.load(state_path)
                self.active_positions = state['active_positions']
                self.trade_history = state['trade_history']
                self.performance_metrics = state['performance_metrics']
            
            self.logger.info(f"Trading system state loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading trading system state: {e}")