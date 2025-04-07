from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from enum import Enum

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class MarkovChainConfig(TechnicalConfig):
    """Configuration for Markov Chain based strategy"""
    # Discretization parameters
    swing_up_threshold: float = 0.015  # 1.5% for swing bullish state
    swing_down_threshold: float = -0.015  # -1.5% for swing bearish state
    swing_neutral_threshold: float = 0.005  # 0.5% for swing neutral state
    
    day_up_threshold: float = 0.005  # 0.5% for day trading bullish state
    day_down_threshold: float = -0.005  # -0.5% for day trading bearish state
    day_neutral_threshold: float = 0.002  # 0.2% for day trading neutral state
    
    # Transition matrix parameters
    swing_lookback_periods: int = 60  # Number of periods to use for swing trading TPM
    day_lookback_periods: int = 30  # Number of periods to use for day trading TPM
    min_transitions: int = 20  # Minimum number of transitions to calculate reliable TPM
    
    # Volatility state parameters
    high_vol_multiplier: float = 2.0  # Current ATR > 2x average indicates high volatility
    low_vol_multiplier: float = 0.5  # Current ATR < 0.5x average indicates low volatility
    atr_lookback: int = 10  # Lookback period for ATR average
    
    # Momentum state parameters
    swing_rsi_period: int = 14  # RSI period for swing trading
    day_rsi_period: int = 14   # RSI period for day trading
    rsi_overbought: float = 70  # RSI overbought threshold
    rsi_oversold: float = 30  # RSI oversold threshold
    
    # Volume and Open Interest parameters
    volume_surge_threshold: float = 2.0  # Volume > 2x average indicates confirmation
    volume_lookback: int = 20  # Lookback period for volume average
    
    # Time parameters
    morning_session_start: int = 9  # Morning session start hour (9 AM)
    morning_session_end: int = 11  # Morning session end hour (11 AM)
    close_session_start: int = 15  # Close session start hour (3 PM)
    
    # Seasonal parameters
    winter_months: List[int] = field(default_factory=lambda: [11, 12, 1, 2, 3])  # Winter months (bullish bias)
    summer_months: List[int] = field(default_factory=lambda: [6, 7, 8])  # Summer months (neutral)
    shoulder_months: List[int] = field(default_factory=lambda: [4, 5, 9, 10])  # Shoulder months (bearish bias)
    
    # Risk management parameters
    day_stop_loss_atr: float = 1.5  # Day trading stop loss in ATR units
    day_take_profit_atr: float = 3.0  # Day trading take profit in ATR units
    swing_stop_loss_atr: float = 2.0  # Swing trading stop loss in ATR units
    swing_take_profit_atr: float = 4.0  # Swing trading take profit in ATR units
    
    # Hidden Markov Model parameters (if extended to HMM)
    use_hmm: bool = False  # Whether to use HMM instead of simple Markov Chain
    n_hidden_states: int = 2  # Number of hidden states in HMM
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'day': {
                    'markov_prediction': 0.35,
                    'rsi_state': 0.20,
                    'volatility_state': 0.15,
                    'volume_confirm': 0.15,
                    'time_of_day': 0.15
                },
                'swing': {
                    'markov_prediction': 0.30,
                    'rsi_state': 0.20,
                    'volatility_state': 0.15,
                    'volume_confirm': 0.15,
                    'seasonal_bias': 0.20
                }
            }


class PriceState(Enum):
    """Price movement states for Markov Chain"""
    BULLISH = 1
    BEARISH = 2
    NEUTRAL = 3


class VolatilityState(Enum):
    """Volatility states for adaptation"""
    HIGH = 1
    NORMAL = 2
    LOW = 3


class MarkovChainStrategy(TechnicalStrategy):
    """
    Trading strategy based on Markov Chains for predicting future price states.
    
    This strategy:
    1. Discretizes price movements into states (bullish, bearish, neutral)
    2. Calculates transition probabilities between states
    3. Predicts the most likely next state
    4. Generates trading signals based on state predictions and confirmations
    """
    
    def __init__(self, config: MarkovChainConfig = None):
        super().__init__(name="markov_chain", config=config or MarkovChainConfig())
        self.config: MarkovChainConfig = self.config
        
        # Initialize transition matrices for different timeframes
        self.swing_tpm = None  # Transition probability matrix for swing trading
        self.day_tpm = None    # Transition probability matrix for day trading
        
        # Initialize current state tracking
        self.current_state = None
        self.current_volatility_state = VolatilityState.NORMAL
        
        # Initialize state history
        self.state_history = []
        self.next_state_prediction = None
        
        # HMM components (if used)
        self.hmm_model = None
        self.hidden_state = None
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for Markov Chain analysis"""
        df = self.prepare_base_features(data)
        
        # Determine if we're working with intraday data
        is_intraday = self.is_intraday_data(df)
        
        # Add discretized price state features
        df = self._add_price_state_features(df, is_intraday)
        
        # Add volatility state features
        df = self._add_volatility_state_features(df)
        
        # Add momentum state features
        df = self._add_momentum_state_features(df, is_intraday)
        
        # Add volume confirmation features
        df = self._add_volume_features(df)
        
        # Add time and seasonal features
        df = self._add_time_features(df)
        
        # Calculate transition matrices
        self._calculate_transition_matrices(df, is_intraday)
        
        # Calculate state predictions
        df = self._add_markov_predictions(df, is_intraday)
        
        return df
    
    def _add_price_state_features(self, df: pd.DataFrame, is_intraday: bool) -> pd.DataFrame:
        """Add discretized price state features based on returns"""
        # Calculate returns if not already present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            
        # Use appropriate thresholds based on timeframe
        if is_intraday:
            up_threshold = self.config.day_up_threshold
            down_threshold = self.config.day_down_threshold
            neutral_threshold = self.config.day_neutral_threshold
        else:
            up_threshold = self.config.swing_up_threshold
            down_threshold = self.config.swing_down_threshold
            neutral_threshold = self.config.swing_neutral_threshold
            
        # Determine price states
        df['price_state'] = None
        
        # Bullish state
        df.loc[df['returns'] >= up_threshold, 'price_state'] = PriceState.BULLISH.value
        
        # Bearish state
        df.loc[df['returns'] <= down_threshold, 'price_state'] = PriceState.BEARISH.value
        
        # Neutral state (between neutral thresholds)
        df.loc[(df['returns'] > down_threshold) & 
               (df['returns'] < neutral_threshold), 'price_state'] = PriceState.NEUTRAL.value
        
        # Fill remaining as neutral
        df['price_state'] = df['price_state'].fillna(PriceState.NEUTRAL.value)
        
        # Store the current state
        if len(df) > 0:
            self.current_state = int(df['price_state'].iloc[-1])
            
        # Store state history
        self.state_history = df['price_state'].tolist()
        
        return df
    
    def _add_volatility_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility state features"""
        if 'atr' not in df.columns:
            # Calculate ATR if not already present
            df['atr'] = self.ti.calculate_atr(
                df['high'], df['low'], df['close'], 
                period=self.config.atr_lookback
            )
            
        # Calculate average ATR
        df['atr_avg'] = df['atr'].rolling(window=self.config.atr_lookback).mean()
        
        # Calculate ATR ratio
        df['atr_ratio'] = df['atr'] / df['atr_avg']
        
        # Determine volatility states
        df['volatility_state'] = VolatilityState.NORMAL.value
        
        # High volatility state
        df.loc[df['atr_ratio'] > self.config.high_vol_multiplier, 'volatility_state'] = VolatilityState.HIGH.value
        
        # Low volatility state
        df.loc[df['atr_ratio'] < self.config.low_vol_multiplier, 'volatility_state'] = VolatilityState.LOW.value
        
        # Store current volatility state
        if len(df) > 0:
            self.current_volatility_state = VolatilityState(df['volatility_state'].iloc[-1])
            
        return df
    
    def _add_momentum_state_features(self, df: pd.DataFrame, is_intraday: bool) -> pd.DataFrame:
        """Add momentum state features based on RSI"""
        # Use appropriate RSI period based on timeframe
        rsi_period = self.config.day_rsi_period if is_intraday else self.config.swing_rsi_period
        
        # Calculate RSI if not already present
        if 'rsi' not in df.columns:
            df['rsi'] = self.ti.calculate_rsi(df['close'], rsi_period)
            
        # Determine momentum states
        df['momentum_state'] = 0  # Neutral
        
        # Overbought state
        df.loc[df['rsi'] > self.config.rsi_overbought, 'momentum_state'] = 1  # Overbought
        
        # Oversold state
        df.loc[df['rsi'] < self.config.rsi_oversold, 'momentum_state'] = -1  # Oversold
        
        # Calculate RSI divergence
        if len(df) > 5:
            # Price making higher highs but RSI making lower highs (bearish divergence)
            df['bearish_div'] = (
                (df['close'] > df['close'].shift(5)) & 
                (df['rsi'] < df['rsi'].shift(5))
            )
            
            # Price making lower lows but RSI making higher lows (bullish divergence)
            df['bullish_div'] = (
                (df['close'] < df['close'].shift(5)) & 
                (df['rsi'] > df['rsi'].shift(5))
            )
            
            # Combined divergence signal
            df['rsi_divergence'] = 0
            df.loc[df['bullish_div'], 'rsi_divergence'] = 1
            df.loc[df['bearish_div'], 'rsi_divergence'] = -1
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume confirmation features"""
        if 'volume' in df.columns:
            # Calculate volume ratio
            df['volume_ma'] = df['volume'].rolling(window=self.config.volume_lookback).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Determine volume surge
            df['volume_surge'] = df['volume_ratio'] > self.config.volume_surge_threshold
            
            # Volume-price alignment (volume increasing with price movement)
            df['volume_aligned'] = (
                (df['returns'] > 0) & (df['volume_ratio'] > 1.2) | 
                (df['returns'] < 0) & (df['volume_ratio'] > 1.2)
            )
            
            # Combined volume confirmation signal
            df['volume_confirm'] = 0
            df.loc[df['volume_surge'] & df['volume_aligned'], 'volume_confirm'] = 1
        else:
            # Default values if volume data not available
            df['volume_ratio'] = 1.0
            df['volume_surge'] = False
            df['volume_confirm'] = 0
            
        # Open Interest features (if available)
        if 'open_interest' in df.columns:
            df['oi_change'] = df['open_interest'].pct_change()
            
            # OI interpretation
            # Rising OI + rising price → bullish (new longs entering)
            df['oi_bullish'] = (df['oi_change'] > 0) & (df['returns'] > 0)
            
            # Falling OI + falling price → bearish (shorts covering)
            df['oi_bearish'] = (df['oi_change'] < 0) & (df['returns'] < 0)
            
            # OI signal
            df['oi_signal'] = 0
            df.loc[df['oi_bullish'], 'oi_signal'] = 1
            df.loc[df['oi_bearish'], 'oi_signal'] = -1
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time and seasonal features"""
        # Time of day features (if datetime index)
        if hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
            
            # Market session flags
            df['morning_session'] = (
                (df['hour'] >= self.config.morning_session_start) & 
                (df['hour'] < self.config.morning_session_end)
            )
            
            df['close_session'] = df['hour'] >= self.config.close_session_start
            
            # Time of day bias
            df['time_of_day_bias'] = 0
            
            # Morning session tends to have higher volatility
            df.loc[df['morning_session'], 'time_of_day_bias'] = 0.3
            
            # Close session tends to have more predictable movement
            df.loc[df['close_session'], 'time_of_day_bias'] = 0.2
            
        # Day of week features
        if hasattr(df.index, 'dayofweek'):
            df['day_of_week'] = df.index.dayofweek
            
            # Tuesday tends to reverse Monday's move ("Turnaround Tuesday")
            df['tuesday'] = df['day_of_week'] == 1
            
            # Friday tends to see risk-off behavior
            df['friday'] = df['day_of_week'] == 4
            
            # EIA report day (Thursday)
            df['report_day'] = df['day_of_week'] == 3
            
        # Seasonal features
        if hasattr(df.index, 'month'):
            df['month'] = df.index.month
            
            # Seasonal bias based on month
            df['seasonal_bias'] = 0
            
            # Winter months (bullish bias)
            df.loc[df['month'].isin(self.config.winter_months), 'seasonal_bias'] = 0.5
            
            # Summer months (mild bullish bias)
            df.loc[df['month'].isin(self.config.summer_months), 'seasonal_bias'] = 0.2
            
            # Shoulder months (bearish bias)
            df.loc[df['month'].isin(self.config.shoulder_months), 'seasonal_bias'] = -0.3
            
        return df
    
    def _calculate_transition_matrices(self, df: pd.DataFrame, is_intraday: bool) -> None:
        """Calculate Markov Chain transition probability matrices"""
        if 'price_state' not in df.columns or len(df) < self.config.min_transitions:
            return
            
        # Get state sequence
        states = df['price_state'].dropna().astype(int).values
        
        # Determine lookback window
        lookback = self.config.day_lookback_periods if is_intraday else self.config.swing_lookback_periods
        
        # Use recent data for transition matrix
        recent_states = states[-min(len(states), lookback):]
        
        if len(recent_states) < self.config.min_transitions:
            return
            
        # Initialize transition counts
        n_states = len(PriceState)
        transition_counts = np.zeros((n_states, n_states))
        
        # Count transitions
        for i in range(len(recent_states) - 1):
            from_state = recent_states[i] - 1  # Adjust for 0-indexing
            to_state = recent_states[i+1] - 1  # Adjust for 0-indexing
            
            # Ensure valid indices
            if 0 <= from_state < n_states and 0 <= to_state < n_states:
                transition_counts[from_state, to_state] += 1
                
        # Calculate probabilities (rows should sum to 1)
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        
        transition_probs = transition_counts / row_sums
        
        # Store the transition matrix based on timeframe
        if is_intraday:
            self.day_tpm = transition_probs
        else:
            self.swing_tpm = transition_probs
            
        # Handle Hidden Markov Model if configured
        if self.config.use_hmm:
            self._update_hmm(df, is_intraday)
    
    def _update_hmm(self, df: pd.DataFrame, is_intraday: bool) -> None:
        """Update Hidden Markov Model (if used)"""
        # This would implement HMM functionality
        # For simplicity, we're using standard Markov Chains in this implementation
        pass
    
    def _add_markov_predictions(self, df: pd.DataFrame, is_intraday: bool) -> pd.DataFrame:
        """Add next state predictions based on Markov Chain model"""
        # Get appropriate transition matrix
        tpm = self.day_tpm if is_intraday else self.swing_tpm
        
        if tpm is None or self.current_state is None:
            # No prediction if we don't have a transition matrix or current state
            df['next_state_prob_bullish'] = np.nan
            df['next_state_prob_bearish'] = np.nan
            df['next_state_prob_neutral'] = np.nan
            df['predicted_next_state'] = np.nan
            df['markov_signal'] = 0
            
            return df
            
        # Current state (adjusted for 0-indexing)
        current_idx = self.current_state - 1
        
        # Get transition probabilities for current state
        if 0 <= current_idx < len(tpm):
            next_state_probs = tpm[current_idx]
            
            # Store probabilities
            df['next_state_prob_bullish'] = np.nan
            df['next_state_prob_bearish'] = np.nan
            df['next_state_prob_neutral'] = np.nan
            
            # Set the last row with predictions
            df.loc[df.index[-1], 'next_state_prob_bullish'] = next_state_probs[PriceState.BULLISH.value - 1]
            df.loc[df.index[-1], 'next_state_prob_bearish'] = next_state_probs[PriceState.BEARISH.value - 1]
            df.loc[df.index[-1], 'next_state_prob_neutral'] = next_state_probs[PriceState.NEUTRAL.value - 1]
            
            # Predict most likely next state
            predicted_state = np.argmax(next_state_probs) + 1  # Adjust for 1-indexing
            df.loc[df.index[-1], 'predicted_next_state'] = predicted_state
            
            # Store prediction
            self.next_state_prediction = predicted_state
            
            # Calculate Markov signal based on state prediction
            markov_signal = 0
            
            if predicted_state == PriceState.BULLISH.value:
                # Predicted bullish - long signal
                markov_signal = next_state_probs[PriceState.BULLISH.value - 1]
            elif predicted_state == PriceState.BEARISH.value:
                # Predicted bearish - short signal
                markov_signal = -next_state_probs[PriceState.BEARISH.value - 1]
                
            # Store signal
            df.loc[df.index[-1], 'markov_signal'] = markov_signal
        
        return df
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate Markov Chain trading signals"""
        if len(features) < self.config.min_transitions:
            return 0, 0, None
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            return self._calculate_day_trading_signals(current, features)
        else:
            return self._calculate_swing_trading_signals(current, features)
    
    def _calculate_day_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate day trading signals using Markov Chain"""
        signals = {}
        
        # 1. Markov prediction signal
        # This is our primary signal based on state transition probabilities
        if 'markov_signal' in current:
            signals['markov_prediction'] = current['markov_signal']
        else:
            signals['markov_prediction'] = 0
            
        # 2. RSI state signal
        # Use RSI to confirm or contradict the Markov prediction
        if 'momentum_state' in current:
            momentum_state = current['momentum_state']
            
            # RSI oversold is bullish, overbought is bearish
            signals['rsi_state'] = momentum_state * -1
            
            # Include divergence if available
            if 'rsi_divergence' in current:
                signals['rsi_state'] = (signals['rsi_state'] + current['rsi_divergence']) / 2
        else:
            signals['rsi_state'] = 0
            
        # 3. Volatility state signal
        # Adjust confidence based on volatility regime
        vol_signal = 0
        
        if 'volatility_state' in current:
            volatility_state = VolatilityState(current['volatility_state'])
            
            if volatility_state == VolatilityState.HIGH:
                # In high volatility, prefer momentum continuation
                vol_signal = np.sign(current['returns']) * 0.5
            elif volatility_state == VolatilityState.LOW:
                # In low volatility, prefer mean reversion
                vol_signal = -np.sign(current['returns']) * 0.3
                
        signals['volatility_state'] = vol_signal
        
        # 4. Volume confirmation signal
        if 'volume_confirm' in current:
            volume_signal = current['volume_confirm']
            
            # Direction based on returns
            if volume_signal == 1:
                volume_signal = np.sign(current['returns'])
                
            signals['volume_confirm'] = volume_signal
        else:
            signals['volume_confirm'] = 0
            
        # 5. Time of day signal
        if 'time_of_day_bias' in current:
            # Use markov signal direction with time of day bias magnitude
            time_signal = np.sign(signals['markov_prediction']) * current['time_of_day_bias']
            signals['time_of_day'] = time_signal
        else:
            signals['time_of_day'] = 0
            
        # Combine signals using weights
        weights = self.config.feature_weights['day']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Boost confidence for stronger Markov signals
            markov_strength = abs(signals['markov_prediction'])
            confidence = agreement_ratio * 0.7 + markov_strength * 0.3
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.DAY_TRADE
    
    def _calculate_swing_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate swing trading signals using Markov Chain"""
        signals = {}
        
        # 1. Markov prediction signal
        if 'markov_signal' in current:
            signals['markov_prediction'] = current['markov_signal']
        else:
            signals['markov_prediction'] = 0
            
        # 2. RSI state signal
        if 'momentum_state' in current:
            momentum_state = current['momentum_state']
            signals['rsi_state'] = momentum_state * -1
            
            # Include divergence if available
            if 'rsi_divergence' in current:
                signals['rsi_state'] = (signals['rsi_state'] + current['rsi_divergence']) / 2
        else:
            signals['rsi_state'] = 0
            
        # 3. Volatility state signal
        vol_signal = 0
        
        if 'volatility_state' in current:
            volatility_state = VolatilityState(current['volatility_state'])
            
            if volatility_state == VolatilityState.HIGH:
                # In high volatility, prefer trend following
                vol_signal = np.sign(signals['markov_prediction']) * 0.5
            elif volatility_state == VolatilityState.LOW:
                # In low volatility, favor range trading signals
                vol_signal = 0.3 if abs(signals['markov_prediction']) > 0.7 else 0
                
        signals['volatility_state'] = vol_signal
        
        # 4. Volume confirmation signal
        if 'volume_confirm' in current and 'oi_signal' in current:
            # Combine volume and open interest signals if available
            volume_signal = (current['volume_confirm'] + current['oi_signal']) / 2
            signals['volume_confirm'] = volume_signal
        elif 'volume_confirm' in current:
            signals['volume_confirm'] = current['volume_confirm']
        else:
            signals['volume_confirm'] = 0
            
        # 5. Seasonal bias signal
        if 'seasonal_bias' in current:
            # Use seasonal bias directly
            signals['seasonal_bias'] = current['seasonal_bias']
            
            # Strengthen signal if it aligns with Markov prediction
            if np.sign(current['seasonal_bias']) == np.sign(signals['markov_prediction']):
                signals['seasonal_bias'] *= 1.2
        else:
            signals['seasonal_bias'] = 0
            
        # Combine signals using weights
        weights = self.config.feature_weights['swing']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Adjust confidence based on volatility regime
            volatility_factor = 1.0
            
            if 'volatility_state' in current:
                volatility_state = VolatilityState(current['volatility_state'])
                
                if volatility_state == VolatilityState.HIGH:
                    volatility_factor = 0.8  # Lower confidence in high volatility
                elif volatility_state == VolatilityState.LOW:
                    volatility_factor = 1.2  # Higher confidence in low volatility
                    
            # Calculate final confidence
            confidence = agreement_ratio * 0.7 + abs(signals['markov_prediction']) * 0.3
            confidence *= volatility_factor
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.SWING_TRADE
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional Markov Chain specific validation"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Common validations
        validations = [
            # Signal strength must be meaningful
            abs(signal) >= 0.2,
            
            # Must have a valid Markov prediction
            'predicted_next_state' in current and not pd.isna(current['predicted_next_state']),
            
            # Markov probability should be reasonably high
            self._get_state_probability(current) > 0.4
        ]
        
        # Add timeframe-specific validations
        if is_intraday:
            day_validations = [
                # For day trading, check time of day
                not (current.get('hour', 0) == 12 and abs(signal) < 0.5),  # Avoid lunch hour unless strong signal
                
                # Verify volume confirms the move
                current.get('volume_ratio', 0) > 0.8,  # Reasonable volume
                
                # RSI shouldn't contradict signal strongly
                not (signal > 0 and current.get('rsi', 50) > 75),  # Avoid longs when severely overbought
                not (signal < 0 and current.get('rsi', 50) < 25)   # Avoid shorts when severely oversold
            ]
            validations.extend(day_validations)
        else:
            # Swing trading validations
            swing_validations = [
                # For swing trading, check seasonal alignment
                not (signal > 0 and current.get('month', 0) in self.config.shoulder_months and abs(signal) < 0.7),
                not (signal < 0 and current.get('month', 0) in self.config.winter_months and abs(signal) < 0.7),
                
                # Check for volume confirmation on swing trades
                current.get('volume_ratio', 0) > 1.0,
                
                # RSI divergence check
                not (signal > 0 and current.get('bearish_div', False)),
                not (signal < 0 and current.get('bullish_div', False))
            ]
            validations.extend(swing_validations)
            
        return all(validations)