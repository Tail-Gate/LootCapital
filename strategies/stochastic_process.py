from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats, optimize
from statsmodels.tsa.stattools import adfuller

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class StochasticProcessConfig(TechnicalConfig):
    """Configuration for stochastic process based strategy"""
    # Mean Reversion (Ornstein-Uhlenbeck) parameters
    ou_lookback_period: int = 60  # Periods for OU parameter estimation
    ou_mean_reversion_strength: float = 0.5  # Threshold for mean reversion signal
    ou_zscore_threshold: float = 2.0  # Z-score for entry signals
    
    # Geometric Brownian Motion (GBM) parameters
    gbm_lookback_period: int = 30  # Periods for GBM parameter estimation
    gbm_drift_weight: float = 0.3  # Weight for drift component
    
    # Jump detection parameters
    jump_zscore_threshold: float = 3.0  # Z-score for jump detection
    post_jump_wait_periods: int = 3  # Periods to wait after jump before trading
    
    # Regime detection parameters
    regime_window: int = 50  # Window for regime detection
    high_vol_threshold: float = 1.5  # Threshold for high volatility regime
    low_vol_threshold: float = 0.7  # Threshold for low volatility regime
    
    # Seasonality parameters
    use_seasonality: bool = True  # Whether to include seasonal components
    seasonal_period: int = 252  # Trading days in yearly seasonal cycle (252 trading days)
    
    # Day trading parameters
    intraday_ou_lookback: int = 30  # Lookback for intraday mean reversion
    intraday_vol_window: int = 15  # Window for intraday volatility estimation
    
    # Swing trading parameters
    swing_ou_lookback: int = 20  # Lookback for swing mean reversion
    swing_vol_window: int = 10  # Window for swing volatility estimation
    
    # Stop loss and take profit parameters
    day_stop_loss_atr: float = 1.5  # Stop loss in ATR units (day trading)
    day_take_profit_atr: float = 3.0  # Take profit in ATR units (day trading)
    swing_stop_loss_atr: float = 2.0  # Stop loss in ATR units (swing trading)
    swing_take_profit_atr: float = 4.0  # Take profit in ATR units (swing trading)
    
    # Hyperparameters for OU process
    ou_theta_min: float = 0.05  # Min mean reversion speed
    ou_theta_max: float = 0.95  # Max mean reversion speed
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'day': {
                    'ou_signal': 0.30,
                    'jump_signal': 0.25,
                    'volatility_regime': 0.20,
                    'momentum': 0.15,
                    'volume': 0.10
                },
                'swing': {
                    'ou_signal': 0.25,
                    'gbm_drift': 0.20,
                    'regime_state': 0.20,
                    'seasonality': 0.20,
                    'roll_yield': 0.15
                }
            }


class StochasticProcessStrategy(TechnicalStrategy):
    """
    Trading strategy based on stochastic processes for modeling financial time series.
    
    This strategy implements:
    1. Mean reversion using Ornstein-Uhlenbeck (OU) process
    2. Trend following using Geometric Brownian Motion (GBM)
    3. Jump detection and exploitation
    4. Regime-switching models for volatility states
    5. Seasonal components for natural gas futures
    """
    
    def __init__(self, config: StochasticProcessConfig = None):
        super().__init__(name="stochastic_process", config=config or StochasticProcessConfig())
        self.config: StochasticProcessConfig = self.config
        
        # Store estimated model parameters
        self.ou_params = {'theta': None, 'mu': None, 'sigma': None}
        self.gbm_params = {'mu': None, 'sigma': None}
        self.current_regime = 'normal'  # 'high_vol', 'low_vol', or 'normal'
        
        # Track jumps
        self.last_jump_idx = None
        self.periods_since_jump = float('inf')
        
        # Store seasonal components
        self.seasonal_factors = {}
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for stochastic process analysis"""
        df = self.prepare_base_features(data)
        
        # Add stochastic process features
        df = self._add_ou_process_features(df)
        df = self._add_gbm_features(df)
        df = self._add_jump_features(df)
        df = self._add_regime_features(df)
        
        # Add seasonality if configured
        if self.config.use_seasonality:
            df = self._add_seasonal_features(df)
        
        return df
    
    def _add_ou_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ornstein-Uhlenbeck process features for mean reversion"""
        if len(df) < self.config.ou_lookback_period:
            return df
        
        # Get price series (log prices for OU process)
        log_prices = np.log(df['close'])
        
        # Estimate OU parameters using log prices
        self.ou_params = self._estimate_ou_parameters(log_prices)
        
        # Calculate OU process residuals (deviation from equilibrium)
        if self.ou_params['theta'] is not None:
            # Calculate expected equilibrium level
            df['ou_equilibrium'] = np.exp(self.ou_params['mu'])
            
            # Calculate residuals for log prices
            log_residuals = log_prices - self.ou_params['mu']
            
            # Convert to price space
            df['ou_residuals'] = df['close'] - df['ou_equilibrium']
            
            # Normalize residuals by volatility (Z-score)
            if self.ou_params['sigma'] > 0:
                df['ou_zscore'] = log_residuals / self.ou_params['sigma']
                
                # Calculate mean reversion signal (-1 to 1)
                # Negative zscore -> bullish (below equilibrium)
                # Positive zscore -> bearish (above equilibrium)
                df['ou_signal'] = -np.clip(df['ou_zscore'] / self.config.ou_zscore_threshold, -1, 1)
                
                # Mean reversion strength (based on speed of mean reversion)
                df['ou_strength'] = min(self.ou_params['theta'] / self.config.ou_theta_max, 1.0)
                
                # Mean reversion confidence (how stationary is the series)
                try:
                    adf_result = adfuller(log_prices.dropna())
                    # Convert p-value to confidence (lower p-value = higher confidence)
                    df['ou_confidence'] = 1 - min(adf_result[1], 1.0)
                except:
                    df['ou_confidence'] = 0.5  # Default middle value
        
        # Add separate features for day trading (shorter lookback)
        if self.is_intraday_data(df) and len(df) >= self.config.intraday_ou_lookback:
            short_log_prices = np.log(df['close'].iloc[-self.config.intraday_ou_lookback:])
            short_ou_params = self._estimate_ou_parameters(short_log_prices)
            
            if short_ou_params['theta'] is not None:
                # Short-term equilibrium
                df['short_ou_equilibrium'] = np.exp(short_ou_params['mu'])
                
                # Short-term residuals
                short_log_residuals = log_prices - short_ou_params['mu']
                df['short_ou_residuals'] = df['close'] - df['short_ou_equilibrium']
                
                if short_ou_params['sigma'] > 0:
                    df['short_ou_zscore'] = short_log_residuals / short_ou_params['sigma']
                    df['short_ou_signal'] = -np.clip(df['short_ou_zscore'] / (self.config.ou_zscore_threshold * 0.8), -1, 1)
        
        return df
    
    def _estimate_ou_parameters(self, log_prices: pd.Series) -> Dict[str, float]:
        """
        Estimate Ornstein-Uhlenbeck process parameters
        
        The OU process follows: dX_t = θ(μ - X_t)dt + σdW_t
        where:
        - θ is the mean reversion speed
        - μ is the equilibrium level
        - σ is the volatility
        """
        try:
            # Skip if not enough data
            if len(log_prices) < 10:
                return {'theta': None, 'mu': None, 'sigma': None}
            
            # Get price differences and lagged prices
            y = log_prices.diff().dropna().values
            x = log_prices.shift().dropna().values
            
            if len(y) != len(x):
                x = x[-len(y):]
            
            # Simple linear regression: y ~ a + b*x
            # For OU: dX_t ~ a + b*X_t where a = θ*μ and b = -θ
            X = np.column_stack((np.ones(len(x)), x))
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            a, b = beta
            theta = -b
            
            # Ensure theta is positive and within reasonable bounds
            theta = max(min(theta, self.config.ou_theta_max), self.config.ou_theta_min)
            
            # Calculate equilibrium level
            mu = a / theta
            
            # Calculate process volatility
            residuals = y - (a + b * x)
            sigma = np.std(residuals)
            
            return {'theta': theta, 'mu': mu, 'sigma': sigma}
        except Exception as e:
            print(f"Error estimating OU parameters: {e}")
            return {'theta': None, 'mu': None, 'sigma': None}
    
    def _add_gbm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Geometric Brownian Motion features for trend analysis"""
        if len(df) < self.config.gbm_lookback_period:
            return df
        
        # Get log returns for GBM
        log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        if len(log_returns) > 0:
            # Estimate GBM parameters: drift (μ) and volatility (σ)
            gbm_window = min(len(log_returns), self.config.gbm_lookback_period)
            recent_returns = log_returns.iloc[-gbm_window:]
            
            mu = recent_returns.mean()
            sigma = recent_returns.std()
            
            self.gbm_params = {'mu': mu, 'sigma': sigma}
            
            # Store parameters in DataFrame
            df['gbm_drift'] = mu
            df['gbm_vol'] = sigma
            
            # Calculate annualized parameters
            trading_days = 252
            df['gbm_drift_annual'] = mu * trading_days
            df['gbm_vol_annual'] = sigma * np.sqrt(trading_days)
            
            # Calculate trend signal based on drift (scaled to -1 to 1)
            # Use drift/volatility ratio (like Sharpe ratio) for signal strength
            if sigma > 0:
                sharpe = mu / sigma
                df['gbm_signal'] = np.clip(sharpe * 5, -1, 1)  # Scale by 5 to normalize
            else:
                df['gbm_signal'] = 0
        
        return df
    
    def _add_jump_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for detecting and quantifying price jumps"""
        if len(df) < 10:
            return df
        
        # Calculate returns
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Calculate historical volatility for normalization
        rolling_std = df['returns'].rolling(window=20).std()
        
        # Detect jumps via z-scores
        df['return_zscore'] = np.abs(df['returns'] / rolling_std)
        
        # Flag jumps when z-score exceeds threshold
        df['is_jump'] = df['return_zscore'] > self.config.jump_zscore_threshold
        
        # Calculate jump frequency
        df['jump_frequency'] = df['is_jump'].rolling(window=50).mean()
        
        # Track jumps and periods since last jump
        if df['is_jump'].any():
            jump_indices = df.index[df['is_jump']]
            last_jump_idx = jump_indices[-1] if len(jump_indices) > 0 else None
            
            if last_jump_idx is not None:
                self.last_jump_idx = last_jump_idx
                
                # Calculate periods since last jump
                df['periods_since_jump'] = 0
                for i, idx in enumerate(df.index):
                    if idx <= last_jump_idx:
                        df.loc[idx, 'periods_since_jump'] = 0
                    else:
                        # Find how many periods after the jump
                        jump_loc = df.index.get_loc(last_jump_idx)
                        current_loc = df.index.get_loc(idx)
                        df.loc[idx, 'periods_since_jump'] = current_loc - jump_loc
                
                # Update instance variable
                self.periods_since_jump = df['periods_since_jump'].iloc[-1]
            
            # Calculate jump direction
            df['jump_direction'] = np.nan
            for idx in jump_indices:
                jump_return = df.loc[idx, 'returns']
                df.loc[idx, 'jump_direction'] = np.sign(jump_return)
                
            # Calculate jump signal (opportunity after a jump)
            df['jump_signal'] = 0
            for i, idx in enumerate(df.index):
                periods_after = df.loc[idx, 'periods_since_jump']
                
                if 1 <= periods_after <= self.config.post_jump_wait_periods:
                    # Recent jump, wait before trading
                    df.loc[idx, 'jump_signal'] = 0
                elif periods_after > self.config.post_jump_wait_periods:
                    # After waiting period, look for reversion or continuation
                    last_jump_loc = df.index.get_loc(self.last_jump_idx)
                    jump_direction = np.sign(df.iloc[last_jump_loc]['returns'])
                    
                    # Check if mean reversion happened after jump
                    if jump_direction > 0:
                        # Bullish jump, check for bearish signal afterward
                        recent_trend = np.sign(df['close'].iloc[-5:].pct_change().mean())
                        df.loc[idx, 'jump_signal'] = -0.5 if recent_trend < 0 else 0
                    else:
                        # Bearish jump, check for bullish signal afterward
                        recent_trend = np.sign(df['close'].iloc[-5:].pct_change().mean())
                        df.loc[idx, 'jump_signal'] = 0.5 if recent_trend > 0 else 0
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime-switching model features"""
        if len(df) < self.config.regime_window:
            return df
        
        # Calculate volatility measures
        if 'volatility' not in df.columns:
            df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Calculate short/long term volatility ratio
        short_vol = df['returns'].rolling(window=5).std()
        long_vol = df['returns'].rolling(window=20).std()
        df['vol_ratio'] = short_vol / long_vol
        
        # Detect volatility regimes
        df['regime'] = 'normal'
        df.loc[df['vol_ratio'] > self.config.high_vol_threshold, 'regime'] = 'high_vol'
        df.loc[df['vol_ratio'] < self.config.low_vol_threshold, 'regime'] = 'low_vol'
        
        # Update current regime
        if len(df) > 0:
            self.current_regime = df['regime'].iloc[-1]
        
        # Calculate regime-based signals
        df['regime_signal'] = 0
        
        # In high volatility regimes: reduce exposure
        df.loc[df['regime'] == 'high_vol', 'regime_signal'] = (
            df.loc[df['regime'] == 'high_vol', 'ou_signal'] * 0.5  # Reduce mean reversion signals
        )
        
        # In low volatility regimes: focus on mean reversion
        df.loc[df['regime'] == 'low_vol', 'regime_signal'] = (
            df.loc[df['regime'] == 'low_vol', 'ou_signal'] * 1.2  # Amplify mean reversion
        )
        
        # In normal regimes: balanced approach
        df.loc[df['regime'] == 'normal', 'regime_signal'] = (
            df.loc[df['regime'] == 'normal', 'ou_signal'] * 0.7 + 
            df.loc[df['regime'] == 'normal', 'gbm_signal'] * 0.3
        )
        
        return df
    
    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal components for natural gas futures"""
        if not hasattr(df.index, 'month') or len(df) < self.config.seasonal_period:
            return df
        
        # Natural gas seasonality factors based on month
        # Higher in winter months (heating demand), lower in spring/fall
        seasonal_factors = {
            1: 0.6,   # January - winter, bullish
            2: 0.4,   # February - winter, bullish
            3: 0.0,   # March - transition, neutral
            4: -0.4,  # April - spring, bearish
            5: -0.3,  # May - spring, bearish
            6: 0.1,   # June - summer AC demand, slightly bullish
            7: 0.3,   # July - summer AC demand, bullish
            8: 0.3,   # August - summer AC demand, bullish
            9: 0.0,   # September - transition, neutral
            10: 0.0,  # October - transition, neutral
            11: 0.3,  # November - early winter, bullish
            12: 0.5   # December - winter, bullish
        }
        
        # Store for later use
        self.seasonal_factors = seasonal_factors
        
        # Add month column
        df['month'] = df.index.month
        
        # Add seasonal factor
        df['seasonal_factor'] = df['month'].map(seasonal_factors)
        
        # Add EIA report day flag (Thursday = Nat Gas storage report)
        if hasattr(df.index, 'dayofweek'):
            df['is_report_day'] = df.index.dayofweek == 3  # 3 = Thursday
            
            # Enhanced volatility on report days
            report_day_factor = 1.5
            df.loc[df['is_report_day'], 'volatility'] = (
                df.loc[df['is_report_day'], 'volatility'] * report_day_factor
            )
        
        # Weekly natural gas storage cycle
        # Storage injections/withdrawals often create weekly patterns
        if hasattr(df.index, 'dayofweek'):
            weekly_pattern = {
                0: 0.1,    # Monday - slight recovery
                1: 0.1,    # Tuesday - continued recovery
                2: -0.1,   # Wednesday - anticipation of report
                3: 0.0,    # Thursday - report day (neutral until data)
                4: -0.1,   # Friday - often selling pressure
                5: 0.0,    # Saturday
                6: 0.0     # Sunday
            }
            
            df['day_of_week'] = df.index.dayofweek
            df['weekly_seasonal'] = df['day_of_week'].map(weekly_pattern)
            
            # Combine seasonal signals
            df['seasonal_signal'] = df['seasonal_factor'] * 0.7 + df['weekly_seasonal'] * 0.3
        else:
            df['seasonal_signal'] = df['seasonal_factor']
            
        # Add natural gas roll yield (contango/backwardation)
        # This would normally use the price difference between front and second month futures
        # Since we don't have that data, we'll approximate based on seasonal patterns
        df['approx_roll_yield'] = 0.0
        
        # Typically backwardation in winter/summer, contango in shoulder months
        winter_summer_months = [1, 2, 7, 8, 12]
        shoulder_months = [3, 4, 5, 9, 10, 11]
        
        df.loc[df['month'].isin(winter_summer_months), 'approx_roll_yield'] = 0.001  # Backwardation
        df.loc[df['month'].isin(shoulder_months), 'approx_roll_yield'] = -0.001  # Contango
        
        return df
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate stochastic process signals"""
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
        """Calculate day trading signals using stochastic process models"""
        signals = {}
        
        # 1. OU Process mean reversion signal
        # Use shorter-term intraday process for day trading
        if 'short_ou_signal' in current:
            signals['ou_signal'] = current['short_ou_signal']
        elif 'ou_signal' in current:
            signals['ou_signal'] = current['ou_signal']
        else:
            signals['ou_signal'] = 0
            
        # 2. Jump detection signal
        if 'jump_signal' in current:
            signals['jump_signal'] = current['jump_signal']
        else:
            signals['jump_signal'] = 0
            
        # 3. Volatility regime signal
        if 'regime_signal' in current:
            signals['volatility_regime'] = current['regime_signal']
        else:
            signals['volatility_regime'] = 0
            
        # 4. Short-term momentum signal
        momentum = features['returns'].rolling(window=5).sum().iloc[-1]
        vol = max(features['volatility'].iloc[-1], 0.001)  # Avoid division by zero
        signals['momentum'] = np.clip(momentum / vol * 2, -1, 1)
            
        # 5. Volume confirmation signal
        if 'volume_ratio' in current:
            # Volume in direction of signal
            base_signal = signals['ou_signal']  # Use OU as base signal
            vol_signal = (current['volume_ratio'] - 1) * np.sign(base_signal)
            signals['volume'] = np.clip(vol_signal, -1, 1)
        else:
            signals['volume'] = 0
            
        # Combine signals using weights
        weights = self.config.feature_weights['day']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            # Calculate agreement ratio
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Adjust confidence based on OU process strength and regime
            ou_confidence = current.get('ou_confidence', 0.5)
            
            # Lower confidence in high volatility regimes
            regime_factor = 1.0
            if current.get('regime') == 'high_vol':
                regime_factor = 0.7
            elif current.get('regime') == 'low_vol':
                regime_factor = 1.2
                
            confidence = agreement_ratio * 0.6 + ou_confidence * 0.4
            confidence *= regime_factor
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.DAY_TRADE
    
    def _calculate_swing_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate swing trading signals using stochastic process models"""
        signals = {}
        
        # 1. OU Process mean reversion signal
        if 'ou_signal' in current:
            signals['ou_signal'] = current['ou_signal']
        else:
            signals['ou_signal'] = 0
            
        # 2. GBM drift signal (trend following)
        if 'gbm_signal' in current:
            signals['gbm_drift'] = current['gbm_signal']
        else:
            signals['gbm_drift'] = 0
            
        # 3. Regime state signal
        if 'regime' in current:
            # Assign signal based on regime
            if current['regime'] == 'high_vol':
                # In high vol, reduce position size and focus on mean reversion
                signals['regime_state'] = current.get('ou_signal', 0) * 0.5
            elif current['regime'] == 'low_vol':
                # In low vol, better for mean reversion
                signals['regime_state'] = current.get('ou_signal', 0) * 1.2
            else:
                # In normal regime, balanced approach
                signals['regime_state'] = (
                    current.get('ou_signal', 0) * 0.5 + 
                    current.get('gbm_signal', 0) * 0.5
                )
        else:
            signals['regime_state'] = 0
            
        # 4. Seasonality signal
        if 'seasonal_signal' in current:
            signals['seasonality'] = current['seasonal_signal']
        else:
            signals['seasonality'] = 0
            
        # 5. Roll yield signal (contango/backwardation)
        if 'approx_roll_yield' in current:
            # Positive roll yield is bullish, negative is bearish
            roll_signal = current['approx_roll_yield'] * 1000  # Scale up small values
            signals['roll_yield'] = np.clip(roll_signal, -1, 1)
        else:
            signals['roll_yield'] = 0
            
        # Combine signals using weights
        weights = self.config.feature_weights['swing']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            # Calculate agreement ratio
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Adjust confidence based on OU process strength
            ou_confidence = current.get('ou_confidence', 0.5)
            
            # Adjust based on seasonality strength
            seasonal_confidence = 0.5
            if 'month' in current:
                month = current['month']
                # Higher confidence in stronger seasonal months
                seasonal_factor = abs(self.seasonal_factors.get(month, 0))
                seasonal_confidence = 0.3 + seasonal_factor * 0.7
                
            confidence = agreement_ratio * 0.5 + ou_confidence * 0.3 + seasonal_confidence * 0.2
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.SWING_TRADE
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional stochastic process specific validation"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Common validations
        validations = [
            # Signal strength must be meaningful
            abs(signal) >= 0.2,
            
            # Don't trade immediately after jumps
            current.get('periods_since_jump', float('inf')) >= self.config.post_jump_wait_periods
        ]
        
        # Add timeframe-specific validations
        if is_intraday:
            # Day trading validations
            day_validations = [
                # OU process signal should be strong enough for day trading
                abs(current.get('short_ou_signal', 0)) >= 0.3,
                
                # Avoid trading when z-score is too extreme
                abs(current.get('short_ou_zscore', 0)) <= self.config.ou_zscore_threshold * 2,
                
                # For mean reversion, need good volume
                current.get('volume_ratio', 0) >= 0.7,
                
                # Avoid trading during extreme regime shifts
                current.get('vol_ratio', 1) <= self.config.high_vol_threshold * 1.2
            ]
            validations.extend(day_validations)
        else:
            # Swing trading validations
            swing_validations = [
                # Mean reversion requires good OU confidence
                not (abs(current.get('ou_signal', 0)) > 0.5 and 
                     current.get('ou_confidence', 0) < 0.4),
                
                # Avoid trading against strong seasonal factors
                not (np.sign(signal) != np.sign(current.get('seasonal_signal', 0)) and 
                     abs(current.get('seasonal_signal', 0)) > 0.5),
                
                # For trend following, need confirmation from GBM drift
                not (abs(current.get('ou_signal', 0)) < 0.3 and 
                     abs(current.get('gbm_signal', 0)) < 0.3)
            ]
            validations.extend(swing_validations)
            
        return all(validations)
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        signal: float, 
        features: pd.DataFrame
    ) -> float:
        """
        Calculate dynamic stop loss using stochastic process models
        """
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Get ATR for volatility-based stop
        atr = current.get('atr', entry_price * 0.01)  # Default to 1% if ATR not available
        
        if is_intraday:
            # Base stop distance for day trading
            base_distance = atr * self.config.day_stop_loss_atr
            
            # Adjust based on volatility regime
            regime = current.get('regime', 'normal')
            if regime == 'high_vol':
                regime_factor = 1.3  # Wider stops in high volatility
            elif regime == 'low_vol':
                regime_factor = 0.8  # Tighter stops in low volatility
            else:
                regime_factor = 1.0
            
            # Adjust based on OU confidence
            # Higher OU confidence = tighter stops for mean reversion
            ou_confidence = current.get('ou_confidence', 0.5)
            ou_factor = 1.0 - (ou_confidence - 0.5) * 0.4  # Scale from 0.8 to 1.2
            
            # Calculate final stop distance
            stop_distance = base_distance * regime_factor * ou_factor
        else:
            # Base stop distance for swing trading
            base_distance = atr * self.config.swing_stop_loss_atr
            
            # Adjust based on volatility regime
            regime = current.get('regime', 'normal')
            if regime == 'high_vol':
                regime_factor = 1.2  # Wider stops in high volatility
            elif regime == 'low_vol':
                regime_factor = 0.9  # Tighter stops in low volatility
            else:
                regime_factor = 1.0
            
            # Adjust based on seasonal confidence
            seasonal_signal = abs(current.get('seasonal_signal', 0))
            seasonal_factor = 1.0 - seasonal_signal * 0.2  # Strong seasonality = tighter stops
            
            # Calculate final stop distance
            stop_distance = base_distance * regime_factor * seasonal_factor
        
        # Apply stop based on signal direction
        if signal > 0:  # Long position
            return max(entry_price - stop_distance, entry_price * 0.95)  # Ensure stop is not too far
        else:  # Short position
            return min(entry_price + stop_distance, entry_price * 1.05)  # Ensure stop is not too far
    
    def calculate_take_profit(
        self, 
        entry_price: float, 
        stop_loss: float, 
        signal: float, 
        features: pd.DataFrame
    ) -> float:
        """
        Calculate dynamic take profit using stochastic process models
        """
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Calculate risk (distance to stop)
        risk = abs(entry_price - stop_loss)
        
        if is_intraday:
            # Base reward:risk ratio for day trading
            base_ratio = self.config.day_take_profit_atr / self.config.day_stop_loss_atr
            
            # Adjust based on mean reversion signal strength
            ou_signal = abs(current.get('short_ou_signal', current.get('ou_signal', 0)))
            ou_factor = 1.0 + ou_signal * 0.5  # Scale from 1.0 to 1.5
            
            # Adjust based on volatility regime
            regime = current.get('regime', 'normal')
            if regime == 'low_vol':
                # In low volatility, more likely to hit targets
                regime_factor = 1.2
            elif regime == 'high_vol':
                # In high volatility, harder to hit precise targets
                regime_factor = 0.8
            else:
                regime_factor = 1.0
                
            # Calculate final reward:risk ratio
            reward_risk_ratio = base_ratio * ou_factor * regime_factor
        else:
            # Base reward:risk ratio for swing trading
            base_ratio = self.config.swing_take_profit_atr / self.config.swing_stop_loss_atr
            
            # Adjust based on mean reversion and trend signals
            ou_signal = abs(current.get('ou_signal', 0))
            gbm_signal = abs(current.get('gbm_signal', 0))
            
            # For mean reversion vs trend following
            if ou_signal > gbm_signal:
                # Mean reversion trades have tighter targets
                signal_factor = 0.9
            else:
                # Trend following trades have wider targets
                signal_factor = 1.1
                
            # Adjust based on seasonality
            month = current.get('month', 0)
            seasonal_factor = 1.0
            
            if month in [1, 2, 12]:  # Winter
                seasonal_factor = 1.2  # Larger moves in winter
            elif month in [4, 5]:  # Spring
                seasonal_factor = 0.9  # Smaller moves in spring
                
            # Calculate final reward:risk ratio
            reward_risk_ratio = base_ratio * signal_factor * seasonal_factor
        
        # Calculate reward based on risk
        reward = risk * reward_risk_ratio
        
        # Apply based on signal direction
        if signal > 0:  # Long position
            return entry_price + reward
        else:  # Short position
            return entry_price - reward
    
    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update stochastic process models with new data"""
        # Re-estimate OU parameters
        log_prices = np.log(features['close'])
        self.ou_params = self._estimate_ou_parameters(log_prices)
        
        # Re-estimate GBM parameters
        log_returns = np.log(features['close'] / features['close'].shift(1)).dropna()
        gbm_window = min(len(log_returns), self.config.gbm_lookback_period)
        recent_returns = log_returns.iloc[-gbm_window:]
        
        mu = recent_returns.mean()
        sigma = recent_returns.std()
        
        self.gbm_params = {'mu': mu, 'sigma': sigma}
        
        # Update jump detection
        self._add_jump_features(features)
        
        # Update regime detection
        self._add_regime_features(features)