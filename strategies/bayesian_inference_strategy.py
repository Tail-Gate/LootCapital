from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class BayesianInferenceConfig(TechnicalConfig):
    """Configuration for Bayesian Inference trading strategy"""
    # Prior distribution parameters
    prior_mean: float = 0.0  # Prior mean for price change
    prior_std: float = 0.02  # Prior standard deviation for price change
    
    # Likelihood function parameters
    likelihood_weight: float = 0.6  # Weight given to new evidence
    
    # Swing trading parameters
    swing_confidence_threshold: float = 0.7  # Minimum posterior probability for swing trade
    swing_target_return: float = 0.04  # Target return for swing trades (4%)
    swing_stop_loss_atr: float = 2.5  # Stop loss in ATR units for swing trading
    
    # Day trading parameters
    day_confidence_threshold: float = 0.7  # Minimum posterior probability for day trade
    day_target_return: float = 0.01  # Target return for day trades (1%)
    day_stop_loss_atr: float = 1.0  # Stop loss in ATR units for day trading
    
    # Feature periods
    ma_short_period: int = 10  # Short moving average period (10-day for swing)
    ma_long_period: int = 50  # Long moving average period (50-day for swing)
    rsi_period: int = 14  # RSI calculation period
    volatility_period: int = 20  # Period for volatility calculation
    
    # Intraday parameters
    intraday_ma_short_period: int = 15  # Short MA period in minutes
    intraday_ma_long_period: int = 35  # Long MA period in minutes
    intraday_rsi_period: int = 14  # Intraday RSI period
    vwap_slope_period: int = 15  # Period for VWAP slope calculation
    
    # Bayesian update parameters
    update_frequency: int = 1  # How often to update the posterior (in bars)
    max_history_bars: int = 100  # Maximum history to keep for Bayesian updates
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Bayesian neural network
    use_bayesian_nn: bool = False  # Whether to use a Bayesian neural network
    nn_hidden_size: int = 32  # Hidden layer size for neural network
    nn_dropout_rate: float = 0.1  # Dropout rate for Bayesian approximation
    
    def __post_init__(self):
        super().validate()
        if not self.feature_weights:
            self.feature_weights = {
                'swing': {
                    'ma_crossover': 0.25,
                    'rsi': 0.20,
                    'trend_strength': 0.20,
                    'volatility': 0.15,
                    'volume': 0.10,
                    'fundamentals': 0.10
                },
                'day': {
                    'vwap_slope': 0.25,
                    'short_ma': 0.20,
                    'intraday_rsi': 0.15,
                    'order_imbalance': 0.15,
                    'volatility': 0.15,
                    'price_momentum': 0.10
                }
            }


class BayesianNN(nn.Module):
    """
    Simple Bayesian Neural Network using dropout for uncertainty estimation
    """
    def __init__(self, input_size, hidden_size, dropout_rate=0.1):
        super(BayesianNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        return self.layer3(x)
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """Run multiple forward passes with dropout to estimate uncertainty"""
        self.train()  # Ensure dropout is active
        
        samples = []
        for _ in range(num_samples):
            samples.append(self.forward(x).detach())
            
        mean = torch.mean(torch.cat(samples, dim=1), dim=1, keepdim=True)
        std = torch.std(torch.cat(samples, dim=1), dim=1, keepdim=True)
        
        return mean, std


class BayesianInferenceStrategy(TechnicalStrategy):
    """
    Strategy using Bayesian inference to update beliefs about market conditions
    and make probabilistic trading decisions.
    
    This strategy:
    1. Defines prior distributions for expected returns
    2. Updates these distributions with new evidence (technical indicators)
    3. Makes trading decisions based on posterior probabilities
    """
    
    def __init__(self, config: BayesianInferenceConfig = None):
        super().__init__(name="bayesian_inference", config=config or BayesianInferenceConfig())
        self.config: BayesianInferenceConfig = self.config
        
        # Bayesian state variables
        self.prior_distribution = None
        self.posterior_distribution = None
        self.likelihood_functions = {}
        
        # History of observations and probabilities
        self.observation_history = []
        self.probability_history = []
        
        # Initialize Bayesian Neural Network if configured
        self.model = None
        if self.config.use_bayesian_nn:
            self.initialize_bayesian_nn()
            
        # Track last update index
        self.last_update_idx = None
        
    def initialize_bayesian_nn(self, input_size=10):
        """Initialize the Bayesian Neural Network"""
        self.model = BayesianNN(
            input_size=input_size,
            hidden_size=self.config.nn_hidden_size,
            dropout_rate=self.config.nn_dropout_rate
        )
        
        # Initialize with reasonable weights
        for param in self.model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)
            
        # Set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for Bayesian analysis"""
        df = self.prepare_base_features(data)
        
        # Add features specific to timeframe
        is_intraday = self.is_intraday_data(df)
        
        if is_intraday:
            df = self._prepare_intraday_features(df)
        else:
            df = self._prepare_swing_features(df)
        
        # Update Bayesian model
        if self._should_update_model(df):
            self._update_bayesian_model(df)
        
        return df
    
    def _prepare_swing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for swing trading"""
        # Moving Averages
        df[f'ma_{self.config.ma_short_period}'] = df['close'].rolling(window=self.config.ma_short_period).mean()
        df[f'ma_{self.config.ma_long_period}'] = df['close'].rolling(window=self.config.ma_long_period).mean()
        
        # MA Crossover
        df['ma_crossover'] = df[f'ma_{self.config.ma_short_period}'] - df[f'ma_{self.config.ma_long_period}']
        
        # RSI
        df['rsi'] = self.ti.calculate_rsi(df['close'], self.config.rsi_period)
        
        # Trend Strength
        df['trend_strength'] = abs(df['ma_crossover']) / df['close']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=self.config.volatility_period).std()
        df['volatility_percentile'] = df['volatility'].rolling(window=90).rank(pct=True)
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period)
        
        # MACD
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Historical volatility
        df['hist_vol'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Add fundamental features (if available)
        # These would be exogenous variables like storage data, weather forecasts, etc.
        if 'storage_level' in df.columns:
            df['storage_zscore'] = (df['storage_level'] - df['storage_level'].rolling(window=52).mean()) / \
                                   df['storage_level'].rolling(window=52).std()
                                   
        # Seasonal factors
        if hasattr(df.index, 'month'):
            # Winter months are typically bullish for natural gas
            df['winter_season'] = df.index.month.isin([11, 12, 1, 2, 3]).astype(int)
            
            # Summer months can also be bullish in high cooling demand
            df['summer_season'] = df.index.month.isin([6, 7, 8]).astype(int)
            
            # Shoulder months are typically bearish
            df['shoulder_season'] = df.index.month.isin([4, 5, 9, 10]).astype(int)
        
        return df
    
    def _prepare_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for day trading"""
        # Short-term Moving Averages
        df[f'ma_{self.config.intraday_ma_short_period}'] = df['close'].rolling(
            window=self.config.intraday_ma_short_period).mean()
        df[f'ma_{self.config.intraday_ma_long_period}'] = df['close'].rolling(
            window=self.config.intraday_ma_long_period).mean()
        
        # Intraday MA Crossover
        df['intraday_ma_crossover'] = df[f'ma_{self.config.intraday_ma_short_period}'] - \
                                      df[f'ma_{self.config.intraday_ma_long_period}']
        
        # Intraday RSI
        df['intraday_rsi'] = self.ti.calculate_rsi(df['close'], self.config.intraday_rsi_period)
        
        # VWAP
        if 'vwap' not in df.columns:
            df['vwap'] = self.ti.calculate_vwap(df)
        
        # VWAP Slope (15-minute)
        df['vwap_slope'] = df['vwap'].pct_change(periods=self.config.vwap_slope_period)
        
        # Short-term volatility
        df['intraday_vol'] = df['returns'].rolling(window=15).std() * np.sqrt(252 * 6.5)
        
        # Order book features (if available)
        if all(col in df.columns for col in ['bid_volume', 'ask_volume']):
            df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        else:
            df['order_imbalance'] = 0
        
        # Micro-MACD (faster periods for intraday)
        ema_fast = df['close'].ewm(span=6).mean()
        ema_slow = df['close'].ewm(span=13).mean()
        df['micro_macd'] = ema_fast - ema_slow
        df['micro_macd_signal'] = df['micro_macd'].ewm(span=4).mean()
        df['micro_macd_hist'] = df['micro_macd'] - df['micro_macd_signal']
        
        # Price momentum
        df['price_momentum'] = df['returns'].rolling(window=5).sum()
        
        # Time of day features (if datetime index)
        if hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            
            # Market session flags
            df['morning_session'] = (df['hour'] >= 9) & (df['hour'] < 11)
            df['mid_session'] = (df['hour'] >= 11) & (df['hour'] < 14)
            df['closing_session'] = (df['hour'] >= 14) & (df['hour'] < 16)
        
        return df
    
    def _should_update_model(self, df: pd.DataFrame) -> bool:
        """Check if we should update the Bayesian model"""
        # Update on first run
        if self.last_update_idx is None:
            return True
        
        # Otherwise, update based on frequency
        current_idx = df.index[-1]
        if self.last_update_idx in df.index:
            periods_since_update = df.index.get_loc(current_idx) - df.index.get_loc(self.last_update_idx)
            return periods_since_update >= self.config.update_frequency
        
        # If index not found, update
        return True
    
    def _update_bayesian_model(self, df: pd.DataFrame) -> None:
        """Update the Bayesian model with new data"""
        # Update state
        self.last_update_idx = df.index[-1]
        
        # Initialize prior if needed
        if self.prior_distribution is None:
            self.prior_distribution = {
                'mean': self.config.prior_mean,
                'std': self.config.prior_std
            }
        
        # Update prior with posterior if we have one
        if self.posterior_distribution is not None:
            self.prior_distribution = self.posterior_distribution
        
        # Calculate likelihoods for different features
        self._calculate_likelihood_functions(df)
        
        # Calculate posterior distribution
        self._calculate_posterior_distribution(df)
        
        # Update Bayesian Neural Network if configured
        if self.config.use_bayesian_nn and self.model is not None:
            self._update_bayesian_nn(df)
    
    def _calculate_likelihood_functions(self, df: pd.DataFrame) -> None:
        """
        Calculate likelihood functions for different features
        This estimates P(E|H) - the probability of observing the evidence given the hypothesis
        """
        is_intraday = self.is_intraday_data(df)
        
        # Clear previous likelihoods
        self.likelihood_functions = {}
        
        if is_intraday:
            # Intraday likelihood functions
            
            # VWAP Slope likelihood
            vwap_slope = df['vwap_slope'].iloc[-1]
            if vwap_slope > 0.005:  # Rising VWAP (0.5%)
                self.likelihood_functions['vwap_slope'] = {'mean': 0.01, 'std': 0.005}  # Likely positive return
            elif vwap_slope < -0.005:
                self.likelihood_functions['vwap_slope'] = {'mean': -0.01, 'std': 0.005}  # Likely negative return
            else:
                self.likelihood_functions['vwap_slope'] = {'mean': 0, 'std': 0.002}  # Neutral
            
            # RSI likelihood
            rsi = df['intraday_rsi'].iloc[-1]
            if rsi > 70:
                self.likelihood_functions['rsi'] = {'mean': -0.005, 'std': 0.003}  # Overbought - likely pullback
            elif rsi < 30:
                self.likelihood_functions['rsi'] = {'mean': 0.005, 'std': 0.003}  # Oversold - likely bounce
            else:
                self.likelihood_functions['rsi'] = {'mean': 0, 'std': 0.005}  # Neutral zone
            
            # Order imbalance likelihood
            if 'order_imbalance' in df.columns:
                imbalance = df['order_imbalance'].iloc[-1]
                if abs(imbalance) > 0.2:  # Significant imbalance
                    self.likelihood_functions['order_imbalance'] = {
                        'mean': 0.005 * np.sign(imbalance),
                        'std': 0.004
                    }
                else:
                    self.likelihood_functions['order_imbalance'] = {'mean': 0, 'std': 0.005}
        
        else:
            # Swing trading likelihood functions
            
            # MA Crossover likelihood
            ma_crossover = df['ma_crossover'].iloc[-1]
            if ma_crossover > 0 and df['ma_crossover'].iloc[-2] <= 0:  # Bullish crossover
                self.likelihood_functions['ma_crossover'] = {'mean': 0.02, 'std': 0.01}
            elif ma_crossover < 0 and df['ma_crossover'].iloc[-2] >= 0:  # Bearish crossover
                self.likelihood_functions['ma_crossover'] = {'mean': -0.02, 'std': 0.01}
            elif ma_crossover > 0:  # Bullish trend
                self.likelihood_functions['ma_crossover'] = {'mean': 0.01, 'std': 0.01}
            elif ma_crossover < 0:  # Bearish trend
                self.likelihood_functions['ma_crossover'] = {'mean': -0.01, 'std': 0.01}
            else:
                self.likelihood_functions['ma_crossover'] = {'mean': 0, 'std': 0.02}
            
            # RSI likelihood
            rsi = df['rsi'].iloc[-1]
            if rsi > 70:
                self.likelihood_functions['rsi'] = {'mean': -0.015, 'std': 0.01}  # Overbought
            elif rsi < 30:
                self.likelihood_functions['rsi'] = {'mean': 0.015, 'std': 0.01}  # Oversold
            else:
                self.likelihood_functions['rsi'] = {'mean': 0, 'std': 0.015}  # Neutral
            
            # Trend strength likelihood
            trend_strength = df['trend_strength'].iloc[-1]
            if trend_strength > 0.02:  # Strong trend
                # Direction based on MA crossover
                direction = np.sign(df['ma_crossover'].iloc[-1]) 
                self.likelihood_functions['trend_strength'] = {
                    'mean': 0.015 * direction,
                    'std': 0.01
                }
            else:
                self.likelihood_functions['trend_strength'] = {'mean': 0, 'std': 0.015}
            
            # Seasonal factors (if available)
            if 'winter_season' in df.columns:
                if df['winter_season'].iloc[-1] == 1:
                    self.likelihood_functions['seasonal'] = {'mean': 0.005, 'std': 0.01}  # Slight bullish bias
                elif df['summer_season'].iloc[-1] == 1:
                    self.likelihood_functions['seasonal'] = {'mean': 0.003, 'std': 0.01}  # Very slight bullish
                elif df['shoulder_season'].iloc[-1] == 1:
                    self.likelihood_functions['seasonal'] = {'mean': -0.003, 'std': 0.01}  # Slight bearish
    
    def _calculate_posterior_distribution(self, df: pd.DataFrame) -> None:
        """
        Calculate posterior distribution using Bayesian updating
        Combines prior with likelihood functions to generate posterior
        """
        if not self.likelihood_functions:
            # No likelihoods, keep prior as posterior
            self.posterior_distribution = self.prior_distribution
            return
        
        # Initialize with prior
        posterior_mean = self.prior_distribution['mean']
        posterior_precision = 1 / (self.prior_distribution['std'] ** 2)  # Precision = 1/variance
        
        # Combine with likelihood functions
        for name, likelihood in self.likelihood_functions.items():
            # Convert to precision (1/variance)
            likelihood_precision = 1 / (likelihood['std'] ** 2)
            
            # Weighted Bayesian update
            weight = self.config.likelihood_weight
            posterior_precision = (1 - weight) * posterior_precision + weight * likelihood_precision
            posterior_mean = (
                (1 - weight) * posterior_mean * posterior_precision + 
                weight * likelihood['mean'] * likelihood_precision
            ) / posterior_precision
        
        # Convert precision back to standard deviation
        posterior_std = np.sqrt(1 / posterior_precision)
        
        # Store posterior
        self.posterior_distribution = {
            'mean': posterior_mean,
            'std': posterior_std
        }
        
        # Store history
        self.observation_history.append({
            'date': df.index[-1],
            'price': df['close'].iloc[-1],
            'prior_mean': self.prior_distribution['mean'],
            'prior_std': self.prior_distribution['std'],
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std
        })
        
        # Limit history size
        if len(self.observation_history) > self.config.max_history_bars:
            self.observation_history = self.observation_history[-self.config.max_history_bars:]
    
    def _update_bayesian_nn(self, df: pd.DataFrame) -> None:
        """Update Bayesian Neural Network with recent data"""
        if len(df) < 30:  # Need sufficient history
            return
            
        # Prepare training data
        X = self._prepare_nn_features(df)
        y = df['returns'].shift(-1).fillna(0).values[-len(X):]  # Next period returns
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        
        # Train for a few epochs
        self.model.train()
        for _ in range(5):  # Just a few updates
            # Forward pass
            y_pred = self.model(X_tensor)
            
            # Loss
            loss = nn.MSELoss()(y_pred, y_tensor)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def _prepare_nn_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for neural network"""
        is_intraday = self.is_intraday_data(df)
        
        # Select relevant columns based on timeframe
        if is_intraday:
            feature_cols = [
                'vwap_slope', 'intraday_rsi', 'intraday_vol',
                'intraday_ma_crossover', 'price_momentum',
                'micro_macd', 'micro_macd_hist'
            ]
        else:
            feature_cols = [
                'ma_crossover', 'rsi', 'volatility',
                'trend_strength', 'macd', 'macd_hist',
                'roc_5', 'roc_10'
            ]
        
        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Extract and normalize features
        X = df[feature_cols].iloc[-30:].values
        
        # Simple normalization
        X_mean = np.nanmean(X, axis=0)
        X_std = np.nanstd(X, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        
        # Replace NaNs with 0
        X = np.nan_to_num(X)
        
        # Normalize
        X_norm = (X - X_mean) / X_std
        
        return X_norm
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate trading signals using Bayesian inference"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # No signal if we don't have a posterior yet
        if self.posterior_distribution is None:
            return 0, 0, None
        
        # Calculate probability of positive/negative returns
        posterior_mean = self.posterior_distribution['mean']
        posterior_std = self.posterior_distribution['std']
        
        # Calculate signal and confidence
        signal, confidence = self._calculate_signal_from_posterior(
            posterior_mean, posterior_std, is_intraday
        )
        
        # Use Bayesian Neural Network if available
        if self.config.use_bayesian_nn and self.model is not None:
            nn_signal, nn_confidence = self._get_nn_signal(features, is_intraday)
            
            # Blend model signals (give 30% weight to NN)
            signal = 0.7 * signal + 0.3 * nn_signal
            confidence = 0.7 * confidence + 0.3 * nn_confidence
        
        # Determine trade type
        trade_type = TradeType.DAY_TRADE if is_intraday else TradeType.SWING_TRADE
        
        return signal, confidence, trade_type
    
    def _calculate_signal_from_posterior(
        self, 
        posterior_mean: float, 
        posterior_std: float,
        is_intraday: bool
    ) -> Tuple[float, float]:
        """Calculate signal strength and confidence from posterior distribution"""
        # Calculate probability of positive/negative returns
        if is_intraday:
            # For day trading, we care about smaller movements
            threshold = 0.005  # 0.5% move
            confidence_threshold = self.config.day_confidence_threshold
        else:
            # For swing trading, we look for larger moves
            threshold = 0.02  # 2% move
            confidence_threshold = self.config.swing_confidence_threshold
        
        # Calculate probabilities using normal CDF
        prob_positive = 1 - stats.norm.cdf(0, loc=posterior_mean, scale=posterior_std)
        prob_negative = stats.norm.cdf(0, loc=posterior_mean, scale=posterior_std)
        
        # Check for significant threshold crossings
        prob_above_threshold = 1 - stats.norm.cdf(threshold, loc=posterior_mean, scale=posterior_std)
        prob_below_neg_threshold = stats.norm.cdf(-threshold, loc=posterior_mean, scale=posterior_std)
        
        # Determine signal direction and strength
        if prob_above_threshold > confidence_threshold:
            signal = 1.0  # Strong long signal
            confidence = prob_above_threshold
        elif prob_below_neg_threshold > confidence_threshold:
            signal = -1.0  # Strong short signal
            confidence = prob_below_neg_threshold
        elif prob_positive > 0.6:  # Moderate conviction for positive
            signal = 0.5  # Moderate long
            confidence = prob_positive
        elif prob_negative > 0.6:  # Moderate conviction for negative
            signal = -0.5  # Moderate short
            confidence = prob_negative
        else:
            signal = 0  # No clear signal
            confidence = max(prob_positive, prob_negative)
        
        return signal, confidence
    
    def _get_nn_signal(self, features: pd.DataFrame, is_intraday: bool) -> Tuple[float, float]:
        """Get signal from Bayesian Neural Network"""
        # Prepare features
        X = self._prepare_nn_features(features)
        X_tensor = torch.tensor(X[-1:], dtype=torch.float32)
        
        # Get prediction with uncertainty
        self.model.eval()
        mean, std = self.model.predict_with_uncertainty(X_tensor)
        
        # Convert to numpy
        pred_mean = mean.item()
        pred_std = std.item()
        
        # Calculate signal from prediction
        signal, confidence = self._calculate_signal_from_posterior(
            pred_mean, pred_std, is_intraday
        )
        
        return signal, confidence
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional Bayesian-specific signal validation"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        # No signal if posterior is missing
        if self.posterior_distribution is None:
            return False
        
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Validate based on Bayesian uncertainty
        # If uncertainty is too high, reject the signal
        if self.posterior_distribution['std'] > 0.03:  # 3% uncertainty is high
            return False
        
        # Additional timeframe-specific validations
        if is_intraday:
            validations = [
                # Signal strength must be meaningful
                abs(signal) >= 0.5,
                
                # VWAP slope should confirm direction
                np.sign(signal) == np.sign(current.get('vwap_slope', 0)) or
                abs(current.get('vwap_slope', 0)) < 0.001,  # Or nearly flat
                
                # Avoid trading against strong momentum
                not (np.sign(signal) != np.sign(current.get('price_momentum', 0)) and 
                     abs(current.get('price_momentum', 0)) > 0.01)
            ]
        else:
            validations = [
                # Signal strength must be meaningful
                abs(signal) >= 0.5,
                
                # MA crossover should not strongly contradict
                not (np.sign(signal) != np.sign(current.get('ma_crossover', 0)) and 
                     abs(current.get('ma_crossover', 0)) > 0.01),
                     
                # Avoid trading against strong RSI readings
                not (signal > 0 and current['rsi'] > 70),  # Avoid longs when overbought
                not (signal < 0 and current['rsi'] < 30)   # Avoid shorts when oversold
            ]
        
        return all(validations)
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        signal: float, 
        features: pd.DataFrame
    ) -> float:
        """
        Calculate dynamic stop loss level using Bayesian uncertainty
        """
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Get ATR for volatility-based stops
        atr = current.get('atr', entry_price * 0.01)  # Default to 1% if ATR not available
        
        # Base stop distance on timeframe
        if is_intraday:
            stop_distance = atr * self.config.day_stop_loss_atr
        else:
            stop_distance = atr * self.config.swing_stop_loss_atr
            
        # Adjust based on Bayesian uncertainty
        # Higher uncertainty = wider stops
        if self.posterior_distribution is not None:
            uncertainty_factor = min(self.posterior_distribution['std'] / 0.01, 2.0)  # Cap at 2x
            stop_distance *= uncertainty_factor
        
        # Apply stop based on direction
        if signal > 0:  # Long position
            return entry_price - stop_distance
        else:  # Short position
            return entry_price + stop_distance
    
    def calculate_take_profit(
        self, 
        entry_price: float, 
        stop_loss: float, 
        signal: float, 
        features: pd.DataFrame
    ) -> float:
        """
        Calculate take profit level based on Bayesian expected return
        """
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Calculate risk (distance to stop)
        risk = abs(entry_price - stop_loss)
        
        # Base reward-to-risk ratio
        reward_risk_ratio = 1.5  # Default
        
        # Use Bayesian posterior for target if available
        if self.posterior_distribution is not None:
            # Expected return from posterior
            expected_return = self.posterior_distribution['mean']
            
            # Scale by timeframe
            if is_intraday:
                target_return = self.config.day_target_return
            else:
                target_return = self.config.swing_target_return
                
            # If expected return is significant and in the same direction
            if np.sign(expected_return) == np.sign(signal) and abs(expected_return) > 0.005:
                # Use expected return to set target (with minimum ratio)
                expected_price = entry_price * (1 + expected_return)
                expected_distance = abs(expected_price - entry_price)
                
                # Ensure minimum reward-to-risk
                reward = max(expected_distance, risk * reward_risk_ratio)
            else:
                # Fall back to standard reward-to-risk
                if is_intraday:
                    reward = risk * 1.5  # 1.5:1 for day trades
                else:
                    reward = risk * 2.0  # 2:1 for swing trades
        else:
            # Without Bayesian data, use standard ratios
            if is_intraday:
                reward = risk * 1.5
            else:
                reward = risk * 2.0
        
        # Apply target based on direction
        if signal > 0:  # Long position
            return entry_price + reward
        else:  # Short position
            return entry_price - reward
    
    def update_model(self, features: pd.DataFrame, trade_result: pd.Series = None) -> None:
        """Update Bayesian model and neural network"""
        # Force Bayesian update
        self.last_update_idx = None
        self._update_bayesian_model(features)
        
        # Update NN with trade results if provided
        if trade_result is not None and self.config.use_bayesian_nn and self.model is not None:
            X = self._prepare_nn_features(features)
            y = trade_result.values
            
            # Convert to tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
            
            # Update NN
            self.model.train()
            for _ in range(10):  # More updates for real results
                y_pred = self.model(X_tensor)
                loss = nn.MSELoss()(y_pred, y_tensor)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    def get_bayesian_stats(self) -> Dict:
        """Get current Bayesian statistics"""
        stats = {
            'prior': self.prior_distribution,
            'posterior': self.posterior_distribution,
            'likelihoods': self.likelihood_functions,
            'history': self.observation_history[-10:] if self.observation_history else []
        }
        
        return stats