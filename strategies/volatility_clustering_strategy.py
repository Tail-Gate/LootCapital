from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from arch import arch_model

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType
from utils.volatility_analyzer import VolatilityAnalyzer

@dataclass
class VolatilityClusteringConfig(TechnicalConfig):
    """Configuration for volatility clustering strategy"""
    # Volatility windows
    short_vol_window: int = 5  # 5-day short-term volatility
    medium_vol_window: int = 10  # 10-day medium-term volatility
    long_vol_window: int = 20  # 20-day long-term volatility
    
    # Volatility ratio thresholds
    high_vol_ratio_threshold: float = 1.2  # Threshold for high volatility regime
    low_vol_ratio_threshold: float = 0.8  # Threshold for low volatility regime
    
    # GARCH parameters
    use_garch: bool = True  # Whether to use GARCH models
    garch_p: int = 1  # GARCH lag order
    garch_q: int = 1  # ARCH lag order
    
    # Jump detection
    jump_zscore_threshold: float = 2.0  # Z-score threshold for jump detection
    
    # RSI parameters
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # ATR parameters
    day_atr_period: int = 5
    swing_atr_period: int = 14
    
    # Volatility clustering trading parameters
    long_entry_vol_percentile: float = 80.0  # Percentile for entering longs
    short_entry_vol_percentile: float = 20.0  # Percentile for entering shorts
    
    # Position sizing
    high_vol_size_reduction: float = 0.5  # Reduce position size in high volatility
    low_vol_size_increase: float = 1.5  # Increase position size in low volatility
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'swing': {
                    'volatility_regime': 0.30,
                    'garch_forecast': 0.20,
                    'volatility_ratio': 0.20,
                    'jump_detection': 0.15,
                    'momentum_signal': 0.15
                },
                'day': {
                    'intraday_vol_pattern': 0.30,
                    'short_term_vol_ratio': 0.25,
                    'volume_volatility': 0.20,
                    'momentum_signal': 0.15,
                    'gap_analysis': 0.10
                }
            }

class VolatilityClusteringStrategy(TechnicalStrategy):
    def __init__(self, config: VolatilityClusteringConfig = None):
        super().__init__(name="volatility_clustering", config=config or VolatilityClusteringConfig())
        self.config: VolatilityClusteringConfig = self.config
        self.vol_analyzer = VolatilityAnalyzer()
        self.garch_model = None
        self.last_garch_update = None
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare volatility clustering specific features"""
        df = self.prepare_base_features(data)
        
        # Get volatility features from analyzer
        df = self.vol_analyzer.prepare_volatility_features(
            df, 
            is_intraday=self.is_intraday_data(df)
        )
        
        # Add volatility ratio features
        df = self._calculate_volatility_ratios(df)
        
        # Add jump detection
        df = self._detect_jumps(df)
        
        # Add GARCH forecasts if enabled
        if self.config.use_garch:
            df = self._add_garch_forecasts(df)
        
        # Add momentum confirmation signals
        df = self._add_momentum_signals(df)
        
        # Add intraday volatility patterns for day trading
        if self.is_intraday_data(df):
            df = self._add_intraday_vol_patterns(df)
        else:
            # Add weekly/seasonal patterns for swing trading
            df = self._add_weekly_patterns(df)
        
        return df
    
    def _calculate_volatility_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ratios between different volatility timeframes"""
        # Calculate historical volatility for different timeframes
        for window in [self.config.short_vol_window, self.config.medium_vol_window, self.config.long_vol_window]:
            df[f'hv_{window}'] = df['returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        # Calculate volatility ratios
        df['vol_ratio_short_medium'] = df[f'hv_{self.config.short_vol_window}'] / df[f'hv_{self.config.medium_vol_window}']
        df['vol_ratio_medium_long'] = df[f'hv_{self.config.medium_vol_window}'] / df[f'hv_{self.config.long_vol_window}']
        
        # Classify volatility regimes
        df['high_vol_regime'] = df['vol_ratio_short_medium'] > self.config.high_vol_ratio_threshold
        df['low_vol_regime'] = df['vol_ratio_short_medium'] < self.config.low_vol_ratio_threshold
        
        # Volatility percentile ranking
        df['vol_percentile'] = df[f'hv_{self.config.short_vol_window}'].rolling(window=100).rank(pct=True) * 100
        
        return df
    
    def _detect_jumps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect price jumps/outliers in the returns"""
        # Calculate z-score of returns
        df['returns_zscore'] = (df['returns'] - df['returns'].rolling(window=100).mean()) / df['returns'].rolling(window=100).std()
        
        # Detect jumps based on z-score threshold
        df['jump_detected'] = abs(df['returns_zscore']) > self.config.jump_zscore_threshold
        
        # Calculate time since last jump
        df['days_since_jump'] = 0
        jump_indices = df.index[df['jump_detected']]
        
        if not jump_indices.empty:
            for i, idx in enumerate(df.index):
                if i == 0:
                    df.loc[idx, 'days_since_jump'] = 100  # Arbitrary large number for first observation
                elif df.loc[idx, 'jump_detected']:
                    df.loc[idx, 'days_since_jump'] = 0
                else:
                    prev_idx = df.index[i-1]
                    df.loc[idx, 'days_since_jump'] = df.loc[prev_idx, 'days_since_jump'] + 1
        
        return df
    
    def _add_garch_forecasts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add GARCH model volatility forecasts"""
        try:
            if len(df) < 100:  # Need sufficient data for GARCH
                return df
                
            # Only update GARCH model periodically to save computation
            current_date = df.index[-1]
            if self.garch_model is None or (
                self.last_garch_update is None or 
                (current_date - self.last_garch_update).days > 5
            ):
                # Fit GARCH model on returns
                returns = df['returns'].dropna().values * 100  # Scale for numerical stability
                model = arch_model(
                    returns, 
                    vol='Garch', 
                    p=self.config.garch_p, 
                    q=self.config.garch_q
                )
                self.garch_model = model.fit(disp='off')
                self.last_garch_update = current_date
                
            # Get conditional volatility (historical volatility from GARCH)
            cond_vol = self.garch_model.conditional_volatility
            garch_vol = pd.Series(
                cond_vol / 100,  # Scale back
                index=df.loc[df['returns'].dropna().index].index
            )
            df['garch_vol'] = garch_vol
            
            # Get forecast for next period
            forecast = self.garch_model.forecast(horizon=1)
            next_vol = forecast.variance.iloc[-1, 0] / 100  # Scale back
            
            # Fill forward the forecast
            df['garch_forecast'] = np.nan
            df.loc[df.index[-1], 'garch_forecast'] = next_vol
            
            # Calculate volatility trend from GARCH
            df['garch_vol_trend'] = df['garch_vol'].pct_change(5)
            
            return df
            
        except Exception as e:
            print(f"Error in GARCH forecasting: {e}")
            # Add empty columns in case of failure
            df['garch_vol'] = np.nan
            df['garch_forecast'] = np.nan
            df['garch_vol_trend'] = np.nan
            return df
    
    def _add_momentum_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators for signal confirmation"""
        # RSI
        if 'rsi' not in df.columns:
            df['rsi'] = self.ti.calculate_rsi(df['close'], self.config.rsi_period)
        
        # MACD
        ema_fast = df['close'].ewm(span=self.config.macd_fast).mean()
        ema_slow = df['close'].ewm(span=self.config.macd_slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # MACD divergence with volatility
        # Positive when MACD is rising but volatility is falling (potential reversal)
        df['macd_vol_divergence'] = np.sign(df['macd_hist'].diff(3)) != np.sign(df[f'hv_{self.config.short_vol_window}'].diff(3))
        
        return df
    
    def _add_intraday_vol_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intraday volatility pattern features for day trading"""
        # Only process if we have datetime index with time component
        if not hasattr(df.index, 'hour'):
            return df
            
        # Extract hour and minute
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        
        # Create time of day feature (hour + minute/60)
        df['time_of_day'] = df['hour'] + df['minute'] / 60
        
        # Flag common volatile periods (e.g., market open, lunch, close)
        df['market_open_period'] = (df['time_of_day'] >= 9.5) & (df['time_of_day'] <= 10.5)
        df['lunch_period'] = (df['time_of_day'] >= 12) & (df['time_of_day'] <= 13)
        df['market_close_period'] = (df['time_of_day'] >= 15) & (df['time_of_day'] <= 16)
        
        # Calculate hourly volatility patterns
        hour_vol = df.groupby('hour')['returns'].std().reset_index()
        hour_vol.columns = ['hour', 'hour_vol']
        df = pd.merge(df, hour_vol, on='hour', how='left')
        
        # Calculate relative volatility (current volatility / typical volatility for this hour)
        df['rel_hour_vol'] = df['volatility'] / df['hour_vol']
        
        # Gap analysis for day trading
        df['overnight_gap'] = df['open'] / df['close'].shift(1) - 1
        df['gap_zscore'] = (df['overnight_gap'] - df['overnight_gap'].rolling(window=20).mean()) / df['overnight_gap'].rolling(window=20).std()
        
        return df
    
    def _add_weekly_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weekly/seasonal patterns for swing trading"""
        # Only process if we have datetime index with date component
        if not hasattr(df.index, 'dayofweek'):
            return df
            
        # Extract day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df.index.dayofweek
        
        # Flag EIA report days (typically Thursdays)
        df['eial_report_day'] = df['day_of_week'] == 3  # Thursday
        
        # Calculate day-of-week volatility patterns
        dow_vol = df.groupby('day_of_week')['returns'].std().reset_index()
        dow_vol.columns = ['day_of_week', 'dow_vol']
        df = pd.merge(df, dow_vol, on='day_of_week', how='left')
        
        # Calculate relative volatility for day of week
        df['rel_dow_vol'] = df['volatility'] / df['dow_vol']
        
        # Month-end effect (last 3 days of month)
        df['month_end'] = df.index.is_month_end | (df.index + pd.Timedelta(days=1)).is_month_end | (df.index + pd.Timedelta(days=2)).is_month_end
        
        return df
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate volatility clustering signals"""
        if len(features) < self.config.medium_vol_window:
            return 0, 0, None
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            return self._calculate_day_trading_signals(current, features)
        else:
            return self._calculate_swing_trading_signals(current, features)
    
    def _calculate_swing_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate swing trading signals based on volatility clustering"""
        signals = {}
        
        # 1. Volatility regime signal
        vol_percentile = current.get('vol_percentile', 50)
        if vol_percentile > self.config.long_entry_vol_percentile:
            # Very high volatility - mean reversion expected (short signal)
            signals['volatility_regime'] = -1
        elif vol_percentile < self.config.short_entry_vol_percentile:
            # Very low volatility - expansion expected (long signal)
            signals['volatility_regime'] = 1
        else:
            signals['volatility_regime'] = 0
            
        # 2. GARCH forecast signal
        if 'garch_forecast' in current and not pd.isna(current['garch_forecast']):
            # Compare forecast to recent volatility
            recent_vol = features[f'hv_{self.config.short_vol_window}'].iloc[-5:].mean()
            forecast_ratio = current['garch_forecast'] / recent_vol
            
            if forecast_ratio > 1.2:  # Volatility expected to increase
                signals['garch_forecast'] = 1  # Long volatility
            elif forecast_ratio < 0.8:  # Volatility expected to decrease
                signals['garch_forecast'] = -1  # Short volatility
            else:
                signals['garch_forecast'] = 0
        else:
            signals['garch_forecast'] = 0
            
        # 3. Volatility ratio signal
        vol_ratio = current.get('vol_ratio_short_medium', 1)
        if vol_ratio > self.config.high_vol_ratio_threshold:
            # Short-term volatility is much higher - mean reversion expected
            signals['volatility_ratio'] = -1
        elif vol_ratio < self.config.low_vol_ratio_threshold:
            # Short-term volatility is much lower - expansion expected
            signals['volatility_ratio'] = 1
        else:
            signals['volatility_ratio'] = 0
            
        # 4. Jump detection signal
        if current['jump_detected']:
            # Recent jump - expect continued high volatility (long vol)
            signals['jump_detection'] = 1
        elif 0 < current['days_since_jump'] <= 3:
            # Recent jump (within 3 days) - still expect high volatility
            signals['jump_detection'] = 0.5
        else:
            signals['jump_detection'] = 0
            
        # 5. Momentum signal for confirmation
        if current['macd_vol_divergence']:
            # MACD diverging from volatility - potential reversal
            signals['momentum_signal'] = -np.sign(current['macd'])
        else:
            # Use RSI for momentum
            if current['rsi'] > self.config.rsi_overbought:
                signals['momentum_signal'] = -1  # Overbought - may reverse
            elif current['rsi'] < self.config.rsi_oversold:
                signals['momentum_signal'] = 1  # Oversold - may reverse
            else:
                signals['momentum_signal'] = 0
                
        # Combine signals
        weights = self.config.feature_weights['swing']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Increase confidence in high volatility regimes
            high_vol_boost = 1 + (vol_percentile / 100) * 0.5
            confidence = agreement_ratio * high_vol_boost
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.SWING_TRADE
    
    def _calculate_day_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate day trading signals based on intraday volatility patterns"""
        signals = {}
        
        # 1. Intraday volatility pattern signal
        rel_hour_vol = current.get('rel_hour_vol', 1)
        if pd.isna(rel_hour_vol):
            rel_hour_vol = 1
            
        if rel_hour_vol > 1.5:  # Much higher volatility than typical for this hour
            # Expect mean reversion in volatility
            signals['intraday_vol_pattern'] = -1
        elif rel_hour_vol < 0.5:  # Much lower volatility than typical
            # Expect increase in volatility
            signals['intraday_vol_pattern'] = 1
        else:
            signals['intraday_vol_pattern'] = 0
            
        # 2. Short-term volatility ratio
        # Use a faster version for intraday
        recent_vol = features['volatility'].iloc[-15:].std()  # Volatility of volatility
        avg_vol = features['volatility'].iloc[-30:].std()
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        
        if vol_ratio > 1.2:
            signals['short_term_vol_ratio'] = -1  # Expect mean reversion
        elif vol_ratio < 0.8:
            signals['short_term_vol_ratio'] = 1  # Expect expansion
        else:
            signals['short_term_vol_ratio'] = 0
            
        # 3. Volume volatility relationship
        vol_spike = current['volume_ratio'] > 2
        price_change = abs(current['returns']) > 0.01  # 1% move
        
        if vol_spike and price_change:
            # Volume and price both moving - expect continuation
            signals['volume_volatility'] = np.sign(current['returns'])
        elif vol_spike and not price_change:
            # Volume spike without price movement - expect breakout
            signals['volume_volatility'] = 1
        else:
            signals['volume_volatility'] = 0
            
        # 4. Momentum signal (similar to swing)
        if current['macd_vol_divergence']:
            signals['momentum_signal'] = -np.sign(current['macd'])
        else:
            if current['rsi'] > self.config.rsi_overbought:
                signals['momentum_signal'] = -1
            elif current['rsi'] < self.config.rsi_oversold:
                signals['momentum_signal'] = 1
            else:
                signals['momentum_signal'] = 0
                
        # 5. Gap analysis
        if 'gap_zscore' in current and not pd.isna(current['gap_zscore']):
            if abs(current['gap_zscore']) > 2:
                # Large gap - expect mean reversion
                signals['gap_analysis'] = -np.sign(current['overnight_gap'])
            else:
                signals['gap_analysis'] = 0
        else:
            signals['gap_analysis'] = 0
            
        # Combine signals
        weights = self.config.feature_weights['day']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Time-of-day adjustment
            time_factor = 1.0
            if current.get('market_open_period', False):
                time_factor = 1.2  # Higher confidence at open
            elif current.get('lunch_period', False):
                time_factor = 0.8  # Lower confidence during lunch
            
            confidence = agreement_ratio * time_factor
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.DAY_TRADE
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional validation for volatility clustering strategy"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Common validations
        validations = [
            # Avoid trading immediately after large jumps
            not current['jump_detected'],
            
            # Require minimum volatility for trading
            current.get(f'hv_{self.config.short_vol_window}', 0) > 0.001
        ]
        
        # Add timeframe-specific validations
        if is_intraday:
            day_validations = [
                # Avoid trading in extremely low volume
                current['volume_ratio'] > 0.5,
                
                # Avoid trading in extreme volatility without confirmation
                not (current.get('rel_hour_vol', 1) > 2 and abs(signal) < 0.7)
            ]
            validations.extend(day_validations)
        else:
            swing_validations = [
                # For swing trades, need stronger signals in high vol
                not (current.get('vol_percentile', 50) > 80 and abs(signal) < 0.7),
                
                # Require volume confirmation for swing trades
                current['volume_ratio'] > 1.0
            ]
            validations.extend(swing_validations)
            
        return all(validations)
    
    def calculate_position_size(self, base_size: float, features: pd.DataFrame) -> float:
        """Adjust position size based on volatility regime"""
        current = features.iloc[-1]
        vol_percentile = current.get('vol_percentile', 50)
        
        if vol_percentile > 80:
            # High volatility - reduce position size
            return base_size * self.config.high_vol_size_reduction
        elif vol_percentile < 20:
            # Low volatility - increase position size
            return base_size * self.config.low_vol_size_increase
        else:
            # Linear scaling between endpoints
            vol_factor = 1 - (vol_percentile - 20) / 60
            size_multiplier = self.config.low_vol_size_increase - (
                (self.config.low_vol_size_increase - self.config.high_vol_size_reduction) * vol_factor
            )
            return base_size * size_multiplier
    
    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update strategy parameters based on recent performance"""
        # Re-fit GARCH model with recent data
        if self.config.use_garch and len(features) >= 100:
            returns = features['returns'].dropna().values * 100
            model = arch_model(
                returns, 
                vol='Garch', 
                p=self.config.garch_p, 
                q=self.config.garch_q
            )
            self.garch_model = model.fit(disp='off')
            self.last_garch_update = features.index[-1]