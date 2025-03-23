from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
import pytz

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class TimeCalendarConfig(TechnicalConfig):
    """Configuration for time of day and calendar-based strategy"""
    # Time of day parameters
    market_open_start: time = time(6, 0)  # 9:00 AM ET
    market_open_end: time = time(5, 30)   # 9:30 AM ET
    midday_start: time = time(12, 0)      # 12:00 PM ET
    midday_end: time = time(13, 0)        # 1:00 PM ET
    market_close_start: time = time(14, 30)  # 2:30 PM ET
    market_close_end: time = time(15, 0)     # 3:00 PM ET
    
    # EIA report parameters (Thursday 10:30 AM ET)
    eia_report_day: int = 3  # Thursday (0=Monday, 6=Sunday)
    eia_report_hour: int = 10
    eia_report_minute: int = 30
    eia_pre_window_minutes: int = 30  # Time window before report
    eia_post_window_minutes: int = 60  # Time window after report
    
    # Calendar event parameters
    winter_months: List[int] = None  # [11, 12, 1, 2, 3]
    summer_months: List[int] = None  # [6, 7, 8]
    shoulder_months: List[int] = None  # [4, 5, 9, 10]
    
    # Contract roll parameters
    days_before_expiry: int = 3  # Trading days before contract expiry
    contract_roll_volatility_factor: float = 1.5  # Expected volatility increase
    
    # Volatility thresholds
    normal_volatility_threshold: float = 0.015  # 1.5% volatility
    high_volatility_threshold: float = 0.025   # 2.5% volatility
    
    # Trading parameters for different time periods
    open_session_factor: float = 1.2  # More aggressive at open
    midday_session_factor: float = 0.8  # Less aggressive midday
    close_session_factor: float = 1.0  # Normal at close
    eia_report_factor: float = 1.5  # More aggressive around EIA report
    
    # Seasonal trading parameters
    winter_factor: float = 1.3  # More aggressive in winter
    summer_factor: float = 1.1  # Slightly more aggressive in summer
    shoulder_factor: float = 0.9  # Less aggressive in shoulder months
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        
        # Set default values for seasonal months if not provided
        if self.winter_months is None:
            self.winter_months = [11, 12, 1, 2, 3]
        if self.summer_months is None:
            self.summer_months = [6, 7, 8]
        if self.shoulder_months is None:
            self.shoulder_months = [4, 5, 9, 10]
            
        if self.feature_weights is None:
            self.feature_weights = {
                'day': {
                    'time_of_day': 0.30,
                    'volume_profile': 0.20,
                    'price_momentum': 0.20,
                    'order_flow': 0.15,
                    'liquidity': 0.15
                },
                'swing': {
                    'seasonal_factor': 0.25,
                    'contract_roll': 0.20,
                    'weekly_pattern': 0.20,
                    'storage_report': 0.20,
                    'weather_impact': 0.15
                }
            }

class TimeCalendarStrategy(TechnicalStrategy):
    """
    Strategy based on time of day patterns and calendar events in natural gas futures.
    
    This strategy focuses on:
    1. Intraday time-based patterns (open, midday, close)
    2. Scheduled events (EIA reports)
    3. Seasonal patterns (winter, summer, shoulder months)
    4. Contract roll effects
    """
    
    def __init__(self, config: TimeCalendarConfig = None):
        super().__init__(name="time_calendar", config=config or TimeCalendarConfig())
        self.config: TimeCalendarConfig = self.config
        
        # Timezone for consistent time handling (ET for US natural gas)
        self.timezone = pytz.timezone('US/Eastern')
        
        # Store upcoming calendar events
        self.upcoming_events = {}
        
        # Store recent EIA report results
        self.recent_eia_reports = []
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time and calendar-based features"""
        df = self.prepare_base_features(data)
        
        # Add time of day features
        df = self._add_time_of_day_features(df)
        
        # Add calendar-based features
        df = self._add_calendar_features(df)
        
        # Add EIA report features
        df = self._add_eia_report_features(df)
        
        # Add contract roll features
        df = self._add_contract_roll_features(df)
        
        # Add seasonal features
        df = self._add_seasonal_features(df)
        
        # Add liquidity and order flow features if available
        if all(col in df.columns for col in ['bid_volume', 'ask_volume']):
            df = self._add_order_flow_features(df)
        
        return df
    
    def _add_time_of_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time of day features"""
        if not hasattr(df.index, 'hour'):
            # Not a datetime index or not intraday data
            return df
        
        # Extract time components
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['time_of_day'] = df['hour'] + df['minute'] / 60
        
        # Session flags
        df['is_pre_market'] = (df['hour'] < 9) | ((df['hour'] == 9) & (df['minute'] < 0))
        df['is_market_open'] = ((df['hour'] == 9) & (df['minute'] >= 0)) | ((df['hour'] == 9) & (df['minute'] <= 30))
        df['is_midday'] = (df['hour'] >= 12) & (df['hour'] < 13)
        df['is_market_close'] = (df['hour'] >= 14) & (df['hour'] <= 15) & (df['minute'] <= 0)
        
        # Time since and until major market times
        current_day_start = pd.Timestamp(df.index[-1].date(), tz=df.index[-1].tz)
        
        market_open = current_day_start + pd.Timedelta(hours=9)
        df['time_since_open'] = (df.index - market_open).total_seconds() / 3600  # in hours
        
        market_close = current_day_start + pd.Timedelta(hours=15)
        df['time_to_close'] = (market_close - df.index).total_seconds() / 3600  # in hours
        
        # Intraday volatility measures
        if len(df) > 20:
            # 1-hour rolling volatility
            df['vol_1h'] = df['returns'].rolling(window=12).std() * np.sqrt(252 * 6.5)  # Annualized
            
            # 5-minute rolling volatility
            df['vol_5m'] = df['returns'].rolling(window=5).std() * np.sqrt(252 * 6.5 * 12)  # Annualized
            
            # Volatility ratio (short-term to 1-hour)
            df['vol_ratio'] = df['vol_5m'] / df['vol_1h']
        
        return df
    
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features"""
        if not hasattr(df.index, 'dayofweek'):
            # Not a datetime index
            return df
        
        # Day of week features
        df['day_of_week'] = df.index.dayofweek
        
        # Flag for EIA report day (Thursday)
        df['is_eia_report_day'] = df['day_of_week'] == self.config.eia_report_day
        
        # Day of month and end-of-month flag
        df['day_of_month'] = df.index.day
        df['is_month_end'] = df.index.is_month_end
        
        # First and last 5 days of the month
        df['is_month_start_period'] = df['day_of_month'] <= 5
        df['is_month_end_period'] = df['day_of_month'] >= 25
        
        # Get previous EIA report times
        self._update_eia_report_times(df)
        
        # Flag for being close to EIA report
        df['near_eia_report'] = False
        
        for report_time in self.recent_eia_reports:
            # Flag periods within the configured windows before and after the report
            pre_window = pd.Timedelta(minutes=self.config.eia_pre_window_minutes)
            post_window = pd.Timedelta(minutes=self.config.eia_post_window_minutes)
            
            df.loc[(df.index >= (report_time - pre_window)) & 
                  (df.index <= (report_time + post_window)), 'near_eia_report'] = True
        
        return df
    
    def _add_eia_report_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to EIA Natural Gas Storage Report"""
        if not df['near_eia_report'].any():
            return df
        
        # Find EIA report reactions
        df['eia_reaction'] = 0
        df['eia_report_candle'] = False
        
        for report_time in self.recent_eia_reports:
            # Define the candle that contains the report time
            try:
                report_candle_idx = df.index.get_indexer([report_time], method='nearest')[0]
                
                if report_candle_idx < len(df) - 1:  # Ensure we have data after report
                    df.loc[df.index[report_candle_idx], 'eia_report_candle'] = True
                    
                    # Calculate reaction over the next several candles
                    pre_report_price = df['close'].iloc[report_candle_idx - 1]
                    post_report_prices = df['close'].iloc[report_candle_idx:report_candle_idx + 5]
                    
                    # Calculate price changes from pre-report to post-report
                    for i in range(len(post_report_prices)):
                        if report_candle_idx + i < len(df):
                            price_change = post_report_prices.iloc[i] / pre_report_price - 1
                            df.loc[df.index[report_candle_idx + i], 'eia_reaction'] = price_change
            except:
                # Handle any issues with indexing
                continue
        
        # Calculate historical EIA report volatility
        if 'eia_report_candle' in df.columns and df['eia_report_candle'].any():
            report_candles = df[df['eia_report_candle']].index
            
            # Calculate average post-report volatility
            post_report_vols = []
            
            for report_time in report_candles:
                idx = df.index.get_loc(report_time)
                
                if idx + 5 < len(df):  # Ensure we have enough data after report
                    post_vol = df['returns'].iloc[idx:idx+5].std() * np.sqrt(252)
                    post_report_vols.append(post_vol)
            
            if post_report_vols:
                df['avg_eia_volatility'] = np.mean(post_report_vols)
            else:
                df['avg_eia_volatility'] = df['volatility']
        
        return df
    
    def _add_contract_roll_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contract roll-related features"""
        # This would require actual contract expiration data
        # For now, we'll simulate with month-end as a proxy for contract roll
        
        # Flag for being in the contract roll period
        df['near_contract_roll'] = df['is_month_end_period']
        
        # If we had actual expiry dates, we would calculate actual days to expiry
        # For now, we'll use a proxy based on day of month
        df['days_to_month_end'] = pd.DatetimeIndex(df.index + pd.DateOffset(months=1)).day - df.index.day
        
        # Simulate days to expiry
        df['simulated_days_to_expiry'] = df['days_to_month_end']
        df.loc[df['simulated_days_to_expiry'] > 20, 'simulated_days_to_expiry'] = 100  # Far from expiry
        
        # Flag for being within the expiry window
        df['in_expiry_window'] = df['simulated_days_to_expiry'] <= self.config.days_before_expiry
        
        return df
    
    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal features"""
        if not hasattr(df.index, 'month'):
            return df
        
        # Add month feature
        df['month'] = df.index.month
        
        # Seasonal flags
        df['is_winter'] = df['month'].isin(self.config.winter_months)
        df['is_summer'] = df['month'].isin(self.config.summer_months)
        df['is_shoulder'] = df['month'].isin(self.config.shoulder_months)
        
        # Seasonal factor (will be used for position sizing)
        df['seasonal_factor'] = 1.0
        df.loc[df['is_winter'], 'seasonal_factor'] = self.config.winter_factor
        df.loc[df['is_summer'], 'seasonal_factor'] = self.config.summer_factor
        df.loc[df['is_shoulder'], 'seasonal_factor'] = self.config.shoulder_factor
        
        return df
    
    def _add_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity and order flow features"""
        # Calculate bid-ask spread
        if 'bid' in df.columns and 'ask' in df.columns:
            df['bid_ask_spread'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)
            
            # Normalize bid-ask spread by time of day
            if 'hour' in df.columns:
                # Group by hour and calculate average spread
                hourly_spread = df.groupby('hour')['bid_ask_spread'].mean()
                
                # Merge back to df
                df = pd.merge(df, hourly_spread.rename('avg_hourly_spread'), 
                              left_on='hour', right_index=True, how='left')
                
                # Calculate relative spread (current spread / average spread for this hour)
                df['relative_spread'] = df['bid_ask_spread'] / df['avg_hourly_spread']
        
        # Calculate volume profile
        if 'volume' in df.columns and 'hour' in df.columns:
            # Group by hour and calculate average volume
            hourly_volume = df.groupby('hour')['volume'].mean()
            
            # Merge back to df
            df = pd.merge(df, hourly_volume.rename('avg_hourly_volume'), 
                          left_on='hour', right_index=True, how='left')
            
            # Calculate relative volume (current volume / average volume for this hour)
            df['relative_volume'] = df['volume'] / df['avg_hourly_volume']
        
        # Calculate order flow imbalance
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
            
            # Calculate cumulative order flow
            df['cumulative_flow'] = df['order_imbalance'].cumsum()
        
        return df
    
    def _update_eia_report_times(self, df: pd.DataFrame) -> None:
        """Update the list of recent EIA report times from the data"""
        # Find Thursdays in the data
        thursdays = df[df['day_of_week'] == self.config.eia_report_day].index
        
        # For each Thursday, add the report time (10:30 AM ET)
        self.recent_eia_reports = []
        for thursday in thursdays:
            # Create a timestamp for the report time on this Thursday
            report_time = pd.Timestamp(
                year=thursday.year,
                month=thursday.month,
                day=thursday.day,
                hour=self.config.eia_report_hour,
                minute=self.config.eia_report_minute,
                tz=thursday.tz
            )
            
            self.recent_eia_reports.append(report_time)
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate time and calendar-based signals"""
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
        """Calculate day trading signals based on time of day"""
        signals = {}
        
        # Time of day signal (based on historical patterns)
        time_signal = self._calculate_time_of_day_signal(current)
        signals['time_of_day'] = time_signal
        
        # Volume profile signal
        if 'relative_volume' in current:
            volume_signal = np.clip(current['relative_volume'] - 1, -1, 1)
            signals['volume_profile'] = volume_signal
        else:
            signals['volume_profile'] = 0
        
        # Price momentum signal
        # For intraday, look at shorter-term momentum
        if 'vwap' in current:
            # VWAP-based momentum
            vwap_deviation = (current['close'] - current['vwap']) / current['atr']
            signals['price_momentum'] = np.clip(vwap_deviation, -1, 1)
        else:
            # Use returns-based momentum
            momentum = features['returns'].rolling(window=5).sum().iloc[-1] / (features['volatility'].iloc[-1] + 1e-8)
            signals['price_momentum'] = np.clip(momentum * 5, -1, 1)
        
        # Order flow signal
        if 'order_imbalance' in current:
            signals['order_flow'] = np.clip(current['order_imbalance'] * 3, -1, 1)
        else:
            signals['order_flow'] = 0
        
        # Liquidity signal (inverse of relative spread)
        if 'relative_spread' in current:
            # Lower spread (higher liquidity) is better for executing
            liquidity_signal = np.clip(1 - current['relative_spread'], -1, 1)
            signals['liquidity'] = liquidity_signal
        else:
            signals['liquidity'] = 0
        
        # EIA report special case - adjust signals if we're near a report
        if current.get('near_eia_report', False):
            # Reduce all signals as we approach the report time
            for key in signals:
                signals[key] *= 0.5
                
            # If we have an actual report candle or reaction, it trumps other signals
            if current.get('eia_report_candle', False) or abs(current.get('eia_reaction', 0)) > 0.01:
                # Use the direction of the EIA reaction as the signal
                eia_signal = np.sign(current.get('eia_reaction', 0)) * min(1, abs(current.get('eia_reaction', 0)) * 20)
                signals['time_of_day'] = eia_signal
                
                # Amplify signal weight for EIA reactions
                weights = self.config.feature_weights['day'].copy()
                weights['time_of_day'] = 0.6  # Increase weight of EIA signal
                weights_sum = sum(weights.values())
                weights = {k: v / weights_sum for k, v in weights.items()}  # Normalize
            else:
                weights = self.config.feature_weights['day']
        else:
            weights = self.config.feature_weights['day']
        
        # Combine signals
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Adjust confidence based on time of day
            time_factor = 1.0
            if current.get('is_market_open', False):
                time_factor = self.config.open_session_factor
            elif current.get('is_midday', False):
                time_factor = self.config.midday_session_factor
            elif current.get('is_market_close', False):
                time_factor = self.config.close_session_factor
                
            # Adjust for EIA report
            if current.get('near_eia_report', False):
                time_factor *= self.config.eia_report_factor
                
            confidence = agreement_ratio * time_factor
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.DAY_TRADE
    
    def _calculate_time_of_day_signal(self, current: pd.Series) -> float:
        """Calculate signal based on time of day patterns"""
        
        # Default to no signal
        signal = 0
        
        # Market open pattern (typically bullish in natural gas)
        if current.get('is_market_open', False):
            signal = 0.5  # Moderate bullish bias at open
            
            # If we have strong momentum, follow it with amplified signal
            if 'returns' in current and abs(current['returns']) > 0.005:  # 0.5% move
                signal = np.sign(current['returns']) * 0.8
        
        # Midday pattern (typically mean-reversion)
        elif current.get('is_midday', False):
            # Mean reversion signal - go against recent momentum
            if 'returns' in current:
                signal = -np.sign(current['returns']) * 0.4
        
        # Market close pattern (typically continuation then reversal)
        elif current.get('is_market_close', False):
            # In the last 15 minutes, look for reversal
            is_last_15min = current.get('hour', 0) == 14 and current.get('minute', 0) >= 45
            
            if is_last_15min:
                # Look for reversal of intraday trend
                if 'returns' in current:
                    daily_return = current['close'] / current['open'] - 1
                    signal = -np.sign(daily_return) * 0.6
            else:
                # Before the last 15 minutes, continuation is more likely
                if 'returns' in current:
                    signal = np.sign(current['returns']) * 0.3
        
        return signal
    
    def _calculate_swing_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate swing trading signals based on calendar events"""
        signals = {}
        
        # Seasonal factor signal
        seasonal_signal = 0
        if current.get('is_winter', False):
            # Winter typically bullish for natural gas
            seasonal_signal = 0.5
        elif current.get('is_shoulder', False):
            # Shoulder months typically bearish
            seasonal_signal = -0.3
            
        signals['seasonal_factor'] = seasonal_signal
        
        # Contract roll signal
        if current.get('in_expiry_window', False):
            # Approaching contract expiry - typically bearish for front month
            roll_signal = -0.4
            
            # If volume is dropping significantly, strengthen the signal
            if 'volume_ratio' in current and current['volume_ratio'] < 0.7:
                roll_signal = -0.7
                
            signals['contract_roll'] = roll_signal
        else:
            signals['contract_roll'] = 0
        
        # Weekly pattern signal (e.g., "Turnaround Tuesday")
        day_of_week = current.get('day_of_week', -1)
        
        if day_of_week == 1:  # Tuesday
            # Check if Monday was a significant move
            if len(features) > 1:
                monday_return = features['returns'].iloc[-2]
                if abs(monday_return) > 0.01:  # 1% move
                    # Reverse Monday's direction on Tuesday
                    signals['weekly_pattern'] = -np.sign(monday_return) * 0.4
                else:
                    signals['weekly_pattern'] = 0
            else:
                signals['weekly_pattern'] = 0
        elif day_of_week == 4:  # Friday
            # Risk-off behavior often seen on Fridays
            signals['weekly_pattern'] = -0.2
        else:
            signals['weekly_pattern'] = 0
        
        # Storage report impact
        if current.get('is_eia_report_day', False):
            # On report day, anticipatory positioning based on market consensus
            if 'volatility' in current and current['volatility'] > self.config.normal_volatility_threshold:
                # Higher volatility indicates uncertainty, which can lead to stronger moves
                signals['storage_report'] = np.sign(features['returns'].rolling(window=3).sum().iloc[-1]) * 0.5
            else:
                signals['storage_report'] = 0.1  # Small bullish bias on report days
        else:
            signals['storage_report'] = 0
        
        # Weather impact (simulated - would be based on actual weather data)
        if current.get('is_winter', False):
            # In winter, colder weather forecasts are bullish
            # Here we're just using a proxy, but would ideally use actual weather data
            signals['weather_impact'] = 0.2
        elif current.get('is_summer', False):
            # In summer, hotter weather forecasts are bullish
            signals['weather_impact'] = 0.1
        else:
            signals['weather_impact'] = 0
        
        # Combine signals
        weights = self.config.feature_weights['swing']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Adjust confidence based on seasonal factors
            seasonal_factor = current.get('seasonal_factor', 1.0)
            
            confidence = agreement_ratio * seasonal_factor
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.SWING_TRADE
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Additional time and calendar-specific validation"""
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Common validations
        validations = [
            # Signal strength must be meaningful
            abs(signal) >= 0.2
        ]
        
        # Add timeframe-specific validations
        if is_intraday:
            # For day trading
            day_validations = [
                # Avoid trading during lunch hour unless signal is very strong
                not (current.get('is_midday', False) and abs(signal) < 0.7),
                
                # Avoid trading right before EIA reports
                not (current.get('near_eia_report', False) and 
                     not current.get('eia_report_candle', False)),
                
                # Require sufficient liquidity
                current.get('relative_volume', 1.0) >= 0.7
            ]
            validations.extend(day_validations)
        else:
            # For swing trading
            swing_validations = [
                # Avoid initiating positions right before contract expiry
                not (current.get('simulated_days_to_expiry', 100) <= 1),
                
                # For bearish positions in winter, require stronger signals
                not (current.get('is_winter', False) and signal < 0 and abs(signal) < 0.5)
            ]
            validations.extend(swing_validations)
            
        return all(validations)
    
    def adjust_position_size(
        self, 
        base_size: float, 
        features: pd.DataFrame
    ) -> float:
        """Adjust position size based on time and seasonal factors"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Start with base size
        adjusted_size = base_size
        
        if is_intraday:
            # Time of day adjustments
            if current.get('is_market_open', False):
                adjusted_size *= self.config.open_session_factor
            elif current.get('is_midday', False):
                adjusted_size *= self.config.midday_session_factor
            elif current.get('is_market_close', False):
                adjusted_size *= self.config.close_session_factor
                
            # EIA report adjustment
            if current.get('near_eia_report', False):
                if current.get('eia_report_candle', False):
                    adjusted_size *= self.config.eia_report_factor
                else:
                    # Reduce size before report release
                    adjusted_size *= 0.5
        else:
            # Seasonal adjustments for swing trading
            seasonal_factor = current.get('seasonal_factor', 1.0)
            adjusted_size *= seasonal_factor
            
            # Contract roll adjustment
            if current.get('in_expiry_window', False):
                adjusted_size *= 0.7  # Reduce size during contract roll
                
            # EIA report day adjustment
            if current.get('is_eia_report_day', False):
                adjusted_size *= 0.8  # Reduce size on report days for swing trades
        
        return adjusted_size
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        signal: float, 
        features: pd.DataFrame
    ) -> float:
        """Calculate dynamic stop loss level based on time and calendar factors"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Get ATR for volatility-based stops
        atr = current.get('atr', entry_price * 0.01)  # Default to 1% if ATR not available
        
        if is_intraday:
            # Base stop distance
            stop_distance = atr * self.config.day_atr_multiplier
            
            # Adjust based on time of day
            if current.get('is_market_open', False):
                # Wider stops during volatile open
                stop_distance *= 1.2
            elif current.get('is_midday', False):
                # Tighter stops during less volatile midday
                stop_distance *= 0.8
                
            # Adjust for EIA reports
            if current.get('near_eia_report', False):
                # Wider stops around reports
                stop_distance *= 1.5
        else:
            # Base stop distance for swing trades
            stop_distance = atr * self.config.swing_atr_multiplier
            
            # Adjust based on seasonal factors
            if current.get('is_winter', False):
                # Wider stops in more volatile winter months
                stop_distance *= 1.2
            elif current.get('is_shoulder', False):
                # Tighter stops in less volatile shoulder months
                stop_distance *= 0.9
                
            # Adjust for contract roll
            if current.get('in_expiry_window', False):
                # Wider stops during volatile roll periods
                stop_distance *= 1.3
        
        # Calculate stop level based on direction
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
        """Calculate take profit level based on time and calendar factors"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Calculate risk (distance to stop)
        risk = abs(entry_price - stop_loss)
        
        if is_intraday:
            # Default reward:risk
            reward_risk_ratio = 1.5
            
            # Adjust based on time of day
            if current.get('is_market_open', False):
                # Higher targets during trending open
                reward_risk_ratio = 2.0
            elif current.get('is_midday', False):
                # Lower targets during choppy midday
                reward_risk_ratio = 1.2
                
            # Adjust for EIA reports
            if current.get('near_eia_report', False) and current.get('eia_report_candle', False):
                # Higher targets for report volatility
                reward_risk_ratio = 2.5
        else:
            # Default reward:risk for swing trades
            reward_risk_ratio = 2.0
            
            # Adjust based on seasonal factors
            if current.get('is_winter', False):
                # Higher targets in trending winter months
                reward_risk_ratio = 2.5
            elif current.get('is_shoulder', False):
                # Lower targets in ranging shoulder months
                reward_risk_ratio = 1.8
                
            # Adjust for contract roll
            if current.get('in_expiry_window', False):
                # Lower targets during choppy roll periods
                reward_risk_ratio = 1.5
        
        # Calculate take profit level
        if signal > 0:  # Long position
            return entry_price + (risk * reward_risk_ratio)
        else:  # Short position
            return entry_price - (risk * reward_risk_ratio)
    
    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update strategy parameters based on recent performance"""
        # Could implement adaptive parameters based on recent market behavior
        # For example, adjusting time-of-day factors based on recent success rates
        pass