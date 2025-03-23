from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
import warnings
from scipy import stats

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class TimeSeriesConfig(TechnicalConfig):
    """Configuration for time series based strategy"""
    # ARIMA parameters
    ar_order: int = 2          # p in ARIMA(p,d,q)
    diff_order: int = 1        # d in ARIMA(p,d,q)
    ma_order: int = 1          # q in ARIMA(p,d,q)
    
    # SARIMA parameters (seasonal components)
    seasonal: bool = False     # Whether to use SARIMA instead of ARIMA
    seasonal_ar_order: int = 1  # P in SARIMA(p,d,q)(P,D,Q)m
    seasonal_diff_order: int = 1  # D in SARIMA(p,d,q)(P,D,Q)m
    seasonal_ma_order: int = 1  # Q in SARIMA(p,d,q)(P,D,Q)m
    seasonal_period: int = 5   # m in SARIMA(p,d,q)(P,D,Q)m (e.g., 5 for weekly, 12 for monthly)
    
    # Forecast parameters
    forecast_horizon_day: int = 5    # Number of periods to forecast for day trading
    forecast_horizon_swing: int = 10  # Number of periods to forecast for swing trading
    
    # Confidence interval for predictions
    confidence_level: float = 0.95  # 95% confidence interval
    
    # Stationarity test parameters
    adf_significance: float = 0.05  # Significance level for ADF test
    
    # Automated model selection
    auto_parameter_selection: bool = True  # Use AIC/BIC for model selection
    max_auto_ar: int = 5       # Maximum AR order to test in auto selection
    max_auto_diff: int = 2     # Maximum differencing order to test
    max_auto_ma: int = 5       # Maximum MA order to test
    
    # Thresholds for signal generation
    min_forecast_change_day: float = 0.03    # Min 3% forecasted change for day trading signal
    min_forecast_change_swing: float = 0.10   # Min 10% forecasted change for swing trading signal
    
    # Feature engineering parameters
    use_log_transform: bool = True   # Apply log transform to stabilize variance
    include_exogenous: bool = True   # Include exogenous variables if available
    
    # Trading parameters
    day_stop_loss_atr: float = 1.5   # Stop loss in ATR units (day trading)
    day_take_profit_atr: float = 3.0  # Take profit in ATR units (day trading)
    swing_stop_loss_atr: float = 2.0  # Stop loss in ATR units (swing trading)
    swing_take_profit_atr: float = 4.0  # Take profit in ATR units (swing trading)
    
    # Retraining frequency
    refit_frequency: int = 20  # Refit model every N periods
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_weights is None:
            self.feature_weights = {
                'day': {
                    'arima_forecast': 0.35,
                    'forecast_confidence': 0.25,
                    'price_momentum': 0.20,
                    'volume_confirmation': 0.10,
                    'volatility_regime': 0.10
                },
                'swing': {
                    'sarima_forecast': 0.35,
                    'seasonal_pattern': 0.25,
                    'trend_direction': 0.20,
                    'stationarity_measure': 0.10,
                    'model_accuracy': 0.10
                }
            }


class TimeSeriesStrategy(TechnicalStrategy):
    """
    Trading strategy based on time series forecasting models (ARIMA, SARIMA).
    
    This strategy uses statistical time series models to predict future price movements
    and generate trading signals based on forecasted changes.
    """
    
    def __init__(self, config: TimeSeriesConfig = None):
        super().__init__(name="time_series", config=config or TimeSeriesConfig())
        self.config: TimeSeriesConfig = self.config
        
        # Initialize models
        self.arima_model = None
        self.sarima_model = None
        
        # Track fitting status
        self.is_fitted = False
        self.last_fit_index = None
        self.fit_count = 0
        
        # Store forecast results
        self.forecasts = {
            'arima': None,
            'sarima': None,
            'forecast_dates': None,
            'confidence_intervals': None
        }
        
        # Performance metrics
        self.model_metrics = {
            'arima_aic': None,
            'sarima_aic': None,
            'forecast_accuracy': None,
            'last_stationarity_pvalue': None
        }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for time series analysis"""
        df = self.prepare_base_features(data)
        
        # Add time series specific features
        df = self._add_time_features(df)
        df = self._add_stationarity_features(df)
        df = self._test_and_transform_stationarity(df)
        
        # Generate forecasts if needed
        if self._should_refit_model(df):
            self._fit_and_forecast(df)
        
        # Add forecast-based features
        df = self._add_forecast_features(df)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features for seasonality detection"""
        if not hasattr(df.index, 'dayofweek'):
            return df
        
        # Add day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df.index.dayofweek
        
        # Add month (1-12)
        df['month'] = df.index.month
        
        # Add quarter (1-4)
        df['quarter'] = df.index.quarter
        
        # Add week of year (1-52)
        df['week_of_year'] = df.index.isocalendar().week
        
        # If intraday data, add hour
        if hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
            
            # Add session flags
            df['morning_session'] = (df['hour'] >= 9) & (df['hour'] < 12)
            df['afternoon_session'] = (df['hour'] >= 12) & (df['hour'] < 16)
            
            # Add time from market open/close (useful for intraday patterns)
            df['hours_since_open'] = df['hour'] - 9  # Assuming market opens at 9 AM
            df['hours_to_close'] = 16 - df['hour']   # Assuming market closes at 4 PM
        
        # Create seasonal dummies (for SARIMAX exogenous variables)
        if self.config.include_exogenous:
            # Day of week dummies
            for i in range(5):  # Monday to Friday
                df[f'day_{i}'] = (df['day_of_week'] == i).astype(int)
                
            # Month dummies (for seasonal patterns like winter demand)
            for i in range(1, 13):
                df[f'month_{i}'] = (df['month'] == i).astype(int)
        
        return df
    
    def _add_stationarity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features related to stationarity of the time series"""
        if len(df) < 30:  # Need enough data for meaningful calculations
            return df
        
        # Calculate rolling means and standard deviations
        df['rolling_mean_30'] = df['close'].rolling(window=30).mean()
        df['rolling_std_30'] = df['close'].rolling(window=30).std()
        
        # Calculate relative position to rolling mean
        df['close_to_mean_ratio'] = df['close'] / df['rolling_mean_30']
        
        # Calculate rate of change in different windows
        for window in [5, 10, 20]:
            df[f'roc_{window}'] = df['close'].pct_change(periods=window)
        
        # Calculate autocorrelation at different lags
        for lag in [1, 5, 10]:
            df[f'autocorr_lag_{lag}'] = df['close'].rolling(window=30).apply(
                lambda x: pd.Series(x).autocorr(lag=lag) if len(pd.Series(x).dropna()) > lag else 0
            )
        
        return df
    
    def _test_and_transform_stationarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Test for stationarity and apply transformations if needed"""
        if len(df) < 30:  # Need enough data
            return df
        
        # Copy price series
        price_series = df['close'].copy()
        
        # Apply log transform if configured
        if self.config.use_log_transform:
            price_series = np.log(price_series)
            df['log_close'] = price_series
        
        # Run Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(price_series.dropna())
            df['adf_pvalue'] = adf_result[1]  # p-value
            
            # Store for later use
            self.model_metrics['last_stationarity_pvalue'] = adf_result[1]
            
            # Flag if series is stationary
            df['is_stationary'] = adf_result[1] < self.config.adf_significance
        except:
            df['adf_pvalue'] = None
            df['is_stationary'] = False
        
        # Calculate differenced series (up to configured order)
        for d in range(1, self.config.max_auto_diff + 1):
            # Regular differencing
            df[f'diff_{d}'] = price_series.diff(d)
            
            # If seasonal, add seasonal differencing
            if self.config.seasonal and self.config.seasonal_period <= len(df) // 3:  # Need enough data
                try:
                    df[f'seasonal_diff_{d}'] = price_series.diff(self.config.seasonal_period * d)
                except:
                    pass
        
        return df
    
    def _should_refit_model(self, df: pd.DataFrame) -> bool:
        """Determine if models should be refitted"""
        # If never fitted, or no last fit index, then fit
        if not self.is_fitted or self.last_fit_index is None:
            return True
        
        # If current index is more than refit_frequency periods past last fit, then refit
        current_index = df.index[-1]
        if self.last_fit_index in df.index:
            last_fit_pos = df.index.get_loc(self.last_fit_index)
            current_pos = len(df) - 1
            periods_since_last_fit = current_pos - last_fit_pos
            
            return periods_since_last_fit >= self.config.refit_frequency
        
        return True
    
    def _fit_and_forecast(self, df: pd.DataFrame) -> None:
        """Fit time series models and generate forecasts"""
        # Need sufficient data
        if len(df) < max(30, self.config.seasonal_period * 3):
            return
        
        # Select series to model (original or transformed)
        if self.config.use_log_transform and 'log_close' in df.columns:
            series = df['log_close'].copy()
        else:
            series = df['close'].copy()
        
        # Select differencing based on stationarity test
        diff_order = self.config.diff_order
        if self.config.auto_parameter_selection and 'adf_pvalue' in df.columns:
            # If not stationary, increment differencing
            if df['adf_pvalue'].iloc[-1] > self.config.adf_significance:
                diff_order = min(self.config.diff_order + 1, self.config.max_auto_diff)
        
        # Select exogenous variables if configured
        exog = None
        if self.config.include_exogenous:
            exog_cols = [col for col in df.columns if col.startswith('day_') or col.startswith('month_')]
            if exog_cols:
                exog = df[exog_cols].copy()
        
        # Silence warning messages
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Fit ARIMA model
            try:
                if self.config.auto_parameter_selection:
                    # Try multiple models and select best (simplified grid search)
                    best_aic = float('inf')
                    best_order = (self.config.ar_order, diff_order, self.config.ma_order)
                    
                    for p in range(0, self.config.max_auto_ar + 1):
                        for q in range(0, self.config.max_auto_ma + 1):
                            if p == 0 and q == 0:
                                continue  # Skip invalid model
                                
                            try:
                                model = ARIMA(
                                    series, 
                                    order=(p, diff_order, q),
                                    enforce_stationarity=False
                                )
                                fit_res = model.fit()
                                
                                if fit_res.aic < best_aic:
                                    best_aic = fit_res.aic
                                    best_order = (p, diff_order, q)
                            except:
                                continue
                    
                    # Fit final model with best parameters
                    self.arima_model = ARIMA(
                        series, 
                        order=best_order,
                        enforce_stationarity=False
                    ).fit()
                else:
                    # Use configured parameters
                    self.arima_model = ARIMA(
                        series, 
                        order=(self.config.ar_order, diff_order, self.config.ma_order),
                        enforce_stationarity=False
                    ).fit()
                
                # Store AIC
                self.model_metrics['arima_aic'] = self.arima_model.aic
                
                # Generate forecast
                forecast_horizon = max(self.config.forecast_horizon_day, self.config.forecast_horizon_swing)
                arima_forecast = self.arima_model.get_forecast(steps=forecast_horizon)
                forecast_mean = arima_forecast.predicted_mean
                forecast_ci = arima_forecast.conf_int(alpha=1-self.config.confidence_level)
                
                # Store forecast results
                self.forecasts['arima'] = forecast_mean
                self.forecasts['forecast_dates'] = pd.date_range(
                    start=df.index[-1], 
                    periods=forecast_horizon+1, 
                    freq=pd.infer_freq(df.index)
                )[1:]  # Skip current date
                
                self.forecasts['confidence_intervals'] = forecast_ci
            except Exception as e:
                print(f"ARIMA model fitting error: {e}")
            
            # Fit SARIMA model if configured
            if self.config.seasonal:
                try:
                    if self.config.auto_parameter_selection:
                        # Try a few common seasonal models (simplified)
                        best_aic = float('inf')
                        best_order = (self.config.ar_order, diff_order, self.config.ma_order)
                        best_seasonal_order = (
                            self.config.seasonal_ar_order,
                            self.config.seasonal_diff_order,
                            self.config.seasonal_ma_order,
                            self.config.seasonal_period
                        )
                        
                        # Simplified grid search
                        for p in [0, 1, 2]:
                            for q in [0, 1]:
                                for P in [0, 1]:
                                    for Q in [0, 1]:
                                        if p == 0 and q == 0 and P == 0 and Q == 0:
                                            continue  # Skip invalid model
                                            
                                        try:
                                            model = SARIMAX(
                                                series,
                                                order=(p, diff_order, q),
                                                seasonal_order=(P, self.config.seasonal_diff_order, Q, self.config.seasonal_period),
                                                enforce_stationarity=False,
                                                exog=exog
                                            )
                                            fit_res = model.fit(disp=False)
                                            
                                            if fit_res.aic < best_aic:
                                                best_aic = fit_res.aic
                                                best_order = (p, diff_order, q)
                                                best_seasonal_order = (
                                                    P,
                                                    self.config.seasonal_diff_order,
                                                    Q,
                                                    self.config.seasonal_period
                                                )
                                        except:
                                            continue
                        
                        # Fit final model with best parameters
                        self.sarima_model = SARIMAX(
                            series,
                            order=best_order,
                            seasonal_order=best_seasonal_order,
                            enforce_stationarity=False,
                            exog=exog
                        ).fit(disp=False)
                    else:
                        # Use configured parameters
                        self.sarima_model = SARIMAX(
                            series,
                            order=(self.config.ar_order, diff_order, self.config.ma_order),
                            seasonal_order=(
                                self.config.seasonal_ar_order,
                                self.config.seasonal_diff_order, 
                                self.config.seasonal_ma_order,
                                self.config.seasonal_period
                            ),
                            enforce_stationarity=False,
                            exog=exog
                        ).fit(disp=False)
                    
                    # Store AIC
                    self.model_metrics['sarima_aic'] = self.sarima_model.aic
                    
                    # Generate forecast
                    # Need to forecast future exogenous variables if used
                    future_exog = None
                    if exog is not None:
                        # This is a simplified approach - in practice you'd need to properly forecast exogenous variables
                        future_exog = exog.tail(self.config.seasonal_period).reset_index(drop=True)
                        # If not enough periods, repeat the pattern
                        while len(future_exog) < forecast_horizon:
                            future_exog = pd.concat([future_exog, future_exog.iloc[0:self.config.seasonal_period]])
                        future_exog = future_exog.iloc[0:forecast_horizon]
                    
                    sarima_forecast = self.sarima_model.get_forecast(steps=forecast_horizon, exog=future_exog)
                    sarima_mean = sarima_forecast.predicted_mean
                    sarima_ci = sarima_forecast.conf_int(alpha=1-self.config.confidence_level)
                    
                    # Store forecast results
                    self.forecasts['sarima'] = sarima_mean
                    
                    # If ARIMA forecast is missing, use SARIMA dates
                    if self.forecasts['forecast_dates'] is None:
                        self.forecasts['forecast_dates'] = pd.date_range(
                            start=df.index[-1], 
                            periods=forecast_horizon+1, 
                            freq=pd.infer_freq(df.index)
                        )[1:]  # Skip current date
                except Exception as e:
                    print(f"SARIMA model fitting error: {e}")
        
        # Update fitting status
        self.is_fitted = (self.arima_model is not None) or (self.sarima_model is not None)
        self.last_fit_index = df.index[-1]
        self.fit_count += 1
    
    def _add_forecast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on model forecasts"""
        if not self.is_fitted:
            return df
            
        # Initialize forecast columns
        for horizon in [1, 5, 10]:
            if horizon <= len(self.forecasts.get('arima', [])):
                # ARIMA forecast
                df[f'arima_forecast_{horizon}'] = None
                df.loc[df.index[-1], f'arima_forecast_{horizon}'] = self.forecasts['arima'][horizon-1]
                
                # Calculate expected return
                if self.config.use_log_transform:
                    # For log-transformed data, need to exponentiate
                    expected_return = np.exp(self.forecasts['arima'][horizon-1]) / df['close'].iloc[-1] - 1
                else:
                    expected_return = self.forecasts['arima'][horizon-1] / df['close'].iloc[-1] - 1
                
                df[f'arima_expected_return_{horizon}'] = None
                df.loc[df.index[-1], f'arima_expected_return_{horizon}'] = expected_return
            
            if self.config.seasonal and horizon <= len(self.forecasts.get('sarima', [])):
                # SARIMA forecast
                df[f'sarima_forecast_{horizon}'] = None
                df.loc[df.index[-1], f'sarima_forecast_{horizon}'] = self.forecasts['sarima'][horizon-1]
                
                # Calculate expected return
                if self.config.use_log_transform:
                    # For log-transformed data, need to exponentiate
                    expected_return = np.exp(self.forecasts['sarima'][horizon-1]) / df['close'].iloc[-1] - 1
                else:
                    expected_return = self.forecasts['sarima'][horizon-1] / df['close'].iloc[-1] - 1
                
                df[f'sarima_expected_return_{horizon}'] = None
                df.loc[df.index[-1], f'sarima_expected_return_{horizon}'] = expected_return
        
        # Add model accuracy metrics
        if self.model_metrics.get('arima_aic') is not None:
            df['arima_aic'] = None
            df.loc[df.index[-1], 'arima_aic'] = self.model_metrics['arima_aic']
            
        if self.model_metrics.get('sarima_aic') is not None:
            df['sarima_aic'] = None
            df.loc[df.index[-1], 'sarima_aic'] = self.model_metrics['sarima_aic']
        
        # Add forecast confidence - narrower CI = higher confidence
        if self.forecasts.get('confidence_intervals') is not None:
            ci = self.forecasts['confidence_intervals']
            
            # Calculate width of CI as percentage of forecast
            for horizon in [1, 5, 10]:
                if horizon <= len(ci):
                    ci_width = ci.iloc[horizon-1, 1] - ci.iloc[horizon-1, 0]
                    
                    if self.config.use_log_transform:
                        # For log-transformed data
                        forecast = np.exp(self.forecasts['arima'][horizon-1])
                        ci_width_pct = (np.exp(ci_width) - 1)
                    else:
                        forecast = self.forecasts['arima'][horizon-1]
                        ci_width_pct = ci_width / forecast if forecast != 0 else float('inf')
                    
                    df[f'forecast_confidence_{horizon}'] = None
                    df.loc[df.index[-1], f'forecast_confidence_{horizon}'] = max(0, 1 - min(ci_width_pct, 1))
        
        return df
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate signals based on time series forecasts"""
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
        """
        Calculate day trading signals using shorter-term forecasts
        """
        signals = {}
        
        # 1. ARIMA Forecast Signal
        forecast_horizon = min(self.config.forecast_horizon_day, 5)  # Use 1-5 step ahead forecast for day trading
        forecast_col = f'arima_expected_return_{forecast_horizon}'
        
        if forecast_col in current and not pd.isna(current[forecast_col]):
            forecast_return = current[forecast_col]
            
            # Signal strength based on forecasted return
            if abs(forecast_return) > self.config.min_forecast_change_day:
                # Scale signal from 0 to 1 based on forecasted return
                max_expected_return = 0.15  # Cap at 15% for scaling
                signal_strength = np.sign(forecast_return) * min(abs(forecast_return) / max_expected_return, 1.0)
                signals['arima_forecast'] = signal_strength
            else:
                signals['arima_forecast'] = 0
        else:
            signals['arima_forecast'] = 0
            
        # 2. Forecast Confidence Signal
        confidence_col = f'forecast_confidence_{forecast_horizon}'
        if confidence_col in current and not pd.isna(current[confidence_col]):
            # Use confidence as a scalar for the forecast signal
            signals['forecast_confidence'] = np.sign(signals['arima_forecast']) * current[confidence_col]
        else:
            signals['forecast_confidence'] = 0
            
        # 3. Price Momentum Signal
        if 'roc_5' in current:
            # Short-term momentum
            momentum = current['roc_5']
            # Scale momentum to [-1, 1]
            signals['price_momentum'] = np.clip(momentum * 20, -1, 1)  # Scale by 20 to normalize
        else:
            signals['price_momentum'] = 0
            
        # 4. Volume Confirmation Signal
        if 'volume_ratio' in current:
            # Volume surge in same direction as forecast
            volume_signal = current['volume_ratio'] - 1  # Above 0 means higher than average volume
            signals['volume_confirmation'] = np.sign(signals['arima_forecast']) * min(volume_signal, 1.0)
        else:
            signals['volume_confirmation'] = 0
            
        # 5. Volatility Regime Signal
        if 'volatility' in current:
            # Calculate volatility z-score
            vol_mean = features['volatility'].iloc[-20:].mean()
            vol_std = features['volatility'].iloc[-20:].std()
            
            if vol_std > 0:
                vol_zscore = (current['volatility'] - vol_mean) / vol_std
                
                # In high volatility, reduce signal strength
                if vol_zscore > 1.0:
                    signals['volatility_regime'] = -np.sign(signals['arima_forecast']) * min(vol_zscore - 1, 1.0)
                else:
                    signals['volatility_regime'] = 0
            else:
                signals['volatility_regime'] = 0
        else:
            signals['volatility_regime'] = 0
        
        # Combine signals using weights
        weights = self.config.feature_weights['day']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Adjust confidence based on model metrics
            model_confidence = current.get(f'forecast_confidence_{forecast_horizon}', 0.5)
            
            confidence = agreement_ratio * 0.6 + model_confidence * 0.4
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.DAY_TRADE
    
    def _calculate_swing_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """
        Calculate swing trading signals using longer-term forecasts and seasonal patterns
        """
        signals = {}
        
        # 1. SARIMA Forecast Signal (if available, otherwise use ARIMA)
        forecast_horizon = min(self.config.forecast_horizon_swing, 10)  # Use 5-10 step ahead forecast
        
        if self.config.seasonal:
            forecast_col = f'sarima_expected_return_{forecast_horizon}'
        else:
            forecast_col = f'arima_expected_return_{forecast_horizon}'
            
        if forecast_col in current and not pd.isna(current[forecast_col]):
            forecast_return = current[forecast_col]
            
            # Signal strength based on forecasted return
            if abs(forecast_return) > self.config.min_forecast_change_swing:
                # Scale signal based on forecasted return
                max_expected_return = 0.25  # Cap at 25% for scaling
                signal_strength = np.sign(forecast_return) * min(abs(forecast_return) / max_expected_return, 1.0)
                signals['sarima_forecast'] = signal_strength
            else:
                signals['sarima_forecast'] = 0
        else:
            signals['sarima_forecast'] = 0
            
        # 2. Seasonal Pattern Signal
        if self.config.seasonal and 'month' in current:
            # Simplified seasonal analysis
            # For natural gas: generally bullish in winter, bearish in spring
            if current['month'] in [11, 12, 1, 2]:  # Winter months
                seasonal_bias = 0.3  # Bullish bias
            elif current['month'] in [3, 4, 5]:     # Spring months
                seasonal_bias = -0.3  # Bearish bias
            else:
                seasonal_bias = 0  # Neutral in other months
                
            signals['seasonal_pattern'] = seasonal_bias
        else:
            signals['seasonal_pattern'] = 0
            
        # 3. Trend Direction Signal
        if 'rolling_mean_30' in current and 'close' in current:
            # Trend direction based on relationship to moving average
            trend_direction = (current['close'] / current['rolling_mean_30']) - 1
            signals['trend_direction'] = np.clip(trend_direction * 10, -1, 1)  # Scale by 10 to normalize
        else:
            signals['trend_direction'] = 0
            
        # 4. Stationarity Measure Signal
        if 'adf_pvalue' in current:
            # Lower p-value means more stationary (better for mean-reversion)
            # Higher p-value means less stationary (better for trend-following)
            stationarity = 1 - min(current['adf_pvalue'], 1)
            
            # If forecast is for uptrend, low stationarity is good
            # If forecast is for downtrend, high stationarity is good
            signals['stationarity_measure'] = np.sign(signals['sarima_forecast']) * (0.5 - stationarity)
        else:
            signals['stationarity_measure'] = 0
            
        # 5. Model Accuracy Signal
        # Use AIC - lower is better
        if 'sarima_aic' in current and not pd.isna(current['sarima_aic']):
            # Scale AIC to a reasonable range (-100 to 100 is common)
            scaled_aic = min(max(current['sarima_aic'], -100), 100) / 100
            
            # Convert to signal (-1 to 1), lower AIC = higher signal
            model_quality = 0.5 - scaled_aic
            signals['model_accuracy'] = np.sign(signals['sarima_forecast']) * model_quality
        else:
            signals['model_accuracy'] = 0
        
        # Combine signals using weights
        weights = self.config.feature_weights['swing']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Adjust confidence based on model metrics
            forecast_confidence = current.get(f'forecast_confidence_{forecast_horizon}', 0.5)
            
            # Seasonality increases confidence for natural gas
            seasonal_boost = 1.2 if current['month'] in [11, 12, 1, 2, 6, 7, 8] else 1.0  # Winter and summer
            
            confidence = (agreement_ratio * 0.5 + forecast_confidence * 0.5) * seasonal_boost
        else:
            confidence = 0
            
        return total_signal, min(confidence, 1.0), TradeType.SWING_TRADE
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """
        Validate time series signals with additional criteria
        """
        if not self.validate_technical_signal(signal, features):
            return False
            
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Common validations
        validations = [
            # Signal strength must be meaningful
            abs(signal) >= 0.3,
            
            # Models must be fitted
            self.is_fitted
        ]
        
        # Add timeframe-specific validations
        if is_intraday:
            # Day trading validations
            forecast_horizon = min(self.config.forecast_horizon_day, 5)
            
            day_validations = [
                # Forecast must be significant
                abs(current.get(f'arima_expected_return_{forecast_horizon}', 0)) >= self.config.min_forecast_change_day,
                
                # Forecast confidence must be reasonable
                current.get(f'forecast_confidence_{forecast_horizon}', 0) > 0.4,
                
                # Don't trade against strong momentum
                not (np.sign(signal) != np.sign(current.get('roc_5', 0)) and 
                     abs(current.get('roc_5', 0)) > 0.03)  # 3% recent move
            ]
            validations.extend(day_validations)
        else:
            # Swing trading validations
            forecast_horizon = min(self.config.forecast_horizon_swing, 10)
            forecast_col = 'sarima_expected_return_' if self.config.seasonal else 'arima_expected_return_'
            
            swing_validations = [
                # Forecast must be significant
                abs(current.get(f'{forecast_col}{forecast_horizon}', 0)) >= self.config.min_forecast_change_swing,
                
                # For natural gas seasonal trades, higher confidence required in shoulder months
                not (current.get('month', 0) in [3, 4, 5, 9, 10] and  # Shoulder months
                     current.get(f'forecast_confidence_{forecast_horizon}', 0) < 0.6),
                     
                # Don't trade against strong long-term momentum
                not (np.sign(signal) != np.sign(current.get('roc_20', 0)) and 
                     abs(current.get('roc_20', 0)) > 0.1)  # 10% longer-term move
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
        Calculate stop loss level based on time series forecast confidence
        """
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Get ATR for volatility-based stop
        atr = current.get('atr', entry_price * 0.01)  # Default to 1% if ATR not available
        
        # Get forecast confidence
        if is_intraday:
            horizon = min(self.config.forecast_horizon_day, 5)
            confidence = current.get(f'forecast_confidence_{horizon}', 0.5)
            base_atr_multiplier = self.config.day_stop_loss_atr
        else:
            horizon = min(self.config.forecast_horizon_swing, 10)
            confidence = current.get(f'forecast_confidence_{horizon}', 0.5)
            base_atr_multiplier = self.config.swing_stop_loss_atr
        
        # Adjust ATR multiplier based on forecast confidence
        # Lower confidence = wider stop
        confidence_factor = 1 + (0.5 - min(confidence, 0.5))  # 1.0 to 1.5
        atr_multiplier = base_atr_multiplier * confidence_factor
        
        # Calculate stop distance
        stop_distance = atr * atr_multiplier
        
        # Apply stop based on signal direction
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
        Calculate take profit level based on forecast
        """
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        # Calculate risk (distance to stop)
        risk = abs(entry_price - stop_loss)
        
        # Default reward-to-risk ratio
        if is_intraday:
            base_reward_risk = self.config.day_take_profit_atr / self.config.day_stop_loss_atr
            horizon = min(self.config.forecast_horizon_day, 5)
            forecast_col = f'arima_expected_return_{horizon}'
        else:
            base_reward_risk = self.config.swing_take_profit_atr / self.config.swing_stop_loss_atr
            horizon = min(self.config.forecast_horizon_swing, 10)
            forecast_col = f'sarima_expected_return_{horizon}' if self.config.seasonal else f'arima_expected_return_{horizon}'
        
        # Check if we have a forecast
        forecast_return = current.get(forecast_col, None)
        
        if forecast_return is not None and not pd.isna(forecast_return):
            # Use forecast to determine take profit
            if self.config.use_log_transform:
                # For log-transformed forecasts
                forecast_price = entry_price * (1 + forecast_return)
            else:
                forecast_price = entry_price * (1 + forecast_return)
                
            # Get distance to forecast price
            forecast_distance = abs(forecast_price - entry_price)
            
            # Set take profit to forecast or standard reward-to-risk, whichever is larger
            standard_target = risk * base_reward_risk
            
            if signal > 0:  # Long
                return entry_price + max(forecast_distance, standard_target)
            else:  # Short
                return entry_price - max(forecast_distance, standard_target)
        else:
            # Use standard reward-to-risk if no forecast available
            if signal > 0:  # Long
                return entry_price + (risk * base_reward_risk)
            else:  # Short
                return entry_price - (risk * base_reward_risk)
    
    def plot_forecast(self, features: pd.DataFrame, forecast_horizon: int = None) -> plt.Figure:
        """
        Generate a plot of the time series forecast
        
        Args:
            features: DataFrame with features
            forecast_horizon: Number of periods to forecast (defaults to max horizon)
            
        Returns:
            Matplotlib figure with forecast plot
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Cannot plot forecast.")
            
        if forecast_horizon is None:
            forecast_horizon = max(self.config.forecast_horizon_day, self.config.forecast_horizon_swing)
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get price data
        price_data = features['close'].iloc[-30:]  # Last 30 periods
        
        # Plot historical data
        ax.plot(price_data.index, price_data.values, 'b-', label='Historical Price')
        
        # Plot forecast
        if self.forecasts['arima'] is not None:
            forecast_dates = self.forecasts['forecast_dates'][:forecast_horizon]
            forecast_values = self.forecasts['arima'][:forecast_horizon]
            
            if self.config.use_log_transform:
                # Convert log forecast back to prices
                forecast_values = np.exp(forecast_values)
            
            ax.plot(forecast_dates, forecast_values, 'r-', label='ARIMA Forecast')
            
            # Plot confidence intervals if available
            if self.forecasts['confidence_intervals'] is not None:
                ci = self.forecasts['confidence_intervals'][:forecast_horizon]
                
                if self.config.use_log_transform:
                    # Convert log CI back to prices
                    lower_ci = np.exp(ci.iloc[:, 0].values)
                    upper_ci = np.exp(ci.iloc[:, 1].values)
                else:
                    lower_ci = ci.iloc[:, 0].values
                    upper_ci = ci.iloc[:, 1].values
                
                ax.fill_between(forecast_dates, lower_ci, upper_ci, color='r', alpha=0.2, label='95% CI')
        
        # Plot SARIMA forecast if available
        if self.config.seasonal and self.forecasts['sarima'] is not None:
            sarima_values = self.forecasts['sarima'][:forecast_horizon]
            
            if self.config.use_log_transform:
                # Convert log forecast back to prices
                sarima_values = np.exp(sarima_values)
            
            ax.plot(forecast_dates, sarima_values, 'g-', label='SARIMA Forecast')
        
        # Customize plot
        ax.set_title('Natural Gas Price Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update time series model with new data"""
        # Force model refit
        self.last_fit_index = None
        self._fit_and_forecast(features)