import numpy as np
import pandas as pd
from typing import Tuple, Dict

class TechnicalIndicators:
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        v = df['volume']
        tp = (df['high'] + df['low'] + df['close']) / 3
        return (tp * v).cumsum() / v.cumsum()

    @staticmethod
    def calculate_bollinger_bands(
        close: pd.Series, 
        period: int = 20, 
        num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_zscore(series: pd.Series, period: int) -> pd.Series:
        """Calculate rolling Z-Score"""
        mean = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return (series - mean) / std

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_volume_surge_factor(
        volume: pd.Series, 
        period: int = 20
    ) -> pd.Series:
        """Calculate Volume Surge Factor"""
        vol_ma = volume.rolling(window=period).mean()
        return volume / vol_ma

    @staticmethod
    def calculate_price_momentum(
        close: pd.Series, 
        period: int
    ) -> pd.Series:
        """Calculate Time-Weighted Price Momentum"""
        returns = close.pct_change()
        return TechnicalIndicators.calculate_zscore(returns, period)

    @staticmethod
    def calculate_volatility_regime(
        close: pd.Series, 
        short_period: int = 5,
        long_period: int = 20
    ) -> pd.Series:
        """Detect Volatility Regime"""
        short_vol = close.pct_change().rolling(window=short_period).std()
        long_vol = close.pct_change().rolling(window=long_period).std()
        return short_vol / long_vol

    @staticmethod
    def calculate_support_resistance(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Dynamic Support and Resistance Levels"""
        # Simple implementation using rolling max/min
        resistance = high.rolling(window=period).max()
        support = low.rolling(window=period).min()
        return support, resistance

    @staticmethod
    def calculate_breakout_intensity(
        close: pd.Series,
        atr: pd.Series,
        threshold: float = 1.5
    ) -> pd.Series:
        """Calculate Breakout Intensity relative to ATR"""
        price_change = close.diff().abs()
        return price_change / atr > threshold

    @staticmethod
    def calculate_bollinger_squeeze(
        close: pd.Series,
        period: int = 20,
        num_std: float = 2
    ) -> pd.Series:
        """Calculate Bollinger Band Squeeze Factor"""
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(
            close, period, num_std
        )
        band_width = (upper - lower) / middle
        return band_width
    
    @staticmethod
    def calculate_directional_movement(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        # Calculate True Range
        tr = pd.DataFrame(index=high.index)
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        # Calculate Directional Movement
        up = high - high.shift(1)
        down = low.shift(1) - low
        
        pos_dm = ((up > down) & (up > 0)).astype(float) * up
        neg_dm = ((down > up) & (down > 0)).astype(float) * down
        
        # Smooth DM and TR
        tr_s = tr['tr'].rolling(window=period).mean()
        pos_dm_s = pos_dm.rolling(window=period).mean()
        neg_dm_s = neg_dm.rolling(window=period).mean()
        
        # Calculate DI
        pos_di = 100 * pos_dm_s / tr_s
        neg_di = 100 * neg_dm_s / tr_s
        
        # Calculate ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    

    @staticmethod
    def calculate_cumulative_delta(
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Calculate Cumulative Delta (simplified version)"""
        # In real implementation, you'd use actual buy/sell volume
        direction = close.diff().apply(lambda x: 1 if x > 0 else -1)
        return (direction * volume).cumsum()
    
    @staticmethod
    def calculate_macd(
        close: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            close: Close price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        # Calculate EMAs
        fast_ema = close.ewm(span=fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Historical Volatility using rolling standard deviation of returns."""
        returns = close.pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)  # Annualized

    @staticmethod
    def calculate_ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series, conversion_period: int = 9, base_period: int = 26, lagging_period: int = 52, displacement: int = 26) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate Ichimoku Cloud components.
        
        Returns:
            Tuple of (conversion_line, base_line, leading_span_a, leading_span_b, lagging_span)
        """
        # Conversion Line (Tenkan-sen)
        conversion_line = (high.rolling(window=conversion_period).max() + low.rolling(window=conversion_period).min()) / 2
        
        # Base Line (Kijun-sen)
        base_line = (high.rolling(window=base_period).max() + low.rolling(window=base_period).min()) / 2
        
        # Leading Span A (Senkou Span A)
        leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
        
        # Leading Span B (Senkou Span B)
        leading_span_b = ((high.rolling(window=lagging_period).max() + low.rolling(window=lagging_period).min()) / 2).shift(displacement)
        
        # Lagging Span (Chikou Span)
        lagging_span = close.shift(-displacement)
        
        return conversion_line, base_line, leading_span_a, leading_span_b, lagging_span
    
    