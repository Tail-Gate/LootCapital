import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
import logging

class TechnicalIndicators:
    """
    Enhanced class for calculating technical indicators with configurable parameters.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TechnicalIndicators with optional configuration.
        
        Args:
            config: Configuration dictionary with feature engineering parameters
        """
        self.config = config or {}
        
        # Set default parameters if not provided in config
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast_period = self.config.get('macd_fast_period', 12)
        self.macd_slow_period = self.config.get('macd_slow_period', 26)
        self.macd_signal_period = self.config.get('macd_signal_period', 9)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_num_std_dev = self.config.get('bb_num_std_dev', 2.0)
        self.atr_period = self.config.get('atr_period', 14)
        self.adx_period = self.config.get('adx_period', 14)
        self.volume_ma_period = self.config.get('volume_ma_period', 20)
        self.price_momentum_lookback = self.config.get('price_momentum_lookback', 5)
    
    def calculate_rsi(self, data: Union[pd.Series, pd.DataFrame], period: Optional[int] = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Series or DataFrame containing price data
            period: RSI period (uses config if not provided)
            
        Returns:
            Series containing RSI values
        """
        if period is None:
            period = self.rsi_period
            
        # Handle both Series and DataFrame input
        if isinstance(data, pd.DataFrame):
            price_data = data['close']
        else:
            price_data = data
            
        delta = price_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def calculate_macd(self, data: Union[pd.Series, pd.DataFrame], 
                      fast_period: Optional[int] = None, 
                      slow_period: Optional[int] = None, 
                      signal_period: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: Series or DataFrame containing price data
            fast_period: Fast EMA period (uses config if not provided)
            slow_period: Slow EMA period (uses config if not provided)
            signal_period: Signal line period (uses config if not provided)
            
        Returns:
            DataFrame containing MACD, signal, and histogram values
        """
        if fast_period is None:
            fast_period = self.macd_fast_period
        if slow_period is None:
            slow_period = self.macd_slow_period
        if signal_period is None:
            signal_period = self.macd_signal_period
            
        # Handle both Series and DataFrame input
        if isinstance(data, pd.DataFrame):
            price_data = data['close']
        else:
            price_data = data
            
        exp1 = price_data.ewm(span=fast_period, adjust=False).mean()
        exp2 = price_data.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        return pd.DataFrame({'macd': macd, 'signal': signal, 'hist': hist})
        
    def calculate_bollinger_bands(self, data: Union[pd.Series, pd.DataFrame], 
                                period: Optional[int] = None, 
                                num_std: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Series or DataFrame containing price data
            period: Moving average period (uses config if not provided)
            num_std: Number of standard deviations (uses config if not provided)
            
        Returns:
            DataFrame containing middle, upper, and lower bands
        """
        if period is None:
            period = self.bb_period
        if num_std is None:
            num_std = self.bb_num_std_dev
            
        # Handle both Series and DataFrame input
        if isinstance(data, pd.DataFrame):
            price_data = data['close']
        else:
            price_data = data
            
        middle = price_data.rolling(window=period).mean()
        std = price_data.rolling(window=period).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower
        })
        
    def calculate_atr(self, data: Union[pd.DataFrame, pd.Series], 
                     high: Optional[pd.Series] = None,
                     low: Optional[pd.Series] = None,
                     close: Optional[pd.Series] = None,
                     period: Optional[int] = None) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame containing OHLC data or Series for single price
            high: High prices (if data is not DataFrame)
            low: Low prices (if data is not DataFrame)
            close: Close prices (if data is not DataFrame)
            period: ATR period (uses config if not provided)
            
        Returns:
            Series containing ATR values
        """
        if period is None:
            period = self.atr_period
            
        if isinstance(data, pd.DataFrame):
            high = data['high']
            low = data['low']
            close = data['close']
        else:
            if high is None or low is None or close is None:
                raise ValueError("If data is not DataFrame, high, low, and close must be provided")
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    def calculate_adx(self, data: Union[pd.DataFrame, pd.Series], 
                     high: Optional[pd.Series] = None,
                     low: Optional[pd.Series] = None,
                     close: Optional[pd.Series] = None,
                     period: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            data: DataFrame containing OHLC data or Series for single price
            high: High prices (if data is not DataFrame)
            low: Low prices (if data is not DataFrame)
            close: Close prices (if data is not DataFrame)
            period: ADX period (uses config if not provided)
            
        Returns:
            DataFrame containing ADX, +DI, and -DI values
        """
        if period is None:
            period = self.adx_period
            
        if isinstance(data, pd.DataFrame):
            high = data['high']
            low = data['low']
            close = data['close']
        else:
            if high is None or low is None or close is None:
                raise ValueError("If data is not DataFrame, high, low, and close must be provided")
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Calculate smoothed averages
        tr_smoothed = tr.rolling(window=period).sum()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).sum() / tr_smoothed
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).sum() / tr_smoothed

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        # Debug logging for NaNs
        logger = logging.getLogger("ADXDebug")
        logger.warning(f"ADX: tr_smoothed NaNs: {tr_smoothed.isna().sum()} (first: {tr_smoothed[tr_smoothed.isna()].index.tolist()[:5]})")
        logger.warning(f"ADX: plus_di NaNs: {plus_di.isna().sum()} (first: {plus_di[plus_di.isna()].index.tolist()[:5]})")
        logger.warning(f"ADX: minus_di NaNs: {minus_di.isna().sum()} (first: {minus_di[minus_di.isna()].index.tolist()[:5]})")
        logger.warning(f"ADX: dx NaNs: {dx.isna().sum()} (first: {dx[dx.isna()].index.tolist()[:5]})")
        logger.warning(f"ADX: adx NaNs: {adx.isna().sum()} (first: {adx[adx.isna()].index.tolist()[:5]})")

        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })
    
    def calculate_volume_ma(self, volume: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Volume Moving Average.
        
        Args:
            volume: Volume data
            period: MA period (uses config if not provided)
            
        Returns:
            Series containing volume MA values
        """
        if period is None:
            period = self.volume_ma_period
        return volume.rolling(window=period).mean()
    
    def calculate_price_momentum(self, data: Union[pd.Series, pd.DataFrame], 
                               lookback: Optional[int] = None) -> pd.Series:
        """
        Calculate Price Momentum.
        
        Args:
            data: Series or DataFrame containing price data
            lookback: Lookback period (uses config if not provided)
            
        Returns:
            Series containing momentum values
        """
        if lookback is None:
            lookback = self.price_momentum_lookback
            
        # Handle both Series and DataFrame input
        if isinstance(data, pd.DataFrame):
            price_data = data['close']
        else:
            price_data = data
            
        return price_data.pct_change(lookback)
    
    def calculate_volatility_regime(self, data: Union[pd.Series, pd.DataFrame], 
                                  period: int = 20) -> pd.Series:
        """
        Calculate Volatility Regime.
        
        Args:
            data: Series or DataFrame containing price data
            period: Period for volatility calculation
            
        Returns:
            Series containing volatility regime values
        """
        # Handle both Series and DataFrame input
        if isinstance(data, pd.DataFrame):
            price_data = data['close']
        else:
            price_data = data
            
        returns = price_data.pct_change()
        volatility = returns.rolling(window=period).std()
        return volatility
    
    def calculate_volume_surge_factor(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Volume Surge Factor.
        
        Args:
            volume: Volume data
            period: Period for volume MA calculation
            
        Returns:
            Series containing volume surge factor values
        """
        volume_ma = volume.rolling(window=period).mean()
        return volume / volume_ma
    
    def calculate_support_resistance(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                   period: int = 20) -> tuple:
        """
        Calculate Support and Resistance levels.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for calculation
            
        Returns:
            Tuple of (support, resistance) Series
        """
        resistance = high.rolling(window=period).max()
        support = low.rolling(window=period).min()
        return support, resistance
    
    def calculate_breakout_intensity(self, close: pd.Series, atr: pd.Series, 
                                   period: int = 20) -> pd.Series:
        """
        Calculate Breakout Intensity.
        
        Args:
            close: Close prices
            atr: ATR values
            period: Period for calculation
            
        Returns:
            Series containing breakout intensity values
        """
        high = close.rolling(window=period).max()
        low = close.rolling(window=period).min()
        breakout_up = (close - high) / atr
        breakout_down = (low - close) / atr
        return pd.concat([breakout_up, breakout_down], axis=1).max(axis=1)
    
    def calculate_directional_movement(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                     period: Optional[int] = None) -> pd.Series:
        """
        Calculate Directional Movement (ADX).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period (uses config if not provided)
            
        Returns:
            Series containing ADX values
        """
        if period is None:
            period = self.adx_period
            
        adx_df = self.calculate_adx(pd.DataFrame({'high': high, 'low': low, 'close': close}), period=period)
        return adx_df['adx']
    
    def calculate_cumulative_delta(self, close: pd.Series, volume: pd.Series, 
                                 period: int = 20) -> pd.Series:
        """
        Calculate Cumulative Delta.
        
        Args:
            close: Close prices
            volume: Volume data
            period: Period for calculation
            
        Returns:
            Series containing cumulative delta values
        """
        price_change = close.diff()
        volume_weighted_change = price_change * volume
        return volume_weighted_change.rolling(window=period).sum() 