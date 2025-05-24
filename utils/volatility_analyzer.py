import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from functools import lru_cache
import logging
import hashlib

logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    """
    Analyzes volatility patterns and provides mean reversion indicators
    for natural gas and other markets with volatility clustering behavior.
    """
    
    def __init__(self, cache_size: int = 128):
        """
        Initialize the volatility analyzer.
        
        Args:
            cache_size: Size of the LRU cache for expensive calculations
        """
        self.cache_size = cache_size
        # Initialize cache for GARCH model results
        self._garch_cache = {}
        
    def _hash_series(self, series: pd.Series) -> str:
        """
        Create a hash of a pandas Series for caching.
        
        Args:
            series: Input Series
            
        Returns:
            String hash of the Series
        """
        # Convert to numpy array and create hash
        arr = series.values
        return hashlib.md5(arr.tobytes()).hexdigest()
        
    def _calculate_volatility_zscore(
        self,
        returns: pd.Series,
        short_window: int,
        long_window: int
    ) -> pd.Series:
        """
        Calculate volatility z-score (cached).
        
        Args:
            returns: Series of returns
            short_window: Short-term window
            long_window: Long-term window
            
        Returns:
            Series with volatility z-scores
        """
        # Create cache key
        cache_key = f"vol_zscore_{self._hash_series(returns)}_{short_window}_{long_window}"
        
        # Check cache
        if hasattr(self, '_vol_zscore_cache') and cache_key in self._vol_zscore_cache:
            logger.debug("Using cached volatility z-score")
            return self._vol_zscore_cache[cache_key]
            
        # Calculate z-score
        short_vol = returns.rolling(window=short_window).std()
        long_vol = returns.rolling(window=long_window).std()
        result = (short_vol - long_vol) / long_vol
        
        # Cache result
        if not hasattr(self, '_vol_zscore_cache'):
            self._vol_zscore_cache = {}
        self._vol_zscore_cache[cache_key] = result
        
        return result
    
    @staticmethod
    def calculate_volatility_rsi(
        volatility_series: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Apply RSI to volatility measure instead of price
        
        Args:
            volatility_series: Series of volatility measurements (e.g., ATR)
            period: RSI calculation period
        
        Returns:
            RSI of volatility
        """
        delta = volatility_series.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate relative strength
        rs = avg_gains / avg_losses
        
        # Calculate RSI
        vol_rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return vol_rsi
    
    def _calculate_atr_ratio(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        short_period: int,
        long_period: int
    ) -> pd.Series:
        """
        Calculate ATR ratio (cached).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            short_period: Short-term period
            long_period: Long-term period
            
        Returns:
            Series with ATR ratios
        """
        # Create cache key
        cache_key = f"atr_ratio_{self._hash_series(high)}_{self._hash_series(low)}_{self._hash_series(close)}_{short_period}_{long_period}"
        
        # Check cache
        if hasattr(self, '_atr_ratio_cache') and cache_key in self._atr_ratio_cache:
            logger.debug("Using cached ATR ratio")
            return self._atr_ratio_cache[cache_key]
            
        # Calculate ATR ratio
        from utils.technical_indicators import TechnicalIndicators
        short_atr = TechnicalIndicators.calculate_atr(high, low, close, period=short_period)
        long_atr = TechnicalIndicators.calculate_atr(high, low, close, period=long_period)
        result = short_atr / long_atr
        
        # Cache result
        if not hasattr(self, '_atr_ratio_cache'):
            self._atr_ratio_cache = {}
        self._atr_ratio_cache[cache_key] = result
        
        return result
    
    @staticmethod
    def calculate_volatility_percentile(
        volatility_series: pd.Series,
        lookback_window: int = 100
    ) -> pd.Series:
        """
        Calculate the percentile rank of current volatility
        
        Args:
            volatility_series: Series of volatility measurements
            lookback_window: Historical window for percentile calculation
            
        Returns:
            Series of volatility percentile ranks (0-100)
        """
        return volatility_series.rolling(window=lookback_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )
    
    def calculate_garch_features(
        self,
        returns: pd.Series,
        p: int = 1,
        q: int = 1
    ) -> pd.DataFrame:
        """
        Calculate GARCH model features for volatility clustering (with caching).
        
        Args:
            returns: Series of price returns
            p: GARCH lag order
            q: ARCH lag order
            
        Returns:
            DataFrame with GARCH features
        """
        # Create cache key
        cache_key = f"garch_{self._hash_series(returns)}_{p}_{q}"
        
        # Check cache
        if cache_key in self._garch_cache:
            logger.debug("Using cached GARCH results")
            return self._garch_cache[cache_key]
            
        try:
            import arch
            from arch import arch_model
            
            # Fit GARCH model
            model = arch_model(returns.dropna(), vol='GARCH', p=p, q=q)
            result = model.fit(disp='off')
            
            # Get conditional volatility
            cond_vol = result.conditional_volatility
            forecast = result.forecast(horizon=5)
            
            # Create features
            features = pd.DataFrame(index=returns.index)
            features['garch_vol'] = pd.Series(
                cond_vol, 
                index=returns.dropna().index
            )
            
            # Calculate volatility trend
            features['garch_vol_change'] = features['garch_vol'].pct_change(5)
            
            # Cache results
            self._garch_cache[cache_key] = features
            
            return features
            
        except ImportError:
            logger.warning("ARCH package not available. GARCH features will not be calculated.")
            return pd.DataFrame(index=returns.index)
        
    def calculate_volume_volatility_features(
        self,
        returns: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate volume-weighted volatility features (cached).
        
        Args:
            returns: Series of price returns
            volume: Series of trading volume
            window: Rolling window for calculations
            
        Returns:
            DataFrame with volume-weighted volatility features
        """
        # Create cache key
        cache_key = f"vol_vol_{self._hash_series(returns)}_{self._hash_series(volume)}_{window}"
        
        # Check cache
        if hasattr(self, '_vol_vol_cache') and cache_key in self._vol_vol_cache:
            logger.debug("Using cached volume volatility features")
            return self._vol_vol_cache[cache_key]
            
        # Calculate features
        features = pd.DataFrame(index=returns.index)
        
        # Volume relative to moving average
        vol_ratio = volume / volume.rolling(window=window).mean()
        
        # Standard deviation of returns
        std_dev = returns.rolling(window=window).std()
        
        # Volume-weighted volatility
        features['vol_weighted_volatility'] = std_dev * vol_ratio
        
        # Volatility adjusted for volume trend
        vol_trend = volume.rolling(window=window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
        )
        features['vol_adjusted_trend'] = std_dev * np.sign(vol_trend)
        
        # Cache results
        if not hasattr(self, '_vol_vol_cache'):
            self._vol_vol_cache = {}
        self._vol_vol_cache[cache_key] = features
        
        return features
    
    def calculate_volatility_regime(
        self,
        returns: pd.Series,
        short_window: int = 5,
        long_window: int = 20,
        vol_z_threshold: float = 1.5
    ) -> pd.Series:
        """
        Classify current volatility regime
        
        Args:
            returns: Series of price returns
            short_window: Short-term window for volatility z-score
            long_window: Long-term window for volatility z-score
            vol_z_threshold: Z-score threshold for regime classification
            
        Returns:
            Series with regime classifications: 
            1 (high vol), 0 (normal vol), -1 (low vol)
        """
        # Calculate volatility z-score
        vol_zscore = self._calculate_volatility_zscore(returns, short_window, long_window)
        
        # Classify regimes
        vol_regime = pd.Series(0, index=returns.index)  # Normal volatility
        vol_regime[vol_zscore > vol_z_threshold] = 1     # High volatility
        vol_regime[vol_zscore < -vol_z_threshold] = -1   # Low volatility
        
        return vol_regime
    
    @staticmethod
    def calculate_volatility_divergence(
        close: pd.Series,
        atr: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        Detect divergence between price and volatility
        
        Args:
            close: Series of closing prices
            atr: Series of ATR values
            window: Lookback window for divergence calculation
            
        Returns:
            Series with divergence indicator:
            1 (bullish divergence), -1 (bearish divergence), 0 (no divergence)
        """
        # Calculate rate of change
        price_roc = close.pct_change(window)
        atr_roc = atr.pct_change(window)
        
        # Identify divergences
        divergence = pd.Series(0, index=close.index)
        
        # Bullish divergence: Price making lower lows but ATR making higher lows
        bullish = (price_roc < 0) & (atr_roc > 0)
        
        # Bearish divergence: Price making higher highs but ATR making lower highs
        bearish = (price_roc > 0) & (atr_roc < 0)
        
        divergence[bullish] = 1
        divergence[bearish] = -1
        
        return divergence
    
    def prepare_volatility_features(
        self,
        data: pd.DataFrame,
        is_intraday: bool = False
    ) -> pd.DataFrame:
        """
        Prepare comprehensive volatility mean reversion features (with caching).
        
        Args:
            data: DataFrame with OHLCV data
            is_intraday: Flag for intraday data
            
        Returns:
            DataFrame with volatility features
        """
        from utils.technical_indicators import TechnicalIndicators
        
        df = data.copy()
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            
        # Calculate log returns for better statistical properties
        df['log_returns'] = np.log1p(df['returns'])
        
        # Set different parameters based on timeframe
        if is_intraday:
            short_vol_window, long_vol_window = 12, 48  # For intraday (hours)
            atr_short, atr_long = 5, 20
            vol_rsi_period = 14
            vol_lookback = 100
        else:
            short_vol_window, long_vol_window = 5, 20  # For daily data
            atr_short, atr_long = 5, 20
            vol_rsi_period = 14
            vol_lookback = 50
            
        # Calculate ATR if not present
        if 'atr' not in df.columns:
            df['atr'] = TechnicalIndicators.calculate_atr(
                df['high'], df['low'], df['close'], period=atr_short
            )
            
        # Add volatility z-score (cached)
        df['vol_zscore'] = self._calculate_volatility_zscore(
            df['returns'], short_vol_window, long_vol_window
        )
        
        # Add ATR ratio (cached)
        df['atr_ratio'] = self._calculate_atr_ratio(
            df['high'], df['low'], df['close'], atr_short, atr_long
        )
        
        # Add ATR-RSI
        df['atr_rsi'] = VolatilityAnalyzer.calculate_volatility_rsi(
            df['atr'], vol_rsi_period
        )
        
        # Add volatility percentile
        df['vol_percentile'] = VolatilityAnalyzer.calculate_volatility_percentile(
            df['atr'], vol_lookback
        )
        
        # Add volatility regime
        df['vol_regime'] = self.calculate_volatility_regime(
            df['returns'], short_vol_window, long_vol_window
        )
        
        # Add volatility divergence
        df['vol_divergence'] = VolatilityAnalyzer.calculate_volatility_divergence(
            df['close'], df['atr']
        )
        
        # Add volume-weighted volatility features if volume data exists (cached)
        if 'volume' in df.columns:
            vol_features = self.calculate_volume_volatility_features(
                df['returns'], df['volume']
            )
            df = pd.concat([df, vol_features], axis=1)
            
        # Add GARCH features for daily data (more stable)
        if not is_intraday and len(df) > 100:
            try:
                garch_features = self.calculate_garch_features(
                    df['returns'].dropna()
                )
                if not garch_features.empty:
                    df = pd.concat([df, garch_features], axis=1)
            except Exception as e:
                logger.warning(f"Could not calculate GARCH features: {e}")
                
        return df
    
    @staticmethod
    def get_volatility_signal(
        features: pd.DataFrame, 
        vol_threshold: float = 80.0,
        vol_oversold: float = 20.0,
        min_confidence: float = 0.6
    ) -> Tuple[float, float]:
        """
        Generate volatility mean reversion signals
        
        Args:
            features: DataFrame with volatility features
            vol_threshold: Upper threshold for volatility percentile
            vol_oversold: Lower threshold for volatility percentile
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (signal_strength, confidence)
            Signal: +1 (volatility expected to increase), -1 (expected to decrease)
        """
        if len(features) < 20:
            return 0, 0
            
        current = features.iloc[-1]
        
        # Signal components
        signal_components = {}
        
        # 1. Volatility percentile extremes
        if 'vol_percentile' in current:
            if current['vol_percentile'] > vol_threshold:
                # High volatility, expect mean reversion downward
                signal_components['vol_percentile'] = -1
            elif current['vol_percentile'] < vol_oversold:
                # Low volatility, expect mean reversion upward
                signal_components['vol_percentile'] = 1
            else:
                signal_components['vol_percentile'] = 0
                
        # 2. Volatility RSI extremes
        if 'atr_rsi' in current:
            if current['atr_rsi'] > 70:
                # Overbought volatility, expect decrease
                signal_components['atr_rsi'] = -1
            elif current['atr_rsi'] < 30:
                # Oversold volatility, expect increase
                signal_components['atr_rsi'] = 1
            else:
                signal_components['atr_rsi'] = 0
                
        # 3. ATR ratio extremes
        if 'atr_ratio' in current:
            if current['atr_ratio'] > 1.5:
                # Short-term ATR much higher than long-term, expect mean reversion
                signal_components['atr_ratio'] = -1
            elif current['atr_ratio'] < 0.7:
                # Short-term ATR much lower than long-term, expect increase
                signal_components['atr_ratio'] = 1
            else:
                signal_components['atr_ratio'] = 0
                
        # 4. Volatility Z-score
        if 'vol_zscore' in current:
            if current['vol_zscore'] > 2.0:
                # Extremely high volatility, expect mean reversion
                signal_components['vol_zscore'] = -1
            elif current['vol_zscore'] < -2.0:
                # Extremely low volatility, expect increase
                signal_components['vol_zscore'] = 1
            else:
                signal_components['vol_zscore'] = 0
                
        # 5. Volatility divergence
        if 'vol_divergence' in current:
            signal_components['vol_divergence'] = current['vol_divergence']
            
        # 6. GARCH forecast if available
        if 'garch_vol_change' in current:
            if not pd.isna(current['garch_vol_change']):
                signal_components['garch_forecast'] = np.sign(current['garch_vol_change'])
            
        # Calculate combined signal
        if signal_components:
            # Use weights: higher weights for more reliable indicators
            weights = {
                'vol_percentile': 0.25,
                'atr_rsi': 0.2,
                'atr_ratio': 0.2,
                'vol_zscore': 0.2,
                'vol_divergence': 0.1,
                'garch_forecast': 0.05
            }
            
            # Compute weighted signal
            signal = sum(
                signal_components.get(k, 0) * weights.get(k, 0)
                for k in signal_components
            )
            
            # Calculate confidence based on agreement
            component_values = [v for v in signal_components.values() if v != 0]
            if component_values:
                agreement = sum(np.sign(v) == np.sign(signal) for v in component_values) / len(component_values)
                confidence = agreement * min(abs(signal) * 2, 1.0)
            else:
                confidence = 0
                
            return signal, confidence
        
        return 0, 0