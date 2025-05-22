import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from datetime import datetime
from .technical_indicators import TechnicalIndicators
from .order_book_features import calculate_order_flow_imbalance, calculate_volume_pressure


class FeatureGenerator:
    """
    Handles feature engineering for the momentum strategy, including technical indicators,
    order book features, and custom momentum-specific features.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        Initialize the feature generator.
        
        Args:
            config: Configuration dictionary for feature generation
            cache_dir: Directory to cache generated features
            version: Version identifier for the feature generator
        """
        self.config = config or {}
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/features")
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize feature importance tracking
        self.feature_importance: Dict[str, float] = {}
        self.feature_interactions: Dict[str, List[str]] = {}
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.indicators = TechnicalIndicators()
    
    def generate_features(
        self,
        ohlcv_data: pd.DataFrame,
        order_book_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate features from OHLCV and order book data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            order_book_data: Optional DataFrame with order book data
            
        Returns:
            DataFrame with generated features
        """
        features = pd.DataFrame(index=ohlcv_data.index)
        
        # Price-based features
        features['returns'] = ohlcv_data['close'].pct_change()
        features['log_returns'] = np.log1p(features['returns'])
        
        # Technical indicators
        features['rsi'] = self.indicators.calculate_rsi(ohlcv_data['close'])
        features['atr'] = self.indicators.calculate_atr(
            ohlcv_data['high'],
            ohlcv_data['low'],
            ohlcv_data['close']
        )
        
        # Bollinger Bands
        upper, middle, lower = self.indicators.calculate_bollinger_bands(ohlcv_data['close'])
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle
        
        # Volume features
        features['volume_ma'] = ohlcv_data['volume'].rolling(window=20).mean()
        features['volume_std'] = ohlcv_data['volume'].rolling(window=20).std()
        features['volume_surge'] = self.indicators.calculate_volume_surge_factor(ohlcv_data['volume'])
        
        # Momentum features
        features['price_momentum'] = self.indicators.calculate_price_momentum(ohlcv_data['close'], 14)
        features['volatility_regime'] = self.indicators.calculate_volatility_regime(ohlcv_data['close'])
        
        # Support/Resistance
        support, resistance = self.indicators.calculate_support_resistance(
            ohlcv_data['high'],
            ohlcv_data['low'],
            ohlcv_data['close']
        )
        features['support'] = support
        features['resistance'] = resistance
        
        # Breakout detection
        features['breakout_intensity'] = self.indicators.calculate_breakout_intensity(
            ohlcv_data['close'],
            features['atr']
        )
        
        # Trend strength
        features['adx'] = self.indicators.calculate_directional_movement(
            ohlcv_data['high'],
            ohlcv_data['low'],
            ohlcv_data['close']
        )
        
        # Cumulative delta
        features['cumulative_delta'] = self.indicators.calculate_cumulative_delta(
            ohlcv_data['close'],
            ohlcv_data['volume']
        )
        
        # Add order book features if available
        if order_book_data is not None:
            features['bid_ask_spread'] = order_book_data['ask_prices'].apply(lambda x: x[0]) - \
                                       order_book_data['bid_prices'].apply(lambda x: x[0])
            
            features['volume_imbalance'] = (order_book_data['bid_volume_total'] - \
                                          order_book_data['ask_volume_total']) / \
                                         (order_book_data['bid_volume_total'] + \
                                          order_book_data['ask_volume_total'])
        
        return features
    
    def generate_technical_features(
        self,
        data: pd.DataFrame,
        price_col: str = 'close',
        volume_col: str = 'volume'
    ) -> pd.DataFrame:
        """
        Generate technical indicators for momentum strategy.
        
        Args:
            data: Input DataFrame with OHLCV data
            price_col: Name of the price column
            volume_col: Name of the volume column
            
        Returns:
            DataFrame with added technical features
        """
        df = data.copy()
        
        # If required columns don't exist, use first two numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError("Data must have at least 2 numeric columns")
        
        # Use first numeric column as price if close not found
        if price_col not in df.columns:
            price_col = numeric_cols[0]
            self.logger.warning(f"Using {price_col} as price column")
        
        # Use second numeric column as volume if volume not found
        if volume_col not in df.columns:
            volume_col = numeric_cols[1]
            self.logger.warning(f"Using {volume_col} as volume column")
        
        try:
            # RSI with multiple periods
            for period in [14, 21, 50]:
                df[f'rsi_{period}'] = self.indicators.calculate_rsi(df[price_col], period=period)
            
            # MACD
            macd, signal, hist = self.indicators.calculate_macd(df[price_col])
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(df[price_col])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # Volume-based features
            df['volume_ma'] = df[volume_col].rolling(window=20).mean()
            df['volume_std'] = df[volume_col].rolling(window=20).std()
            df['volume_ratio'] = df[volume_col] / df['volume_ma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating technical features: {str(e)}")
            raise
    
    def generate_momentum_features(
        self,
        data: pd.DataFrame,
        price_col: str = 'close',
        volume_col: str = 'volume'
    ) -> pd.DataFrame:
        """
        Generate momentum-specific features.
        
        Args:
            data: Input DataFrame with OHLCV data
            price_col: Name of the price column
            volume_col: Name of the volume column
            
        Returns:
            DataFrame with added momentum features
        """
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError("Data must have at least 2 numeric columns")
        # Use first numeric column as price if close not found
        if price_col not in df.columns:
            price_col = numeric_cols[0]
            self.logger.warning(f"Using {price_col} as price column in momentum features")
        # Use second numeric column as volume if volume not found
        if volume_col not in df.columns:
            volume_col = numeric_cols[1]
            self.logger.warning(f"Using {volume_col} as volume column in momentum features")
        # Use third and fourth numeric columns as high and low if not found
        if 'high' not in df.columns:
            if len(numeric_cols) > 2:
                high_col = numeric_cols[2]
                self.logger.warning(f"Using {high_col} as high column in momentum features")
            else:
                high_col = price_col
                self.logger.warning(f"Not enough columns for high; using {price_col}")
        else:
            high_col = 'high'
        if 'low' not in df.columns:
            if len(numeric_cols) > 3:
                low_col = numeric_cols[3]
                self.logger.warning(f"Using {low_col} as low column in momentum features")
            else:
                low_col = price_col
                self.logger.warning(f"Not enough columns for low; using {price_col}")
        else:
            low_col = 'low'
        # Price momentum
        for period in [1, 3, 5, 10]:
            df[f'returns_{period}'] = df[price_col].pct_change(period)
            df[f'log_returns_{period}'] = np.log(df[price_col] / df[price_col].shift(period))
        # Volume momentum
        df['volume_returns'] = df[volume_col].pct_change()
        df['volume_ma_ratio'] = df[volume_col] / df[volume_col].rolling(window=20).mean()
        # Trend strength (only if high, low, close available)
        try:
            df['adx'] = self.indicators.calculate_directional_movement(
                df[high_col],
                df[low_col],
                df[price_col]
            )
            df['trend_strength'] = abs(df['returns_5']) * df['adx']
        except Exception as e:
            self.logger.warning(f"Could not compute ADX/trend_strength: {str(e)}")
            df['adx'] = np.nan
            df['trend_strength'] = np.nan
        # Momentum divergence
        df['price_momentum'] = df[price_col].pct_change(5)
        df['volume_momentum'] = df[volume_col].pct_change(5)
        df['momentum_divergence'] = np.sign(df['price_momentum']) != np.sign(df['volume_momentum'])
        return df
    
    def generate_order_book_features(
        self,
        data: pd.DataFrame,
        order_book_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate order book features for momentum strategy.
        
        Args:
            data: Input DataFrame with OHLCV data
            order_book_data: Optional DataFrame with order book data
            
        Returns:
            DataFrame with added order book features
        """
        df = data.copy()
        
        if order_book_data is not None:
            # Order flow imbalance
            df['order_flow_imbalance'] = calculate_order_flow_imbalance(order_book_data)
            
            # Volume pressure
            df['volume_pressure'] = calculate_volume_pressure(order_book_data)
            
            # Market depth features
            df['depth_imbalance'] = self._calculate_depth_imbalance(order_book_data)
            df['spread'] = order_book_data['ask_price'] - order_book_data['bid_price']
        
        return df
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            data: Input DataFrame with OHLCV data
            period: ADX period
            
        Returns:
            Series with ADX values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
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
        tr_smoothed = tr.rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / tr_smoothed
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / tr_smoothed
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_depth_imbalance(
        self,
        order_book_data: pd.DataFrame,
        levels: int = 5
    ) -> pd.Series:
        """
        Calculate order book depth imbalance.
        
        Args:
            order_book_data: DataFrame with order book data
            levels: Number of levels to consider
            
        Returns:
            Series with depth imbalance values
        """
        bid_volume = order_book_data[[f'bid_volume_{i}' for i in range(levels)]].sum(axis=1)
        ask_volume = order_book_data[[f'ask_volume_{i}' for i in range(levels)]].sum(axis=1)
        
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    def update_feature_importance(self, importance_dict: Dict[str, float]) -> None:
        """
        Update feature importance tracking.
        
        Args:
            importance_dict: Dictionary of feature names and their importance scores
        """
        self.feature_importance.update(importance_dict)
    
    def detect_feature_interactions(
        self,
        data: pd.DataFrame,
        target_col: str,
        threshold: float = 0.3
    ) -> Dict[str, List[str]]:
        """
        Detect significant feature interactions.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of the target column
            threshold: Correlation threshold for interaction detection
            
        Returns:
            Dictionary of feature interactions
        """
        corr_matrix = data.corr()
        interactions = {}
        
        for col in data.columns:
            if col != target_col:
                # Find features that correlate with this feature
                correlated = corr_matrix[col][
                    (abs(corr_matrix[col]) > threshold) & 
                    (corr_matrix[col].index != col)
                ].index.tolist()
                
                if correlated:
                    interactions[col] = correlated
        
        self.feature_interactions = interactions
        return interactions
    
    def save_state(self, path: Optional[str] = None) -> None:
        """
        Save the feature generator state.
        
        Args:
            path: Path to save state (defaults to cache directory)
        """
        path = Path(path) if path else self.cache_dir / f"feature_generator_state_{self.version}"
        path.mkdir(parents=True, exist_ok=True)
        
        # Save feature importance
        with open(path / "feature_importance.json", "w") as f:
            json.dump(self.feature_importance, f, indent=2)
        
        # Save feature interactions
        with open(path / "feature_interactions.json", "w") as f:
            json.dump(self.feature_interactions, f, indent=2)
        
        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Saved feature generator state to {path}")
    
    def load_state(self, path: str) -> None:
        """
        Load the feature generator state.
        
        Args:
            path: Path to load state from
        """
        path = Path(path)
        
        # Load feature importance
        with open(path / "feature_importance.json", "r") as f:
            self.feature_importance = json.load(f)
        
        # Load feature interactions
        with open(path / "feature_interactions.json", "r") as f:
            self.feature_interactions = json.load(f)
        
        # Load config
        with open(path / "config.json", "r") as f:
            self.config = json.load(f)
        
        self.logger.info(f"Loaded feature generator state from {path}")
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the feature generator.
        """
        try:
            # Clear feature importance
            self.feature_importance.clear()
            
            # Clear feature interactions
            self.feature_interactions.clear()
            
            # Clear indicators
            self.indicators = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error during feature generator cleanup: {str(e)}")
            raise 