import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json
from datetime import datetime
from market_analysis.technical_indicators import TechnicalIndicators
from .order_book_features import calculate_order_flow_imbalance, calculate_volume_pressure
import multiprocessing as mp
import gc
import time
import psutil
from tqdm import tqdm

def generate_features_chunk(chunk: pd.DataFrame, order_book_data: Optional[pd.DataFrame], indicators, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Top-level function for multiprocessing: generate features for a single chunk.
    Args:
        chunk: DataFrame chunk with OHLCV data
        order_book_data: Optional DataFrame with order book data
        indicators: TechnicalIndicators instance with configurable parameters
        config: Optional configuration dictionary with feature parameters
    Returns:
        DataFrame with generated features
    """
    features = pd.DataFrame(index=chunk.index)
    
    # Get config parameters with defaults
    config = config or {}
    
    # Price-based features
    returns_period = config.get('returns_period', 1)
    log_returns_period = config.get('log_returns_period', 1)
    
    features.loc[:, 'returns'] = chunk['close'].pct_change(periods=returns_period)
    features.loc[:, 'returns'] = features['returns'].replace([np.inf, -np.inf], np.nan)
    features.loc[:, 'log_returns'] = np.log1p(features['returns'])
    features.loc[:, 'log_returns'] = features['log_returns'].replace([np.inf, -np.inf], np.nan)
    
    # Technical indicators using configurable parameters
    rsi_period = config.get('rsi_period', 14)
    atr_period = config.get('atr_period', 14)
    
    rsi = indicators.calculate_rsi(chunk['close'], period=rsi_period)
    atr = indicators.calculate_atr(chunk, high=chunk['high'], low=chunk['low'], close=chunk['close'], period=atr_period)
    features.loc[:, 'rsi'] = rsi.astype(np.float32)
    features.loc[:, 'atr'] = atr.astype(np.float32)
    
    # Bollinger Bands with configurable parameters
    bb_period = config.get('bollinger_period', 20)
    bb_std_dev = config.get('bollinger_std_dev', 2.0)
    
    bb_result = indicators.calculate_bollinger_bands(chunk['close'], period=bb_period, num_std=bb_std_dev)
    features.loc[:, 'bb_upper'] = bb_result['upper'].astype(np.float32)
    features.loc[:, 'bb_middle'] = bb_result['middle'].astype(np.float32)
    features.loc[:, 'bb_lower'] = bb_result['lower'].astype(np.float32)
    features.loc[:, 'bb_width'] = ((bb_result['upper'] - bb_result['lower']) / bb_result['middle']).astype(np.float32)
    features.loc[:, 'bb_width'] = features['bb_width'].replace([np.inf, -np.inf], np.nan)
    
    # Volume features with configurable parameters
    volume_ma_period = config.get('volume_ma_period', 20)
    volume_std_period = config.get('volume_std_period', 20)
    volume_surge_period = config.get('volume_surge_period', 20)
    
    features.loc[:, 'volume'] = chunk['volume'].astype(np.float32)
    features.loc[:, 'volume_ma'] = indicators.calculate_volume_ma(chunk['volume'], period=volume_ma_period).astype(np.float32)
    features.loc[:, 'volume_std'] = chunk['volume'].rolling(window=volume_std_period).std().astype(np.float32)
    features.loc[:, 'volume_surge'] = indicators.calculate_volume_surge_factor(chunk['volume'], period=volume_surge_period).astype(np.float32)
    features.loc[:, 'volume_ratio'] = (chunk['volume'] / features['volume_ma']).astype(np.float32)
    
    # MACD features with configurable parameters
    macd_fast_period = config.get('macd_fast_period', 12)
    macd_slow_period = config.get('macd_slow_period', 26)
    macd_signal_period = config.get('macd_signal_period', 9)
    
    macd_result = indicators.calculate_macd(chunk['close'], fast_period=macd_fast_period, slow_period=macd_slow_period, signal_period=macd_signal_period)
    features.loc[:, 'macd'] = macd_result['macd'].astype(np.float32)
    features.loc[:, 'macd_signal'] = macd_result['signal'].astype(np.float32)
    features.loc[:, 'macd_hist'] = macd_result['hist'].astype(np.float32)
    
    # Moving Average Crossover with configurable parameters
    short_ma_period = config.get('short_ma_period', 5)
    long_ma_period = config.get('long_ma_period', 10)
    
    short_ma = chunk['close'].rolling(window=short_ma_period).mean()
    long_ma = chunk['close'].rolling(window=long_ma_period).mean()
    features.loc[:, 'ma_crossover'] = (short_ma - long_ma).astype(np.float32)
    
    # Swing RSI with configurable parameters
    swing_rsi_period = config.get('swing_rsi_period', 14)
    features.loc[:, 'swing_rsi'] = indicators.calculate_rsi(chunk['close'], period=swing_rsi_period).astype(np.float32)
    
    # VWAP Ratio with configurable parameters
    vwap_period = config.get('vwap_period', 20)
    vwap = (chunk['volume'] * (chunk['high'] + chunk['low'] + chunk['close']) / 3).rolling(window=vwap_period).sum() / chunk['volume'].rolling(window=vwap_period).sum()
    features.loc[:, 'vwap_ratio'] = (chunk['close'] / vwap).astype(np.float32)
    
    # Momentum features with configurable parameters
    price_momentum_lookback = config.get('price_momentum_lookback', 5)
    volatility_regime_period = config.get('volatility_regime_period', 20)
    
    momentum = indicators.calculate_price_momentum(chunk['close'], lookback=price_momentum_lookback)
    vol_regime = indicators.calculate_volatility_regime(chunk['close'], period=volatility_regime_period)
    features.loc[:, 'price_momentum'] = momentum.astype(np.float32)
    features.loc[:, 'volatility_regime'] = vol_regime.astype(np.float32)
    
    # Support/Resistance with configurable parameters
    support_resistance_period = config.get('support_resistance_period', 20)
    support, resistance = indicators.calculate_support_resistance(chunk['high'], chunk['low'], chunk['close'], period=support_resistance_period)
    features.loc[:, 'support'] = support.astype(np.float32)
    features.loc[:, 'resistance'] = resistance.astype(np.float32)
    
    # Breakout detection with configurable parameters
    breakout_period = config.get('breakout_period', 20)
    features.loc[:, 'breakout_intensity'] = indicators.calculate_breakout_intensity(chunk['close'], atr, period=breakout_period).astype(np.float32)
    
    # Trend strength with configurable parameters
    adx_period = config.get('adx_period', 14)
    features.loc[:, 'adx'] = indicators.calculate_directional_movement(chunk['high'], chunk['low'], chunk['close'], period=adx_period).astype(np.float32)
    
    # Cumulative delta with configurable parameters
    cumulative_delta_period = config.get('cumulative_delta_period', 20)
    features.loc[:, 'cumulative_delta'] = indicators.calculate_cumulative_delta(chunk['close'], chunk['volume'], period=cumulative_delta_period).astype(np.float32)
    
    # Add order book features if available
    if order_book_data is not None:
        features.loc[:, 'bid_ask_spread'] = (
            order_book_data['ask_prices'].apply(lambda x: x[0]) -
            order_book_data['bid_prices'].apply(lambda x: x[0])
        ).astype(np.float32)
        features.loc[:, 'volume_imbalance'] = (
            (order_book_data['bid_volume_total'] - order_book_data['ask_volume_total']) /
            (order_book_data['bid_volume_total'] + order_book_data['ask_volume_total'])
        ).astype(np.float32)
        features.loc[:, 'volume_imbalance'] = features['volume_imbalance'].replace([np.inf, -np.inf], np.nan)
    
    # --- START OF NEW/MODIFIED CODE: Aggressive NaN/Inf cleanup ---
    # Convert any remaining infinities to NaN first
    features = features.replace([np.inf, -np.inf], np.nan)

    # Fill leading NaNs using backward fill (bfill) from subsequent valid data
    # This is often best for initial NaNs from lookback periods.
    features = features.bfill() 

    # Fill any remaining NaNs using forward fill (ffill) from previous valid data
    features = features.ffill()

    # As a last resort, fill any remaining NaNs with 0 (e.g., if an entire column was NaN)
    features = features.fillna(0)

    # Double-check: assert no NaNs/Infs remain
    if features.isnull().any().any() or (features == np.inf).any().any() or (features == -np.inf).any().any():
        # This assert should ideally not be hit if cleanup is robust
        problematic_cols_after_cleanup = []
        for col in features.columns:
            if features[col].isnull().any() or (features[col] == np.inf).any() or (features[col] == -np.inf).any():
                problematic_cols_after_cleanup.append(col)
        raise ValueError(f"CRITICAL ERROR: NaNs/Infs still present in features after cleanup in generate_features_chunk for columns: {problematic_cols_after_cleanup}")
    # --- END OF NEW/MODIFIED CODE ---

    return features

class FeatureGenerator:
    """
    Enhanced feature generator with configurable parameters for hyperparameter optimization.
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
            config: Configuration dictionary for feature generation (includes feature engineering parameters)
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
        
        # Initialize TechnicalIndicators with configurable parameters
        # Extract feature engineering parameters from config
        feature_config = {}
        if self.config:
            feature_params = [
                'rsi_period', 'macd_fast_period', 'macd_slow_period', 'macd_signal_period',
                'bb_period', 'bb_num_std_dev', 'atr_period', 'adx_period',
                'volume_ma_period', 'price_momentum_lookback'
            ]
            for param in feature_params:
                if param in self.config:
                    feature_config[param] = self.config[param]
        
        self.indicators = TechnicalIndicators(feature_config)
        
        # Memory optimization settings
        self.optimize_memory = True
        self.chunk_size = 10000  # Process data in chunks of 10k rows
        self.dtype_map = {
            'float32': np.float32,
            'float64': np.float64,
            'int32': np.int32,
            'int64': np.int64
        }
        
        # Batch processing settings
        self.batch_size = 50000  # Default batch size
        self.max_batches_in_memory = 2  # Maximum number of batches to keep in memory
        self.batch_cache = {}  # Cache for batch results
        
        # Performance monitoring
        self.performance_metrics = {
            'total_processing_time': 0.0,
            'total_memory_usage': 0.0,
            'rows_processed': 0,
            'feature_generation_times': {},
            'memory_usage_by_feature': {},
            'batch_processing_metrics': [],
            'errors': []
        }
        self.monitoring_enabled = True
        self.metrics_file = self.cache_dir / f"performance_metrics_{self.version}.json"
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by converting to appropriate dtypes.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized dtypes
        """
        if not self.optimize_memory:
            return df
            
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
            
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype(np.int32)
            
        return df
    
    def _process_in_chunks(
        self,
        data: pd.DataFrame,
        func: callable,
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process DataFrame in chunks to reduce memory usage.
        
        Args:
            data: Input DataFrame
            func: Function to apply to each chunk
            chunk_size: Size of chunks (defaults to self.chunk_size)
            
        Returns:
            Processed DataFrame
        """
        if not self.optimize_memory:
            return func(data)
            
        chunk_size = chunk_size or self.chunk_size
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        results = []
        
        for chunk in chunks:
            result = func(chunk)
            results.append(result)
            gc.collect()  # Force garbage collection after each chunk
            
        return pd.concat(results)
    
    def _update_performance_metrics(
        self,
        operation: str,
        start_time: float,
        start_memory: float,
        rows_processed: int,
        error: Optional[str] = None
    ) -> None:
        """
        Update performance metrics for an operation.
        
        Args:
            operation: Name of the operation
            start_time: Start time of the operation
            start_memory: Start memory usage
            rows_processed: Number of rows processed
            error: Optional error message
        """
        if not self.monitoring_enabled:
            return
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Update operation-specific metrics
        if operation not in self.performance_metrics['feature_generation_times']:
            self.performance_metrics['feature_generation_times'][operation] = []
            self.performance_metrics['memory_usage_by_feature'][operation] = []
            
        self.performance_metrics['feature_generation_times'][operation].append(end_time - start_time)
        self.performance_metrics['memory_usage_by_feature'][operation].append(end_memory - start_memory)
        
        # Update total metrics
        self.performance_metrics['total_processing_time'] += end_time - start_time
        self.performance_metrics['total_memory_usage'] += end_memory - start_memory
        self.performance_metrics['rows_processed'] += rows_processed
        
        # Record error if any
        if error:
            self.performance_metrics['errors'].append({
                'operation': operation,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
            
        # Save metrics periodically
        if self.performance_metrics['rows_processed'] % 100000 == 0:
            self.save_performance_metrics()
            
    def save_performance_metrics(self) -> None:
        """Save current performance metrics to file."""
        if not self.monitoring_enabled:
            return
            
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            self.logger.info(f"Saved performance metrics to {self.metrics_file}")
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {str(e)}")
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.monitoring_enabled:
            return {}
            
        summary = {
            'total_rows_processed': self.performance_metrics['rows_processed'],
            'total_processing_time': self.performance_metrics['total_processing_time'],
            'total_memory_usage': self.performance_metrics['total_memory_usage'],
            'average_processing_speed': (
                self.performance_metrics['rows_processed'] / 
                self.performance_metrics['total_processing_time']
                if self.performance_metrics['total_processing_time'] > 0 else 0
            ),
            'average_memory_per_row': (
                self.performance_metrics['total_memory_usage'] / 
                self.performance_metrics['rows_processed']
                if self.performance_metrics['rows_processed'] > 0 else 0
            ),
            'feature_performance': {},
            'error_count': len(self.performance_metrics['errors'])
        }
        
        # Calculate per-feature metrics
        for feature in self.performance_metrics['feature_generation_times']:
            times = self.performance_metrics['feature_generation_times'][feature]
            memory = self.performance_metrics['memory_usage_by_feature'][feature]
            
            summary['feature_performance'][feature] = {
                'average_time': np.mean(times) if times else 0,
                'average_memory': np.mean(memory) if memory else 0,
                'total_calls': len(times)
            }
            
        return summary
    
    def process_batch(
        self,
        batch_data: pd.DataFrame,
        order_book_batch: Optional[pd.DataFrame] = None,
        batch_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process a single batch of data with memory optimization.
        
        Args:
            batch_data: DataFrame with OHLCV data for the batch
            order_book_batch: Optional DataFrame with order book data for the batch
            batch_id: Optional identifier for the batch (for caching)
            
        Returns:
            DataFrame with generated features for the batch
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Check cache if batch_id is provided
            if batch_id and batch_id in self.batch_cache:
                self.logger.debug(f"Using cached results for batch {batch_id}")
                return self.batch_cache[batch_id]
            
            # Optimize input data types
            batch_data = self._optimize_dtypes(batch_data)
            if order_book_batch is not None:
                order_book_batch = self._optimize_dtypes(order_book_batch)
            
            # Generate features for the batch
            features = generate_features_chunk(batch_data, order_book_batch, self.indicators, self.config)
            
            # Cache results if batch_id is provided
            if batch_id:
                # Manage cache size
                if len(self.batch_cache) >= self.max_batches_in_memory:
                    # Remove oldest batch
                    oldest_batch = next(iter(self.batch_cache))
                    del self.batch_cache[oldest_batch]
                self.batch_cache[batch_id] = features
            
            # Update performance metrics
            self._update_performance_metrics(
                'batch_processing',
                start_time,
                start_memory,
                len(batch_data)
            )
            
            return features
            
        except Exception as e:
            error_msg = f"Error processing batch {batch_id}: {str(e)}"
            self._update_performance_metrics(
                'batch_processing',
                start_time,
                start_memory,
                len(batch_data),
                error_msg
            )
            raise

    def generate_features_batched(self, ohlcv_data: pd.DataFrame, order_book_data: Optional[pd.DataFrame] = None, batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Generate features in batches to manage memory usage.
        Args:
            ohlcv_data: DataFrame with OHLCV data
            order_book_data: Optional DataFrame with order book data
            batch_size: Size of batches to process (default: 10% of data)
        Returns:
            DataFrame with generated features
        """
        if batch_size is None:
            batch_size = max(1, len(ohlcv_data) // 10)
        
        # Initialize empty DataFrame for results
        all_features = pd.DataFrame(index=ohlcv_data.index)
        
        # Process in batches
        num_batches = (len(ohlcv_data) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), desc="Generating features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(ohlcv_data))
            
            # Get batch data
            batch_ohlcv = ohlcv_data.iloc[start_idx:end_idx]
            batch_order_book = order_book_data.iloc[start_idx:end_idx] if order_book_data is not None else None
            
            # Generate features for batch
            batch_features = generate_features_chunk(batch_ohlcv, batch_order_book, self.indicators, self.config)
            
            # Store results
            for col in batch_features.columns:
                all_features.loc[batch_features.index, col] = batch_features[col].astype(np.float32)
            
            # Force garbage collection if memory optimization is enabled
            if self.optimize_memory:
                gc.collect()
        
        return all_features

    def generate_features(
        self,
        ohlcv_data: pd.DataFrame,
        order_book_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate features from OHLCV and order book data. Uses batch processing for large datasets.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            order_book_data: Optional DataFrame with order book data
            
        Returns:
            DataFrame with generated features
        """
        # Define thresholds for different processing methods
        small_threshold = 10000  # Use simple processing
        medium_threshold = 100000  # Use parallel processing
        large_threshold = 500000  # Use batch processing
        
        total_rows = len(ohlcv_data)
        
        if total_rows <= small_threshold:
            # Small dataset: use simple processing
            return generate_features_chunk(ohlcv_data, order_book_data, self.indicators, self.config)
        elif total_rows <= medium_threshold:
            # Medium dataset: use parallel processing
            return self.parallel_feature_generation(ohlcv_data, order_book_data)
        else:
            # Large dataset: use batch processing
            return self.generate_features_batched(ohlcv_data, order_book_data)
    
    def parallel_feature_generation(
        self,
        ohlcv_data: pd.DataFrame,
        order_book_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate features in parallel for large datasets with memory optimization.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            order_book_data: Optional DataFrame with order book data
            
        Returns:
            DataFrame with generated features
        """
        # Determine the number of processes to use
        num_processes = mp.cpu_count()
        
        # Calculate optimal chunk size based on available memory
        total_rows = len(ohlcv_data)
        chunk_size = max(10000, total_rows // (num_processes * 4))  # Ensure chunks aren't too small
        
        # Split the data into chunks
        chunks = [ohlcv_data[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
        
        # Split order book data if available
        ob_chunks = None
        if order_book_data is not None:
            ob_chunks = [order_book_data[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
        else:
            ob_chunks = [None] * len(chunks)
        
        # Create a pool of processes
        with mp.Pool(processes=num_processes) as pool:
            # Map the feature generation function to each chunk
            results = pool.starmap(
                generate_features_chunk,
                zip(chunks, ob_chunks, [self.indicators]*len(chunks), [self.config]*len(chunks))
            )
        
        # Combine the results and optimize memory
        combined = pd.concat(results)
        return self._optimize_dtypes(combined)
    
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
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error during feature generator cleanup: {str(e)}")
            raise
    
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
            
            # --- NaN/Inf cleanup for technical features ---
            # Convert any remaining infinities to NaN first
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill leading NaNs using backward fill from subsequent valid data
            df = df.bfill()
            
            # Fill any remaining NaNs using forward fill from previous valid data
            df = df.ffill()
            
            # As a last resort, fill any remaining NaNs with 0
            df = df.fillna(0)
            # --- END OF CLEANUP ---
            
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
        
        # --- NaN/Inf cleanup for momentum features ---
        # Convert any remaining infinities to NaN first
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill leading NaNs using backward fill from subsequent valid data
        df = df.bfill()
        
        # Fill any remaining NaNs using forward fill from previous valid data
        df = df.ffill()
        
        # As a last resort, fill any remaining NaNs with 0
        df = df.fillna(0)
        # --- END OF CLEANUP ---
        
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
            df['spread'] = order_book_data['ask_prices'].apply(lambda x: x[0]) - order_book_data['bid_prices'].apply(lambda x: x[0])
        
        # --- NaN/Inf cleanup for order book features ---
        # Convert any remaining infinities to NaN first
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill leading NaNs using backward fill from subsequent valid data
        df = df.bfill()
        
        # Fill any remaining NaNs using forward fill from previous valid data
        df = df.ffill()
        
        # As a last resort, fill any remaining NaNs with 0
        df = df.fillna(0)
        # --- END OF CLEANUP ---
        
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
    
    def calculate_historical_volatility(self, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Historical Volatility using rolling standard deviation of returns."""
        return TechnicalIndicators.calculate_historical_volatility(close, period)

    def calculate_ichimoku_cloud(self, high: pd.Series, low: pd.Series, close: pd.Series, conversion_period: int = 9, base_period: int = 26, lagging_period: int = 52, displacement: int = 26) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        return TechnicalIndicators.calculate_ichimoku_cloud(high, low, close, conversion_period, base_period, lagging_period, displacement) 