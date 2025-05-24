import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import pandas as pd
import numpy as np
import time
import cProfile
import pstats
import io
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import psutil
import gc

from utils.feature_generator import FeatureGenerator
from utils.volatility_analyzer import VolatilityAnalyzer
from utils.synthetic_data_generator import SyntheticDataGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def profile_batch_processing(
    data_size: int = 200000,
    batch_size: int = 50000,
    interval_minutes: int = 15,
    n_levels: int = 10
) -> Dict[str, float]:
    """
    Profile batch processing performance.
    
    Args:
        data_size: Number of data points to generate
        batch_size: Size of each batch
        interval_minutes: Time interval in minutes
        n_levels: Number of order book levels
        
    Returns:
        Dictionary with timing and memory results
    """
    # Generate synthetic data
    generator = SyntheticDataGenerator(
        start_date=datetime.now() - timedelta(days=data_size * interval_minutes / 1440),
        end_date=datetime.now()
    )
    
    ohlcv_data = generator.generate_ohlcv(interval_minutes)
    order_book_data = generator.generate_order_book(ohlcv_data, n_levels)
    
    # Add total volume columns expected by FeatureGenerator
    order_book_data['bid_volume_total'] = order_book_data['bid_volumes'].apply(np.sum)
    order_book_data['ask_volume_total'] = order_book_data['ask_volumes'].apply(np.sum)
    
    # Expand bid_volumes and ask_volumes arrays into separate columns
    for i in range(n_levels):
        order_book_data[f'bid_volume_{i}'] = order_book_data['bid_volumes'].apply(lambda x: x[i] if len(x) > i else np.nan)
        order_book_data[f'ask_volume_{i}'] = order_book_data['ask_volumes'].apply(lambda x: x[i] if len(x) > i else np.nan)
    
    # Initialize feature generator
    feature_gen = FeatureGenerator()
    feature_gen.batch_size = batch_size
    
    # Profile batch processing
    results = {}
    
    # Force garbage collection before test
    gc.collect()
    initial_memory = get_memory_usage()
    
    # Time the batch processing
    start_time = time.time()
    features = feature_gen.generate_features_batched(ohlcv_data, order_book_data)
    processing_time = time.time() - start_time
    
    # Calculate memory usage
    final_memory = get_memory_usage()
    memory_used = final_memory - initial_memory
    
    # Store results
    results['batch_processing_time'] = processing_time
    results['batch_processing_memory'] = memory_used
    results['rows_per_second'] = len(features) / processing_time
    results['memory_per_row'] = memory_used / len(features)
    
    # Verify feature generation
    required_columns = [
        'returns', 'log_returns', 'rsi', 'atr', 'bb_upper', 'bb_middle',
        'bb_lower', 'bb_width', 'volume_ma', 'volume_std', 'volume_surge',
        'price_momentum', 'volatility_regime', 'support', 'resistance',
        'breakout_intensity', 'adx', 'cumulative_delta'
    ]
    
    missing_columns = [col for col in required_columns if col not in features.columns]
    if missing_columns:
        logger.error(f"Missing columns in batch processing: {missing_columns}")
        results['status'] = 'failed'
    else:
        results['status'] = 'success'
    
    return results

def profile_feature_generation(
    data_size: int = 1000,
    interval_minutes: int = 15,
    n_levels: int = 10
) -> Dict[str, float]:
    """
    Profile feature generation performance.
    
    Args:
        data_size: Number of data points to generate
        interval_minutes: Time interval in minutes
        n_levels: Number of order book levels
        
    Returns:
        Dictionary with timing results
    """
    # Generate synthetic data
    generator = SyntheticDataGenerator(
        start_date=datetime.now() - timedelta(days=data_size * interval_minutes / 1440),
        end_date=datetime.now()
    )
    
    ohlcv_data = generator.generate_ohlcv(interval_minutes)
    order_book_data = generator.generate_order_book(ohlcv_data, n_levels)
    
    # Add total volume columns expected by FeatureGenerator
    order_book_data['bid_volume_total'] = order_book_data['bid_volumes'].apply(np.sum)
    order_book_data['ask_volume_total'] = order_book_data['ask_volumes'].apply(np.sum)
    
    # Expand bid_volumes and ask_volumes arrays into separate columns
    for i in range(n_levels):
        order_book_data[f'bid_volume_{i}'] = order_book_data['bid_volumes'].apply(lambda x: x[i] if len(x) > i else np.nan)
        order_book_data[f'ask_volume_{i}'] = order_book_data['ask_volumes'].apply(lambda x: x[i] if len(x) > i else np.nan)
    
    # Initialize feature generators
    feature_gen = FeatureGenerator()
    vol_analyzer = VolatilityAnalyzer()
    
    # Profile feature generation
    results = {}
    
    # Force garbage collection before each test
    gc.collect()
    initial_memory = get_memory_usage()
    
    # 1. Profile basic feature generation
    start_time = time.time()
    features = feature_gen.generate_features(ohlcv_data, order_book_data)
    results['basic_features_time'] = time.time() - start_time
    results['basic_features_memory'] = get_memory_usage() - initial_memory
    logger.info(f"Basic features shape: {features.shape}")
    
    # Force garbage collection
    gc.collect()
    initial_memory = get_memory_usage()
    
    # 2. Profile technical features
    start_time = time.time()
    tech_features = feature_gen.generate_technical_features(ohlcv_data)
    results['technical_features_time'] = time.time() - start_time
    results['technical_features_memory'] = get_memory_usage() - initial_memory
    logger.info(f"Technical features shape: {tech_features.shape}")
    
    # Force garbage collection
    gc.collect()
    initial_memory = get_memory_usage()
    
    # 3. Profile momentum features
    start_time = time.time()
    momentum_features = feature_gen.generate_momentum_features(ohlcv_data)
    results['momentum_features_time'] = time.time() - start_time
    results['momentum_features_memory'] = get_memory_usage() - initial_memory
    logger.info(f"Momentum features shape: {momentum_features.shape}")
    
    # Force garbage collection
    gc.collect()
    initial_memory = get_memory_usage()
    
    # 4. Profile order book features
    start_time = time.time()
    ob_features = feature_gen.generate_order_book_features(ohlcv_data, order_book_data)
    results['order_book_features_time'] = time.time() - start_time
    results['order_book_features_memory'] = get_memory_usage() - initial_memory
    logger.info(f"Order book features shape: {ob_features.shape}")
    
    # Force garbage collection
    gc.collect()
    initial_memory = get_memory_usage()
    
    # 5. Profile volatility features
    start_time = time.time()
    vol_features = vol_analyzer.prepare_volatility_features(ohlcv_data, is_intraday=True)
    results['volatility_features_time'] = time.time() - start_time
    results['volatility_features_memory'] = get_memory_usage() - initial_memory
    logger.info(f"Volatility features shape: {vol_features.shape}")
    
    return results

def profile_memory_usage(
    data_size: int = 1000,
    interval_minutes: int = 15,
    n_levels: int = 10
) -> Dict[str, float]:
    """
    Profile memory usage of feature generation.
    
    Args:
        data_size: Number of data points to generate
        interval_minutes: Time interval in minutes
        n_levels: Number of order book levels
        
    Returns:
        Dictionary with memory usage results
    """
    # Generate synthetic data
    generator = SyntheticDataGenerator(
        start_date=datetime.now() - timedelta(days=data_size * interval_minutes / 1440),
        end_date=datetime.now()
    )
    
    ohlcv_data = generator.generate_ohlcv(interval_minutes)
    order_book_data = generator.generate_order_book(ohlcv_data, n_levels)
    
    # Add total volume columns expected by FeatureGenerator
    order_book_data['bid_volume_total'] = order_book_data['bid_volumes'].apply(np.sum)
    order_book_data['ask_volume_total'] = order_book_data['ask_volumes'].apply(np.sum)
    
    # Expand bid_volumes and ask_volumes arrays into separate columns
    for i in range(n_levels):
        order_book_data[f'bid_volume_{i}'] = order_book_data['bid_volumes'].apply(lambda x: x[i] if len(x) > i else np.nan)
        order_book_data[f'ask_volume_{i}'] = order_book_data['ask_volumes'].apply(lambda x: x[i] if len(x) > i else np.nan)
    
    # Initialize feature generators
    feature_gen = FeatureGenerator()
    vol_analyzer = VolatilityAnalyzer()
    
    # Profile memory usage
    results = {}
    
    # Force garbage collection before each test
    gc.collect()
    initial_memory = get_memory_usage()
    
    # 1. Basic features
    features = feature_gen.generate_features(ohlcv_data, order_book_data)
    results['basic_features'] = get_memory_usage() - initial_memory
    
    # Force garbage collection
    gc.collect()
    initial_memory = get_memory_usage()
    
    # 2. Technical features
    tech_features = feature_gen.generate_technical_features(ohlcv_data)
    results['technical_features'] = get_memory_usage() - initial_memory
    
    # Force garbage collection
    gc.collect()
    initial_memory = get_memory_usage()
    
    # 3. Momentum features
    momentum_features = feature_gen.generate_momentum_features(ohlcv_data)
    results['momentum_features'] = get_memory_usage() - initial_memory
    
    # Force garbage collection
    gc.collect()
    initial_memory = get_memory_usage()
    
    # 4. Order book features
    ob_features = feature_gen.generate_order_book_features(ohlcv_data, order_book_data)
    results['order_book_features'] = get_memory_usage() - initial_memory
    
    # Force garbage collection
    gc.collect()
    initial_memory = get_memory_usage()
    
    # 5. Volatility features
    vol_features = vol_analyzer.prepare_volatility_features(ohlcv_data, is_intraday=True)
    results['volatility_features'] = get_memory_usage() - initial_memory
    
    return results

def main():
    """Main profiling function."""
    # Create results directory
    results_dir = Path("profiling_results")
    results_dir.mkdir(exist_ok=True)
    
    # Profile different data sizes
    data_sizes = [1000, 5000, 10000, 200000]
    timing_results = {}
    memory_results = {}
    batch_results = {}
    
    for size in data_sizes:
        logger.info(f"\nProfiling with {size} data points...")
        
        # Profile timing and memory
        results = profile_feature_generation(
            data_size=size,
            interval_minutes=15,
            n_levels=10
        )
        
        # Split results into timing and memory
        timing_results[size] = {k: v for k, v in results.items() if k.endswith('_time')}
        memory_results[size] = {k: v for k, v in results.items() if k.endswith('_memory')}
        
        # Profile batch processing for large datasets
        if size >= 10000:
            logger.info(f"\nProfiling batch processing with {size} data points...")
            batch_results[size] = profile_batch_processing(
                data_size=size,
                batch_size=50000,
                interval_minutes=15,
                n_levels=10
            )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save timing results
    timing_df = pd.DataFrame(timing_results).T
    timing_df.to_csv(results_dir / f"timing_results_{timestamp}.csv")
    
    # Save memory results
    memory_df = pd.DataFrame(memory_results).T
    memory_df.to_csv(results_dir / f"memory_results_{timestamp}.csv")
    
    # Save batch processing results
    if batch_results:
        batch_df = pd.DataFrame(batch_results).T
        batch_df.to_csv(results_dir / f"batch_results_{timestamp}.csv")
    
    # Print summary
    logger.info("\nTiming Results (seconds):")
    logger.info(timing_df)
    
    logger.info("\nMemory Usage (MB):")
    logger.info(memory_df)
    
    if batch_results:
        logger.info("\nBatch Processing Results:")
        logger.info(batch_df)
    
    # Calculate and print memory optimization metrics
    logger.info("\nMemory Optimization Metrics:")
    for size in data_sizes:
        total_memory = sum(memory_results[size].values())
        logger.info(f"\nData size: {size}")
        logger.info(f"Total memory usage: {total_memory:.2f} MB")
        logger.info(f"Memory per row: {total_memory / size:.4f} MB")
        
        # Calculate memory reduction compared to baseline (1000 rows)
        if size > 1000:
            baseline_memory = sum(memory_results[1000].values())
            memory_reduction = (baseline_memory * size - total_memory) / (baseline_memory * size) * 100
            logger.info(f"Memory reduction vs. baseline: {memory_reduction:.2f}%")
        
        # Print batch processing metrics if available
        if size in batch_results:
            batch_metrics = batch_results[size]
            logger.info(f"\nBatch Processing Metrics:")
            logger.info(f"Processing time: {batch_metrics['batch_processing_time']:.2f} seconds")
            logger.info(f"Memory usage: {batch_metrics['batch_processing_memory']:.2f} MB")
            logger.info(f"Rows per second: {batch_metrics['rows_per_second']:.2f}")
            logger.info(f"Memory per row: {batch_metrics['memory_per_row']:.4f} MB")
            logger.info(f"Status: {batch_metrics['status']}")

if __name__ == "__main__":
    main() 