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

from utils.feature_generator import FeatureGenerator
from utils.volatility_analyzer import VolatilityAnalyzer
from utils.synthetic_data_generator import SyntheticDataGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # 1. Profile basic feature generation
    start_time = time.time()
    features = feature_gen.generate_features(ohlcv_data, order_book_data)
    results['basic_features'] = time.time() - start_time
    logger.info(f"Basic features shape: {features.shape}")
    
    # 2. Profile technical features
    start_time = time.time()
    tech_features = feature_gen.generate_technical_features(ohlcv_data)
    results['technical_features'] = time.time() - start_time
    logger.info(f"Technical features shape: {tech_features.shape}")
    
    # 3. Profile momentum features
    start_time = time.time()
    momentum_features = feature_gen.generate_momentum_features(ohlcv_data)
    results['momentum_features'] = time.time() - start_time
    logger.info(f"Momentum features shape: {momentum_features.shape}")
    
    # 4. Profile order book features
    start_time = time.time()
    ob_features = feature_gen.generate_order_book_features(ohlcv_data, order_book_data)
    results['order_book_features'] = time.time() - start_time
    logger.info(f"Order book features shape: {ob_features.shape}")
    
    # 5. Profile volatility features
    start_time = time.time()
    vol_features = vol_analyzer.prepare_volatility_features(ohlcv_data, is_intraday=True)
    results['volatility_features'] = time.time() - start_time
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
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
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
    
    # 1. Basic features
    features = feature_gen.generate_features(ohlcv_data, order_book_data)
    results['basic_features'] = process.memory_info().rss / 1024 / 1024 - initial_memory
    
    # 2. Technical features
    tech_features = feature_gen.generate_technical_features(ohlcv_data)
    results['technical_features'] = process.memory_info().rss / 1024 / 1024 - initial_memory
    
    # 3. Momentum features
    momentum_features = feature_gen.generate_momentum_features(ohlcv_data)
    results['momentum_features'] = process.memory_info().rss / 1024 / 1024 - initial_memory
    
    # 4. Order book features
    ob_features = feature_gen.generate_order_book_features(ohlcv_data, order_book_data)
    results['order_book_features'] = process.memory_info().rss / 1024 / 1024 - initial_memory
    
    # 5. Volatility features
    vol_features = vol_analyzer.prepare_volatility_features(ohlcv_data, is_intraday=True)
    results['volatility_features'] = process.memory_info().rss / 1024 / 1024 - initial_memory
    
    return results

def main():
    """Main profiling function."""
    # Create results directory
    results_dir = Path("profiling_results")
    results_dir.mkdir(exist_ok=True)
    
    # Profile different data sizes
    data_sizes = [1000, 5000, 10000]
    timing_results = {}
    memory_results = {}
    
    for size in data_sizes:
        logger.info(f"\nProfiling with {size} data points...")
        
        # Profile timing
        timing_results[size] = profile_feature_generation(
            data_size=size,
            interval_minutes=15,
            n_levels=10
        )
        
        # Profile memory
        memory_results[size] = profile_memory_usage(
            data_size=size,
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
    
    # Print summary
    logger.info("\nTiming Results (seconds):")
    logger.info(timing_df)
    
    logger.info("\nMemory Usage (MB):")
    logger.info(memory_df)

if __name__ == "__main__":
    main() 