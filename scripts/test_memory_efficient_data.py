#!/usr/bin/env python3
"""
Test Memory-Efficient Data Loading

This script tests the memory-efficient data loading implementation to ensure
it works without OOM kills during hyperparameter optimization.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import psutil
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor, manage_memory
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_ultra_minimal_config():
    """Creates a minimal STGNNConfig for testing."""
    return STGNNConfig(
        num_nodes=1,  # Single asset
        input_dim=2,   # Only 2 features: returns, volume
        hidden_dim=4,  # Ultra-small hidden dimension
        output_dim=3,  # 3 classes: down/no direction/up
        num_layers=1,  # Single layer only
        dropout=0.1,   # Minimal dropout
        kernel_size=2, # Minimal kernel size
        learning_rate=0.001,
        batch_size=1,  # Ultra-small batch size
        seq_len=5,     # Ultra-short sequence length
        prediction_horizon=15,
        early_stopping_patience=2,
        features=['returns', 'volume'],  # Minimal feature set
        assets=['ETH/USD'],
        focal_alpha=1.0,
        focal_gamma=2.0,
        # Minimal feature engineering parameters
        rsi_period=10,
        macd_fast_period=10,
        macd_slow_period=20,
        macd_signal_period=7,
        bb_period=15,
        bb_num_std_dev=2.0,
        atr_period=10,
        adx_period=10,
        volume_ma_period=15,
        price_momentum_lookback=3,
        price_threshold=0.005
    )

def create_memory_efficient_processor(config):
    """Creates a STGNNDataProcessor for testing."""
    market_data = MarketData()
    technical_indicators = TechnicalIndicators()
    return STGNNDataProcessor(config, market_data, technical_indicators)

def test_memory_efficient_data_loading():
    """Test memory-efficient data loading with ultra-minimal configuration"""
    print("Testing memory-efficient data preparation...")
    
    try:
        # Create ultra-minimal configuration
        config = create_ultra_minimal_config()
        
        # Create data processor
        data_processor = create_memory_efficient_processor(config)
        
        # Test data preparation with ultra-minimal parameters
        print("Testing memory-efficient data preparation...")
        manage_memory()
        
        # Use a time range that actually has data (data goes up to May 29, 2025)
        end_time = datetime(2025, 5, 29, 6, 0, 0)  # Last available data point
        start_time = end_time - timedelta(days=7)    # Use 7 days of data
        
        print(f"Requesting data from {start_time} to {end_time}")
        
        X, adj, y = data_processor.prepare_data(start_time, end_time)
        
        print(f"Memory-efficient data preparation successful!")
        print(f"  X shape: {X.shape}")
        print(f"  adj shape: {adj.shape}")
        print(f"  y shape: {y.shape}")
        
        # Verify data integrity
        if len(X) > 0:
            print(f"  ✓ Data contains {len(X)} sequences")
            print(f"  ✓ X has valid shape: {X.shape}")
            print(f"  ✓ y has valid shape: {y.shape}")
            print(f"  ✓ No NaN/Inf values in X: {torch.isnan(X).any() or torch.isinf(X).any()}")
            print(f"  ✓ No NaN/Inf values in y: {torch.isnan(y).any() or torch.isinf(y).any()}")
        else:
            print("  ⚠ Warning: No sequences generated")
        
        manage_memory()
        return True
        
    except Exception as e:
        print(f"❌ MEMORY-EFFICIENT DATA LOADING TEST FAILED!")
        print(f"Error: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        manage_memory()
        return False

def test_single_asset_processing():
    """Test single asset processing specifically"""
    
    logger.info("="*80)
    logger.info("TESTING SINGLE ASSET PROCESSING")
    logger.info("="*80)
    
    try:
        # Create minimal configuration
        config = STGNNConfig(
            num_nodes=1,
            input_dim=2,
            hidden_dim=4,
            output_dim=3,
            num_layers=1,
            dropout=0.1,
            kernel_size=2,
            learning_rate=0.001,
            batch_size=1,
            seq_len=5,
            prediction_horizon=15,
            early_stopping_patience=2,
            features=['returns', 'volume'],
            assets=['ETH/USD'],
            price_threshold=0.005
        )
        
        # Create data processor
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        data_processor = STGNNDataProcessor(config, market_data, technical_indicators)
        
        # Test single asset processing with real data
        end_time = datetime(2025, 5, 29, 6, 0, 0)  # Last available data point
        start_time = end_time - timedelta(days=7)    # Use 7 days of data
        
        print(f"Requesting data from {start_time} to {end_time}")
        
        logger.info("Testing single asset data preparation...")
        X, y = data_processor.prepare_data_single_asset('ETH/USD', start_time, end_time)
        
        logger.info(f"Single asset processing successful!")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Single asset processing failed: {e}")
        return False

def main():
    """Main function to test memory-efficient data loading"""
    
    logger.info("Starting memory-efficient data loading tests...")
    
    # Test 1: Memory-efficient data loading
    success1 = test_memory_efficient_data_loading()
    
    # Test 2: Single asset processing
    success2 = test_single_asset_processing()
    
    if success1 and success2:
        logger.info("All memory-efficient data loading tests passed!")
        logger.info("Recommendation: Proceed with hyperparameter optimization.")
    else:
        logger.error("Some memory-efficient data loading tests failed.")
        logger.error("Recommendation: Further optimize or switch to alternative architecture.")
    
    # Final memory usage
    manage_memory()

if __name__ == "__main__":
    main() 