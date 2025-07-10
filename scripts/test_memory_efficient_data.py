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

def test_memory_efficient_data_loading():
    """Test memory-efficient data loading implementation"""
    
    logger.info("="*80)
    logger.info("TESTING MEMORY-EFFICIENT DATA LOADING")
    logger.info("="*80)
    
    # Log initial memory
    manage_memory()
    
    try:
        # Create ultra-minimal configuration
        logger.info("Creating ultra-minimal STGNN configuration...")
        
        config = STGNNConfig(
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
        
        manage_memory()
        
        # Create data processor
        logger.info("Creating memory-efficient data processor...")
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        data_processor = STGNNDataProcessor(config, market_data, technical_indicators)
        
        manage_memory()
        
        # Use ultra-small time window
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)  # Use only 1 day of data
        
        # Test data preparation
        logger.info("Testing memory-efficient data preparation...")
        X, adj, y = data_processor.prepare_data(start_time, end_time)
        
        manage_memory()
        
        logger.info(f"Data preparation successful!")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  adj shape: {adj.shape}")
        logger.info(f"  y shape: {y.shape}")
        
        # Test data splitting
        logger.info("Testing data splitting...")
        X_train, y_train, X_val, y_val = data_processor.split_data(X, y)
        
        manage_memory()
        
        logger.info(f"Data splitting successful!")
        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  y_train shape: {y_train.shape}")
        logger.info(f"  X_val shape: {X_val.shape}")
        logger.info(f"  y_val shape: {y_val.shape}")
        
        # Test dataloader creation
        logger.info("Testing dataloader creation...")
        train_loader = data_processor.create_dataloader(X_train, y_train, batch_size=1)
        val_loader = data_processor.create_dataloader(X_val, y_val, batch_size=1)
        
        manage_memory()
        
        logger.info(f"Dataloader creation successful!")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        
        # Test data iteration
        logger.info("Testing data iteration...")
        for i, (batch_X, batch_y) in enumerate(train_loader):
            logger.info(f"  Batch {i+1}: X shape {batch_X.shape}, y shape {batch_y.shape}")
            if i >= 2:  # Only test first 3 batches
                break
        
        manage_memory()
        
        # Success!
        logger.info("="*80)
        logger.info("✅ MEMORY-EFFICIENT DATA LOADING TEST PASSED!")
        logger.info("="*80)
        logger.info("The memory-efficient data loading implementation works correctly.")
        logger.info("Memory usage stayed under control throughout the process.")
        logger.info("Ready to proceed with hyperparameter optimization.")
        
        return True
        
    except Exception as e:
        logger.error("="*80)
        logger.error("❌ MEMORY-EFFICIENT DATA LOADING TEST FAILED!")
        logger.error("="*80)
        logger.error(f"Error: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        manage_memory()
        
        logger.error("The memory-efficient data loading still has issues.")
        logger.error("Recommendation: Further reduce parameters or switch to alternative architecture.")
        
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
        
        # Test single asset processing
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
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