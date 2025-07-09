#!/usr/bin/env python3
"""
Memory Profiler for STGNN Hyperparameter Optimization

This script helps identify memory bottlenecks in the hyperparameter optimization process.
"""

import psutil
import gc
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators
from scripts.train_stgnn_improved import ClassificationSTGNNTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return memory_gb

def log_memory_usage(stage: str):
    """Log memory usage at a specific stage"""
    memory_gb = get_memory_usage()
    logger.info(f"[{stage}] Memory usage: {memory_gb:.2f} GB")

def profile_data_loading():
    """Profile memory usage during data loading"""
    logger.info("=== Profiling Data Loading ===")
    
    # Initial memory
    log_memory_usage("Start")
    
    # Create config
    config = STGNNConfig(
        num_nodes=1,
        input_dim=15,
        hidden_dim=64,
        output_dim=3,
        num_layers=2,
        dropout=0.2,
        kernel_size=3,
        learning_rate=0.001,
        batch_size=16,
        seq_len=30,
        prediction_horizon=15,
        early_stopping_patience=3,
        features=['price', 'volume', 'rsi', 'macd', 'bollinger', 'atr', 'adx', 'stoch', 'williams_r', 'cci', 'mfi', 'obv', 'vwap', 'support', 'resistance'],
        assets=['ETH/USD'],
        price_threshold=0.005
    )
    
    log_memory_usage("After config creation")
    
    # Create data processor
    market_data = MarketData()
    technical_indicators = TechnicalIndicators()
    data_processor = STGNNDataProcessor(config, market_data, technical_indicators)
    
    log_memory_usage("After data processor creation")
    
    # Load data with 30-day window
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    try:
        X, adj, y = data_processor.prepare_data(start_time, end_time)
        log_memory_usage("After data preparation")
        
        logger.info(f"Data shapes - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
        
        # Convert to classification targets
        y_flat = y.flatten().numpy()
        classes = np.ones(len(y_flat), dtype=int)
        classes[y_flat > config.price_threshold] = 2
        classes[y_flat < -config.price_threshold] = 0
        y_classes = torch.LongTensor(classes.reshape(y.shape))
        
        log_memory_usage("After classification conversion")
        
        # Clean up
        del X, adj, y, y_classes, data_processor
        gc.collect()
        
        log_memory_usage("After cleanup")
        
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        log_memory_usage("After error")

def profile_trainer_creation():
    """Profile memory usage during trainer creation"""
    logger.info("=== Profiling Trainer Creation ===")
    
    log_memory_usage("Start")
    
    # Create config
    config = STGNNConfig(
        num_nodes=1,
        input_dim=15,
        hidden_dim=64,
        output_dim=3,
        num_layers=2,
        dropout=0.2,
        kernel_size=3,
        learning_rate=0.001,
        batch_size=16,
        seq_len=30,
        prediction_horizon=15,
        early_stopping_patience=3,
        features=['price', 'volume', 'rsi', 'macd', 'bollinger', 'atr', 'adx', 'stoch', 'williams_r', 'cci', 'mfi', 'obv', 'vwap', 'support', 'resistance'],
        assets=['ETH/USD'],
        price_threshold=0.005
    )
    
    log_memory_usage("After config creation")
    
    # Create data processor
    market_data = MarketData()
    technical_indicators = TechnicalIndicators()
    data_processor = STGNNDataProcessor(config, market_data, technical_indicators)
    
    log_memory_usage("After data processor creation")
    
    # Create trainer
    trainer = ClassificationSTGNNTrainer(
        config=config,
        data_processor=data_processor,
        price_threshold=0.005,
        focal_alpha=1.0,
        focal_gamma=2.0,
        class_weights=None,
        start_time=datetime.now() - timedelta(days=30),
        end_time=datetime.now()
    )
    
    log_memory_usage("After trainer creation")
    
    # Clean up
    del trainer, data_processor
    gc.collect()
    
    log_memory_usage("After cleanup")

def profile_smote_processing():
    """Profile memory usage during SMOTE processing"""
    logger.info("=== Profiling SMOTE Processing ===")
    
    log_memory_usage("Start")
    
    # Create synthetic data for SMOTE testing
    batch_size = 1000
    num_nodes = 1
    seq_len = 30
    input_dim = 15
    
    # Create synthetic X and y
    X = torch.randn(batch_size, num_nodes, seq_len, input_dim)
    y = torch.randint(0, 3, (batch_size, num_nodes))
    
    log_memory_usage("After synthetic data creation")
    
    # Simulate SMOTE processing
    from imblearn.over_sampling import SMOTE
    
    # Reshape for SMOTE
    X_reshaped = X.reshape(batch_size * num_nodes, seq_len * input_dim)
    y_reshaped = y.reshape(-1)
    
    log_memory_usage("After reshaping for SMOTE")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X_reshaped.numpy(), y_reshaped.numpy())
    
    log_memory_usage("After SMOTE processing")
    
    # Reshape back
    new_batch_size = len(y_balanced) // num_nodes
    X_balanced_reshaped = X_balanced.reshape(new_batch_size, num_nodes, seq_len, input_dim)
    y_balanced_reshaped = y_balanced.reshape(new_batch_size, num_nodes)
    
    log_memory_usage("After reshaping back")
    
    logger.info(f"Original batch size: {batch_size}, Balanced batch size: {new_batch_size}")
    logger.info(f"Memory increase factor: {new_batch_size / batch_size:.2f}x")
    
    # Clean up
    del X, y, X_reshaped, y_reshaped, X_balanced, y_balanced, X_balanced_reshaped, y_balanced_reshaped
    gc.collect()
    
    log_memory_usage("After cleanup")

def main():
    """Run memory profiling"""
    logger.info("Starting memory profiling for STGNN hyperparameter optimization")
    
    # Profile each component
    profile_data_loading()
    profile_trainer_creation()
    profile_smote_processing()
    
    logger.info("Memory profiling completed")

if __name__ == "__main__":
    main() 