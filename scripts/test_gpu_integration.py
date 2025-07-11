#!/usr/bin/env python3
"""
Test GPU Integration for STGNN Hyperparameter Optimization

This script tests GPU integration with ultra-minimal parameters to ensure:
1. GPU is detected and used correctly
2. Memory usage is monitored properly
3. Training works with GPU acceleration
4. No OOM issues occur with minimal parameters
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import gc
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators
from scripts.train_stgnn_improved import ClassificationSTGNNTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_device():
    """Get the best available device for training"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB")
        return device
    else:
        device = torch.device('cpu')
        logger.info("GPU not available, using CPU")
        return device

def manage_memory():
    """Force garbage collection and log memory usage"""
    gc.collect()
    
    # Log CPU memory usage
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"CPU Memory usage: {memory_mb:.1f} MB")
    
    # Log GPU memory usage if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory_mb = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
            gpu_memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            gpu_memory_cached = torch.cuda.memory_reserved(i) / 1024 / 1024
            logger.info(f"GPU {i} Memory: {gpu_memory_allocated:.1f}MB allocated, {gpu_memory_cached:.1f}MB cached, {gpu_memory_mb:.1f}MB total")
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_ultra_minimal_config():
    """Create ultra-minimal configuration for GPU testing"""
    
    # Ultra-minimal parameters with proper tensor shapes
    config = STGNNConfig(
        num_nodes=1,  # Single asset
        input_dim=2,  # Only 2 features
        hidden_dim=8,  # Small but sufficient for tensor shapes
        output_dim=3,  # 3 classes for classification
        num_layers=1,  # Single layer
        dropout=0.1,  # Minimal dropout
        kernel_size=3,  # Standard kernel size
        learning_rate=0.001,  # Standard learning rate
        batch_size=2,  # Small batch size
        num_epochs=1,  # Single epoch for testing
        early_stopping_patience=2,
        seq_len=10,  # Slightly longer sequences for proper shapes
        prediction_horizon=15,
        features=['returns', 'volume'],  # Only 2 features
        assets=['ETH/USD'],  # Single asset
        confidence_threshold=0.51,
        buy_threshold=0.6,
        sell_threshold=0.4,
        retrain_interval=24,
        focal_alpha=1.0,
        focal_gamma=2.0,
        # Feature engineering parameters (minimal)
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
        price_threshold=0.005  # 0.5% threshold
    )
    
    return config

def test_gpu_integration():
    """Test GPU integration with ultra-minimal configuration"""
    
    logger.info("="*60)
    logger.info("TESTING GPU INTEGRATION")
    logger.info("="*60)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Log initial memory
    manage_memory()
    
    try:
        # Create ultra-minimal configuration
        config = create_ultra_minimal_config()
        logger.info("Ultra-minimal configuration created")
        logger.info(f"Hidden dim: {config.hidden_dim}, Layers: {config.num_layers}")
        logger.info(f"Batch size: {config.batch_size}, Seq len: {config.seq_len}")
        logger.info(f"Features: {config.features}")
        
        # Initialize components
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        
        # Create data processor
        data_processor = STGNNDataProcessor(config, market_data, technical_indicators)
        logger.info("Data processor created")
        
        # Use valid date range in March 2025 (data goes up to May 2025)
        end_time = datetime(2025, 3, 15, 12, 0, 0)  # March 15, 2025
        start_time = end_time - timedelta(days=1)  # March 14, 2025
        logger.info(f"Using date range: {start_time} to {end_time}")
        
        # Create trainer with device
        trainer = ClassificationSTGNNTrainer(
            config=config,
            data_processor=data_processor,
            price_threshold=config.price_threshold,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            class_weights=None,
            start_time=start_time,
            end_time=end_time,
            device=device
        )
        logger.info("Trainer created with device")
        
        # Monitor memory after trainer creation
        manage_memory()
        
        # Prepare data
        logger.info("Preparing classification data...")
        X, adj, y_classes = trainer.prepare_classification_data()
        logger.info(f"Data prepared - X: {X.shape}, adj: {adj.shape}, y: {y_classes.shape}")
        
        # Monitor memory after data preparation
        manage_memory()
        
        # Split data
        X_train, y_train, X_val, y_val = data_processor.split_data(X, y_classes)
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Create dataloaders
        train_loader = data_processor.create_dataloader(X_train, y_train, drop_last=True)
        val_loader = data_processor.create_dataloader(X_val, y_val, drop_last=False)
        logger.info("Dataloaders created")
        
        # Monitor memory after dataloader creation
        manage_memory()
        
        # Test single training epoch
        logger.info("Testing single training epoch...")
        train_loss, train_acc = trainer.train_epoch(train_loader)
        logger.info(f"Training epoch completed - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # Monitor memory after training
        manage_memory()
        
        # Test validation
        logger.info("Testing validation...")
        val_loss, val_acc = trainer.validate(val_loader)
        logger.info(f"Validation completed - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Monitor memory after validation
        manage_memory()
        
        # Test evaluation
        logger.info("Testing evaluation...")
        evaluation_results = trainer.evaluate(X_val, y_val)
        logger.info("Evaluation completed")
        logger.info(f"F1 scores: {evaluation_results['f1']}")
        
        # Final memory check
        manage_memory()
        
        logger.info("="*60)
        logger.info("GPU INTEGRATION TEST COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Device used: {device}")
        logger.info(f"Training loss: {train_loss:.4f}")
        logger.info(f"Training accuracy: {train_acc:.4f}")
        logger.info(f"Validation loss: {val_loss:.4f}")
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        logger.info(f"F1 scores: {evaluation_results['f1']}")
        
        return True
        
    except Exception as e:
        logger.error(f"GPU integration test failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main function to test GPU integration"""
    
    logger.info("Starting GPU integration test...")
    
    # Test GPU integration
    success = test_gpu_integration()
    
    if success:
        logger.info("✅ GPU integration test PASSED")
        logger.info("Ready to proceed with GPU hyperparameter optimization")
    else:
        logger.error("❌ GPU integration test FAILED")
        logger.error("Need to fix issues before proceeding")
    
    return success

if __name__ == "__main__":
    main() 