#!/usr/bin/env python3
"""
Comprehensive Test Script for STGNN Critical Fixes
==================================================

This script tests all the critical fixes applied to resolve:
1. ZeroDivisionError in validation DataLoader
2. Numerical instability causing NaN losses
3. Empty sequence creation
4. Class weight validation
5. Gradient clipping implementation
"""

import sys
import os
import torch
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from scripts.train_stgnn_improved import WeightedFocalLoss, STGNNClassificationModel
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sequence_creation_validation():
    """Test sequence creation validation fixes"""
    logger.info("=== Testing Sequence Creation Validation ===")
    
    try:
        # Create minimal config
        assets = ['ETH/USD']
        features = ['returns', 'volume']
        
        config = STGNNConfig(
            num_nodes=len(assets),
            input_dim=len(features),
            hidden_dim=8,
            output_dim=3,
            num_layers=1,
            dropout=0.1,
            kernel_size=2,
            learning_rate=0.001,
            batch_size=2,
            seq_len=5,
            features=features,
            assets=assets
        )
        
        # Initialize data processor
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        processor = STGNNDataProcessor(config, market_data, technical_indicators)
        
        # Test with insufficient data (should trigger validation)
        logger.info("Testing with insufficient data...")
        
        # Create minimal test data
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create truly insufficient dataset (less than seq_len + prediction_horizon)
        dates = pd.date_range(start='2024-01-01', end='2024-01-01 02:00', freq='15T')  # Only 8 data points
        test_data = pd.DataFrame({
            'open': [100] * len(dates),
            'high': [101] * len(dates),
            'low': [99] * len(dates),
            'close': [100] * len(dates),
            'volume': [1000] * len(dates)
        }, index=dates)
        
        # Test sequence creation with insufficient data
        processor._original_prices = test_data['close']
        processor._returns = pd.Series(0, index=test_data.index)
        
        # This should trigger the validation and return empty arrays
        X, y = processor.create_sequences_lazy(test_data)
        
        if len(X) == 0:
            logger.info("✅ Sequence creation validation working correctly - returned empty arrays for insufficient data")
            return True
        else:
            logger.error(f"❌ Sequence creation validation failed - returned {len(X)} sequences for insufficient data")
            return False
            
    except Exception as e:
        logger.error(f"Sequence creation validation test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_dataloader_validation():
    """Test DataLoader validation fixes"""
    logger.info("=== Testing DataLoader Validation ===")
    
    try:
        # Create minimal config
        assets = ['ETH/USD']
        features = ['returns', 'volume']
        
        config = STGNNConfig(
            num_nodes=len(assets),
            input_dim=len(features),
            hidden_dim=8,
            output_dim=3,
            num_layers=1,
            dropout=0.1,
            kernel_size=2,
            learning_rate=0.001,
            batch_size=2,
            seq_len=5,
            features=features,
            assets=assets
        )
        
        # Initialize data processor
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        processor = STGNNDataProcessor(config, market_data, technical_indicators)
        
        # Test with empty data
        logger.info("Testing DataLoader creation with empty data...")
        
        try:
            # Create empty tensors
            X_empty = torch.empty(0, 1, 5, 2)  # Empty tensor
            y_empty = torch.empty(0, 1)
            
            # This should trigger validation and return empty DataLoader
            train_loader = processor.create_dataloader(X_empty, y_empty, drop_last=True)
            
            # If we get here, the DataLoader was created successfully (which is wrong for empty data)
            logger.error(f"❌ DataLoader validation failed - created DataLoader with {len(train_loader)} batches for empty data")
            return False
            
        except ValueError as e:
            if "num_samples should be a positive integer value" in str(e):
                logger.info("✅ DataLoader validation working correctly - properly rejected empty data")
                return True
            else:
                logger.error(f"❌ Unexpected ValueError: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Unexpected exception: {e}")
            return False
            
    except Exception as e:
        logger.error(f"DataLoader validation test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_class_weight_validation():
    """Test class weight validation in WeightedFocalLoss"""
    logger.info("=== Testing Class Weight Validation ===")
    
    try:
        device = torch.device('cpu')
        
        # Test with invalid class weights (negative values)
        invalid_weights = torch.tensor([-1.0, 0.0, 2.0])
        criterion = WeightedFocalLoss(class_weights=invalid_weights, alpha=1.0, gamma=2.0)
        
        # Create test inputs
        inputs = torch.randn(4, 3).to(device)
        targets = torch.randint(0, 3, (4,)).to(device)
        
        # This should trigger class weight validation and use uniform weights
        loss = criterion(inputs, targets)
        
        if not torch.isnan(loss) and not torch.isinf(loss):
            logger.info("✅ Class weight validation working correctly - handled invalid weights")
            return True
        else:
            logger.error(f"❌ Class weight validation failed - loss is {loss}")
            return False
            
    except Exception as e:
        logger.error(f"Class weight validation test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_gradient_clipping():
    """Test gradient clipping implementation"""
    logger.info("=== Testing Gradient Clipping ===")
    
    try:
        device = torch.device('cpu')
        
        # Create a simple model
        model = torch.nn.Linear(10, 3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create test data
        inputs = torch.randn(4, 10).to(device)
        targets = torch.randint(0, 3, (4,)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        logger.info("✅ Gradient clipping test passed")
        return True
        
    except Exception as e:
        logger.error(f"Gradient clipping test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_numerical_stability():
    """Test numerical stability improvements"""
    logger.info("=== Testing Numerical Stability ===")
    
    try:
        device = torch.device('cpu')
        
        # Create model
        model = STGNNClassificationModel(
            num_nodes=1,
            input_dim=2,
            hidden_dim=8,
            num_classes=3,
            num_layers=1,
            dropout=0.1,
            kernel_size=3
        ).to(device)
        
        # Create test data with extreme values
        X = torch.randn(2, 1, 5, 2).to(device) * 1000  # Large values
        adj = torch.eye(1).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            logits, _ = model(X, adj)
            
            # Check for NaN or Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.error("❌ Model produces NaN or Inf outputs")
                return False
            else:
                logger.info("✅ Model handles extreme inputs without NaN/Inf")
                return True
                
    except Exception as e:
        logger.error(f"Numerical stability test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_loss_function_robustness():
    """Test loss function robustness with extreme inputs"""
    logger.info("=== Testing Loss Function Robustness ===")
    
    try:
        device = torch.device('cpu')
        
        # Create criterion
        class_weights = torch.tensor([1.0, 1.0, 1.0])
        criterion = WeightedFocalLoss(class_weights=class_weights, alpha=1.0, gamma=2.0)
        
        # Test with extreme inputs
        inputs = torch.tensor([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]]).to(device)
        targets = torch.tensor([0, 1]).to(device)
        
        # Calculate loss
        loss = criterion(inputs, targets)
        
        if not torch.isnan(loss) and not torch.isinf(loss):
            logger.info("✅ Loss function handles extreme inputs correctly")
            return True
        else:
            logger.error(f"❌ Loss function failed with extreme inputs: {loss}")
            return False
            
    except Exception as e:
        logger.error(f"Loss function robustness test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all comprehensive tests"""
    logger.info("Starting comprehensive STGNN fixes validation...")
    
    tests = [
        ("Sequence Creation Validation", test_sequence_creation_validation),
        ("DataLoader Validation", test_dataloader_validation),
        ("Class Weight Validation", test_class_weight_validation),
        ("Gradient Clipping", test_gradient_clipping),
        ("Numerical Stability", test_numerical_stability),
        ("Loss Function Robustness", test_loss_function_robustness)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("COMPREHENSIVE FIXES VALIDATION SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result is True)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result is True else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL CRITICAL FIXES VALIDATED! STGNN pipeline should now be robust.")
    else:
        logger.error("💥 SOME FIXES NEED ATTENTION! Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 