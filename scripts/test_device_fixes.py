#!/usr/bin/env python3
"""
Test Device Fixes for STGNN Hyperparameter Optimization
=======================================================

This script tests all the critical device handling fixes applied to resolve the cuda:0 vs cuda mismatch:
1. Device detection and placement
2. Tensor movement to GPU
3. Model device placement
4. DataLoader pin_memory
5. Loss calculation on GPU
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

from utils.stgnn_hyperopt import get_device, manage_memory
from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from scripts.train_stgnn_improved import STGNNClassificationModel, ClassificationSTGNNTrainer
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_device_detection():
    """Test the fixed device detection function"""
    logger.info("=== Testing Device Detection ===")
    
    device = get_device()
    logger.info(f"Detected device: {device}")
    logger.info(f"Device type: {device.type}")
    
    if device.type == 'cuda':
        logger.info(f"GPU device index: {device.index}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(device.index)}")
        
        # Test GPU memory allocation
        test_tensor = torch.randn(1000, 1000).to(device)
        logger.info(f"Test tensor device: {test_tensor.device}")
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(device.index) / 1024 / 1024:.1f} MB")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        
    return device

def test_model_device_placement(device):
    """Test model device placement"""
    logger.info("=== Testing Model Device Placement ===")
    
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
            seq_len=10,
            features=features,
            assets=assets
        )
        
        # Create model
        model = STGNNClassificationModel(
            num_nodes=config.num_nodes,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_classes=3,
            num_layers=1,
            dropout=0.1,
            kernel_size=2
        )
        
        logger.info(f"Model created on device: {next(model.parameters()).device}")
        
        # Move model to device
        model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # Verify model is on correct device
        model_device = next(model.parameters()).device
        if model_device != device:
            logger.error(f"Model device mismatch: {model_device} vs {device}")
            return False
        else:
            logger.info(f"Model successfully on device: {device}")
            return True
            
    except Exception as e:
        logger.error(f"Model device placement test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_tensor_device_movement(device):
    """Test tensor movement to device"""
    logger.info("=== Testing Tensor Device Movement ===")
    
    try:
        # Create test tensors
        batch_size, num_nodes, seq_len, input_dim = 2, 1, 10, 2
        X = torch.randn(batch_size, num_nodes, seq_len, input_dim)
        adj = torch.eye(num_nodes)
        y = torch.randn(batch_size, num_nodes)
        
        logger.info(f"Original tensor devices - X: {X.device}, adj: {adj.device}, y: {y.device}")
        
        # Move tensors to device
        X = X.to(device)
        adj = adj.to(device)
        y = y.to(device)
        
        logger.info(f"Tensors moved to device: {device}")
        logger.info(f"Tensor devices after move - X: {X.device}, adj: {adj.device}, y: {y.device}")
        
        # Verify all tensors are on the same device
        if X.device != device or adj.device != device or y.device != device:
            logger.error("Not all tensors moved to correct device")
            return False
            
        # Test classification target creation
        y_flat = y.flatten().cpu().numpy()
        classes = np.ones(len(y_flat), dtype=int)
        classes[y_flat > 0.005] = 2
        classes[y_flat < -0.005] = 0
        y_classes = torch.LongTensor(classes.reshape(y.shape)).to(device)
        
        logger.info(f"Classification targets device: {y_classes.device}")
        
        if y_classes.device != device:
            logger.error("Classification targets not on correct device")
            return False
            
        logger.info("All tensors successfully moved to device")
        return True
        
    except Exception as e:
        logger.error(f"Tensor device movement test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_dataloader_pin_memory(device):
    """Test DataLoader pin_memory setting"""
    logger.info("=== Testing DataLoader pin_memory ===")
    
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
            seq_len=10,
            features=features,
            assets=assets
        )
        
        # Initialize data processor
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        processor = STGNNDataProcessor(config, market_data, technical_indicators)
        
        # Create test data
        X = torch.randn(10, 1, 10, 2)
        y = torch.randint(0, 3, (10, 1))
        
        # Create dataloader
        dataloader = processor.create_dataloader(X, y, batch_size=2)
        
        # Check if pin_memory is set correctly
        pin_memory_expected = torch.cuda.is_available()
        pin_memory_actual = dataloader.pin_memory
        
        logger.info(f"Expected pin_memory: {pin_memory_expected}")
        logger.info(f"Actual pin_memory: {pin_memory_actual}")
        
        if pin_memory_expected != pin_memory_actual:
            logger.error("DataLoader pin_memory not set correctly")
            return False
            
        logger.info("DataLoader pin_memory set correctly")
        return True
        
    except Exception as e:
        logger.error(f"DataLoader pin_memory test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_trainer_device_placement(device):
    """Test trainer device placement"""
    logger.info("=== Testing Trainer Device Placement ===")
    
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
            seq_len=10,
            features=features,
            assets=assets
        )
        
        # Initialize data processor
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        processor = STGNNDataProcessor(config, market_data, technical_indicators)
        
        # Create trainer with explicit device
        trainer = ClassificationSTGNNTrainer(
            config=config,
            data_processor=processor,
            price_threshold=0.005,
            focal_alpha=1.0,
            focal_gamma=2.0,
            device=device
        )
        
        logger.info(f"Trainer device: {trainer.device}")
        logger.info(f"Model device: {next(trainer.model.parameters()).device}")
        
        # Verify trainer and model are on correct device
        if trainer.device != device:
            logger.error(f"Trainer device mismatch: {trainer.device} vs {device}")
            return False
            
        model_device = next(trainer.model.parameters()).device
        if model_device != device:
            logger.error(f"Model device mismatch: {model_device} vs {device}")
            return False
            
        logger.info("Trainer and model successfully on correct device")
        return True
        
    except Exception as e:
        logger.error(f"Trainer device placement test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_loss_calculation_on_gpu(device):
    """Test loss calculation on GPU"""
    logger.info("=== Testing Loss Calculation on GPU ===")
    
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
            seq_len=10,
            features=features,
            assets=assets
        )
        
        # Initialize data processor
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        processor = STGNNDataProcessor(config, market_data, technical_indicators)
        
        # Create trainer
        trainer = ClassificationSTGNNTrainer(
            config=config,
            data_processor=processor,
            price_threshold=0.005,
            focal_alpha=1.0,
            focal_gamma=2.0,
            device=device
        )
        
        # Create test data
        X = torch.randn(2, 1, 10, 2).to(device)
        adj = torch.eye(1).to(device)
        y = torch.randint(0, 3, (2, 1)).to(device)
        
        logger.info(f"Test data devices - X: {X.device}, adj: {adj.device}, y: {y.device}")
        
        # Test forward pass
        trainer.model.eval()
        with torch.no_grad():
            logits, _ = trainer.model(X, adj)
            logger.info(f"Logits device: {logits.device}")
            logger.info(f"Logits shape: {logits.shape}")
            
            # Test loss calculation
            y_flat = y.view(-1)
            loss = trainer.criterion(logits, y_flat)
            logger.info(f"Loss device: {loss.device}")
            logger.info(f"Loss value: {loss.item()}")
            
            # Check for infinite or NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss detected: {loss.item()}")
                return False
                
        logger.info("Loss calculation on GPU successful")
        return True
        
    except Exception as e:
        logger.error(f"Loss calculation test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all device fix tests"""
    logger.info("Starting comprehensive device fix testing...")
    
    # Test device detection
    device = test_device_detection()
    
    tests = [
        ("Model Device Placement", lambda: test_model_device_placement(device)),
        ("Tensor Device Movement", lambda: test_tensor_device_movement(device)),
        ("DataLoader pin_memory", lambda: test_dataloader_pin_memory(device)),
        ("Trainer Device Placement", lambda: test_trainer_device_placement(device)),
        ("Loss Calculation on GPU", lambda: test_loss_calculation_on_gpu(device))
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
    logger.info("DEVICE FIX TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result is True)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result is True else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL DEVICE FIXES WORKING! GPU utilization should now work correctly.")
    else:
        logger.error("💥 SOME DEVICE FIXES FAILED! Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 