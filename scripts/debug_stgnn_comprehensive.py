#!/usr/bin/env python3
"""
Comprehensive STGNN Debugging Script
====================================

This script tests all the critical fixes applied to the STGNN pipeline:
1. GPU device placement
2. Tensor shape handling
3. Loss calculation
4. Model forward pass
5. Data loading and processing
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
from utils.stgnn_utils import STGNNModel, GraphConvolution
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
    """Test GPU device detection and placement"""
    logger.info("=== Testing Device Detection ===")
    
    # Test device detection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB")
        
        # Test GPU memory allocation
        test_tensor = torch.randn(1000, 1000).to(device)
        logger.info(f"Test tensor device: {test_tensor.device}")
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 / 1024:.1f} MB")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        
    else:
        device = torch.device('cpu')
        logger.info("GPU not available, using CPU")
    
    return device

def test_graph_convolution():
    """Test GraphConvolution layer with device placement"""
    logger.info("=== Testing GraphConvolution ===")
    
    device = test_device_detection()
    
    try:
        # Create test data
        batch_size, num_nodes, features = 4, 1, 10
        x = torch.randn(batch_size, num_nodes, features).to(device)
        adj = torch.eye(num_nodes).to(device)
        
        logger.info(f"Input shapes - x: {x.shape}, adj: {adj.shape}")
        logger.info(f"Device - x: {x.device}, adj: {adj.device}")
        
        # Create GraphConvolution layer
        graph_conv = GraphConvolution(in_features=features, out_features=5).to(device)
        logger.info(f"GraphConvolution device: {next(graph_conv.parameters()).device}")
        
        # Test forward pass without attention
        output = graph_conv(x, adj, return_attention=False)
        logger.info(f"Output shape (no attention): {output.shape}")
        logger.info(f"Output device: {output.device}")
        
        # Test forward pass with attention
        output, attention = graph_conv(x, adj, return_attention=True)
        logger.info(f"Output shape (with attention): {output.shape}")
        logger.info(f"Attention shape: {attention.shape}")
        logger.info(f"Output device: {output.device}, Attention device: {attention.device}")
        
        # Check for None values
        if output is None:
            logger.error("Output is None!")
            return False
        if attention is None:
            logger.error("Attention is None!")
            return False
            
        logger.info("GraphConvolution test passed!")
        return True
        
    except Exception as e:
        logger.error(f"GraphConvolution test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_stgnn_model():
    """Test STGNNModel with device placement"""
    logger.info("=== Testing STGNNModel ===")
    
    device = test_device_detection()
    
    try:
        # Create test data
        batch_size, num_nodes, seq_len, input_dim = 2, 1, 5, 10
        x = torch.randn(batch_size, num_nodes, seq_len, input_dim).to(device)
        adj = torch.eye(num_nodes).to(device)
        
        logger.info(f"Input shapes - x: {x.shape}, adj: {adj.shape}")
        logger.info(f"Device - x: {x.device}, adj: {adj.device}")
        
        # Create STGNNModel
        model = STGNNModel(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=3,
            num_layers=1,
            dropout=0.1,
            kernel_size=2
        ).to(device)
        
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        # Test forward pass
        output, attention_dict = model(x, adj)
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Output device: {output.device}")
        logger.info(f"Attention dict keys: {list(attention_dict.keys())}")
        
        # Check for None values
        if output is None:
            logger.error("Output is None!")
            return False
        if attention_dict is None:
            logger.error("Attention dict is None!")
            return False
            
        logger.info("STGNNModel test passed!")
        return True
        
    except Exception as e:
        logger.error(f"STGNNModel test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_classification_model():
    """Test STGNNClassificationModel with device placement"""
    logger.info("=== Testing STGNNClassificationModel ===")
    
    device = test_device_detection()
    
    try:
        # Create test data
        batch_size, num_nodes, seq_len, input_dim = 2, 1, 5, 10
        x = torch.randn(batch_size, num_nodes, seq_len, input_dim).to(device)
        adj = torch.eye(num_nodes).to(device)
        
        logger.info(f"Input shapes - x: {x.shape}, adj: {adj.shape}")
        logger.info(f"Device - x: {x.device}, adj: {adj.device}")
        
        # Create STGNNClassificationModel
        model = STGNNClassificationModel(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=8,
            num_classes=3,
            num_layers=1,
            dropout=0.1,
            kernel_size=2
        ).to(device)
        
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        # Test forward pass
        logits, attention_dict = model(x, adj)
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Logits device: {logits.device}")
        logger.info(f"Attention dict keys: {list(attention_dict.keys())}")
        
        # Check for None values
        if logits is None:
            logger.error("Logits is None!")
            return False
        if attention_dict is None:
            logger.error("Attention dict is None!")
            return False
            
        # Test loss calculation
        y_batch = torch.randint(0, 3, (batch_size * num_nodes,)).to(device)
        logger.info(f"Target shape: {y_batch.shape}")
        logger.info(f"Target device: {y_batch.device}")
        
        # Create focal loss
        from scripts.train_stgnn_improved import FocalLoss
        criterion = FocalLoss(alpha=1.0, gamma=2.0).to(device)
        
        loss = criterion(logits, y_batch)
        logger.info(f"Loss value: {loss.item()}")
        logger.info(f"Loss device: {loss.device}")
        
        # Check for infinite or NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Invalid loss detected: {loss.item()}")
            return False
            
        logger.info("STGNNClassificationModel test passed!")
        return True
        
    except Exception as e:
        logger.error(f"STGNNClassificationModel test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_data_loading():
    """Test data loading with device placement"""
    logger.info("=== Testing Data Loading ===")
    
    device = test_device_detection()
    
    try:
        # Create minimal config
        assets = ['ETH/USD']
        features = ['returns', 'volume']  # Minimal features for testing
        
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
        
        # Load minimal data
        data = processor.prepare_data(
            start_time='2024-01-01',
            end_time='2024-01-02'
        )
        
        X, adj, y = data
        logger.info(f"Data shapes - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
        
        # Move to device
        X = X.to(device)
        adj = adj.to(device)
        y = y.to(device)
        
        logger.info(f"Device - X: {X.device}, adj: {adj.device}, y: {y.device}")
        
        # Test with classification model
        model = STGNNClassificationModel(
            num_nodes=config.num_nodes,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_classes=3,
            num_layers=1,
            dropout=0.1,
            kernel_size=2
        ).to(device)
        
        # Test forward pass
        logits, _ = model(X, adj)
        logger.info(f"Model output shape: {logits.shape}")
        logger.info(f"Model output device: {logits.device}")
        
        logger.info("Data loading test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_trainer():
    """Test ClassificationSTGNNTrainer with device placement"""
    logger.info("=== Testing ClassificationSTGNNTrainer ===")
    
    device = test_device_detection()
    
    try:
        # Create minimal config
        assets = ['ETH/USD']
        features = ['returns', 'volume']  # Minimal features for testing
        
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
        
        # Create trainer
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
        
        # Test data preparation
        X, adj, y_classes = trainer.prepare_classification_data()
        logger.info(f"Prepared data shapes - X: {X.shape}, adj: {adj.shape}, y: {y_classes.shape}")
        logger.info(f"Data devices - X: {X.device}, adj: {adj.device}, y: {y_classes.device}")
        
        # Test single forward pass
        trainer.model.eval()
        with torch.no_grad():
            logits, _ = trainer.model(X[:2], adj)  # Use first 2 samples
            logger.info(f"Forward pass output shape: {logits.shape}")
            logger.info(f"Forward pass output device: {logits.device}")
            
            # Test loss calculation
            y_batch = y_classes[:2].view(-1)
            loss = trainer.criterion(logits, y_batch)
            logger.info(f"Loss value: {loss.item()}")
            logger.info(f"Loss device: {loss.device}")
            
            # Check for infinite or NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss detected: {loss.item()}")
                return False
        
        logger.info("ClassificationSTGNNTrainer test passed!")
        return True
        
    except Exception as e:
        logger.error(f"ClassificationSTGNNTrainer test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all comprehensive tests"""
    logger.info("Starting comprehensive STGNN debugging...")
    
    tests = [
        ("Device Detection", test_device_detection),
        ("GraphConvolution", test_graph_convolution),
        ("STGNNModel", test_stgnn_model),
        ("STGNNClassificationModel", test_classification_model),
        ("Data Loading", test_data_loading),
        ("ClassificationSTGNNTrainer", test_trainer)
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
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result is True)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result is True else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! STGNN pipeline is working correctly.")
    else:
        logger.error("💥 SOME TESTS FAILED! Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 