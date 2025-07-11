#!/usr/bin/env python3
"""
Debug STGNN None Tensor Error - Comprehensive Analysis
=====================================================

This script performs detailed debugging to identify the exact cause of the
'NoneType' object has no attribute 'dim' error in the STGNN model.
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
from utils.stgnn_utils import STGNNModel
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_tensor_shapes_and_values():
    """Debug tensor shapes and values to identify None tensor issues"""
    logger.info("=== STGNN Tensor Shape and Value Debugging ===")
    
    try:
        # Initialize dependencies
        logger.info("1. Initializing dependencies...")
        
        # Create proper config with all required parameters
        assets = ['ETH/USD']
        features = [
            'returns', 'log_returns',
            'rsi', 'atr', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'volume', 'volume_ma', 'volume_std', 'volume_surge', 'volume_ratio',
            'ma_crossover', 'price_momentum', 'volatility_regime',
            'support', 'resistance', 'breakout_intensity',
            'vwap_ratio', 'cumulative_delta'
        ]
        
        config = STGNNConfig(
            num_nodes=len(assets),
            input_dim=len(features),
            hidden_dim=128,
            output_dim=3,
            num_layers=3,
            dropout=0.3,
            kernel_size=3,
            learning_rate=0.0005,
            batch_size=16,
            num_epochs=100,
            early_stopping_patience=10,
            seq_len=200,
            prediction_horizon=15,
            features=features,
            assets=assets,
            confidence_threshold=0.51,
            buy_threshold=0.6,
            sell_threshold=0.4,
            retrain_interval=24,
            focal_alpha=1.0,
            focal_gamma=3.0
        )
        
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        
        # Initialize data processor
        logger.info("2. Initializing STGNNDataProcessor...")
        processor = STGNNDataProcessor(config, market_data, technical_indicators)
        
        # Load data
        logger.info("3. Loading data...")
        data = processor.prepare_data(
            start_time='2020-01-01',  # Full 5-year dataset
            end_time='2025-05-29'
        )
        
        logger.info(f"4. Data loaded successfully:")
        logger.info(f"   X shape: {data[0].shape}")
        logger.info(f"   adj shape: {data[1].shape}")
        logger.info(f"   y shape: {data[2].shape}")
        
        # Check for None values
        logger.info("5. Checking for None values...")
        if data[0] is None:
            logger.error("X is None!")
        if data[1] is None:
            logger.error("adj is None!")
        if data[2] is None:
            logger.error("y is None!")
            
        # Check tensor properties
        logger.info("6. Checking tensor properties...")
        logger.info(f"   X dtype: {data[0].dtype}")
        logger.info(f"   adj dtype: {data[1].dtype}")
        logger.info(f"   y dtype: {data[2].dtype}")
        
        logger.info(f"   X device: {data[0].device}")
        logger.info(f"   adj device: {data[1].device}")
        logger.info(f"   y device: {data[2].device}")
        
        # Check for NaN or inf values
        logger.info("7. Checking for NaN or inf values...")
        if torch.isnan(data[0]).any():
            nan_count = torch.isnan(data[0]).sum().item()
            total_elements = data[0].numel()
            nan_percentage = (nan_count / total_elements) * 100
            logger.error(f"X contains {nan_count} NaN values ({nan_percentage:.2f}% of all elements)")
            
            # Analyze NaN patterns by feature
            X_reshaped = data[0].reshape(-1, data[0].shape[-1])  # Flatten batch and time dimensions
            for i in range(X_reshaped.shape[1]):
                feature_nan_count = torch.isnan(X_reshaped[:, i]).sum().item()
                feature_nan_percentage = (feature_nan_count / X_reshaped.shape[0]) * 100
                if feature_nan_percentage > 0:
                    logger.warning(f"  Feature {i}: {feature_nan_count} NaN values ({feature_nan_percentage:.2f}%)")
        else:
            logger.info("X contains no NaN values")
            
        if torch.isinf(data[0]).any():
            logger.error("X contains inf values!")
            
        if torch.isnan(data[1]).any():
            logger.error("adj contains NaN values!")
        if torch.isinf(data[1]).any():
            logger.error("adj contains inf values!")
            
        if torch.isnan(data[2]).any():
            logger.error("y contains NaN values!")
        if torch.isinf(data[2]).any():
            logger.error("y contains inf values!")
        
        return {'X': data[0], 'adj': data[1], 'y': data[2]}, config
        
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise

def debug_model_creation(data, config):
    """Debug model creation and forward pass"""
    logger.info("=== Model Creation and Forward Pass Debugging ===")
    
    try:
        # Create model using config parameters
        logger.info("1. Creating STGNN model...")
        model = STGNNModel(
            num_nodes=config.num_nodes,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            kernel_size=config.kernel_size
        )
        
        logger.info(f"2. Model created successfully:")
        logger.info(f"   Model type: {type(model)}")
        logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"   Config - num_nodes: {config.num_nodes}, input_dim: {config.input_dim}")
        logger.info(f"   Config - hidden_dim: {config.hidden_dim}, output_dim: {config.output_dim}")
        
        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"3. Moving model to device: {device}")
        model = model.to(device)
        
        # Move data to device
        logger.info("4. Moving data to device...")
        X = data['X'].to(device)
        adj = data['adj'].to(device)
        y = data['y'].to(device)
        
        logger.info(f"   X device: {X.device}")
        logger.info(f"   adj device: {adj.device}")
        logger.info(f"   y device: {y.device}")
        
        # Test forward pass
        logger.info("5. Testing forward pass...")
        model.eval()
        with torch.no_grad():
            output, attention_dict = model(X, adj)
            
        logger.info(f"6. Forward pass successful:")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Attention dict keys: {list(attention_dict.keys())}")
        
        return model, X, adj, y
        
    except Exception as e:
        logger.error(f"Error in model creation/forward pass: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def debug_graph_convolution_specifically(data):
    """Debug GraphConvolution layer specifically"""
    logger.info("=== GraphConvolution Specific Debugging ===")
    
    try:
        from utils.stgnn_utils import GraphConvolution
        
        # Create a simple GraphConvolution layer with correct input features
        logger.info("1. Creating GraphConvolution layer...")
        # Get the actual input features dimension from the data
        input_features = data['X'].shape[-1]  # Last dimension is features
        logger.info(f"   Input features dimension: {input_features}")
        
        graph_conv = GraphConvolution(in_features=input_features, out_features=4)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph_conv = graph_conv.to(device)
        
        # Prepare test data
        logger.info("2. Preparing test data...")
        X = data['X'][:10, :, :, :].to(device)  # Take first 10 samples
        adj = data['adj'].to(device)
        
        logger.info(f"   Test X shape: {X.shape}")
        logger.info(f"   Test adj shape: {adj.shape}")
        
        # Test each time step separately
        logger.info("3. Testing GraphConvolution on each time step...")
        for t in range(X.shape[2]):  # For each time step
            logger.info(f"   Testing time step {t}...")
            x_t = X[:, :, t, :]  # [batch_size, num_nodes, features]
            
            logger.info(f"   x_t shape: {x_t.shape}")
            logger.info(f"   x_t device: {x_t.device}")
            logger.info(f"   adj device: {adj.device}")
            
            # Test without return_attention
            logger.info(f"   Testing without return_attention...")
            output = graph_conv(x_t, adj, return_attention=False)
            logger.info(f"   Output shape: {output.shape}")
            
            # Test with return_attention
            logger.info(f"   Testing with return_attention...")
            output, attention = graph_conv(x_t, adj, return_attention=True)
            logger.info(f"   Output shape: {output.shape}")
            logger.info(f"   Attention shape: {attention.shape}")
            
            # Check for None values
            if output is None:
                logger.error(f"   Output is None at time step {t}!")
            if attention is None:
                logger.error(f"   Attention is None at time step {t}!")
                
            break  # Only test first time step for brevity
            
    except Exception as e:
        logger.error(f"Error in GraphConvolution debugging: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def debug_adjacency_matrix(data):
    """Debug adjacency matrix specifically"""
    logger.info("=== Adjacency Matrix Debugging ===")
    
    try:
        adj = data['adj']
        
        logger.info(f"1. Adjacency matrix properties:")
        logger.info(f"   Shape: {adj.shape}")
        logger.info(f"   Dtype: {adj.dtype}")
        logger.info(f"   Device: {adj.device}")
        logger.info(f"   Requires grad: {adj.requires_grad}")
        
        # Check values
        logger.info(f"2. Adjacency matrix values:")
        logger.info(f"   Min value: {adj.min().item()}")
        logger.info(f"   Max value: {adj.max().item()}")
        logger.info(f"   Mean value: {adj.mean().item()}")
        logger.info(f"   Std value: {adj.std().item()}")
        
        # Check for special values
        logger.info(f"3. Checking for special values:")
        logger.info(f"   NaN count: {torch.isnan(adj).sum().item()}")
        logger.info(f"   Inf count: {torch.isinf(adj).sum().item()}")
        logger.info(f"   Zero count: {(adj == 0).sum().item()}")
        logger.info(f"   One count: {(adj == 1).sum().item()}")
        
        # Check if matrix is symmetric
        if adj.shape[0] == adj.shape[1]:
            is_symmetric = torch.allclose(adj, adj.t())
            logger.info(f"4. Matrix symmetry: {is_symmetric}")
        
        # Check if matrix is valid (no negative values for adjacency)
        has_negative = (adj < 0).any()
        logger.info(f"5. Has negative values: {has_negative}")
        
    except Exception as e:
        logger.error(f"Error in adjacency matrix debugging: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    """Main debugging function"""
    logger.info("Starting comprehensive STGNN debugging...")
    
    try:
        # Step 1: Debug data loading and get config
        data, config = debug_tensor_shapes_and_values()
        
        # Step 2: Debug adjacency matrix
        debug_adjacency_matrix(data)
        
        # Step 3: Debug GraphConvolution specifically
        debug_graph_convolution_specifically(data)
        
        # Step 4: Debug full model
        model, X, adj, y = debug_model_creation(data, config)
        
        logger.info("=== All debugging tests completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Debugging failed with error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 