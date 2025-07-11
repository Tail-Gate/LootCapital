#!/usr/bin/env python3
"""
Test GraphConvolution Fix - Validate None Tensor Error Resolution
===============================================================

This script tests the fix for the GraphConvolution.forward() method to ensure
it properly handles return_attention=True and doesn't return None values.
"""

import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.stgnn_utils import GraphConvolution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_graph_convolution_fix():
    """Test the GraphConvolution fix for return_attention=True"""
    logger.info("🧪 Testing GraphConvolution fix...")
    
    # Test parameters
    batch_size = 2
    num_nodes = 3
    in_features = 4
    out_features = 5
    
    # Create test data
    x = torch.randn(batch_size, num_nodes, in_features)
    adj = torch.randn(num_nodes, num_nodes)
    adj = torch.softmax(adj, dim=-1)  # Make it a valid adjacency matrix
    
    logger.info(f"Input shapes - x: {x.shape}, adj: {adj.shape}")
    
    # Create GraphConvolution layer
    graph_conv = GraphConvolution(in_features, out_features)
    
    # Test 1: Without return_attention
    logger.info("Testing without return_attention...")
    try:
        output = graph_conv(x, adj, return_attention=False)
        logger.info(f"✅ Success - Output shape: {output.shape}")
        assert output.shape == (batch_size, num_nodes, out_features), f"Expected shape {(batch_size, num_nodes, out_features)}, got {output.shape}"
    except Exception as e:
        logger.error(f"❌ Failed without return_attention: {e}")
        return False
    
    # Test 2: With return_attention
    logger.info("Testing with return_attention=True...")
    try:
        output, attention = graph_conv(x, adj, return_attention=True)
        logger.info(f"✅ Success - Output shape: {output.shape}, Attention shape: {attention.shape}")
        
        # Validate shapes
        assert output.shape == (batch_size, num_nodes, out_features), f"Expected output shape {(batch_size, num_nodes, out_features)}, got {output.shape}"
        assert attention.shape == (batch_size, num_nodes, num_nodes), f"Expected attention shape {(batch_size, num_nodes, num_nodes)}, got {attention.shape}"
        
        # Validate that attention is not None
        assert attention is not None, "Attention should not be None"
        
        # Validate that attention has the correct dimension
        assert attention.dim() == 3, f"Attention should be 3D, got {attention.dim()}D"
        
        logger.info("✅ All validations passed!")
        
    except Exception as e:
        logger.error(f"❌ Failed with return_attention=True: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False
    
    # Test 3: Test with different input shapes
    logger.info("Testing with different input shapes...")
    try:
        # Test with 2D input
        x_2d = torch.randn(num_nodes, in_features)
        output_2d, attention_2d = graph_conv(x_2d, adj, return_attention=True)
        logger.info(f"✅ 2D input success - Output shape: {output_2d.shape}, Attention shape: {attention_2d.shape}")
        
        # Test with sparse adjacency
        adj_sparse = adj.to_sparse()
        output_sparse, attention_sparse = graph_conv(x, adj_sparse, return_attention=True)
        logger.info(f"✅ Sparse adjacency success - Output shape: {output_sparse.shape}, Attention shape: {attention_sparse.shape}")
        
    except Exception as e:
        logger.error(f"❌ Failed with different input shapes: {e}")
        return False
    
    logger.info("🎉 All GraphConvolution tests passed!")
    return True

def test_stgnn_model_integration():
    """Test the fix with STGNNModel integration"""
    logger.info("🧪 Testing STGNNModel integration...")
    
    try:
        from utils.stgnn_utils import STGNNModel
        
        # Create test data
        batch_size = 2
        num_nodes = 3
        seq_len = 5
        input_dim = 4
        hidden_dim = 6
        output_dim = 3
        
        x = torch.randn(batch_size, num_nodes, seq_len, input_dim)
        adj = torch.randn(num_nodes, num_nodes)
        adj = torch.softmax(adj, dim=-1)
        
        # Create STGNNModel
        model = STGNNModel(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=1,
            dropout=0.1,
            kernel_size=3
        )
        
        # Test forward pass
        output, attention_dict = model(x, adj)
        logger.info(f"✅ STGNNModel forward pass success - Output shape: {output.shape}")
        logger.info(f"✅ Attention dict keys: {list(attention_dict.keys())}")
        
        # Validate output
        assert output.shape == (batch_size, num_nodes, output_dim), f"Expected output shape {(batch_size, num_nodes, output_dim)}, got {output.shape}"
        assert len(attention_dict) > 0, "Attention dict should not be empty"
        
        logger.info("🎉 STGNNModel integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ STGNNModel integration test failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting GraphConvolution fix validation...")
    
    # Test 1: GraphConvolution fix
    test1_success = test_graph_convolution_fix()
    
    # Test 2: STGNNModel integration
    test2_success = test_stgnn_model_integration()
    
    if test1_success and test2_success:
        logger.info("🎉 ALL TESTS PASSED! The GraphConvolution fix is working correctly.")
        logger.info("✅ The 'NoneType' object has no attribute 'dim' error should now be resolved.")
    else:
        logger.error("❌ Some tests failed. The fix may not be complete.")
        sys.exit(1) 