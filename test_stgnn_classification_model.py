#!/usr/bin/env python3
"""
Test script for STGNNClassificationModel

This script tests the STGNNClassificationModel with actual input dimensions
from the training configuration to ensure it works correctly.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scripts.train_stgnn_improved import STGNNClassificationModel, create_improved_config
from utils.stgnn_utils import STGNNModel

def test_stgnn_classification_model():
    """Test STGNNClassificationModel with actual input dimensions"""
    
    print("="*60)
    print("TESTING STGNN CLASSIFICATION MODEL")
    print("="*60)
    
    # Create configuration
    config = create_improved_config()
    print(f"Configuration created successfully")
    print(f"Assets: {config.assets}")
    print(f"Features: {len(config.features)} features")
    print(f"Feature list: {config.features}")
    
    # Model parameters
    num_nodes = len(config.assets)  # Should be 1 for ETH/USD
    input_dim = len(config.features)  # Number of features
    hidden_dim = config.hidden_dim
    num_classes = 3  # Down, No Direction, Up
    num_layers = config.num_layers
    dropout = config.dropout
    kernel_size = config.kernel_size
    batch_size = config.batch_size
    seq_len = config.seq_len
    
    print(f"\nModel Parameters:")
    print(f"  num_nodes: {num_nodes}")
    print(f"  input_dim: {input_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_classes: {num_classes}")
    print(f"  num_layers: {num_layers}")
    print(f"  dropout: {dropout}")
    print(f"  kernel_size: {kernel_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    
    # Create dummy input data
    print(f"\nCreating dummy input data...")
    
    # Create X with shape [batch_size, num_nodes, seq_len, input_dim]
    X = torch.randn(batch_size, num_nodes, seq_len, input_dim)
    print(f"X shape: {X.shape}")
    print(f"X stats: min={X.min().item():.4f}, max={X.max().item():.4f}, mean={X.mean().item():.4f}")
    
    # Create adjacency matrix with shape [num_nodes, num_nodes]
    # For single asset, this is just a 1x1 matrix with value 1 (self-connection)
    adj = torch.eye(num_nodes)
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Adjacency matrix:\n{adj}")
    
    # Test for NaN/Inf in inputs
    print(f"\nInput Validation:")
    print(f"X has NaN: {torch.isnan(X).any().item()}")
    print(f"X has Inf: {torch.isinf(X).any().item()}")
    print(f"Adj has NaN: {torch.isnan(adj).any().item()}")
    print(f"Adj has Inf: {torch.isinf(adj).any().item()}")
    
    # Create model
    print(f"\nCreating STGNNClassificationModel...")
    model = STGNNClassificationModel(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        kernel_size=kernel_size
    )
    
    # Print model summary
    print(f"Model created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Test forward pass
    print(f"\nPerforming forward pass...")
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():
        try:
            logits, attention_dict = model(X, adj)
            
            print(f"Forward pass successful!")
            print(f"Logits shape: {logits.shape}")
            print(f"Expected logits shape: [{batch_size * num_nodes}, {num_classes}]")
            print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
            
            # Test for NaN/Inf in output
            print(f"\nOutput Validation:")
            print(f"Logits has NaN: {torch.isnan(logits).any().item()}")
            print(f"Logits has Inf: {torch.isinf(logits).any().item()}")
            
            # Assertions
            assert not torch.isnan(logits).any(), "NaN detected in logits output"
            assert not torch.isinf(logits).any(), "Inf detected in logits output"
            assert logits.shape == (batch_size * num_nodes, num_classes), f"Unexpected logits shape: {logits.shape}"
            
            print(f"✓ All assertions passed!")
            
            # Test softmax probabilities
            probabilities = torch.softmax(logits, dim=1)
            print(f"Probabilities shape: {probabilities.shape}")
            print(f"Probabilities sum per sample: {probabilities.sum(dim=1)}")
            print(f"Probabilities stats: min={probabilities.min().item():.4f}, max={probabilities.max().item():.4f}, mean={probabilities.mean().item():.4f}")
            
            # Test predictions
            _, predictions = torch.max(logits, 1)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Unique predictions: {torch.unique(predictions)}")
            
            # Test attention dict
            print(f"Attention dict keys: {list(attention_dict.keys())}")
            for key, value in attention_dict.items():
                print(f"  {key}: {value.shape}")
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            raise
    
    # Test with different batch sizes
    print(f"\nTesting with different batch sizes...")
    test_batch_sizes = [1, 4, 8]
    
    for test_batch_size in test_batch_sizes:
        print(f"Testing batch size: {test_batch_size}")
        X_test = torch.randn(test_batch_size, num_nodes, seq_len, input_dim)
        
        with torch.no_grad():
            logits_test, _ = model(X_test, adj)
            expected_shape = (test_batch_size * num_nodes, num_classes)
            assert logits_test.shape == expected_shape, f"Batch size {test_batch_size}: Expected {expected_shape}, got {logits_test.shape}"
            assert not torch.isnan(logits_test).any(), f"NaN detected in batch size {test_batch_size}"
            assert not torch.isinf(logits_test).any(), f"Inf detected in batch size {test_batch_size}"
            print(f"  ✓ Batch size {test_batch_size} passed")
    
    # Test model components individually
    print(f"\nTesting model components...")
    
    # Test STGNN base model
    print(f"Testing STGNN base model...")
    stgnn_features, stgnn_attention = model.stgnn(X, adj)
    print(f"STGNN features shape: {stgnn_features.shape}")
    print(f"Expected STGNN features shape: [{batch_size}, {num_nodes}, {hidden_dim}]")
    assert stgnn_features.shape == (batch_size, num_nodes, hidden_dim), f"Unexpected STGNN features shape: {stgnn_features.shape}"
    assert not torch.isnan(stgnn_features).any(), "NaN detected in STGNN features"
    assert not torch.isinf(stgnn_features).any(), "Inf detected in STGNN features"
    print(f"  ✓ STGNN base model passed")
    
    # Test classifier
    print(f"Testing classifier...")
    reshaped_features = stgnn_features.view(-1, hidden_dim)
    classifier_output = model.classifier(reshaped_features)
    print(f"Classifier output shape: {classifier_output.shape}")
    print(f"Expected classifier output shape: [{batch_size * num_nodes}, {num_classes}]")
    assert classifier_output.shape == (batch_size * num_nodes, num_classes), f"Unexpected classifier output shape: {classifier_output.shape}"
    assert not torch.isnan(classifier_output).any(), "NaN detected in classifier output"
    assert not torch.isinf(classifier_output).any(), "Inf detected in classifier output"
    print(f"  ✓ Classifier passed")
    
    print(f"\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("STGNNClassificationModel is working correctly with actual input dimensions.")
    print("="*60)

if __name__ == "__main__":
    test_stgnn_classification_model() 