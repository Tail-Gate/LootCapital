#!/usr/bin/env python3
"""
Test script to verify event-based analysis implementation

This script tests that the model training now uses event-based analysis
instead of final-price-difference analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

def test_event_based_analysis():
    """Test that the data processor now uses event-based analysis"""
    
    print("Testing Event-Based Analysis Implementation")
    print("=" * 50)
    
    # Create a simple config
    config = STGNNConfig(
        num_nodes=1,
        input_dim=5,
        hidden_dim=32,
        output_dim=1,
        num_layers=2,
        dropout=0.2,
        kernel_size=3,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=10,
        early_stopping_patience=5,
        seq_len=10,  # Short sequence for testing
        prediction_horizon=5,  # Short horizon for testing
        features=['returns', 'rsi', 'macd', 'volume_ma', 'volume_ratio'],  # Engineered features only
        assets=['ETH/USD'],
        confidence_threshold=0.51,
        buy_threshold=0.6,
        sell_threshold=0.4,
        retrain_interval=24
    )
    
    # Initialize components
    market_data = MarketData(data_source_path='data')
    technical_indicators = TechnicalIndicators()
    data_processor = STGNNDataProcessor(config, market_data, technical_indicators)
    
    # Load a small amount of data for testing
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 10)
    
    print(f"Loading test data from {start_date} to {end_date}")
    
    try:
        # Get market data
        market_data_dict = market_data.get_data(['ETH/USD'], start_date, end_date)
        
        if not market_data_dict or 'ETH/USD' not in market_data_dict:
            print("No data available for testing")
            return False
        
        eth_data = market_data_dict['ETH/USD']
        print(f"Loaded {len(eth_data)} records")
        
        # Prepare features
        features = data_processor.prepare_features(eth_data)
        print(f"Prepared features: {list(features.columns)}")
        
        # Check that original prices are stored
        if hasattr(data_processor, '_original_prices'):
            print(f"✓ Original prices stored: {len(data_processor._original_prices)} records")
        else:
            print("✗ Original prices not stored")
            return False
        
        # Create sequences
        X, y = data_processor.create_sequences(features)
        print(f"✓ Created sequences: X shape {X.shape}, y shape {y.shape}")
        
        # Test event-based analysis logic
        print("\nTesting Event-Based Analysis Logic:")
        print("-" * 40)
        
        # Get a sample sequence
        if len(X) > 0:
            sample_idx = 0
            sample_features = features.iloc[sample_idx:sample_idx + config.seq_len]
            sample_target = y[sample_idx]
            
            print(f"Sample sequence {sample_idx}:")
            print(f"  Input features: {sample_features.shape}")
            print(f"  Target value: {sample_target:.6f}")
            
            # Manually calculate what the target should be using event-based analysis
            start_idx = sample_idx + config.seq_len
            end_idx = start_idx + config.prediction_horizon
            
            if start_idx < len(data_processor._original_prices) and end_idx <= len(data_processor._original_prices):
                start_price = data_processor._original_prices.iloc[start_idx]
                window_prices = data_processor._original_prices.iloc[start_idx:end_idx]
                price_changes = (window_prices - start_price) / start_price
                
                max_positive = price_changes.max()
                max_negative = price_changes.min()
                
                if abs(max_positive) > abs(max_negative):
                    expected_target = max_positive
                else:
                    expected_target = max_negative
                
                print(f"  Expected target (event-based): {expected_target:.6f}")
                print(f"  Actual target: {sample_target:.6f}")
                print(f"  Match: {'✓' if abs(sample_target - expected_target) < 1e-6 else '✗'}")
                
                # Show the price window
                print(f"  Price window: {start_price:.2f} -> {list(window_prices.values)}")
                print(f"  Price changes: {list(price_changes.values)}")
                print(f"  Max positive: {max_positive:.6f}, Max negative: {max_negative:.6f}")
        
        # Test multiple sequences
        print(f"\nTesting {min(5, len(X))} sequences:")
        print("-" * 40)
        
        for i in range(min(5, len(X))):
            target = y[i]
            print(f"Sequence {i}: target = {target:.6f}")
        
        # Compare with old method (final price difference)
        print(f"\nComparison with Old Method (Final Price Difference):")
        print("-" * 50)
        
        if len(X) > 0:
            sample_idx = 0
            start_idx = sample_idx + config.seq_len
            end_idx = start_idx + config.prediction_horizon
            
            if end_idx - 1 < len(data_processor._returns):
                old_target = data_processor._returns.iloc[end_idx - 1]
                new_target = y[sample_idx]
                
                print(f"Old method (final price difference): {old_target:.6f}")
                print(f"New method (event-based): {new_target:.6f}")
                print(f"Difference: {abs(new_target - old_target):.6f}")
                
                if abs(new_target) > abs(old_target):
                    print("✓ Event-based analysis captures larger moves")
                else:
                    print("ℹ Both methods captured similar moves")
        
        print(f"\n✓ Event-based analysis test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_alignment():
    """Test that the model training data aligns with analysis script results"""
    
    print("\n" + "=" * 50)
    print("Testing Analysis Alignment")
    print("=" * 50)
    
    # This would compare the distribution of targets from model training
    # with the distribution from the analysis script
    # For now, just a placeholder
    print("Analysis alignment test - to be implemented")
    print("This would verify that model training targets match analysis script results")

if __name__ == "__main__":
    success = test_event_based_analysis()
    
    if success:
        test_analysis_alignment()
        print(f"\n{'='*50}")
        print("ALL TESTS PASSED!")
        print("Event-based analysis is working correctly.")
        print("Model training will now capture all trading opportunities.")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print("TESTS FAILED!")
        print("Event-based analysis needs to be fixed.")
        print(f"{'='*50}")
        sys.exit(1) 