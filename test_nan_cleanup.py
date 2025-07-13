#!/usr/bin/env python3
"""
Test script to verify NaN/Inf cleanup in feature generator.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature_generator import generate_features_chunk
from market_analysis.technical_indicators import TechnicalIndicators

def test_nan_cleanup():
    """Test that generate_features_chunk properly handles NaNs and Infs."""
    
    # Create sample OHLCV data with some problematic values
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    
    # Create data with some edge cases that could produce NaNs/Infs
    data = pd.DataFrame({
        'open': np.random.randn(100) * 100 + 1000,
        'high': np.random.randn(100) * 100 + 1000,
        'low': np.random.randn(100) * 100 + 1000,
        'close': np.random.randn(100) * 100 + 1000,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Add some edge cases that could cause problems
    data.loc[0, 'volume'] = 0  # Zero volume
    data.loc[1, 'close'] = data.loc[1, 'close'] * 0  # Zero price
    data.loc[2, 'high'] = data.loc[2, 'low']  # Same high/low
    
    # Initialize technical indicators
    indicators = TechnicalIndicators()
    
    print("Testing generate_features_chunk with edge case data...")
    
    try:
        # Generate features
        features = generate_features_chunk(data, None, indicators)
        
        # Check for NaNs and Infs
        nan_count = features.isnull().sum().sum()
        inf_count = (features == np.inf).sum().sum() + (features == -np.inf).sum().sum()
        
        print(f"Generated {len(features.columns)} features")
        print(f"NaN count: {nan_count}")
        print(f"Inf count: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print("✅ SUCCESS: No NaNs or Infs found in generated features!")
            return True
        else:
            print("❌ FAILURE: NaNs or Infs still present!")
            print("Columns with NaNs:", features.columns[features.isnull().any()].tolist())
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_edge_cases():
    """Test with more extreme edge cases."""
    
    print("\nTesting with extreme edge cases...")
    
    # Create data with more extreme cases
    dates = pd.date_range('2023-01-01', periods=50, freq='1H')
    
    data = pd.DataFrame({
        'open': [1000] * 50,
        'high': [1000] * 50,
        'low': [1000] * 50,
        'close': [1000] * 50,  # Constant price
        'volume': [0] * 50  # Zero volume
    }, index=dates)
    
    indicators = TechnicalIndicators()
    
    try:
        features = generate_features_chunk(data, None, indicators)
        
        nan_count = features.isnull().sum().sum()
        inf_count = (features == np.inf).sum().sum() + (features == -np.inf).sum().sum()
        
        print(f"Edge case test - NaN count: {nan_count}, Inf count: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print("✅ SUCCESS: Edge cases handled correctly!")
            return True
        else:
            print("❌ FAILURE: Edge cases not handled!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR in edge case test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing NaN/Inf cleanup in feature generator...")
    
    test1_passed = test_nan_cleanup()
    test2_passed = test_edge_cases()
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! NaN/Inf cleanup is working correctly.")
    else:
        print("\n💥 SOME TESTS FAILED! Check the implementation.")
        sys.exit(1) 