#!/usr/bin/env python3
"""
STGNN Data Analysis Script

This script analyzes the training data to check for class imbalance and data distribution issues.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

def create_config():
    """Create configuration for STGNN data analysis"""
    
    # Multiple assets for analysis
    assets = ['BTC/USD', 'ETH/USD', 'BNB/USD', 'SOL/USD', 'ADA/USD']
    
    # Engineered features only - NO raw OHLCV data
    # These features are derived from price/volume data and are suitable for ML training
    features = [
        # Price-derived features (safe for ML)
        'returns', 'log_returns',
        
        # Technical indicators (derived from price)
        'rsi', 'macd', 'bollinger_upper', 'bollinger_lower',
        'atr', 'adx', 'vwap', 'volume_ma', 'volume_ratio'
    ]
    
    # Create configuration
    config = STGNNConfig(
        num_nodes=len(assets),
        input_dim=len(features),
        hidden_dim=64,
        output_dim=1,
        num_layers=2,
        dropout=0.2,
        kernel_size=3,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100,
        early_stopping_patience=10,
        seq_len=20,
        prediction_horizon=15,
        features=features,
        assets=assets,
        confidence_threshold=0.51,
        buy_threshold=0.6,
        sell_threshold=0.4,
        retrain_interval=24
    )
    
    return config

def analyze_target_distribution(y, config):
    """Analyze the distribution of target values"""
    
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    # Flatten all targets
    y_flat = y.flatten().numpy()
    
    print(f"Total samples: {len(y_flat)}")
    print(f"Mean target: {np.mean(y_flat):.6f}")
    print(f"Std target: {np.std(y_flat):.6f}")
    print(f"Min target: {np.min(y_flat):.6f}")
    print(f"Max target: {np.max(y_flat):.6f}")
    
    # Analyze distribution
    print(f"\nTarget Distribution:")
    print(f"  Zero values (exactly 0): {np.sum(y_flat == 0)} ({np.sum(y_flat == 0)/len(y_flat)*100:.2f}%)")
    print(f"  Near-zero values (|x| < 0.001): {np.sum(np.abs(y_flat) < 0.001)} ({np.sum(np.abs(y_flat) < 0.001)/len(y_flat)*100:.2f}%)")
    print(f"  Near-zero values (|x| < 0.01): {np.sum(np.abs(y_flat) < 0.01)} ({np.sum(np.abs(y_flat) < 0.01)/len(y_flat)*100:.2f}%)")
    
    # Analyze by asset
    print(f"\nTarget Distribution by Asset:")
    for i, asset in enumerate(config.assets):
        asset_targets = y[:, i].numpy()
        print(f"\n{asset}:")
        print(f"  Mean: {np.mean(asset_targets):.6f}")
        print(f"  Std: {np.std(asset_targets):.6f}")
        print(f"  Zero values: {np.sum(asset_targets == 0)} ({np.sum(asset_targets == 0)/len(asset_targets)*100:.2f}%)")
        print(f"  Near-zero (|x| < 0.01): {np.sum(np.abs(asset_targets) < 0.01)} ({np.sum(np.abs(asset_targets) < 0.01)/len(asset_targets)*100:.2f}%)")
    
    # Analyze trading signals
    print(f"\nTrading Signal Analysis:")
    buy_signals = np.sum(y_flat > config.buy_threshold)
    sell_signals = np.sum(y_flat < -config.sell_threshold)
    hold_signals = len(y_flat) - buy_signals - sell_signals
    
    print(f"  Buy signals (> {config.buy_threshold}): {buy_signals} ({buy_signals/len(y_flat)*100:.2f}%)")
    print(f"  Sell signals (< -{config.sell_threshold}): {sell_signals} ({sell_signals/len(y_flat)*100:.2f}%)")
    print(f"  Hold signals: {hold_signals} ({hold_signals/len(y_flat)*100:.2f}%)")
    
    # Check for extreme imbalance
    total_signals = buy_signals + sell_signals
    if total_signals < len(y_flat) * 0.1:  # Less than 10% are actual signals
        print(f"\n⚠️  WARNING: Severe class imbalance detected!")
        print(f"   Only {total_signals/len(y_flat)*100:.2f}% of samples have actionable signals")
        print(f"   {hold_signals/len(y_flat)*100:.2f}% of samples are 'hold' signals")
    
    return {
        'total_samples': len(y_flat),
        'mean': np.mean(y_flat),
        'std': np.std(y_flat),
        'zero_count': np.sum(y_flat == 0),
        'near_zero_count': np.sum(np.abs(y_flat) < 0.01),
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'hold_signals': hold_signals
    }

def analyze_feature_distribution(X, config):
    """Analyze the distribution of input features"""
    
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Analyze each feature
    for i, feature in enumerate(config.features):
        feature_data = X[:, :, :, i].flatten().numpy()
        
        print(f"\n{feature}:")
        print(f"  Mean: {np.mean(feature_data):.6f}")
        print(f"  Std: {np.std(feature_data):.6f}")
        print(f"  Min: {np.min(feature_data):.6f}")
        print(f"  Max: {np.max(feature_data):.6f}")
        print(f"  Zero values: {np.sum(feature_data == 0)} ({np.sum(feature_data == 0)/len(feature_data)*100:.2f}%)")
        print(f"  NaN values: {np.sum(np.isnan(feature_data))}")
        print(f"  Inf values: {np.sum(np.isinf(feature_data))}")

def suggest_improvements(analysis_results):
    """Suggest improvements based on analysis"""
    
    print("\n" + "="*60)
    print("SUGGESTED IMPROVEMENTS")
    print("="*60)
    
    hold_ratio = analysis_results['hold_signals'] / analysis_results['total_samples']
    
    if hold_ratio > 0.8:
        print("🔴 CRITICAL: Severe class imbalance detected!")
        print("\nRecommended solutions:")
        print("1. **Adjust thresholds**: Lower buy_threshold and sell_threshold")
        print("   - Current: buy_threshold=0.6, sell_threshold=0.4")
        print("   - Suggested: buy_threshold=0.1, sell_threshold=0.1")
        
        print("\n2. **Use different target variable**:")
        print("   - Predict price direction (up/down) instead of returns")
        print("   - Use volatility-adjusted returns")
        print("   - Predict price levels instead of returns")
        
        print("\n3. **Data augmentation**:")
        print("   - Oversample minority classes")
        print("   - Use synthetic data generation")
        
        print("\n4. **Loss function modifications**:")
        print("   - Use weighted loss function")
        print("   - Use focal loss for imbalanced data")
        print("   - Use class-balanced loss")
        
        print("\n5. **Feature engineering**:")
        print("   - Add more volatile features")
        print("   - Use higher frequency data")
        print("   - Add market regime indicators")
    
    elif hold_ratio > 0.6:
        print("🟡 WARNING: Moderate class imbalance detected")
        print("Consider adjusting thresholds or using weighted loss")
    
    else:
        print("🟢 Data distribution looks reasonable")

def main():
    """Main analysis function"""
    
    print("Starting STGNN data analysis...")
    
    try:
        # Create configuration
        config = create_config()
        
        # Initialize components
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        
        # Create data processor
        data_processor = STGNNDataProcessor(config, market_data, technical_indicators)
        
        # Prepare data
        print("Preparing data for analysis...")
        X, adj, y = data_processor.prepare_data()
        
        print(f"Data shapes:")
        print(f"  X: {X.shape}")
        print(f"  adj: {adj.shape}")
        print(f"  y: {y.shape}")
        
        # Analyze target distribution
        analysis_results = analyze_target_distribution(y, config)
        
        # Analyze feature distribution
        analyze_feature_distribution(X, config)
        
        # Suggest improvements
        suggest_improvements(analysis_results)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 