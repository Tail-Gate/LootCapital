#!/usr/bin/env python3
"""
Test script for the three-class XGBoost momentum strategy
"""

import pandas as pd
import numpy as np
from strategies.xgBoost_momentum_strategy import MomentumStrategy, MomentumConfig

def create_sample_data(n_samples=1000):
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Generate sample price data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1h')
    
    # Create trending price data with more realistic patterns
    trend = np.cumsum(np.random.randn(n_samples) * 0.01)
    noise = np.random.randn(n_samples) * 0.005
    
    close_prices = 100 + trend + noise
    
    # Create more realistic OHLC data
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(n_samples) * 0.002,
        'high': close_prices + abs(np.random.randn(n_samples) * 0.005),
        'low': close_prices - abs(np.random.randn(n_samples) * 0.005),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    # Add some order book data if needed
    data['bid_volume'] = data['volume'] * 0.4 + np.random.randn(n_samples) * 100
    data['ask_volume'] = data['volume'] * 0.6 + np.random.randn(n_samples) * 100
    
    return data

def test_three_class_strategy():
    """Test the three-class classification strategy"""
    print("Testing three-class XGBoost momentum strategy...")
    
    # Load real data but only use 1 month
    print("Loading real ETH-USDT data (1 month)...")
    ohlcv_data = pd.read_csv('data/historical/ETH-USDT-SWAP_ohlcv_15m.csv')
    ohlcv_data['timestamp'] = pd.to_datetime(ohlcv_data['timestamp'])
    ohlcv_data.set_index('timestamp', inplace=True)
    
    # Use only the last month of data (approximately 2880 15-minute candles)
    one_month_data = ohlcv_data.tail(2880)
    print(f"Using {len(one_month_data)} samples (1 month of 15-minute data)")
    print(f"Date range: {one_month_data.index.min()} to {one_month_data.index.max()}")
    
    # Initialize strategy with new parameters
    config = MomentumConfig(
        num_classes=3,
        lookforward_periods=4,  # Predict 4 candlesticks ahead
        price_threshold_pct=0.02,  # 2% threshold
        min_confidence_threshold=0.5  # 0.5 minimum confidence
    )
    strategy = MomentumStrategy(config)
    
    # Prepare features
    print("\nPreparing features...")
    features = strategy.prepare_features(one_month_data)
    print(f"Prepared features with shape: {features.shape}")
    
    # Use all available features from the feature generator
    print("\nAvailable features:")
    print(f"Total features generated: {len(features.columns)}")
    
    # Get all features that are in our default feature list
    available_features = [f for f in strategy.config.feature_list if f in features.columns]
    print(f"Features from our list that are available: {len(available_features)}")
    
    # Use all available features
    strategy.config.feature_list = available_features
    
    # Simply drop rows with NaN values
    features_clean = features[available_features].dropna()
    print(f"Clean features after dropping NaNs: {len(features_clean)} samples")
    
    # Create three-class labels with new parameters
    print("\nCreating labels...")
    labels = strategy.create_three_class_labels(one_month_data, lookforward_periods=4, threshold_pct=0.02)
    labels_clean = labels.loc[features_clean.index]
    
    # Drop NaN values from labels
    labels_clean = labels_clean.dropna()
    features_clean = features_clean.loc[labels_clean.index]
    
    print(f"Label distribution:")
    for value, count in labels_clean.value_counts().items():
        percentage = (count / len(labels_clean)) * 100
        label_name = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}[value]
        print(f"  {label_name}: {count} ({percentage:.1f}%)")
    
    print(f"Final clean dataset: {len(features_clean)} samples")
    
    if len(features_clean) < 100:
        print("Not enough clean data for training")
        return
    
    # Train the model
    print("\nTraining model...")
    strategy.train(features_clean, labels_clean, num_boost_round=50)
    print("Model training completed!")
    
    # Test predictions
    print("\nTesting predictions...")
    class_probabilities, class_predictions = strategy.predict(features_clean.tail(10))
    
    print(f"Prediction results:")
    print(f"  Probabilities shape: {class_probabilities.shape}")
    print(f"  Predictions shape: {class_predictions.shape}")
    
    # Show detailed prediction analysis
    print(f"\nDetailed prediction analysis:")
    class_names = ['SHORT', 'HOLD', 'LONG']
    for i in range(min(5, len(class_probabilities))):
        probs = class_probabilities[i]
        pred_class = class_predictions[i]
        confidence = probs[pred_class]
        
        print(f"  Sample {i+1}:")
        print(f"    Probabilities: SHORT={probs[0]:.3f}, HOLD={probs[1]:.3f}, LONG={probs[2]:.3f}")
        print(f"    Prediction: {class_names[pred_class]} (confidence: {confidence:.3f})")
    
    # Analyze predictions
    pred_dist = pd.Series(class_predictions).value_counts()
    print(f"\nPrediction distribution:")
    for value, count in pred_dist.items():
        percentage = (count / len(class_predictions)) * 100
        label_name = class_names[value]
        print(f"  {label_name}: {count} ({percentage:.1f}%)")
    
    # Test signal generation with confidence threshold
    print(f"\nTesting signal generation with confidence threshold...")
    signal, confidence, trade_type = strategy.calculate_signals(features_clean.tail(20))
    
    signal_map = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}
    print(f"Latest signal: {signal_map[signal]} (confidence: {confidence:.3f}, type: {trade_type})")
    
    # Test multiple signals to see confidence threshold in action
    print(f"\nTesting multiple signals:")
    test_features = features_clean.tail(20)
    signals = []
    confidences = []
    
    for i in range(len(test_features)):
        if i < len(test_features) - 1:  # Need at least 2 samples for signal generation
            test_data = test_features.iloc[:i+1]
            sig, conf, tt = strategy.calculate_signals(test_data)
            signals.append(sig)
            confidences.append(conf)
    
    # Show signals above and below confidence threshold
    high_conf_signals = [(i, sig, conf) for i, (sig, conf) in enumerate(zip(signals, confidences)) if conf > 0.5]
    low_conf_signals = [(i, sig, conf) for i, (sig, conf) in enumerate(zip(signals, confidences)) if conf <= 0.5]
    
    print(f"  Signals with confidence > 0.5: {len(high_conf_signals)}")
    print(f"  Signals with confidence <= 0.5: {len(low_conf_signals)}")
    
    if high_conf_signals:
        print(f"  Example high confidence signal: {signal_map[high_conf_signals[0][1]]} (confidence: {high_conf_signals[0][2]:.3f})")
    
    # Test feature importance
    print("\nFeature importance (top 10):")
    importance = strategy.get_feature_importance()
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, score in sorted_importance:
            print(f"  {feature}: {score:.4f}")
    
    print("\nThree-class strategy test completed successfully!")

if __name__ == "__main__":
    test_three_class_strategy() 