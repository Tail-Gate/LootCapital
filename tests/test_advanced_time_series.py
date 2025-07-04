import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from strategies.advanced_time_series import (
    AdvancedTimeSeriesConfig,
    AdvancedTimeSeriesStrategy,
    BiLSTMModel,
    TCNModel,
    TransformerModel,
    TimeSeriesDataset
)

@pytest.fixture
def sample_data():
    """Create sample OHLCV data"""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 1000),
        'high': np.random.normal(101, 1, 1000),
        'low': np.random.normal(99, 1, 1000),
        'close': np.random.normal(100, 1, 1000),
        'volume': np.random.normal(1000, 100, 1000)
    }, index=dates)
    
    # Add some order book data
    data['bid_volume'] = np.random.normal(500, 50, 1000)
    data['ask_volume'] = np.random.normal(500, 50, 1000)
    data['bid_depth'] = np.random.normal(10000, 1000, 1000)
    data['ask_depth'] = np.random.normal(10000, 1000, 1000)
    
    return data

@pytest.fixture
def strategy():
    """Create strategy instance"""
    config = AdvancedTimeSeriesConfig(
        model_type='bi_lstm',
        sequence_length=60,
        batch_size=32,
        hidden_size=64
    )
    return AdvancedTimeSeriesStrategy(config)

def test_strategy_initialization(strategy):
    """Test strategy initialization"""
    assert strategy.name == "advanced_time_series"
    assert isinstance(strategy.config, AdvancedTimeSeriesConfig)
    assert strategy.config.model_type == 'bi_lstm'
    assert not strategy.is_fitted

def test_feature_preparation(strategy, sample_data):
    """Test feature preparation"""
    features = strategy.prepare_features(sample_data)
    
    # Check time features
    assert 'hour_sin' in features.columns
    assert 'hour_cos' in features.columns
    assert 'day_sin' in features.columns
    assert 'day_cos' in features.columns
    
    # Check technical features
    assert 'roc_5' in features.columns
    assert 'sma_10' in features.columns
    assert 'std_20' in features.columns
    assert 'volume_ma' in features.columns
    assert 'atr' in features.columns
    
    # Check market features
    assert 'price_impact' in features.columns
    assert 'order_imbalance' in features.columns
    assert 'depth_imbalance' in features.columns

def test_model_creation(sample_data):
    """Test creation of different model types"""
    # Test Bi-LSTM
    config = AdvancedTimeSeriesConfig(model_type='bi_lstm')
    strategy = AdvancedTimeSeriesStrategy(config)
    strategy.prepare_features(sample_data)
    assert isinstance(strategy.model, BiLSTMModel)
    
    # Test TCN
    config = AdvancedTimeSeriesConfig(model_type='tcn')
    strategy = AdvancedTimeSeriesStrategy(config)
    strategy.prepare_features(sample_data)
    assert isinstance(strategy.model, TCNModel)
    
    # Test Transformer
    config = AdvancedTimeSeriesConfig(model_type='transformer')
    strategy = AdvancedTimeSeriesStrategy(config)
    strategy.prepare_features(sample_data)
    assert isinstance(strategy.model, TransformerModel)

def test_sequence_preparation(strategy, sample_data):
    """Test sequence preparation for model input"""
    features = strategy.prepare_features(sample_data)
    sequences = strategy._prepare_sequences(features)
    
    # Check sequence shape
    expected_shape = (len(features) - strategy.config.sequence_length + 1,
                     strategy.config.sequence_length,
                     len(features.columns))
    assert sequences.shape == expected_shape
    assert isinstance(sequences, torch.Tensor)

def test_dataset_creation(strategy, sample_data):
    """Test TimeSeriesDataset creation and usage"""
    features = strategy.prepare_features(sample_data)
    X = strategy._prepare_sequences(features)
    y = torch.FloatTensor(np.random.normal(0, 1, len(X)))
    
    dataset = TimeSeriesDataset(X.numpy(), y.numpy(), strategy.config.sequence_length)
    
    # Test dataset length
    assert len(dataset) == len(X)
    
    # Test item retrieval
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (strategy.config.sequence_length, X.shape[2])
    assert y.shape == (1,)

def test_signal_generation(strategy, sample_data):
    """Test trading signal generation"""
    features = strategy.prepare_features(sample_data)
    
    # Test without fitted model
    signal, confidence, trade_type = strategy.calculate_technical_signals(features)
    assert signal == 0.0
    assert confidence == 0.0
    assert trade_type.name == 'NONE'
    
    # Mock fitted model
    strategy.is_fitted = True
    features['model_prediction'] = 0.5
    features['prediction_confidence'] = 0.8
    
    signal, confidence, trade_type = strategy.calculate_technical_signals(features)
    assert signal == 0.8
    assert confidence == 0.5
    assert trade_type.name == 'LONG'

def test_stop_loss_calculation(strategy, sample_data):
    """Test stop loss calculation"""
    features = strategy.prepare_features(sample_data)
    entry_price = 100.0
    signal = 0.5
    
    stop_loss = strategy.calculate_stop_loss(entry_price, signal, features)
    assert isinstance(stop_loss, float)
    assert stop_loss < entry_price  # For long position

def test_take_profit_calculation(strategy, sample_data):
    """Test take profit calculation"""
    features = strategy.prepare_features(sample_data)
    entry_price = 100.0
    stop_loss = 95.0
    signal = 0.5
    
    take_profit = strategy.calculate_take_profit(entry_price, stop_loss, signal, features)
    assert isinstance(take_profit, float)
    assert take_profit > entry_price  # For long position

def test_model_update(strategy, sample_data):
    """Test model update functionality"""
    features = strategy.prepare_features(sample_data)
    target = pd.Series(np.random.normal(0, 1, len(features)), index=features.index)
    
    # Test model update
    strategy.update_model(features, target)
    assert strategy.is_fitted
    assert strategy.last_train_index is not None 