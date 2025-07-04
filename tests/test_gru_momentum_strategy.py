import pytest
import numpy as np
import pandas as pd
import torch
from strategies.gru_momentum_strategy import GRUMomentumStrategy, GRUMomentumConfig
from utils.gru_utils import GRUAttentionModel

@pytest.fixture
def config():
    """Create a test configuration"""
    return GRUMomentumConfig(
        feature_list=['returns', 'log_returns', 'volume', 'volume_ma', 'volume_ratio', 'rsi', 'macd'],
        gru_hidden_dim=32,
        gru_num_layers=1,
        gru_sequence_length=5,
        gru_dropout=0.1,
        gru_bidirectional=True,
        batch_size=16,
        learning_rate=0.001,
        weight_decay=1e-5,
        early_stopping_patience=3,
        probability_threshold=0.6
    )

@pytest.fixture
def strategy(config):
    """Create a test strategy instance"""
    return GRUMomentumStrategy(config)

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create sample OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    return data

def test_model_initialization(strategy):
    """Test GRU model initialization"""
    # Initialize model
    input_dim = len(strategy.config.feature_list)
    model = GRUAttentionModel(
        input_dim=input_dim,
        hidden_dim=strategy.config.gru_hidden_dim,
        num_layers=strategy.config.gru_num_layers,
        output_dim=1,
        dropout=strategy.config.gru_dropout,
        bidirectional=strategy.config.gru_bidirectional
    )
    
    # Verify model architecture
    assert isinstance(model, GRUAttentionModel)
    assert model.hidden_dim == strategy.config.gru_hidden_dim
    assert model.num_layers == strategy.config.gru_num_layers
    assert model.bidirectional == strategy.config.gru_bidirectional

def test_sequence_preparation(strategy, sample_data):
    """Test sequence data preparation"""
    # Prepare features first
    features = strategy.prepare_features(sample_data)
    
    # Prepare sequences
    sequences = strategy._prepare_sequence_data(features)
    
    # Verify sequence shape
    expected_shape = (
        len(features) - strategy.config.gru_sequence_length + 1,
        strategy.config.gru_sequence_length,
        len(strategy.config.feature_list)
    )
    assert sequences.shape == expected_shape
    assert isinstance(sequences, torch.Tensor)

def test_model_training(strategy, sample_data):
    """Test model training"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    
    # Create target variable (next period's return)
    targets = features['returns'].shift(-1).dropna()
    
    # Train model
    strategy.train(features, targets)
    
    # Verify model was trained
    assert strategy.model is not None
    assert isinstance(strategy.model, GRUAttentionModel)

def test_model_prediction(strategy, sample_data):
    """Test model prediction"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    
    # Train model first
    targets = features['returns'].shift(-1).dropna()
    strategy.train(features, targets)
    
    # Make predictions
    probabilities, attention_weights = strategy.predict(features)
    
    # Verify predictions
    assert isinstance(probabilities, np.ndarray)
    assert isinstance(attention_weights, np.ndarray)
    assert len(probabilities) == len(features) - strategy.config.gru_sequence_length + 1

def test_model_save_load(strategy, sample_data, tmp_path):
    """Test model saving and loading"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    
    # Train model
    targets = features['returns'].shift(-1).dropna()
    strategy.train(features, targets)
    
    # Save model
    save_path = tmp_path / "test_model.pt"
    strategy.save_model(str(save_path))
    
    # Create new strategy instance
    new_strategy = GRUMomentumStrategy(strategy.config)
    
    # Load model
    new_strategy.load_model(str(save_path))
    
    # Verify model was loaded
    assert new_strategy.model is not None
    assert isinstance(new_strategy.model, GRUAttentionModel)

def test_model_explanation(strategy, sample_data):
    """Test model explanation"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    
    # Train model first
    targets = features['returns'].shift(-1).dropna()
    strategy.train(features, targets)
    
    # Get explanations
    explanation = strategy.explain(features)
    
    # Verify explanation format
    assert isinstance(explanation, dict)
    assert 'attention_weights' in explanation
    assert 'attention_summary' in explanation
    assert isinstance(explanation['attention_weights'], np.ndarray)
    assert isinstance(explanation['attention_summary'], np.ndarray)

def test_signal_generation(strategy, sample_data):
    """Test trading signal generation"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    # Train model first
    targets = features['returns'].shift(-1).dropna()
    strategy.train(features, targets)
    # Generate signals (pass sample_data, not features)
    signals = strategy.calculate_signals(sample_data)
    # Verify signals
    assert isinstance(signals, pd.DataFrame)
    assert 'probability' in signals.columns
    assert 'signal' in signals.columns
    assert signals['signal'].isin([-1, 0, 1]).all() 