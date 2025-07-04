import pytest
import pandas as pd
import numpy as np
import torch
from strategies.transformer_strategy import TransformerStrategy

@pytest.fixture
def sample_data():
    # Create sample OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data

@pytest.fixture
def transformer_strategy():
    return TransformerStrategy(
        input_size=5,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        sequence_length=10
    )

def test_prepare_data(transformer_strategy, sample_data):
    X, y = transformer_strategy.prepare_data(sample_data)
    
    # Check shapes
    expected_samples = len(sample_data) - transformer_strategy.sequence_length
    assert X.shape == (expected_samples, transformer_strategy.sequence_length, 5)
    assert y.shape == (expected_samples,)
    
    # Check data types
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)

def test_train(transformer_strategy, sample_data):
    # Train with minimal epochs for testing
    history = transformer_strategy.train(
        sample_data,
        num_epochs=2,
        batch_size=16
    )
    
    # Check history
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2
    
    # Check that losses are decreasing
    assert history['train_loss'][-1] <= history['train_loss'][0]

def test_predict(transformer_strategy, sample_data):
    # Train first
    transformer_strategy.train(sample_data, num_epochs=2)
    
    # Make predictions
    predictions, attention_dict = transformer_strategy.predict(sample_data)
    
    # Check predictions
    expected_samples = len(sample_data) - transformer_strategy.sequence_length
    assert predictions.shape == (expected_samples, 1)
    
    # Check attention weights
    assert isinstance(attention_dict, dict)
    assert len(attention_dict) == transformer_strategy.num_layers

def test_generate_signals(transformer_strategy, sample_data):
    # Train first
    transformer_strategy.train(sample_data, num_epochs=2)
    
    # Generate signals
    signals = transformer_strategy.generate_signals(sample_data)
    
    # Check signals DataFrame
    assert isinstance(signals, pd.DataFrame)
    assert 'prediction' in signals.columns
    assert 'signal' in signals.columns
    
    # Check signal values
    assert signals['signal'].isin([-1, 0, 1]).all()
    
    # Check index alignment
    expected_index = sample_data.index[transformer_strategy.sequence_length:]
    assert signals.index.equals(expected_index)

def test_save_load_model(transformer_strategy, sample_data, tmp_path):
    # Train first
    transformer_strategy.train(sample_data, num_epochs=2)
    
    # Save model
    save_path = tmp_path / "transformer_model.pt"
    transformer_strategy.save_model(str(save_path))
    
    # Create new strategy and load model
    new_strategy = TransformerStrategy(
        input_size=5,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        sequence_length=10
    )
    new_strategy.load_model(str(save_path))
    
    # Compare predictions
    original_predictions, _ = transformer_strategy.predict(sample_data)
    loaded_predictions, _ = new_strategy.predict(sample_data)
    
    np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)

def test_attention_visualization(transformer_strategy, sample_data):
    # Train first
    transformer_strategy.train(sample_data, num_epochs=2)
    
    # Make predictions to get attention weights
    transformer_strategy.predict(sample_data)
    
    # Get attention visualization
    feature_names = ['open', 'high', 'low', 'close', 'volume']
    attention_viz = transformer_strategy.get_attention_visualization(feature_names)
    
    # Check visualization data
    assert isinstance(attention_viz, dict)
    assert len(attention_viz) == transformer_strategy.num_layers
    
    for layer_name, attention_data in attention_viz.items():
        assert 'attention_matrix' in attention_data
        assert 'time_steps' in attention_data
        assert 'feature_names' in attention_data
        assert attention_data['feature_names'] == feature_names

def test_invalid_attention_visualization(transformer_strategy):
    # Try to get attention visualization without predictions
    with pytest.raises(ValueError):
        transformer_strategy.get_attention_visualization() 