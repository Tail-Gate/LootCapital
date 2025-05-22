import numpy as np
import pandas as pd
import torch
import pytest
from strategies.lstm_swing_mean_reversion import (
    LSTMSwingMeanReversionStrategy, LSTMSwingMeanReversionConfig
)

class DummyModel:
    def __init__(self, output=0.5):
        self.output = output
        self.eval_called = False
    def eval(self):
        self.eval_called = True
    def to(self, device):
        return self
    def __call__(self, X):
        batch_size = X.shape[0]
        return torch.tensor([[self.output]] * batch_size, dtype=torch.float32), torch.ones((batch_size, X.shape[1], 1))

def make_synthetic_data(n=30):
    np.random.seed(42)
    data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n)) + 100,
        'high': np.cumsum(np.random.randn(n)) + 101,
        'low': np.cumsum(np.random.randn(n)) + 99,
        'volume': np.abs(np.random.randn(n) * 1000),
        'bid_volume': np.abs(np.random.randn(n) * 500),
        'ask_volume': np.abs(np.random.randn(n) * 500),
    })
    return data

def test_prepare_features():
    config = LSTMSwingMeanReversionConfig(sequence_length=5)
    strat = LSTMSwingMeanReversionStrategy(config)
    data = make_synthetic_data(20)
    features = config.prepare_features(data)
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] > 0
    # Check some expected feature columns
    assert any('rsi_seq' in c for c in features.columns)
    assert any('adx_seq' in c for c in features.columns)

def test_signal_calculation_with_mocked_model():
    config = LSTMSwingMeanReversionConfig(sequence_length=5)
    strat = LSTMSwingMeanReversionStrategy(config)
    data = make_synthetic_data(20)
    features = config.prepare_features(data)
    # Patch the model to always return a strong long signal
    strat.model = DummyModel(output=1.0)
    signal, confidence, trade_type = strat.calculate_signals(features)
    assert signal == 1
    assert confidence >= 0
    # Patch the model to always return a strong short signal
    strat.model = DummyModel(output=-1.0)
    signal, confidence, trade_type = strat.calculate_signals(features)
    assert signal == -1
    # Patch the model to return a weak signal
    strat.model = DummyModel(output=0.0)
    signal, confidence, trade_type = strat.calculate_signals(features)
    assert signal == 0

def test_explain_attention():
    config = LSTMSwingMeanReversionConfig(sequence_length=5)
    strat = LSTMSwingMeanReversionStrategy(config)
    data = make_synthetic_data(10)
    features = config.prepare_features(data)
    # Patch the model to return uniform attention
    strat.model = DummyModel(output=0.5)
    X = features.values[-config.sequence_length:]
    X = X.reshape(1, config.sequence_length, -1)
    result = strat.explain(X)
    assert 'attention_weights' in result
    assert 'attention_summary' in result
    assert result['attention_weights'].shape[0] == config.sequence_length
    assert np.allclose(result['attention_weights'], 1.0)

def test_end_to_end_workflow():
    config = LSTMSwingMeanReversionConfig(sequence_length=5)
    strat = LSTMSwingMeanReversionStrategy(config)
    data = make_synthetic_data(30)
    features = config.prepare_features(data)
    # Patch the model for deterministic output
    strat.model = DummyModel(output=0.8)
    signal, confidence, trade_type = strat.calculate_signals(features)
    assert signal in [-1, 0, 1]
    assert 0 <= confidence <= 1.5 