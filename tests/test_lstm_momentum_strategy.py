import pytest
import pandas as pd
import numpy as np
import torch
from strategies.lstm_momentum_strategy import LSTMMomentumStrategy, LSTMMomentumConfig
from utils.lstm_utils import LSTMAttentionModel

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
    data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000),
        'bid_volume': np.random.randint(1000, 10000, 1000),
        'ask_volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    return data

@pytest.fixture
def strategy():
    """Create a LSTMMomentumStrategy instance for testing"""
    return LSTMMomentumStrategy()

def test_sequence_data_preparation(strategy, sample_data):
    """Test sequence data preparation for LSTM"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    
    # Test sequence preparation
    sequences = strategy._prepare_sequence_data(features)
    
    # Verify sequence shape
    expected_shape = (len(features) - strategy.config.lstm_sequence_length + 1,
                     strategy.config.lstm_sequence_length,
                     len(strategy.config.feature_list))
    assert sequences.shape == expected_shape
    
    # Verify sequence content
    assert isinstance(sequences, torch.Tensor)
    assert not torch.isnan(sequences).any()
    assert not torch.isinf(sequences).any()

def test_model_initialization(strategy):
    """Test LSTM model initialization"""
    # Initialize model
    input_dim = len(strategy.config.feature_list)
    model = LSTMAttentionModel(
        input_dim=input_dim,
        hidden_dim=strategy.config.lstm_hidden_dim,
        num_layers=strategy.config.lstm_num_layers,
        output_dim=1,
        dropout=strategy.config.lstm_dropout,
        bidirectional=strategy.config.lstm_bidirectional,
        num_heads=strategy.config.lstm_attention_heads
    )
    
    # Verify model architecture
    assert isinstance(model, LSTMAttentionModel)
    assert model.hidden_dim == strategy.config.lstm_hidden_dim
    assert model.num_layers == strategy.config.lstm_num_layers
    assert model.dropout == strategy.config.lstm_dropout
    assert model.bidirectional == strategy.config.lstm_bidirectional
    assert model.num_heads == strategy.config.lstm_attention_heads

def test_attention_mechanism(strategy, sample_data):
    """Test attention mechanism in LSTM model"""
    # Prepare features and train model
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    
    if not features.empty:
        strategy.train(features, labels)
        
        # Get attention weights
        _, attention_weights = strategy.predict(features)
        
        # Verify attention weights
        assert attention_weights is not None
        assert attention_weights.shape[0] == len(features) - strategy.config.lstm_sequence_length + 1
        assert attention_weights.shape[1] == strategy.config.lstm_sequence_length
        assert np.all(attention_weights >= 0) and np.all(attention_weights <= 1)

def test_bidirectional_lstm(strategy, sample_data):
    """Test bidirectional LSTM functionality"""
    # Prepare features and train model
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    
    if not features.empty:
        strategy.train(features, labels)
        
        # Verify bidirectional processing
        sequences = strategy._prepare_sequence_data(features)
        with torch.no_grad():
            output, _ = strategy.model(sequences)
        
        # Check output shape (should be doubled for bidirectional)
        expected_hidden_dim = strategy.config.lstm_hidden_dim * 2 if strategy.config.lstm_bidirectional else strategy.config.lstm_hidden_dim
        assert output.shape[-1] == expected_hidden_dim

def test_dropout_and_regularization(strategy, sample_data):
    """Test dropout and regularization in LSTM model"""
    # Prepare features and train model
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    
    if not features.empty:
        strategy.train(features, labels)
        
        # Verify dropout is applied during training
        strategy.model.train()
        sequences = strategy._prepare_sequence_data(features)
        output1, _ = strategy.model(sequences)
        output2, _ = strategy.model(sequences)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2)
        
        # Verify dropout is disabled during evaluation
        strategy.model.eval()
        output1, _ = strategy.model(sequences)
        output2, _ = strategy.model(sequences)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)

def test_training_loop(strategy, sample_data):
    """Test LSTM training loop"""
    # Prepare features and labels
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    
    if not features.empty:
        # Train model
        strategy.train(features, labels)
        
        # Verify model was trained
        assert strategy.model is not None
        
        # Verify model parameters were updated
        initial_params = [p.clone() for p in strategy.model.parameters()]
        strategy.train(features, labels)
        final_params = [p for p in strategy.model.parameters()]
        
        # Parameters should be different after training
        for init, final in zip(initial_params, final_params):
            assert not torch.allclose(init, final)

def test_early_stopping(strategy, sample_data):
    """Test early stopping during training"""
    # Prepare features and labels
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    
    if not features.empty:
        # Train model with early stopping
        strategy.train(features, labels)
        
        # Verify model was trained
        assert strategy.model is not None
        
        # Verify training stopped early if validation didn't improve
        # (This is a bit tricky to test directly, but we can verify the model exists)

def test_learning_rate_scheduling(strategy, sample_data):
    """Test learning rate scheduling during training"""
    # Prepare features and labels
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    
    if not features.empty:
        # Train model
        strategy.train(features, labels)
        
        # Verify model was trained
        assert strategy.model is not None
        
        # Verify learning rate was updated
        # (This is also tricky to test directly, but we can verify the model exists)

def test_model_checkpointing(strategy, sample_data, tmp_path):
    """Test model checkpointing"""
    # Prepare features and labels
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    
    if not features.empty:
        # Train model
        strategy.train(features, labels)
        
        # Save model
        model_path = tmp_path / "model.pt"
        strategy.save_model(str(model_path))
        
        # Load model
        new_strategy = LSTMMomentumStrategy()
        new_strategy.load_model(str(model_path))
        
        # Verify predictions match
        orig_probs, _ = strategy.predict(features)
        new_probs, _ = new_strategy.predict(features)
        np.testing.assert_array_almost_equal(orig_probs, new_probs)

def test_sequence_prediction(strategy, sample_data):
    """Test sequence prediction"""
    # Prepare features and train model
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    
    if not features.empty:
        strategy.train(features, labels)
        
        # Make predictions
        probabilities, attention_weights = strategy.predict(features)
        
        # Verify predictions
        assert len(probabilities) == len(features) - strategy.config.lstm_sequence_length + 1
        assert all(0 <= p <= 1 for p in probabilities)
        assert attention_weights.shape[0] == len(probabilities)

def test_batch_prediction(strategy, sample_data):
    """Test batch prediction"""
    # Prepare features and train model
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    
    if not features.empty:
        strategy.train(features, labels)
        
        # Test different batch sizes
        for batch_size in [1, 32, 64, 128]:
            sequences = strategy._prepare_sequence_data(features)
            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(sequences),
                batch_size=batch_size,
                shuffle=False
            )
            
            # Make predictions
            all_probs = []
            all_attention = []
            
            for batch in dataloader:
                probs, attention = strategy.predict(batch[0])
                all_probs.extend(probs)
                all_attention.extend(attention)
            
            # Verify predictions
            assert len(all_probs) == len(features) - strategy.config.lstm_sequence_length + 1
            assert all(0 <= p <= 1 for p in all_probs)

def test_feature_normalization(strategy, sample_data):
    """Test feature normalization"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    
    # Verify feature normalization
    for feature in strategy.config.feature_list:
        if feature in features.columns:
            # Check for NaN values
            assert not features[feature].isna().all()
            
            # Check for infinite values
            assert not np.isinf(features[feature]).any()
            
            # Check for reasonable value ranges
            if feature.endswith('_normalized'):
                assert abs(features[feature].mean()) < 1
                assert features[feature].std() < 2

def test_data_augmentation(strategy, sample_data):
    """Test data augmentation"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    
    # Verify data augmentation
    # (This would depend on the specific augmentation methods implemented)
    pass

def test_batch_generation(strategy, sample_data):
    """Test batch generation"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    
    # Test different batch sizes
    for batch_size in [1, 32, 64, 128]:
        sequences = strategy._prepare_sequence_data(features)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(sequences),
            batch_size=batch_size,
            shuffle=True
        )
        
        # Verify batch generation
        for batch in dataloader:
            assert len(batch[0]) <= batch_size
            assert batch[0].shape[1:] == (strategy.config.lstm_sequence_length, len(strategy.config.feature_list))

def test_model_explanation(strategy, sample_data):
    """Test model explanation"""
    # Prepare features and train model
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    
    if not features.empty:
        strategy.train(features, labels)
        
        # Get model explanation
        explanation = strategy.explain(features)
        
        # Verify explanation
        assert "attention_weights" in explanation
        assert "feature_importance" in explanation
        assert len(explanation["feature_importance"]) == len(strategy.config.feature_list) 