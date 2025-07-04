import pytest
import torch
import numpy as np
from utils.transformer_utils import (
    EnhancedPositionalEncoding,
    TransformerAttentionModel,
    train_transformer,
    predict_transformer,
    save_transformer,
    load_transformer,
    visualize_attention
)

@pytest.fixture
def sample_data():
    # Create sample data
    batch_size = 32
    seq_len = 10
    input_size = 5
    X = torch.randn(batch_size, seq_len, input_size)
    y = torch.randn(batch_size)
    return X, y

@pytest.fixture
def transformer_model():
    # Create a small transformer model for testing
    model = TransformerAttentionModel(
        input_size=5,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64
    )
    return model

def test_enhanced_positional_encoding():
    # Test positional encoding
    d_model = 32
    seq_len = 10
    batch_size = 4
    
    pos_encoder = EnhancedPositionalEncoding(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test forward pass
    output = pos_encoder(x)
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Test learnable parameters
    assert pos_encoder.alpha.requires_grad
    assert pos_encoder.beta.requires_grad

def test_transformer_attention_model(transformer_model, sample_data):
    X, _ = sample_data
    
    # Test forward pass
    output, attention_dict = transformer_model(X)
    assert output.shape == (X.shape[0], 1)
    assert isinstance(attention_dict, dict)
    assert len(attention_dict) == transformer_model.num_layers
    
    # Test attention weights shape
    for layer_name, attention in attention_dict.items():
        assert attention.shape == (X.shape[0], X.shape[1], X.shape[1])

def test_train_transformer(transformer_model, sample_data):
    X, y = sample_data
    
    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
    
    # Setup training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(transformer_model.parameters())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    trained_model = train_transformer(
        transformer_model,
        dataloader,
        criterion,
        optimizer,
        device,
        num_epochs=2
    )
    
    assert isinstance(trained_model, TransformerAttentionModel)

def test_predict_transformer(transformer_model, sample_data):
    X, _ = sample_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Make predictions
    predictions, attention_dict = predict_transformer(transformer_model, X, device)
    
    assert predictions.shape == (X.shape[0], 1)
    assert isinstance(attention_dict, dict)
    assert len(attention_dict) == transformer_model.num_layers

def test_save_load_transformer(transformer_model, tmp_path):
    # Save model
    save_path = tmp_path / "transformer_model.pt"
    save_transformer(transformer_model, str(save_path))
    
    # Load model
    loaded_model = load_transformer(str(save_path))
    
    # Check model parameters
    assert loaded_model.input_size == transformer_model.input_size
    assert loaded_model.d_model == transformer_model.d_model
    assert loaded_model.nhead == transformer_model.nhead
    assert loaded_model.num_layers == transformer_model.num_layers

def test_visualize_attention():
    # Create sample attention weights
    batch_size = 2
    seq_len = 5
    attention_dict = {
        'layer_0_self_attention': np.random.rand(batch_size, seq_len, seq_len),
        'layer_1_self_attention': np.random.rand(batch_size, seq_len, seq_len)
    }
    
    # Test visualization
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    processed_attention = visualize_attention(
        attention_dict,
        seq_len,
        feature_names
    )
    
    assert isinstance(processed_attention, dict)
    assert len(processed_attention) == len(attention_dict)
    
    for layer_name, attention_data in processed_attention.items():
        assert 'attention_matrix' in attention_data
        assert 'time_steps' in attention_data
        assert 'feature_names' in attention_data
        assert len(attention_data['time_steps']) == seq_len
        assert attention_data['feature_names'] == feature_names 