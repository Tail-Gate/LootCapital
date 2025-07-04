import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from utils.feature_generator import FeatureGenerator

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    return data

@pytest.fixture
def sample_order_book_data():
    """Create sample order book data for testing."""
    n = 100
    n_levels = 5
    # Generate random prices and volumes for each level
    bid_prices = np.random.randn(n, n_levels).cumsum(axis=1) + 99
    ask_prices = np.random.randn(n, n_levels).cumsum(axis=1) + 101
    bid_volumes = np.random.randint(100, 1000, (n, n_levels))
    ask_volumes = np.random.randint(100, 1000, (n, n_levels))
    data = pd.DataFrame({
        'bid_prices': [list(row) for row in bid_prices],
        'ask_prices': [list(row) for row in ask_prices],
        'bid_volumes': [list(row) for row in bid_volumes],
        'ask_volumes': [list(row) for row in ask_volumes],
    })
    # Also add bid_volume_0...bid_volume_4 and ask_volume_0...ask_volume_4 for depth imbalance
    for i in range(n_levels):
        data[f'bid_volume_{i}'] = bid_volumes[:, i]
        data[f'ask_volume_{i}'] = ask_volumes[:, i]
    # Add total volume columns
    data['bid_volume_total'] = bid_volumes.sum(axis=1)
    data['ask_volume_total'] = ask_volumes.sum(axis=1)
    return data

@pytest.fixture
def feature_generator():
    """Create a FeatureGenerator instance for testing."""
    return FeatureGenerator()

def test_generate_technical_features(feature_generator, sample_ohlcv_data):
    """Test technical feature generation."""
    features = feature_generator.generate_technical_features(sample_ohlcv_data)
    
    # Check RSI features
    assert 'rsi_14' in features.columns
    assert 'rsi_21' in features.columns
    assert 'rsi_50' in features.columns
    
    # Check MACD features
    assert 'macd' in features.columns
    assert 'macd_signal' in features.columns
    assert 'macd_hist' in features.columns
    
    # Check Bollinger Bands features
    assert 'bb_upper' in features.columns
    assert 'bb_middle' in features.columns
    assert 'bb_lower' in features.columns
    assert 'bb_width' in features.columns
    
    # Check volume features
    assert 'volume_ma' in features.columns
    assert 'volume_std' in features.columns
    assert 'volume_ratio' in features.columns
    
    # Verify feature calculations
    assert not features['rsi_14'].isnull().all()
    assert not features['macd'].isnull().all()
    assert not features['bb_width'].isnull().all()

def test_generate_momentum_features(feature_generator, sample_ohlcv_data):
    """Test momentum feature generation."""
    features = feature_generator.generate_momentum_features(sample_ohlcv_data)
    
    # Check returns features
    for period in [1, 3, 5, 10]:
        assert f'returns_{period}' in features.columns
        assert f'log_returns_{period}' in features.columns
    
    # Check volume momentum features
    assert 'volume_returns' in features.columns
    assert 'volume_ma_ratio' in features.columns
    
    # Check trend features
    assert 'adx' in features.columns
    assert 'trend_strength' in features.columns
    
    # Check momentum divergence
    assert 'price_momentum' in features.columns
    assert 'volume_momentum' in features.columns
    assert 'momentum_divergence' in features.columns
    
    # Verify feature calculations
    assert not features['returns_5'].isnull().all()
    assert not features['adx'].isnull().all()
    assert features['momentum_divergence'].dtype == bool

def test_generate_order_book_features(feature_generator, sample_ohlcv_data, sample_order_book_data):
    """Test order book feature generation."""
    features = feature_generator.generate_order_book_features(
        sample_ohlcv_data,
        sample_order_book_data
    )
    
    # Check order book features
    assert 'order_flow_imbalance' in features.columns
    assert 'volume_pressure' in features.columns
    assert 'depth_imbalance' in features.columns
    assert 'spread' in features.columns
    
    # Verify feature calculations
    assert not features['order_flow_imbalance'].isnull().all()
    assert not features['depth_imbalance'].isnull().all()
    assert features['spread'].mean() > 0  # Spread should be positive

def test_feature_importance_tracking(feature_generator):
    """Test feature importance tracking."""
    importance_dict = {
        'rsi_14': 0.5,
        'macd': 0.3,
        'volume_ratio': 0.2
    }
    
    feature_generator.update_feature_importance(importance_dict)
    assert feature_generator.feature_importance == importance_dict

def test_feature_interaction_detection(feature_generator, sample_ohlcv_data):
    """Test feature interaction detection."""
    # Add some correlated features
    sample_ohlcv_data['correlated_1'] = sample_ohlcv_data['close'] * 1.1
    sample_ohlcv_data['correlated_2'] = sample_ohlcv_data['close'] * 0.9
    
    interactions = feature_generator.detect_feature_interactions(
        sample_ohlcv_data,
        target_col='close',
        threshold=0.3
    )
    
    assert isinstance(interactions, dict)
    assert len(interactions) > 0

def test_save_load_state(feature_generator, sample_ohlcv_data):
    """Test saving and loading feature generator state."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate some features and update importance
        features = feature_generator.generate_technical_features(sample_ohlcv_data)
        feature_generator.update_feature_importance({'rsi_14': 0.5})
        
        # Save state
        feature_generator.save_state(temp_dir)
        
        # Create new generator and load state
        new_generator = FeatureGenerator()
        new_generator.load_state(temp_dir)
        
        # Verify state was loaded correctly
        assert new_generator.feature_importance == feature_generator.feature_importance
        assert new_generator.config == feature_generator.config 

def test_calculate_historical_volatility(feature_generator, sample_ohlcv_data):
    """Test historical volatility calculation."""
    volatility = feature_generator.calculate_historical_volatility(sample_ohlcv_data['close'])
    assert not volatility.isnull().all()
    assert volatility.dtype == float

def test_calculate_ichimoku_cloud(feature_generator, sample_ohlcv_data):
    """Test Ichimoku Cloud calculation."""
    conversion_line, base_line, leading_span_a, leading_span_b, lagging_span = feature_generator.calculate_ichimoku_cloud(
        sample_ohlcv_data['high'],
        sample_ohlcv_data['low'],
        sample_ohlcv_data['close']
    )
    assert not conversion_line.isnull().all()
    assert not base_line.isnull().all()
    assert not leading_span_a.isnull().all()
    assert not leading_span_b.isnull().all()
    assert not lagging_span.isnull().all() 