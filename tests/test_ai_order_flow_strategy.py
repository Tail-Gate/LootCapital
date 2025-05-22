import pytest
import pandas as pd
import numpy as np
from strategies.ai_order_flow_strategy import AIOrderFlowStrategy, AIOrderFlowConfig

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create a DataFrame with required columns
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.normal(1000, 100, 100),
        'bid': np.random.normal(99.9, 0.5, 100),
        'ask': np.random.normal(100.1, 0.5, 100),
        'bid_volume': np.random.normal(500, 50, 100),
        'ask_volume': np.random.normal(500, 50, 100)
    }
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def strategy():
    """Create strategy instance for testing"""
    return AIOrderFlowStrategy()

def test_strategy_initialization(strategy):
    """Test strategy initialization"""
    assert strategy is not None
    assert strategy.config is not None
    assert isinstance(strategy.config, AIOrderFlowConfig)

def test_feature_preparation(strategy, sample_data):
    """Test feature preparation"""
    features = strategy.prepare_features(sample_data)
    
    # Check that all required features are present
    lstm_features = strategy._get_lstm_features()
    xgboost_features = strategy._get_xgboost_features()
    
    for feature in lstm_features + xgboost_features:
        assert feature in features.columns, f"Missing feature: {feature}"

def test_lstm_data_preparation(strategy, sample_data):
    """Test LSTM data preparation"""
    features = strategy.prepare_features(sample_data)
    X_lstm = strategy._prepare_lstm_data(features)
    
    # Check tensor shape
    expected_shape = (len(features) - strategy.config.lstm_sequence_length + 1,
                     strategy.config.lstm_sequence_length,
                     len(strategy._get_lstm_features()))
    assert X_lstm.shape == expected_shape

def test_xgboost_data_preparation(strategy, sample_data):
    """Test XGBoost data preparation"""
    features = strategy.prepare_features(sample_data)
    X_xgb = strategy._prepare_xgboost_data(features)
    
    # Check DataFrame shape
    expected_shape = (len(features), len(strategy._get_xgboost_features()))
    assert X_xgb.shape == expected_shape

def test_signal_calculation(strategy, sample_data):
    """Test signal calculation"""
    features = strategy.prepare_features(sample_data)
    signal, confidence, trade_type = strategy.calculate_technical_signals(features)
    
    # Check signal properties
    assert isinstance(signal, float)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1
    assert trade_type in ['DAY_TRADE', 'SWING_TRADE']

def test_signal_explanation(strategy, sample_data):
    """Test signal explanation"""
    features = strategy.prepare_features(sample_data)
    explanation = strategy.explain_signal(features)
    
    # Check explanation structure
    assert 'model' in explanation
    assert 'features' in explanation
    assert 'confidence' in explanation
    assert 'reasoning' in explanation
    
    # Check feature importances
    assert isinstance(explanation['features'], dict)
    for feature, importance in explanation['features'].items():
        assert isinstance(importance, float)
        assert 0 <= importance <= 1

def test_model_training(strategy, sample_data):
    """Test model training"""
    features = strategy.prepare_features(sample_data)
    
    # Create dummy targets (1 for price increase, 0 for decrease)
    targets = (features['close'].shift(-1) > features['close']).astype(int)
    targets = targets[:-1]  # Remove last row (no future price)
    features = features[:-1]  # Remove last row to match targets
    
    # Train models
    strategy.train_models(features, targets)
    
    # Check that models were created
    assert strategy.lstm_model is not None
    assert strategy.xgboost_model is not None

def test_model_persistence(strategy, sample_data):
    """Test model saving and loading"""
    features = strategy.prepare_features(sample_data)
    
    # Create dummy targets
    targets = (features['close'].shift(-1) > features['close']).astype(int)
    targets = targets[:-1]
    features = features[:-1]
    
    # Train models
    strategy.train_models(features, targets)
    
    # Create new strategy instance
    new_strategy = AIOrderFlowStrategy()
    
    # Check that models were loaded
    assert new_strategy.lstm_model is not None
    assert new_strategy.xgboost_model is not None

def test_ai_features(strategy, sample_data):
    """Test AI-specific feature calculation"""
    features = strategy.prepare_features(sample_data)
    
    # Check AI-specific features
    assert 'price_zscore' in features.columns
    assert 'liquidity_zone_score' in features.columns
    assert 'volume_profile' in features.columns
    
    # Check feature calculations
    assert not features['price_zscore'].isnull().any()
    assert not features['volume_profile'].isnull().any()

def test_signal_validation(strategy, sample_data):
    """Test signal validation"""
    features = strategy.prepare_features(sample_data)
    signal, confidence, trade_type = strategy.calculate_technical_signals(features)
    
    # Validate signal
    is_valid = strategy.validate_signal(signal, features)
    assert isinstance(is_valid, bool)

def test_feature_weights(strategy):
    """Test feature weight configuration"""
    config = strategy.config
    
    # Check day trading weights
    day_weights = config.feature_weights['day']
    assert 'lstm_signal' in day_weights
    assert sum(day_weights.values()) == 1.0
    
    # Check swing trading weights
    swing_weights = config.feature_weights['swing']
    assert 'xgboost_signal' in swing_weights
    assert sum(swing_weights.values()) == 1.0 