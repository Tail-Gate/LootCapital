import pytest
import pandas as pd
import numpy as np
from strategies.momentum_strategy import MomentumStrategy, MomentumConfig

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
    """Create a MomentumStrategy instance for testing"""
    return MomentumStrategy()

def test_lagged_returns_calculation(strategy, sample_data):
    """Test that lagged returns are calculated correctly"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    
    # Check that lagged returns columns exist
    assert 'returns_1' in features.columns
    assert 'returns_2' in features.columns
    assert 'returns_3' in features.columns
    
    # Verify calculations
    expected_returns_1 = sample_data['close'].pct_change(periods=1).rename('returns_1')
    expected_returns_2 = sample_data['close'].pct_change(periods=2).rename('returns_2')
    expected_returns_3 = sample_data['close'].pct_change(periods=3).rename('returns_3')
    
    pd.testing.assert_series_equal(features['returns_1'], expected_returns_1)
    pd.testing.assert_series_equal(features['returns_2'], expected_returns_2)
    pd.testing.assert_series_equal(features['returns_3'], expected_returns_3)
    
    # Check for NaN values in the first few rows (expected due to lag)
    assert np.isnan(features['returns_1'].iloc[0])
    assert np.isnan(features['returns_2'].iloc[0:2]).all()
    assert np.isnan(features['returns_3'].iloc[0:3]).all()
    
    # Check that non-lagged values are not NaN
    assert not features['returns_1'].iloc[1:].isna().any()
    assert not features['returns_2'].iloc[2:].isna().any()
    assert not features['returns_3'].iloc[3:].isna().any()

def test_rolling_statistics_calculation(strategy, sample_data):
    """Test that rolling statistics are calculated correctly"""
    features = strategy.prepare_features(sample_data)
    
    # Check that rolling statistics columns exist
    assert 'rolling_mean' in features.columns
    assert 'rolling_std' in features.columns
    
    # Verify calculations
    expected_rolling_mean = sample_data['close'].pct_change(periods=1).rolling(window=20).mean().rename('rolling_mean')
    expected_rolling_std = sample_data['close'].pct_change(periods=1).rolling(window=20).std().rename('rolling_std')
    
    pd.testing.assert_series_equal(features['rolling_mean'], expected_rolling_mean)
    pd.testing.assert_series_equal(features['rolling_std'], expected_rolling_std)

def test_roc_calculation(strategy, sample_data):
    """Test that ROC calculations are correct for multiple periods"""
    features = strategy.prepare_features(sample_data)
    
    # Check that ROC columns exist for all periods
    for period in [1, 3, 5, 10]:
        assert f'roc_{period}' in features.columns
        assert f'norm_roc_{period}' in features.columns
    
    # Verify calculations for each period
    for period in [1, 3, 5, 10]:
        expected_roc = sample_data['close'].pct_change(periods=period).rename(f'roc_{period}')
        expected_norm_roc = expected_roc / features['volatility']
        expected_norm_roc = expected_norm_roc.rename(f'norm_roc_{period}')
        
        pd.testing.assert_series_equal(features[f'roc_{period}'], expected_roc)
        pd.testing.assert_series_equal(features[f'norm_roc_{period}'], expected_norm_roc)
        
        # Check for NaN values in the first few rows (expected due to lag)
        assert np.isnan(features[f'roc_{period}'].iloc[0:period]).all()
        assert not features[f'roc_{period}'].iloc[period:].isna().any()

def test_volume_roc_calculation(strategy, sample_data):
    """Test that Volume ROC calculations are correct"""
    features = strategy.prepare_features(sample_data)
    
    # Check that Volume ROC columns exist
    for period in [1, 3, 5]:
        assert f'volume_roc_{period}' in features.columns
        assert f'norm_volume_roc_{period}' in features.columns
    
    # Calculate expected values
    volume_ma = sample_data['volume'].rolling(window=20).mean()
    
    # Verify calculations for each period
    for period in [1, 3, 5]:
        expected_volume_roc = sample_data['volume'].pct_change(periods=period).rename(f'volume_roc_{period}')
        expected_norm_volume_roc = expected_volume_roc / volume_ma
        expected_norm_volume_roc = expected_norm_volume_roc.rename(f'norm_volume_roc_{period}')
        
        pd.testing.assert_series_equal(features[f'volume_roc_{period}'], expected_volume_roc)
        pd.testing.assert_series_equal(features[f'norm_volume_roc_{period}'], expected_norm_volume_roc)
        
        # Check for NaN values in the first few rows (expected due to lag)
        assert np.isnan(features[f'volume_roc_{period}'].iloc[0:period]).all()
        assert not features[f'volume_roc_{period}'].iloc[period:].isna().any()

def test_order_book_features_calculation(strategy, sample_data):
    """Test that order book features are calculated correctly"""
    features = strategy.prepare_features(sample_data)
    
    # Check that order book feature columns exist
    assert 'order_book_imbalance' in features.columns
    assert 'volume_spike' in features.columns
    assert 'market_depth' in features.columns
    assert 'depth_imbalance' in features.columns
    assert 'volume_delta' in features.columns
    assert 'cumulative_delta' in features.columns
    assert 'volume_pressure' in features.columns
    assert 'significant_imbalance' in features.columns
    
    # Verify order book imbalance calculation
    expected_imbalance = (sample_data['bid_volume'] - sample_data['ask_volume']) / (sample_data['bid_volume'] + sample_data['ask_volume'])
    expected_imbalance = expected_imbalance.rename('order_book_imbalance')
    pd.testing.assert_series_equal(features['order_book_imbalance'], expected_imbalance)
    
    # Verify volume spike detection
    volume_mean = sample_data['volume'].rolling(window=20).mean()
    volume_std = sample_data['volume'].rolling(window=20).std()
    expected_spike = sample_data['volume'] > (volume_mean + 2 * volume_std)
    expected_spike = expected_spike.rename('volume_spike')
    pd.testing.assert_series_equal(features['volume_spike'], expected_spike)
    
    # Verify market depth calculation
    expected_depth = sample_data['bid_volume'] + sample_data['ask_volume']
    expected_depth = expected_depth.rename('market_depth')
    pd.testing.assert_series_equal(features['market_depth'], expected_depth)
    
    # Verify depth imbalance calculation
    expected_depth_imbalance = (sample_data['bid_volume'] - sample_data['ask_volume']) / expected_depth
    expected_depth_imbalance = expected_depth_imbalance.rename('depth_imbalance')
    pd.testing.assert_series_equal(features['depth_imbalance'], expected_depth_imbalance)
    
    # Verify volume delta calculation
    expected_delta = sample_data['bid_volume'] - sample_data['ask_volume']
    expected_delta = expected_delta.rename('volume_delta')
    pd.testing.assert_series_equal(features['volume_delta'], expected_delta)
    
    # Verify cumulative delta calculation
    expected_cumulative_delta = expected_delta.cumsum()
    expected_cumulative_delta = expected_cumulative_delta.rename('cumulative_delta')
    pd.testing.assert_series_equal(features['cumulative_delta'], expected_cumulative_delta)
    
    # Verify volume pressure calculation
    expected_pressure = expected_delta * features['returns_1']
    expected_pressure = expected_pressure.rename('volume_pressure')
    pd.testing.assert_series_equal(features['volume_pressure'], expected_pressure)
    
    # Verify significant imbalance detection
    expected_significant = abs(expected_imbalance) > strategy.config.obi_threshold
    expected_significant = expected_significant.rename('significant_imbalance')
    pd.testing.assert_series_equal(features['significant_imbalance'], expected_significant)

def test_order_book_features_without_data(strategy, sample_data):
    """Test that order book features handle missing data correctly"""
    # Remove order book data
    data_without_ob = sample_data.drop(['bid_volume', 'ask_volume'], axis=1)
    features = strategy.prepare_features(data_without_ob)
    
    # Check that order book feature columns exist with default values
    assert 'order_book_imbalance' in features.columns
    assert 'volume_spike' in features.columns
    assert 'market_depth' in features.columns
    assert 'depth_imbalance' in features.columns
    assert 'volume_delta' in features.columns
    assert 'cumulative_delta' in features.columns
    assert 'volume_pressure' in features.columns
    assert 'significant_imbalance' in features.columns
    
    # Verify default values
    assert (features['order_book_imbalance'] == 0).all()
    assert not features['volume_spike'].any()
    assert (features['market_depth'] == features['volume']).all()
    assert (features['depth_imbalance'] == 0).all()
    assert (features['volume_delta'] == 0).all()
    assert (features['cumulative_delta'] == 0).all()
    assert (features['volume_pressure'] == 0).all()
    assert not features['significant_imbalance'].any()

def test_volatility_features_calculation(strategy, sample_data):
    """Test that volatility features are calculated correctly"""
    features = strategy.prepare_features(sample_data)
    
    # Check that all volatility features exist
    expected_features = [
        'historical_vol',
        'realized_vol',
        'vol_ratio',
        'volatility_regime',
        'volatility_percentile',
        'volatility_divergence'
    ]
    for feature in expected_features:
        assert feature in features.columns, f"Missing feature: {feature}"
    
    # Test historical volatility calculation
    expected_vol = features['returns_1'].rolling(window=20).std() * np.sqrt(252)
    pd.testing.assert_series_equal(
        features['historical_vol'],
        expected_vol,
        check_names=False
    )
    
    # Test realized volatility calculation
    expected_realized_vol = np.sqrt(
        features['returns_1'].rolling(window=20).apply(lambda x: np.sum(x**2))
    ) * np.sqrt(252)
    pd.testing.assert_series_equal(
        features['realized_vol'],
        expected_realized_vol,
        check_names=False
    )
    
    # Test volatility ratio
    expected_ratio = features['historical_vol'] / features['historical_vol'].rolling(window=50).mean()
    pd.testing.assert_series_equal(
        features['vol_ratio'],
        expected_ratio,
        check_names=False
    )
    
    # Test volatility regime (ignore NaNs)
    non_nan_regimes = features['volatility_regime'].dropna()
    assert non_nan_regimes.isin(['low', 'normal', 'high']).all()
    
    # Test volatility percentile (ignore NaNs)
    non_nan_percentiles = features['volatility_percentile'].dropna()
    assert (non_nan_percentiles >= 0).all()
    assert (non_nan_percentiles <= 1).all()
    
    # Test volatility divergence
    assert features['volatility_divergence'].isin([-1, 0, 1]).all()
    
    # Test for NaN values in early periods
    assert features['historical_vol'].iloc[:20].isna().all()
    assert features['realized_vol'].iloc[:20].isna().all()
    assert features['vol_ratio'].iloc[:50].isna().all()
    assert features['volatility_percentile'].iloc[:100].isna().all()

def test_xgboost_training(strategy, sample_data):
    """Test XGBoost model training"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    # Create binary labels (1 for price increase, 0 for decrease)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    # Drop rows with NaNs in required features
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    if features.empty:
        pytest.skip("Not enough valid rows after dropping NaNs in required features.")
    # Train model
    strategy.train(features, labels)
    # Verify model was trained
    assert strategy.model is not None

def test_xgboost_prediction(strategy, sample_data):
    """Test XGBoost model prediction"""
    # Prepare features
    features = strategy.prepare_features(sample_data)
    # Create binary labels
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    # Drop rows with NaNs in required features
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    if features.empty:
        pytest.skip("Not enough valid rows after dropping NaNs in required features.")
    # Train model
    strategy.train(features, labels)
    # Make predictions
    probabilities, predictions = strategy.predict(features)
    # Verify predictions
    assert len(probabilities) == len(features)
    assert len(predictions) == len(features)
    assert all(p in [0, 1] for p in predictions)
    assert all(0 <= p <= 1 for p in probabilities)

def test_xgboost_save_load(strategy, sample_data, tmp_path):
    """Test XGBoost model save and load"""
    # Prepare features and labels
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    # Drop rows with NaNs in required features
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    if features.empty:
        pytest.skip("Not enough valid rows after dropping NaNs in required features.")
    # Train model
    strategy.train(features, labels)
    # Save model
    model_path = tmp_path / "model.json"
    strategy.save_model(str(model_path))
    # Create new strategy instance
    new_strategy = MomentumStrategy()
    # Load model
    new_strategy.load_model(str(model_path))
    # Verify predictions match
    orig_probs, _ = strategy.predict(features)
    new_probs, _ = new_strategy.predict(features)
    np.testing.assert_array_almost_equal(orig_probs, new_probs)

def test_xgboost_feature_importance(strategy, sample_data):
    """Test XGBoost feature importance calculation"""
    # Prepare features and labels
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    # Drop rows with NaNs in required features
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    if features.empty:
        pytest.skip("Not enough valid rows after dropping NaNs in required features.")
    # Train model
    strategy.train(features, labels)
    # Get feature importance
    importance = strategy.get_feature_importance()
    # Verify importance scores
    assert importance is not None
    assert len(importance) > 0
    assert all(score >= 0 for score in importance.values())

def test_xgboost_explanation(strategy, sample_data):
    """Test XGBoost model explanation"""
    # Prepare features and labels
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    # Drop rows with NaNs in required features
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    if features.empty:
        pytest.skip("Not enough valid rows after dropping NaNs in required features.")
    # Train model
    strategy.train(features, labels)
    # Get explanation
    shap_values, importance = strategy.explain(features)
    # Verify explanation
    assert shap_values is not None
    assert importance is not None
    assert len(importance) > 0

def test_xgboost_signal_generation(strategy, sample_data):
    """Test XGBoost-based signal generation"""
    # Prepare features and labels
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    # Drop rows with NaNs in required features
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    if features.empty:
        pytest.skip("Not enough valid rows after dropping NaNs in required features.")
    # Train model
    strategy.train(features, labels)
    # Generate signals
    signal, confidence, trade_type = strategy.calculate_signals(features)
    # Verify signal properties
    assert signal in [-1, 0, 1]
    assert 0 <= confidence <= 1
    assert trade_type in [strategy.TradeType.DAY_TRADE, strategy.TradeType.SWING_TRADE]

def test_xgboost_signal_validation(strategy, sample_data):
    """Test XGBoost signal validation"""
    # Prepare features and labels
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    # Drop rows with NaNs in required features
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    if features.empty:
        pytest.skip("Not enough valid rows after dropping NaNs in required features.")
    # Train model
    strategy.train(features, labels)
    # Generate signals
    signal, confidence, trade_type = strategy.calculate_signals(features)
    # Validate signal
    is_valid = strategy.validate_signal(signal, features)
    assert isinstance(is_valid, bool)

def test_xgboost_model_update(strategy, sample_data):
    """Test XGBoost model update functionality"""
    # Prepare features and labels
    features = strategy.prepare_features(sample_data)
    labels = (features['close'].shift(-1) > features['close']).astype(int)
    # Drop rows with NaNs in required features
    features = features.dropna(subset=strategy.config.feature_list)
    labels = labels.loc[features.index]
    if features.empty:
        pytest.skip("Not enough valid rows after dropping NaNs in required features.")
    # Train model
    strategy.train(features, labels)
    # Get initial predictions
    initial_probs, _ = strategy.predict(features)
    # Update model with new data
    new_features = features.copy()
    new_features['close'] = new_features['close'] * 1.1  # Simulate price increase
    new_labels = (new_features['close'].shift(-1) > new_features['close']).astype(int)
    new_features = new_features.dropna(subset=strategy.config.feature_list)
    new_labels = new_labels.loc[new_features.index]
    if new_features.empty:
        pytest.skip("Not enough valid rows after dropping NaNs in required features (update phase).")
    strategy.update_model(new_features, new_labels)
    # Get updated predictions
    updated_probs, _ = strategy.predict(new_features)
    # Verify predictions changed
    assert not np.array_equal(initial_probs, updated_probs) 