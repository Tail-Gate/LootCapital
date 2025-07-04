# Feature Engineering Best Practices for ML Trading Systems

## The Problem: Raw OHLCV Data in ML Training

### What Was Wrong

The original code was incorrectly including raw OHLCV (Open, High, Low, Close, Volume) data as features in the ML model:

```python
# ❌ WRONG - Raw OHLCV data as features
features = [
    'close', 'volume', 'returns', 'log_returns',  # 'close' and 'volume' are raw data
    'rsi', 'macd', 'bb_upper', 'bb_lower',
    # ...
]
```

### Why This Is Problematic

1. **Look-ahead Bias**: Raw price data can introduce look-ahead bias if not handled carefully
2. **Scale Issues**: Raw prices have different scales than engineered features (e.g., $50,000 vs 0.02)
3. **Live Trading Incompatibility**: In live trading, you won't have the "close" price until the candle closes
4. **Feature Leakage**: Raw prices can leak information about the target variable
5. **Non-stationarity**: Raw prices are non-stationary, making them poor ML features

## The Solution: Engineered Features Only

### What We Fixed

```python
# ✅ CORRECT - Engineered features only
features = [
    # Price-derived features (safe for ML)
    'returns', 'log_returns',
    
    # Technical indicators (derived from price)
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    
    # Volume-derived features (safe for ML)
    'volume_ma', 'volume_std', 'volume_surge', 'volume_ratio',
    
    # Momentum and regime features
    'price_momentum', 'volatility_regime',
    
    # Support/Resistance levels (derived from price)
    'support', 'resistance',
    
    # VWAP analysis
    'vwap_ratio',
    
    # Additional technical features
    'ma_crossover', 'swing_rsi', 'breakout_intensity',
    'adx', 'cumulative_delta', 'atr'
]
```

## Feature Engineering Best Practices

### 1. Use Derived Features, Not Raw Data

| ❌ Avoid | ✅ Use Instead |
|----------|----------------|
| `'close'` | `'returns'`, `'log_returns'` |
| `'volume'` | `'volume_ma'`, `'volume_ratio'`, `'volume_surge'` |
| `'high'`, `'low'` | `'atr'`, `'bb_width'`, `'support'`, `'resistance'` |
| `'open'` | `'gap'`, `'overnight_return'` |

### 2. Ensure Live Trading Compatibility

**Question**: "In live trading, we won't have the close price until the candle closes, so how do we handle this?"

**Answer**: 
- Use only features that can be calculated from **completed** candles
- For real-time predictions, use the most recent **completed** candle's data
- Never use features that depend on the current (incomplete) candle

### 3. Feature Categories for Trading

#### Price-Derived Features (Safe)
- **Returns**: `pct_change()`, `log_returns`
- **Momentum**: Price momentum over different periods
- **Volatility**: Rolling standard deviation, ATR
- **Trend**: Moving averages, trend strength indicators

#### Volume-Derived Features (Safe)
- **Volume Analysis**: Volume moving averages, volume ratios
- **Volume Patterns**: Volume surge, volume momentum
- **Order Flow**: VWAP, volume-weighted indicators

#### Technical Indicators (Safe)
- **Oscillators**: RSI, MACD, Stochastic
- **Bands**: Bollinger Bands, Keltner Channels
- **Trend**: ADX, Moving Average Crossovers

#### Market Microstructure (Safe)
- **Support/Resistance**: Dynamic levels
- **Breakout Detection**: Price level analysis
- **Market Regime**: Volatility regime, trend regime

### 4. Feature Scaling Considerations

```python
# Different scaling strategies for different feature types
scaler_config = {
    'minmax': ['rsi', 'bb_width', 'volume_ratio'],  # Bounded features
    'standard': ['returns', 'log_returns', 'momentum'],  # Unbounded features
    'robust': ['volume', 'atr']  # Features with outliers
}
```

### 5. Temporal Feature Engineering

#### Lag Features
```python
# Use lagged features to avoid look-ahead bias
features['returns_lag1'] = features['returns'].shift(1)
features['volume_lag1'] = features['volume_ma'].shift(1)
```

#### Rolling Window Features
```python
# Rolling statistics over different windows
features['returns_ma_5'] = features['returns'].rolling(5).mean()
features['volatility_20'] = features['returns'].rolling(20).std()
```

## Implementation Guidelines

### 1. Feature Validation

Before using any feature, ask:
- [ ] Can this be calculated from completed candles only?
- [ ] Is this feature stationary or made stationary?
- [ ] Does this feature have predictive power?
- [ ] Is this feature robust to market changes?

### 2. Feature Selection Process

```python
def validate_features(features_list):
    """Validate that features are suitable for ML trading"""
    
    raw_data_features = ['open', 'high', 'low', 'close', 'volume']
    
    for feature in features_list:
        if feature in raw_data_features:
            raise ValueError(f"Feature '{feature}' is raw OHLCV data. Use engineered features instead.")
    
    return True
```

### 3. Live Trading Feature Pipeline

```python
def prepare_live_features(latest_completed_candle):
    """Prepare features for live trading using only completed data"""
    
    # Only use data from the most recent COMPLETED candle
    features = {
        'returns': latest_completed_candle['returns'],
        'volume_ratio': latest_completed_candle['volume_ratio'],
        'rsi': latest_completed_candle['rsi'],
        # ... other engineered features
    }
    
    return features
```

## Files Fixed

The following files were updated to remove raw OHLCV data:

1. `scripts/walk_forward_optimization.py` - Main walk-forward optimization
2. `scripts/train_stgnn_improved.py` - STGNN training script
3. `scripts/analyze_stgnn_data.py` - Data analysis script
4. `scripts/test_event_based_analysis.py` - Event-based analysis test
5. `tests/test_stgnn_data.py` - Unit tests

## Testing the Fix

To verify the fix works:

```bash
# Run the walk-forward optimization
python scripts/walk_forward_optimization.py

# Check that no warnings about missing 'close' or 'volume' features appear
# The FeatureGenerator should now work correctly with engineered features only
```

## Benefits of This Fix

1. **Eliminates Look-ahead Bias**: No raw price data that could leak future information
2. **Improves Model Robustness**: Engineered features are more stable and predictive
3. **Enables Live Trading**: All features can be calculated from completed candles
4. **Better Feature Scaling**: Engineered features have more consistent scales
5. **Reduces Overfitting**: Raw price data can cause models to memorize price levels

## Future Considerations

1. **Feature Importance Analysis**: Regularly analyze which features contribute most to predictions
2. **Feature Stability**: Monitor feature distributions over time to detect regime changes
3. **Feature Engineering Pipeline**: Automate the creation of new features based on market analysis
4. **Cross-Validation**: Use walk-forward validation to ensure features work across different time periods

## Conclusion

By removing raw OHLCV data and using only properly engineered features, we've created a more robust, live-trading-compatible ML system. This change eliminates potential sources of bias and ensures that the model can be deployed in real-world trading scenarios. 