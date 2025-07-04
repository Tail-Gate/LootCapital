import pandas as pd
import numpy as np
import pytest
from utils.labeling import LabelGenerator

def test_binary_label_generation():
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'close': [101, 102, 101, 104, 105]
    })
    gen = LabelGenerator(lookahead=1, threshold=0.005, label_type='binary')
    labels = gen.generate(df)
    # Should be 1 if next close > current close + 0.5%
    expected = pd.Series([1, 0, 1, 1, np.nan])
    pd.testing.assert_series_equal(labels.reset_index(drop=True), expected, check_names=False)

def test_multiclass_label_generation():
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'close': [101, 102, 101, 104, 105]
    })
    gen = LabelGenerator(lookahead=1, threshold=0.005, label_type='multiclass')
    labels = gen.generate(df)
    # Default bins: [-inf, -0.005, 0.005, inf], labels: [-1, 0, 1]
    expected = pd.Series([1, -1, 1, 1, np.nan])
    pd.testing.assert_series_equal(labels.reset_index(drop=True).astype('float'), expected.astype('float'), check_names=False)

def test_regression_label_generation():
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'close': [101, 102, 101, 104, 105]
    })
    gen = LabelGenerator(lookahead=1, label_type='regression')
    labels = gen.generate(df)
    expected = ((df['close'].shift(-1) - df['close']) / df['close']).astype(float)
    expected.iloc[-1] = np.nan
    pd.testing.assert_series_equal(labels.reset_index(drop=True), expected.reset_index(drop=True), check_names=False) 