import pandas as pd
import numpy as np
from typing import Optional, Union, List, Literal

class LabelGenerator:
    """
    Generates labels for trading data for ML tasks (classification/regression).
    Supports binary, multi-class, and regression labeling with configurable lookahead and threshold.
    """
    def __init__(
        self,
        lookahead: int = 1,
        threshold: float = 0.0,
        label_type: Literal['binary', 'multiclass', 'regression'] = 'binary',
        price_diff: Literal['close_to_close', 'open_to_close', 'custom'] = 'close_to_close',
        custom_price_cols: Optional[List[str]] = None,
        multiclass_bins: Optional[List[float]] = None,
        multiclass_labels: Optional[List[Union[int, str]]] = None
    ):
        """
        Args:
            lookahead: Number of bars to look ahead for label generation
            threshold: Threshold for binary/multiclass labeling (e.g., 0.005 for 0.5% move)
            label_type: 'binary', 'multiclass', or 'regression'
            price_diff: Which price columns to use for return calculation
            custom_price_cols: If price_diff='custom', provide [col_now, col_future]
            multiclass_bins: For multiclass, list of bin edges (e.g., [-np.inf, -0.01, 0.01, np.inf])
            multiclass_labels: For multiclass, list of labels for each bin
        """
        self.lookahead = lookahead
        self.threshold = threshold
        self.label_type = label_type
        self.price_diff = price_diff
        self.custom_price_cols = custom_price_cols
        self.multiclass_bins = multiclass_bins
        self.multiclass_labels = multiclass_labels

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate labels for the given DataFrame.
        Args:
            df: DataFrame with price columns (must have 'close', 'open' if needed)
        Returns:
            pd.Series of labels (same index as df)
        """
        if self.price_diff == 'close_to_close':
            price_now = df['close']
            price_future = df['close'].shift(-self.lookahead)
        elif self.price_diff == 'open_to_close':
            price_now = df['open']
            price_future = df['close'].shift(-self.lookahead)
        elif self.price_diff == 'custom' and self.custom_price_cols:
            price_now = df[self.custom_price_cols[0]]
            price_future = df[self.custom_price_cols[1]].shift(-self.lookahead)
        else:
            raise ValueError('Invalid price_diff or missing custom_price_cols')

        returns = (price_future - price_now) / price_now

        if self.label_type == 'binary':
            # 1 = up, 0 = down/flat (or use -1/1 if preferred)
            labels = (returns > self.threshold).astype(int)
        elif self.label_type == 'multiclass':
            if self.multiclass_bins is None or self.multiclass_labels is None:
                # Default: 3-class (down, flat, up)
                bins = [-np.inf, -self.threshold, self.threshold, np.inf]
                labels = [-1, 0, 1]
            else:
                bins = self.multiclass_bins
                labels = self.multiclass_labels
            labels = pd.cut(returns, bins=bins, labels=labels)
        elif self.label_type == 'regression':
            labels = returns
        else:
            raise ValueError('Invalid label_type')

        # Remove labels for last lookahead bars (no future data)
        labels.iloc[-self.lookahead:] = np.nan
        return labels

    @staticmethod
    def event_based_label(df: pd.DataFrame, lookahead: int = 10, threshold: float = 0.02, price_col: str = 'close') -> pd.Series:
        """
        Label each bar as 1 if a +threshold move occurs within lookahead bars,
        -1 if a -threshold move occurs within lookahead bars, 0 otherwise.
        Args:
            df: DataFrame with price data (must have price_col)
            lookahead: Number of bars to look ahead
            threshold: Price move threshold (e.g., 0.02 for 2%)
            price_col: Column to use for price
        Returns:
            pd.Series of labels (1, -1, 0)
        """
        prices = df[price_col].values
        labels = np.zeros(len(prices), dtype=float)
        for i in range(len(prices) - lookahead):
            window = prices[i+1:i+1+lookahead]
            pct_change = (window - prices[i]) / prices[i]
            if np.any(pct_change >= threshold):
                labels[i] = 1
            elif np.any(pct_change <= -threshold):
                labels[i] = -1
            else:
                labels[i] = 0
        # Last lookahead bars cannot be labeled
        labels[-lookahead:] = np.nan
        return pd.Series(labels, index=df.index) 