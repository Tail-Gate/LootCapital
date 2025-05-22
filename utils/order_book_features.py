import pandas as pd
import numpy as np
from typing import Optional

def calculate_order_flow_imbalance(order_book_data: pd.DataFrame, bid_prefix: str = 'bid_volume_', ask_prefix: str = 'ask_volume_', levels: int = 5) -> pd.Series:
    """
    Calculate order flow imbalance from order book data.
    Args:
        order_book_data: DataFrame with bid/ask volume columns (e.g., bid_volume_0, ask_volume_0, ...)
        bid_prefix: Prefix for bid volume columns
        ask_prefix: Prefix for ask volume columns
        levels: Number of levels to use
    Returns:
        Series with order flow imbalance values
    """
    bid_cols = [f'{bid_prefix}{i}' for i in range(levels)]
    ask_cols = [f'{ask_prefix}{i}' for i in range(levels)]
    bid_volume = order_book_data[bid_cols].sum(axis=1)
    ask_volume = order_book_data[ask_cols].sum(axis=1)
    return (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-9)

def calculate_volume_pressure(order_book_data: pd.DataFrame, price_col: Optional[str] = None, bid_prefix: str = 'bid_volume_', ask_prefix: str = 'ask_volume_', levels: int = 5) -> pd.Series:
    """
    Calculate volume pressure as the difference between bid and ask volumes, optionally weighted by price movement.
    Args:
        order_book_data: DataFrame with bid/ask volume columns
        price_col: Optional price column to weight pressure (not used by default)
        bid_prefix: Prefix for bid volume columns
        ask_prefix: Prefix for ask volume columns
        levels: Number of levels to use
    Returns:
        Series with volume pressure values
    """
    bid_cols = [f'{bid_prefix}{i}' for i in range(levels)]
    ask_cols = [f'{ask_prefix}{i}' for i in range(levels)]
    bid_volume = order_book_data[bid_cols].sum(axis=1)
    ask_volume = order_book_data[ask_cols].sum(axis=1)
    pressure = bid_volume - ask_volume
    if price_col and price_col in order_book_data:
        price_diff = order_book_data[price_col].diff().fillna(0)
        return pressure * np.sign(price_diff)
    return pressure 