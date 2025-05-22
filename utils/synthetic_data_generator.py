import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    """
    Generates synthetic market data for testing purposes.
    Creates realistic OHLCV data, order book data, and derived features.
    """
    
    def __init__(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        base_price: float = 100.0,
        volatility: float = 0.02,
        trend_strength: float = 0.1,
        volume_base: float = 1000.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            base_price: Base price for the asset
            volatility: Base volatility level
            trend_strength: Strength of the trend component
            volume_base: Base volume level
            seed: Random seed for reproducibility
        """
        self.start_date = start_date or (datetime.now() - timedelta(days=30))
        self.end_date = end_date or datetime.now()
        self.base_price = base_price
        self.volatility = volatility
        self.trend_strength = trend_strength
        self.volume_base = volume_base
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_ohlcv(
        self,
        interval_minutes: int = 15,
        add_noise: bool = True
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data.
        
        Args:
            interval_minutes: Time interval in minutes
            add_noise: Whether to add noise to the data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate timestamps
        timestamps = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=f"{interval_minutes}T"
        )
        
        # Generate price series with trend and noise
        n_periods = len(timestamps)
        trend = np.linspace(0, self.trend_strength, n_periods)
        noise = np.random.normal(0, self.volatility, n_periods)
        price_series = self.base_price * (1 + trend + noise)
        
        # Generate OHLCV data
        data = []
        for i in range(n_periods):
            # Add some intra-period volatility
            if add_noise:
                period_noise = np.random.normal(0, self.volatility/2, 4)
            else:
                period_noise = np.zeros(4)
            
            open_price = price_series[i] * (1 + period_noise[0])
            high_price = price_series[i] * (1 + max(period_noise))
            low_price = price_series[i] * (1 + min(period_noise))
            close_price = price_series[i] * (1 + period_noise[-1])
            
            # Generate volume with some correlation to price movement
            price_change = abs(close_price - open_price) / open_price
            volume = self.volume_base * (1 + price_change * 10) * (1 + np.random.normal(0, 0.2))
            
            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def generate_order_book(
        self,
        ohlcv_data: pd.DataFrame,
        n_levels: int = 10,
        spread_pct: float = 0.001
    ) -> pd.DataFrame:
        """
        Generate synthetic order book data.
        
        Args:
            ohlcv_data: OHLCV data to base order book on
            n_levels: Number of price levels in the order book
            spread_pct: Base spread percentage
            
        Returns:
            DataFrame with order book data
        """
        order_book_data = []
        
        for _, row in ohlcv_data.iterrows():
            mid_price = row['close']
            spread = mid_price * spread_pct
            
            # Generate bid side
            bid_prices = np.linspace(mid_price - spread, mid_price - spread * n_levels, n_levels)
            bid_volumes = np.random.lognormal(mean=5, sigma=1, size=n_levels)
            
            # Generate ask side
            ask_prices = np.linspace(mid_price + spread, mid_price + spread * n_levels, n_levels)
            ask_volumes = np.random.lognormal(mean=5, sigma=1, size=n_levels)
            
            # Add some noise to volumes
            bid_volumes *= (1 + np.random.normal(0, 0.1, n_levels))
            ask_volumes *= (1 + np.random.normal(0, 0.1, n_levels))
            
            order_book_data.append({
                'timestamp': row['timestamp'],
                'bid_prices': bid_prices,
                'bid_volumes': bid_volumes,
                'ask_prices': ask_prices,
                'ask_volumes': ask_volumes
            })
        
        return pd.DataFrame(order_book_data)
    
    def generate_features(
        self,
        ohlcv_data: pd.DataFrame,
        order_book_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate synthetic features from OHLCV and order book data.
        
        Args:
            ohlcv_data: OHLCV data
            order_book_data: Order book data
            
        Returns:
            DataFrame with generated features
        """
        # Calculate returns
        ohlcv_data['returns'] = ohlcv_data['close'].pct_change()
        
        # Calculate rolling statistics
        ohlcv_data['rolling_mean'] = ohlcv_data['close'].rolling(window=20).mean()
        ohlcv_data['rolling_std'] = ohlcv_data['close'].rolling(window=20).std()
        
        # Calculate volume features
        ohlcv_data['volume_ma'] = ohlcv_data['volume'].rolling(window=20).mean()
        ohlcv_data['volume_std'] = ohlcv_data['volume'].rolling(window=20).std()
        
        # Calculate order book features
        order_book_data['bid_ask_spread'] = order_book_data['ask_prices'].apply(lambda x: x[0]) - \
                                           order_book_data['bid_prices'].apply(lambda x: x[0])
        
        order_book_data['bid_volume_total'] = order_book_data['bid_volumes'].apply(sum)
        order_book_data['ask_volume_total'] = order_book_data['ask_volumes'].apply(sum)
        
        order_book_data['volume_imbalance'] = (order_book_data['bid_volume_total'] - \
                                             order_book_data['ask_volume_total']) / \
                                            (order_book_data['bid_volume_total'] + \
                                             order_book_data['ask_volume_total'])
        
        # Merge features
        features = pd.merge(
            ohlcv_data,
            order_book_data[['timestamp', 'bid_ask_spread', 'volume_imbalance']],
            on='timestamp'
        )
        
        # Add some synthetic features
        features['price_momentum'] = features['returns'].rolling(window=5).sum()
        features['volume_momentum'] = features['volume'].pct_change().rolling(window=5).sum()
        
        # Add some noise to features
        noise_columns = ['returns', 'rolling_mean', 'rolling_std', 'volume_ma', 'volume_std',
                        'bid_ask_spread', 'volume_imbalance', 'price_momentum', 'volume_momentum']
        
        for col in noise_columns:
            if col in features.columns:
                features[col] = features[col] * (1 + np.random.normal(0, 0.01, len(features)))
        
        return features
    
    def generate_target(
        self,
        features: pd.DataFrame,
        lookahead: int = 5,
        threshold: float = 0.001
    ) -> pd.Series:
        """
        Generate synthetic target variable.
        
        Args:
            features: Feature DataFrame
            lookahead: Number of periods to look ahead
            threshold: Threshold for binary classification
            
        Returns:
            Series with target values
        """
        # Calculate future returns
        future_returns = features['close'].shift(-lookahead) / features['close'] - 1
        
        # Generate binary target
        target = (future_returns > threshold).astype(int)
        
        # Add some noise to make it more realistic
        target = target * (1 + np.random.normal(0, 0.1, len(target)))
        target = (target > 0.5).astype(int)
        
        return target
    
    def generate_dataset(
        self,
        interval_minutes: int = 15,
        n_levels: int = 10,
        lookahead: int = 5,
        threshold: float = 0.001
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Generate complete synthetic dataset.
        
        Args:
            interval_minutes: Time interval in minutes
            n_levels: Number of price levels in order book
            lookahead: Number of periods to look ahead for target
            threshold: Threshold for binary classification
            
        Returns:
            Tuple of (OHLCV data, Order book data, Features, Target)
        """
        # Generate base data
        ohlcv_data = self.generate_ohlcv(interval_minutes)
        order_book_data = self.generate_order_book(ohlcv_data, n_levels)
        
        # Generate features and target
        features = self.generate_features(ohlcv_data, order_book_data)
        target = self.generate_target(features, lookahead, threshold)
        
        return ohlcv_data, order_book_data, features, target 