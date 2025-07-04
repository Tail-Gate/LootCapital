import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.stgnn_config import STGNNConfig
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

class STGNNDataProcessor:
    """Data processor for STGNN model"""
    
    def __init__(self, config: STGNNConfig, market_data: MarketData, technical_indicators: TechnicalIndicators):
        """
        Initialize data processor
        
        Args:
            config: STGNN configuration
            market_data: Market data provider
            technical_indicators: Technical indicators calculator
        """
        self.config = config
        self.market_data = market_data
        self.technical_indicators = technical_indicators
        
        # Initialize scaler (can be MinMaxScaler or StandardScaler)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # Default to MinMaxScaler
        self.scaler_fitted = False
        
    def set_scaler(self, scaler_type: str = 'minmax'):
        """
        Set the scaler type
        
        Args:
            scaler_type: 'minmax' for MinMaxScaler or 'standard' for StandardScaler
        """
        if scaler_type.lower() == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif scaler_type.lower() == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Use 'minmax' or 'standard'")
        
        self.scaler_fitted = False
        print(f"Scaler set to: {scaler_type}")
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for STGNN model
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with prepared features
        """
        # Store original price data for event-based analysis
        self._original_prices = data['close'].copy()
        
        # Calculate basic price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log1p(data['returns'] + 1e-6)
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / (data['volume_ma'] + 1e-6)
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-6)
            data['rsi'] = 100 - (100 / (1 + rs))
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = exp1 - exp2
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            data['bb_std'] = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
            data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
        except Exception as e:
            print(f"Warning: Failed to calculate technical indicators: {e}")
            data['rsi'] = 50
            data['macd'] = 0
            data['macd_signal'] = 0
            data['bb_middle'] = data['close']
            data['bb_upper'] = data['close']
            data['bb_lower'] = data['close']
        # Always return features in the order specified by config, fill missing with zeros
        features = pd.DataFrame()
        for feat in self.config.features:
            col = 'bb_middle' if feat == 'bollinger_bands' else feat
            if col in data.columns:
                features[feat] = data[col]
            else:
                features[feat] = 0
        # Handle missing/infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill()
        features = features.fillna(0)
        # Store returns separately for target calculation
        self._returns = data['returns'] if 'returns' in data.columns else pd.Series(0, index=data.index)
        return features
        
    def fit_scaler(self, features: pd.DataFrame):
        """
        Fit the scaler on training data
        
        Args:
            features: DataFrame with features to fit scaler on
        """
        if not self.scaler_fitted:
            # Remove any remaining NaN values before fitting
            features_clean = features.replace([np.inf, -np.inf], np.nan).fillna(0)
            self.scaler.fit(features_clean.values)
            self.scaler_fitted = True
            print(f"Scaler fitted on {len(features)} samples with {len(features.columns)} features")
            
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler
        
        Args:
            features: DataFrame with features to transform
            
        Returns:
            DataFrame with scaled features
        """
        if not self.scaler_fitted:
            raise ValueError("Scaler must be fitted before transforming features")
            
        # Remove any remaining NaN values before transforming
        features_clean = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Transform features
        scaled_features = self.scaler.transform(features_clean.values)
        
        # Convert back to DataFrame with original column names and index
        scaled_df = pd.DataFrame(
            scaled_features, 
            columns=features.columns, 
            index=features.index
        )
        
        return scaled_df
        
    def create_sequences(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training using event-based analysis
        
        Args:
            features: DataFrame with technical features
            
        Returns:
            Tuple of (X, y) where:
            - X: Input sequences of shape [batch_size, seq_len, num_features]
            - y: Target values of shape [batch_size] (event-based analysis results)
        """
        X, y = [], []
        input_features = [f for f in self.config.features if f in features.columns]
        
        for i in range(len(features) - self.config.seq_len - self.config.prediction_horizon + 1):
            # Input sequence
            X.append(features[input_features].iloc[i:i + self.config.seq_len].values)
            
            # Event-based target calculation
            start_idx = i + self.config.seq_len
            end_idx = start_idx + self.config.prediction_horizon
            
            if start_idx < len(self._original_prices) and end_idx <= len(self._original_prices):
                # Get the starting price
                start_price = self._original_prices.iloc[start_idx]
                
                # Get all prices within the prediction horizon window
                window_prices = self._original_prices.iloc[start_idx:end_idx]
                
                # Calculate all price changes from the starting price
                price_changes = (window_prices - start_price) / start_price
                
                # Find the maximum positive and negative moves within the window
                max_positive_move = price_changes.max()
                max_negative_move = price_changes.min()
                
                # Use the maximum move in either direction as the target
                # This captures any significant move that occurred within the window
                if abs(max_positive_move) > abs(max_negative_move):
                    target_return = max_positive_move
                else:
                    target_return = max_negative_move
                
                y.append(target_return)
            else:
                # Fallback to original method if index is out of bounds
                if i + self.config.seq_len + self.config.prediction_horizon - 1 < len(self._returns):
                    y.append(self._returns.iloc[i + self.config.seq_len + self.config.prediction_horizon - 1])
                else:
                    y.append(0.0)  # Default to no change if out of bounds
        
        return np.array(X), np.array(y)
        
    def create_adjacency_matrix(self, market_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Create adjacency matrix based on correlation between assets
        
        Args:
            market_data: Dictionary mapping asset symbols to market data
            
        Returns:
            Adjacency matrix of shape [num_nodes, num_nodes]
        """
        # Calculate correlation matrix
        returns = pd.DataFrame({
            asset: data['close'].pct_change().fillna(0)
            for asset, data in market_data.items()
        })
        correlation = returns.corr()
        correlation = correlation.fillna(0)  # Fill NaN with 0
        print(f"Correlation matrix:\n{correlation}")
        
        # Convert to adjacency matrix (only keep positive correlations)
        adj = np.maximum(correlation.values, 0)
        print(f"Adjacency matrix after zeroing negatives:\n{adj}")
        
        # Add self-loops (diagonal elements)
        np.fill_diagonal(adj, 1.0)
        print(f"Adjacency matrix after adding self-loops:\n{adj}")
        
        # Normalize adjacency matrix
        row_sum = adj.sum(axis=1)
        print(f"Row sums before normalization: {row_sum}")
        row_sum[row_sum == 0] = 1  # Avoid division by zero
        adj = adj / row_sum[:, np.newaxis]
        print(f"Adjacency matrix after normalization:\n{adj}")
        
        # Ensure all values are non-negative after normalization
        adj[adj < 0] = 0
        print(f"Final adjacency matrix (non-negative):\n{adj}")
        
        return adj
  
        
    def prepare_data(self, start_time=None, end_time=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for training and prediction with feature scaling
        
        Args:
            start_time: Optional start time for data range
            end_time: Optional end time for data range
            
        Returns:
            Tuple of (X, adj, y) where:
            - X: Input features of shape [batch_size, num_nodes, seq_len, input_dim]
            - adj: Adjacency matrix of shape [num_nodes, num_nodes]
            - y: Target values of shape [batch_size, num_nodes]
        """
        # Get market data with optional time window
        if start_time is not None or end_time is not None:
            market_data = self.market_data.get_data(self.config.assets, start_time, end_time)
        else:
            market_data = self.market_data.get_data(self.config.assets)
        
        # Prepare features for each asset
        features_dict = {}
        for asset in self.config.assets:
            features_dict[asset] = self.prepare_features(market_data[asset])
        
        # Fit scaler on training data (use first 80% of data)
        if not self.scaler_fitted:
            # Combine features from all assets for fitting
            all_features = pd.concat([features_dict[asset] for asset in self.config.assets], axis=0)
            train_size = int(len(all_features) * 0.8)
            train_features = all_features.iloc[:train_size]
            self.fit_scaler(train_features)
        
        # Transform features for all assets
        scaled_features_dict = {}
        for asset in self.config.assets:
            scaled_features_dict[asset] = self.transform_features(features_dict[asset])
        
        # Create sequences for each asset
        X_dict = {}
        y_dict = {}
        for asset, features in scaled_features_dict.items():
            X_dict[asset], y_dict[asset] = self.create_sequences(features)
        
        # Stack features and targets
        X = np.stack([X_dict[asset] for asset in self.config.assets], axis=1)  # [batch_size, num_nodes, seq_len, input_dim]
        y = np.stack([y_dict[asset] for asset in self.config.assets], axis=1)  # [batch_size, num_nodes]
        
        # Create adjacency matrix
        adj = self.create_adjacency_matrix(market_data)
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        y = torch.FloatTensor(y)
        
        # Remove inf/nan from X and y
        if torch.isinf(X).any() or torch.isnan(X).any():
            print("Warning: Inf or NaN values found in X, replacing with 0.")
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isinf(y).any() or torch.isnan(y).any():
            print("Warning: Inf or NaN values found in y, replacing with 0.")
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, adj, y
        

    def split_data(self, X: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data into training and validation sets
        
        Args:
            X: Input features
            y: Target values
            train_ratio: Ratio of training data
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        train_size = int(len(X) * train_ratio)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_val, y_val
        
    def create_dataloader(self, X: torch.Tensor, y: torch.Tensor, batch_size: Optional[int] = None, drop_last: bool = False) -> torch.utils.data.DataLoader:
        """
        Create dataloader for training
        
        Args:
            X: Input features
            y: Target values
            batch_size: Batch size (uses config if None)
            drop_last: Whether to drop the last incomplete batch (important for BatchNorm)
        
        Returns:
            DataLoader for training
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last
        ) 