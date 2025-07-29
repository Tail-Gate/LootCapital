import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.stgnn_config import STGNNConfig
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators
import gc
import psutil
import os
from datetime import timedelta
from datetime import datetime
from utils.feature_generator import FeatureGenerator

def manage_memory():
    """Force garbage collection and log memory usage"""
    gc.collect()
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

class STGNNDataProcessor:
    """Memory-efficient data processor for STGNN model"""
    
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
        
        # Memory optimization settings
        self.max_sequences = 1000  # Maximum sequences to create
        self.chunk_size = 1000     # Data points per chunk
        
        self.feature_generator = FeatureGenerator(config=config.to_dict())
        
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
        
    def load_data_in_chunks(self, start_time=None, end_time=None, chunk_size=None):
        """
        Load data in small chunks to prevent memory explosion
        
        Args:
            start_time: Start time for data range
            end_time: End time for data range
            chunk_size: Number of data points per chunk
            
        Returns:
            Dictionary mapping asset symbols to concatenated data
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        print(f"Loading data in chunks of {chunk_size} data points...")
        
        # Load data with optional time window
        if start_time is not None or end_time is not None:
            market_data = self.market_data.get_data(self.config.assets, start_time, end_time)
        else:
            market_data = self.market_data.get_data(self.config.assets)
        
        # Process each asset separately to minimize memory
        processed_data = {}
        for asset in self.config.assets:
            print(f"Processing asset: {asset}")
            asset_data = market_data[asset]
            
            # Process in chunks if data is large
            if len(asset_data) > chunk_size:
                chunks = []
                for i in range(0, len(asset_data), chunk_size):
                    chunk = asset_data.iloc[i:i + chunk_size]
                    chunks.append(chunk)
                    manage_memory()
                
                # Concatenate chunks
                processed_data[asset] = pd.concat(chunks, axis=0)
                del chunks
                manage_memory()
            else:
                processed_data[asset] = asset_data
                
        return processed_data
        
    def prepare_features_memory_efficient(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features with minimal memory overhead
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with prepared features
        """
        print("Preparing features with memory-efficient approach...")
        
        # Store original price data for event-based analysis
        self._original_prices = data['close'].copy()
        
        # Use only essential features to minimize memory
        essential_features = ['returns', 'volume']
        if 'rsi' in self.config.features:
            essential_features.append('rsi')
        if 'macd' in self.config.features:
            essential_features.append('macd')
        if 'bollinger' in self.config.features:
            essential_features.append('bollinger')
            
        features = pd.DataFrame(index=data.index)
        
        # Calculate features one by one to avoid large intermediate objects
        print("Calculating returns...")
        features['returns'] = data['close'].pct_change()
        manage_memory()
        
        print("Calculating volume...")
        features['volume'] = data['volume']
        manage_memory()
        
        # Only calculate essential technical indicators
        if 'rsi' in essential_features:
            print("Calculating RSI...")
            try:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-6)
                features['rsi'] = 100 - (100 / (1 + rs))
                del delta, gain, loss, rs
                manage_memory()
            except Exception as e:
                print(f"Warning: Failed to calculate RSI: {e}")
                features['rsi'] = 50
                
        if 'macd' in essential_features:
            print("Calculating MACD...")
            try:
                exp1 = data['close'].ewm(span=12, adjust=False).mean()
                exp2 = data['close'].ewm(span=26, adjust=False).mean()
                features['macd'] = exp1 - exp2
                del exp1, exp2
                manage_memory()
            except Exception as e:
                print(f"Warning: Failed to calculate MACD: {e}")
                features['macd'] = 0
                
        if 'bollinger' in essential_features:
            print("Calculating Bollinger Bands...")
            try:
                bb_middle = data['close'].rolling(window=20).mean()
                bb_std = data['close'].rolling(window=20).std()
                features['bollinger'] = (data['close'] - bb_middle) / (bb_std + 1e-6)
                del bb_middle, bb_std
                manage_memory()
            except Exception as e:
                print(f"Warning: Failed to calculate Bollinger Bands: {e}")
                features['bollinger'] = 0
        
        # Handle missing/infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill()
        features = features.fillna(0)
        
        # Store returns separately for target calculation
        self._returns = features['returns'] if 'returns' in features.columns else pd.Series(0, index=data.index)
        
        print(f"Features prepared: {list(features.columns)}")
        return features
        
    def create_sequences_lazy(self, features: pd.DataFrame, max_sequences=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences on-demand instead of all at once
        
        Args:
            features: DataFrame with technical features
            max_sequences: Maximum number of sequences to create
            
        Returns:
            Tuple of (X, y) where:
            - X: Input sequences of shape [batch_size, seq_len, num_features]
            - y: Target values of shape [batch_size] (event-based analysis results)
        """
        if max_sequences is None:
            max_sequences = self.max_sequences
            
        print(f"Creating sequences with lazy generation (max: {max_sequences})...")
        
        input_features = [f for f in self.config.features if f in features.columns]
        total_sequences = len(features) - self.config.seq_len - self.config.prediction_horizon + 1
        
        # CRITICAL FIX: Validate total_sequences before proceeding
        if total_sequences <= 0:
            print(f"ERROR: Cannot create sequences - insufficient data!")
            print(f"Available data points: {len(features)}")
            print(f"Required: seq_len={self.config.seq_len} + prediction_horizon={self.config.prediction_horizon} + 1")
            print(f"Calculated total_sequences: {total_sequences}")
            
            # Try to adjust parameters to create at least some sequences
            min_required = self.config.seq_len + self.config.prediction_horizon
            if len(features) < min_required:
                print(f"ERROR: Not enough data points ({len(features)}) for minimum requirements ({min_required})")
                # Return empty arrays to trigger proper error handling upstream
                return np.array([]), np.array([])
            
            # Try with shorter sequence length
            original_seq_len = self.config.seq_len
            self.config.seq_len = max(1, len(features) - self.config.prediction_horizon)
            print(f"Adjusted sequence length to: {self.config.seq_len}")
            total_sequences = len(features) - self.config.seq_len - self.config.prediction_horizon + 1
            
            if total_sequences <= 0:
                print(f"ERROR: Still cannot create sequences after adjustment")
                self.config.seq_len = original_seq_len  # Restore original
                return np.array([]), np.array([])
        
        print(f"Creating {total_sequences} sequences from {len(features)} data points")
        
        if total_sequences > max_sequences:
            # Sample sequences instead of creating all
            print(f"Sampling {max_sequences} sequences from {total_sequences} possible sequences")
            indices = np.random.choice(total_sequences, max_sequences, replace=False)
            indices = np.sort(indices)  # Sort for reproducibility
        else:
            indices = range(total_sequences)
            
        X, y = [], []
        
        for i in indices:
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
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created {len(X)} sequences with shapes X: {X.shape}, y: {y.shape}")
        manage_memory()
        
        return X, y
        
    def prepare_data_single_asset(self, asset: str, start_time=None, end_time=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process one asset at a time to minimize memory
        
        Args:
            asset: Asset symbol to process
            start_time: Start time for data range
            end_time: End time for data range
            
        Returns:
            Tuple of (X, y) for single asset
        """
        print(f"Processing single asset: {asset}")
        
        # Load only one asset
        if start_time is not None or end_time is not None:
            data = self.market_data.get_data([asset], start_time, end_time)
        else:
            data = self.market_data.get_data([asset])
        
        asset_data = data[asset]
        print(f"Loaded {len(asset_data)} data points for {asset}")
        
        # Check if we have enough data - use real data only
        if len(asset_data) < 100:  # Need at least 100 data points
            print(f"Warning: Insufficient real data ({len(asset_data)} points).")
            print(f"Available data range: {asset_data.index.min()} to {asset_data.index.max()}")
            print(f"Requested range: {start_time} to {end_time}")
            
            # Use a larger time window to get more data
            if len(asset_data) == 0:
                # If no data loaded, try loading more historical data
                print("No data loaded. Trying to load more historical data...")
                end_time = datetime.now() - timedelta(days=1)  # Use yesterday as end
                start_time = end_time - timedelta(days=30)     # Use 30 days of data
                data = self.market_data.get_data([asset], start_time, end_time)
                asset_data = data[asset]
                print(f"Loaded {len(asset_data)} data points with extended range")
        
        # Use full feature generator
        features = self.feature_generator.generate_features(asset_data)
        
        # CRITICAL DEBUG: Add comprehensive NaN/Inf checks after feature generation
        if features.isnull().any().any() or (features == np.inf).any().any() or (features == -np.inf).any().any():
            print(f"DEBUG: NaN/Inf detected in 'features' DataFrame after feature generation. Shape: {features.shape}")
            print("--- Head of problematic features ---")
            print(features.head())
            print("--- Tail of problematic features ---")
            print(features.tail())
            print("--- Columns with NaN/Inf values ---")
            nan_inf_cols = []
            for col in features.columns:
                if features[col].isnull().any() or (features[col] == np.inf).any() or (features[col] == -np.inf).any():
                    nan_inf_cols.append(col)
                    print(f"  Column '{col}' has NaN/Inf.")
                    # Print statistics for the problematic column
                    col_data = features[col].replace([np.inf, -np.inf], np.nan)
                    print(f"    Stats for {col}: min={col_data.min()}, max={col_data.max()}, mean={col_data.mean()}, NaNs={col_data.isnull().sum()}")
            # Optionally, save the problematic DataFrame to a CSV for manual inspection
            # features.to_csv("problematic_features_after_generation.csv")
            raise ValueError("NaN/Inf detected in features after generation. Stopping to debug.")
        
        # Store original price data for event-based analysis
        self._original_prices = asset_data['close'].copy()
        
        # FIXED: Add proper feature scaling like in multi-asset path
        # Fit scaler on training data (use first 80% of data)
        if not self.scaler_fitted:
            print("Fitting scaler on single asset data...")
            train_size = int(len(features) * 0.8)
            train_features = features.iloc[:train_size]
            self.fit_scaler(train_features)
            print(f"Scaler fitted on {len(train_features)} samples with {len(train_features.columns)} features")
        
        # Transform features using fitted scaler
        print("Transforming features using fitted scaler...")
        scaled_features = self.transform_features(features)
        
        # Create sequences from scaled features
        X, y = self.create_sequences_lazy(scaled_features)
        
        # CRITICAL FIX: Validate sequences after creation
        if len(X) == 0:
            print("ERROR: No sequences created from real data.")
            print(f"Features shape: {features.shape}")
            print(f"Sequence length: {self.config.seq_len}")
            print(f"Prediction horizon: {self.config.prediction_horizon}")
            print(f"Available data points: {len(features)}")
            
            # Try with shorter sequence length
            original_seq_len = self.config.seq_len
            self.config.seq_len = max(1, min(5, len(features) // 2))
            print(f"Trying with shorter sequence length: {self.config.seq_len}")
            X, y = self.create_sequences_lazy(features)
            self.config.seq_len = original_seq_len  # Restore original
            
            # If still no sequences, return empty arrays to trigger proper error handling
            if len(X) == 0:
                print("ERROR: Still no sequences after adjustment. Returning empty arrays.")
                return np.array([]), np.array([])
        
        # Clean up immediately
        del data, asset_data, features, scaled_features
        manage_memory()
        
        return X, y
        
    def _create_synthetic_data(self, asset: str, num_points: int) -> pd.DataFrame:
        """Create synthetic data for testing when real data is not available"""
        print(f"Creating synthetic data for {asset} with {num_points} points...")
        
        # Create time index
        end_time = datetime.now()
        start_time = end_time - timedelta(days=num_points // 96)  # 96 15-min intervals per day
        time_index = pd.date_range(start=start_time, end=end_time, periods=num_points, freq='15T')
        
        # Create synthetic OHLCV data
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price
        base_price = 2000.0  # ETH-like price
        prices = [base_price]
        
        # Generate price movements
        for i in range(1, num_points):
            # Random walk with some trend
            change = np.random.normal(0, 0.01)  # 1% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, num_points)
        }, index=time_index)
        
        # Ensure high >= open, close and low <= open, close
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        print(f"Synthetic data created with shape: {data.shape}")
        return data
        
    def _create_synthetic_sequences(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic sequences when no real sequences can be created"""
        print("Creating synthetic sequences...")
        
        # Create minimal sequences for testing
        num_sequences = min(100, len(features) - self.config.seq_len - self.config.prediction_horizon + 1)
        
        if num_sequences <= 0:
            # Create completely synthetic sequences
            num_sequences = 50
            X = np.random.randn(num_sequences, self.config.seq_len, len(self.config.features))
            y = np.random.randn(num_sequences)
            print(f"Created {num_sequences} completely synthetic sequences")
        else:
            # Use available features to create sequences
            X, y = [], []
            for i in range(num_sequences):
                if i + self.config.seq_len <= len(features):
                    X.append(features.iloc[i:i + self.config.seq_len].values)
                    y.append(np.random.normal(0, 0.01))  # Small random return
                else:
                    break
            
            X = np.array(X)
            y = np.array(y)
            print(f"Created {len(X)} sequences from available features")
        
        return X, y
        
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
        Create sequences for training using event-based analysis (legacy method)
        
        Args:
            features: DataFrame with technical features
            
        Returns:
            Tuple of (X, y) where:
            - X: Input sequences of shape [batch_size, seq_len, num_features]
            - y: Target values of shape [batch_size] (event-based analysis results)
        """
        # Use lazy sequence generation instead
        return self.create_sequences_lazy(features)
        
    def create_adjacency_matrix(self, market_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Create adjacency matrix based on correlation between assets
        
        Args:
            market_data: Dictionary mapping asset symbols to market data
            
        Returns:
            Adjacency matrix of shape [num_nodes, num_nodes]
        """
        # For single asset, create simple adjacency matrix
        if len(self.config.assets) == 1:
            adj = np.array([[1.0]])
            print("Single asset adjacency matrix: [[1.0]]")
            return adj
            
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
        Memory-efficient data preparation for training and prediction
        
        Args:
            start_time: Optional start time for data range
            end_time: Optional end time for data range
            
        Returns:
            Tuple of (X, adj, y) where:
            - X: Input features of shape [batch_size, num_nodes, seq_len, input_dim]
            - adj: Adjacency matrix of shape [num_nodes, num_nodes]
            - y: Target values of shape [batch_size, num_nodes]
        """
        print("Starting full feature generator data preparation...")
        manage_memory()
        
        # For single asset, use optimized processing
        if len(self.config.assets) == 1:
            print("Using single asset processing with full feature generator...")
            asset = self.config.assets[0]
            X, y = self.prepare_data_single_asset(asset, start_time, end_time)
            
            # Check if we have valid sequences
            if len(X) == 0:
                print("Warning: No sequences created from real data.")
                print("This indicates insufficient real data for the requested time range.")
                raise ValueError("No sequences could be created from real data. Check data availability and time range.")
            
            # Reshape for single asset: [batch_size, 1, seq_len, input_dim]
            if len(X.shape) == 3:  # (batch, seq_len, features)
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
                y = y.reshape(y.shape[0], 1)
            else:
                print(f"Warning: Unexpected X shape: {X.shape}")
                raise ValueError(f"Unexpected X shape: {X.shape}. Expected 3D array.")
                
            # Create simple adjacency matrix for single asset
            adj = np.array([[1.0]])
            
            # Convert to tensors
            X = torch.FloatTensor(X)
            adj = torch.FloatTensor(adj)
            y = torch.FloatTensor(y)
            
            print(f"Single asset data prepared - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
            manage_memory()
            
            return X, adj, y
        
        # For multiple assets, use chunked processing
        print("Using chunked processing for multiple assets with full feature generator...")
        
        # Load data in chunks
        market_data = self.load_data_in_chunks(start_time, end_time)
        
        # Prepare features for each asset
        features_dict = {}
        for asset in self.config.assets:
            print(f"Preparing features for {asset} with full feature generator...")
            features_dict[asset] = self.feature_generator.generate_features(market_data[asset])
            manage_memory()
        
        # Fit scaler on training data (use first 80% of data)
        if not self.scaler_fitted:
            print("Fitting scaler...")
            # Combine features from all assets for fitting
            all_features = pd.concat([features_dict[asset] for asset in self.config.assets], axis=0)
            train_size = int(len(all_features) * 0.8)
            train_features = all_features.iloc[:train_size]
            self.fit_scaler(train_features)
            del all_features, train_features
            manage_memory()
        
        # Transform features for all assets
        scaled_features_dict = {}
        for asset in self.config.assets:
            print(f"Transforming features for {asset}...")
            scaled_features_dict[asset] = self.transform_features(features_dict[asset])
            del features_dict[asset]  # Clean up original features
            manage_memory()
        
        # Create sequences for each asset
        X_dict = {}
        y_dict = {}
        # Collect keys to avoid modifying dict during iteration
        assets_to_process = list(scaled_features_dict.keys())
        for asset in assets_to_process:
            print(f"Creating sequences for {asset}...")
            features = scaled_features_dict[asset]
            X_dict[asset], y_dict[asset] = self.create_sequences_lazy(features)
            del scaled_features_dict[asset]  # Clean up scaled features
            manage_memory()
        
        # Stack features and targets
        print("Stacking features and targets...")
        X = np.stack([X_dict[asset] for asset in self.config.assets], axis=1)  # [batch_size, num_nodes, seq_len, input_dim]
        y = np.stack([y_dict[asset] for asset in self.config.assets], axis=1)  # [batch_size, num_nodes]
        
        # Create adjacency matrix
        print("Creating adjacency matrix...")
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
        
        print(f"Data preparation completed - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
        manage_memory()
        
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
        
        # Dynamically get number of CPU cores with fallback
        num_workers = 0  # Force single-process data loading to avoid CUDA context issues
        
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,  # Use dynamic CPU core count
            pin_memory=False,  # Disable pin_memory for CPU operations
            drop_last=drop_last
        ) 