import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # We'll still use this for initialization
import os
import joblib
from scipy.stats import percentileofscore

class FinancialDataset(Dataset):
    """Dataset for financial time series"""
    
    def __init__(self, features: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

class AutoencoderCluster(nn.Module):
    """Autoencoder for dimensionality reduction and clustering"""
    
    def __init__(self, input_dim, hidden_dim=10, latent_dim=5, n_clusters=4):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Clustering layer
        self.n_clusters = n_clusters
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def cluster_assign(self, z):
        """Compute distance to cluster centers"""
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_centers, 2), dim=2) / 1.0)
        q = q.pow(2)  # Student's t-distribution
        return q / torch.sum(q, dim=1, keepdim=True)


class MarketRegimeDetector:
    """
    Detects market regimes using unsupervised deep learning.
    Market regimes might include: trending, mean-reverting, high volatility,
    low volatility, or combinations of these characteristics.
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        lookback_period: int = 60,
        latent_dim: int = 5,
        hidden_dim: int = 20,
        model_path: Optional[str] = None
    ):
        """
        Initialize the market regime detector
        
        Args:
            n_regimes: Number of distinct market regimes to identify
            lookback_period: Number of periods to use for feature calculation
            latent_dim: Dimension of latent space in autoencoder
            hidden_dim: Dimension of hidden layer in autoencoder
            model_path: Path to saved model file (if loading existing model)
        """
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Initialize models
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.model = None  # Will be initialized after feature extraction
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Regime characteristics (to be learned)
        self.regime_profiles = {}
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime detection
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with extracted features
        """
        df = data.copy()
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Feature set 1: Return characteristics
        features = pd.DataFrame(index=df.index)
        
        # Rolling window statistics
        for window in [5, 10, 20]:
            # Return characteristics
            features[f'returns_mean_{window}'] = df['returns'].rolling(window=window).mean()
            features[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
            features[f'returns_skew_{window}'] = df['returns'].rolling(window=window).skew()
            features[f'returns_kurt_{window}'] = df['returns'].rolling(window=window).kurt()
            
            # Absolute return characteristics (volatility)
            features[f'abs_returns_mean_{window}'] = df['returns'].abs().rolling(window=window).mean()
            
            # Trend strength
            features[f'trend_strength_{window}'] = df['close'].rolling(window=window).apply(
                lambda x: np.abs(x.iloc[-1] / x.iloc[0] - 1) / x.std()
                if len(x.dropna()) > 1 else 0
            )
        
        # Feature set 2: Volatility characteristics
        if 'volatility' in df.columns:
            features['volatility'] = df['volatility']
        else:
            features['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volatility of volatility
        features['vol_of_vol'] = features['volatility'].rolling(window=20).std()
        
        # Volatility regime
        features['vol_ratio'] = features['volatility'] / features['volatility'].rolling(window=50).mean()
        
        # Feature set 3: Market efficiency/predictability
        for window in [10, 20]:
            # Auto-correlation (predictability)
            features[f'autocorr_{window}'] = df['returns'].rolling(window=window).apply(
                lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else 0
            )
        
        # Feature set 4: Volume characteristics
        if 'volume' in df.columns:
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            features['volume_trend'] = df['volume'].rolling(window=10).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean() 
                if len(x.dropna()) > 1 else 0
            )
            
            # Price-volume correlation
            features['price_volume_corr'] = df.rolling(window=20).apply(
                lambda x: x['returns'].corr(x['volume']) if len(x.dropna()) > 1 else 0
            )
        
        # Drop NaN values from initialization of rolling windows
        features = features.dropna()
        
        return features
    
    def fit(self, data: pd.DataFrame, epochs=100, batch_size=64, learning_rate=0.001) -> None:
        """
        Fit the regime detection model
        
        Args:
            data: DataFrame with OHLCV data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        # Extract features
        features_df = self.extract_features(data)
        feature_cols = features_df.columns
        
        # Standardize features
        features = self.scaler.fit_transform(features_df)
        
        # Initialize model if not exists
        if self.model is None:
            input_dim = features.shape[1]
            self.model = AutoencoderCluster(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                n_clusters=self.n_regimes
            ).to(self.device)
        
        # Create data loader
        dataset = FinancialDataset(features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize cluster centers using K-means
        with torch.no_grad():
            # Get encoded representations
            encoded = []
            for batch in DataLoader(dataset, batch_size=batch_size):
                batch = batch.to(self.device)
                _, z = self.model(batch)
                encoded.append(z.cpu())
            
            encoded = torch.cat(encoded).numpy()
            
            # Initialize with KMeans
            self.kmeans.fit(encoded)
            self.model.cluster_centers.data = torch.tensor(
                self.kmeans.cluster_centers_, 
                dtype=torch.float32
            ).to(self.device)
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion_ae = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Forward pass
                x_recon, z = self.model(batch)
                q = self.model.cluster_assign(z)
                
                # Compute losses
                recon_loss = criterion_ae(x_recon, batch)
                # The lower the cluster assignment entropy, the better
                cluster_loss = torch.mean(-torch.sum(q * torch.log(q + 1e-10), dim=1))
                
                # Total loss
                loss = recon_loss + 0.1 * cluster_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        # Create regime profiles
        self._create_regime_profiles(features_df)
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict market regimes for given data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with regime labels
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
            
        # Extract features
        features_df = self.extract_features(data)
        
        # Standardize features
        features = self.scaler.transform(features_df)
        
        # Create dataset and loader
        dataset = FinancialDataset(features)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        # Get cluster assignments
        self.model.eval()
        cluster_assignments = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                _, z = self.model(batch)
                q = self.model.cluster_assign(z)
                assignments = torch.argmax(q, dim=1).cpu().numpy()
                cluster_assignments.extend(assignments)
        
        # Convert to Series
        regime_series = pd.Series(cluster_assignments, index=features_df.index)
        
        return regime_series
    
    def _create_regime_profiles(self, features: pd.DataFrame) -> None:
        """
        Create profiles for each detected regime
        
        Args:
            features: DataFrame with extracted features
        """
        # Predict regimes on training data
        regimes = self.predict(features)
        
        # Add regime labels to features
        features_with_regimes = features.copy()
        features_with_regimes['regime'] = regimes
        
        # Calculate average feature values for each regime
        for regime in range(self.n_regimes):
            regime_data = features_with_regimes[features_with_regimes['regime'] == regime]
            
            if len(regime_data) > 0:
                # Calculate basic statistics for each feature
                profile = {}
                for col in features.columns:
                    profile[f"{col}_mean"] = regime_data[col].mean()
                    profile[f"{col}_std"] = regime_data[col].std()
                
                # Determine regime characteristics
                profile['volatility_level'] = self._classify_value(
                    regime_data['volatility'].mean(), 
                    features['volatility']
                )
                
                profile['trend_strength'] = self._classify_value(
                    regime_data['trend_strength_20'].mean(), 
                    features['trend_strength_20']
                )
                
                profile['mean_reversion'] = self._classify_value(
                    -regime_data['autocorr_20'].mean(),  # Negative autocorrelation = mean reversion
                    -features['autocorr_20']
                )
                
                # Store profile
                self.regime_profiles[regime] = profile
    
    def _classify_value(self, value: float, distribution: pd.Series) -> str:
        """
        Classify a value relative to its distribution
        
        Args:
            value: Value to classify
            distribution: Reference distribution
            
        Returns:
            Classification as "high", "medium", or "low"
        """
        percentile = percentileofscore(distribution, value)
        
        if percentile > 75:
            return "high"
        elif percentile < 25:
            return "low"
        else:
            return "medium"
    
    def get_regime_name(self, regime_id: int) -> str:
        """
        Get descriptive name for a regime based on its characteristics
        
        Args:
            regime_id: Regime identifier
            
        Returns:
            Human-readable name for the regime
        """
        if regime_id not in self.regime_profiles:
            return f"Regime {regime_id}"
            
        profile = self.regime_profiles[regime_id]
        
        # Create name based on key characteristics
        vol_desc = profile['volatility_level']
        trend_desc = profile['trend_strength']
        mr_desc = profile['mean_reversion']
        
        if trend_desc == "high":
            if vol_desc == "high":
                return "High Volatility Trending"
            else:
                return "Low Volatility Trending"
        elif mr_desc == "high":
            if vol_desc == "high":
                return "High Volatility Mean Reversion"
            else:
                return "Low Volatility Mean Reversion"
        elif vol_desc == "high":
            return "High Volatility Choppy"
        elif vol_desc == "low":
            return "Low Volatility Consolidation"
        else:
            return "Neutral Market"
    
    def save_model(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: File path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save PyTorch model
        torch.save(self.model.state_dict(), f"{path}_model.pt")
        
        # Save other components
        model_data = {
            'scaler': self.scaler,
            'regime_profiles': self.regime_profiles,
            'n_regimes': self.n_regimes,
            'lookback_period': self.lookback_period,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'input_dim': next(self.model.encoder.parameters()).shape[1]
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """
        Load the model from disk
        
        Args:
            path: File path to load the model from
        """
        # Load auxiliary data
        model_data = joblib.load(path)
        
        self.scaler = model_data['scaler']
        self.regime_profiles = model_data['regime_profiles']
        self.n_regimes = model_data['n_regimes']
        self.lookback_period = model_data['lookback_period']
        self.latent_dim = model_data['latent_dim']
        self.hidden_dim = model_data['hidden_dim']
        
        # Initialize model architecture
        input_dim = model_data['input_dim']
        self.model = AutoencoderCluster(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_clusters=self.n_regimes
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(f"{path}_model.pt", map_location=self.device))
        self.model.eval()