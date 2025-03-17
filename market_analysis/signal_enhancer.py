import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import joblib
import os

class LSTMSignalEnhancer(nn.Module):
    """
    LSTM model for enhancing trading signals specific to natural gas futures
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMSignalEnhancer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for focusing on important time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM output
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size)
        
        # Final prediction
        out = self.fc(context_vector)  # (batch_size, 1)
        
        return out, attention_weights


class NaturalGasSignalEnhancer:
    """
    Enhances trading signals for natural gas futures using deep learning
    """
    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        model_path: Optional[str] = None
    ):
        """
        Initialize the signal enhancer
        
        Args:
            sequence_length: Number of time steps to use for prediction
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            model_path: Path to load pre-trained model
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Preprocessing
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model will be initialized during training
        self.model = None
        self.feature_columns = None
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features specific to natural gas futures
        
        Args:
            data: DataFrame with OHLCV and other features
            
        Returns:
            DataFrame with extracted features
        """
        df = data.copy()
        
        # Ensure we have basic price and returns data
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Feature Group 1: Price action features
        # Relative price to moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}_ratio'] = df['close'] / df['close'].rolling(window=window).mean()
        
        # Feature Group 2: Volatility features (crucial for natural gas)
        if 'volatility' not in df.columns:
            df['volatility'] = df['returns'].rolling(window=20).std()
            
        # Volatility ratios (for capturing regime shifts)
        df['vol_ratio_5_20'] = df['returns'].rolling(window=5).std() / df['returns'].rolling(window=20).std()
        
        # Feature Group 3: Seasonality features (important for natural gas)
        # Extract month and day of month
        if hasattr(df.index, 'month'):
            df['month'] = df.index.month
            df['day_of_month'] = df.index.day
            
            # Winter vs summer season (heating vs cooling)
            df['winter_season'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)
            
            # Beginning, middle, or end of month (contract rolling effects)
            df['start_of_month'] = (df['day_of_month'] <= 10).astype(int)
            df['end_of_month'] = (df['day_of_month'] >= 21).astype(int)
        
        # Feature Group 4: EIA report day effects
        if hasattr(df.index, 'dayofweek'):
            # EIA Natural Gas Storage Report (typically Thursday)
            df['report_day'] = (df.index.dayofweek == 3).astype(int)
            df['day_after_report'] = (df.index.dayofweek == 4).astype(int)
        
        # Feature Group 5: Technical indicators specific to natural gas
        # RSI (captures overbought/oversold conditions)
        from utils.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        df['rsi'] = ti.calculate_rsi(df['close'], 14)
        
        # Calculate momentum indicators
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Feature Group 6: Volume analysis
        if 'volume' in df.columns:
            # Volume relative to moving average
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # Volume-price relationship (important for spotting reversals)
            df['volume_price_trend'] = df['volume_ratio'] * np.sign(df['returns'])
            
            # OBV-like measure
            df['obv_change'] = df['volume'] * np.sign(df['returns'])
            df['obv'] = df['obv_change'].cumsum()
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def _create_sequences(self, features: pd.DataFrame, target: pd.Series, future_periods: int = 1):
        """
        Create sequences for LSTM training/prediction
        
        Args:
            features: DataFrame with features
            target: Series with target values
            future_periods: How many periods ahead to predict
            
        Returns:
            X tensor, y tensor
        """
        X, y = [], []
        
        features_array = features.values
        target_array = target.shift(-future_periods).values[:-future_periods]
        
        for i in range(len(features_array) - self.sequence_length - future_periods + 1):
            X.append(features_array[i:i + self.sequence_length])
            y.append(target_array[i + self.sequence_length - 1])
        
        # Convert to tensors
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        
        return X_tensor, y_tensor
    
    def train(
        self, 
        data: pd.DataFrame, 
        target_column: str = 'returns', 
        epochs: int = 100, 
        batch_size: int = 32, 
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ):
        """
        Train the signal enhancement model
        
        Args:
            data: DataFrame with features and target
            target_column: Column name to use as prediction target
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Proportion of data to use for validation
        """
        # Prepare features
        features = self.prepare_features(data)
        self.feature_columns = features.columns.tolist()
        
        # Scale features and target
        features_scaled = self.feature_scaler.fit_transform(features)
        target = data[target_column].shift(-1)  # Predict next period's return
        target = target.iloc[features.index]
        target_scaled = self.target_scaler.fit_transform(target.values.reshape(-1, 1))
        
        # Create sequences
        X, y = self._create_sequences(
            pd.DataFrame(features_scaled, index=features.index, columns=features.columns),
            pd.Series(target_scaled.flatten(), index=features.index)
        )
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = X.shape[2]
        self.model = LSTMSignalEnhancer(
            input_size=input_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs, _ = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Val Loss: {val_loss/len(val_loader):.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pt'))
        self.model.eval()
    
    def predict(self, data: pd.DataFrame, return_attention: bool = False):
        """
        Make predictions using the trained model
        
        Args:
            data: DataFrame with raw input data
            return_attention: Whether to return attention weights
            
        Returns:
            Series with predictions and optionally attention weights
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Check that we have the expected columns
        missing_cols = set(self.feature_columns) - set(features.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # Use only the columns that were used during training
        features = features[self.feature_columns]
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        
        # Create sequences (without target)
        X = []
        for i in range(len(features_scaled) - self.sequence_length + 1):
            X.append(features_scaled[i:i + self.sequence_length])
        
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions, attention_weights = self.model(X_tensor)
        
        # Convert to numpy and rescale
        predictions_np = predictions.cpu().numpy()
        predictions_rescaled = self.target_scaler.inverse_transform(predictions_np)
        
        # Create result Series
        prediction_idx = features.index[self.sequence_length - 1:]
        predictions_series = pd.Series(predictions_rescaled.flatten(), index=prediction_idx)
        
        if return_attention:
            attention_np = attention_weights.cpu().numpy()
            return predictions_series, attention_np
        else:
            return predictions_series
    
    def enhance_signals(self, data: pd.DataFrame, signals: pd.Series):
        """
        Enhance trading signals using model predictions
        
        Args:
            data: DataFrame with raw input data
            signals: Original strategy signals (-1 to 1)
            
        Returns:
            Series with enhanced signals
        """
        # Get predictions (expected returns)
        predictions = self.predict(data)
        
        # Align predictions with signals
        predictions = predictions.reindex(signals.index)
        
        # Calculate signal enhancement
        # Higher predicted returns should amplify long signals and dampen short signals
        # Lower predicted returns should amplify short signals and dampen long signals
        pred_zscore = (predictions - predictions.mean()) / predictions.std()
        
        # Enhance signals (weighted combination)
        enhanced_signals = 0.7 * signals + 0.3 * pred_zscore
        
        # Ensure signals remain in [-1, 1] range
        enhanced_signals = enhanced_signals.clip(-1, 1)
        
        return enhanced_signals
    
    def save_model(self, path: str):
        """
        Save the model to disk
        
        Args:
            path: File path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save PyTorch model weights
        torch.save(self.model.state_dict(), f"{path}_weights.pt")
        
        # Save other components
        model_data = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """
        Load the model from disk
        
        Args:
            path: File path to load the model from
        """
        # Load components
        model_data = joblib.load(path)
        
        self.feature_scaler = model_data['feature_scaler']
        self.target_scaler = model_data['target_scaler']
        self.feature_columns = model_data['feature_columns']
        self.sequence_length = model_data['sequence_length']
        self.hidden_size = model_data['hidden_size']
        self.num_layers = model_data['num_layers']
        
        # Initialize model architecture
        input_size = len(self.feature_columns)
        self.model = LSTMSignalEnhancer(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(torch.load(f"{path}_weights.pt", map_location=self.device))
        self.model.eval()