from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import warnings
import math

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType

@dataclass
class AdvancedTimeSeriesConfig(TechnicalConfig):
    """Configuration for advanced time series models (Bi-LSTM, TCN, Transformer)"""
    # Model selection
    model_type: str = 'bi_lstm'  # Options: 'bi_lstm', 'tcn', 'transformer'
    
    # Common parameters
    sequence_length: int = 60  # Number of time steps to look back
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout: float = 0.2
    hidden_size: int = 64
    num_layers: int = 2
    
    # Bi-LSTM specific
    lstm_bidirectional: bool = True
    lstm_attention: bool = True
    
    # TCN specific
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2
    tcn_num_channels: List[int] = None  # Will be set in __post_init__
    
    # Transformer specific
    transformer_nhead: int = 4
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1
    
    # Training parameters
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Feature engineering
    use_log_transform: bool = True
    include_exogenous: bool = True
    
    # Trading parameters
    min_confidence: float = 0.7  # Minimum model confidence for trading
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 4.0
    
    def __post_init__(self):
        super().validate()
        if self.tcn_num_channels is None:
            self.tcn_num_channels = [self.hidden_size] * self.num_layers

class TimeSeriesDataset(Dataset):
    """Dataset for time series models"""
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.targets[idx]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class BiLSTMModel(pl.LightningModule):
    """Bidirectional LSTM with optional attention, Lightning version"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, bidirectional: bool = True, use_attention: bool = True, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.learning_rate = learning_rate
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention layer
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2 if bidirectional else hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
        
        # Output layer
        self.fc = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size,
            1
        )
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        lstm_out, _ = self.lstm(x)
        if self.use_attention:
            attention_weights = self.attention(lstm_out)
            attention_weights = F.softmax(attention_weights, dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)
        else:
            context = lstm_out[:, -1, :]
            attention_weights = None
        out = self.fc(context)
        return out, attention_weights
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.forward(x)
        loss = self.loss_fn(y_hat.squeeze(), y.squeeze())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.forward(x)
        loss = self.loss_fn(y_hat.squeeze(), y.squeeze())
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class TCNBlock(nn.Module):
    """Temporal Convolutional Network block"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class TCNModel(pl.LightningModule):
    """Temporal Convolutional Network, Lightning version"""
    def __init__(self, input_size: int, num_channels: List[int], kernel_size: int, 
                 dropout: float = 0.2, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.network(x)
        x = x[:, :, -1]  # Take last time step
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat.squeeze(), y.squeeze())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat.squeeze(), y.squeeze())
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class TransformerModel(pl.LightningModule):
    """Transformer model for time series, Lightning version"""
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_layers: int, dim_feedforward: int, dropout: float = 0.1, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, 1)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Take last time step
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat.squeeze(), y.squeeze())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat.squeeze(), y.squeeze())
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AdvancedTimeSeriesStrategy(TechnicalStrategy):
    """
    Advanced time series strategy using Bi-LSTM, TCN, or Transformer models.
    """
    
    def __init__(self, config: AdvancedTimeSeriesConfig = None):
        super().__init__(name="advanced_time_series", config=config or AdvancedTimeSeriesConfig())
        self.config: AdvancedTimeSeriesConfig = self.config
        self.model = None
        self.is_fitted = False
        self.trainer = None
        self.last_train_index = None
        self.feature_dim = None
        
    def _create_model(self, input_size: int) -> nn.Module:
        """Create the selected model type with the correct input size"""
        if self.config.model_type == 'bi_lstm':
            return BiLSTMModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                bidirectional=self.config.lstm_bidirectional,
                use_attention=self.config.lstm_attention,
                learning_rate=self.config.learning_rate
            )
        elif self.config.model_type == 'tcn':
            return TCNModel(
                input_size=input_size,
                num_channels=self.config.tcn_num_channels,
                kernel_size=self.config.tcn_kernel_size,
                dropout=self.config.dropout,
                learning_rate=self.config.learning_rate
            )
        elif self.config.model_type == 'transformer':
            return TransformerModel(
                input_size=input_size,
                d_model=self.config.hidden_size,
                nhead=self.config.transformer_nhead,
                num_layers=self.config.num_layers,
                dim_feedforward=self.config.transformer_dim_feedforward,
                dropout=self.config.transformer_dropout,
                learning_rate=self.config.learning_rate
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for advanced time series models"""
        df = self.prepare_base_features(data)
        
        # Add time series specific features
        df = self._add_time_features(df)
        df = self._add_technical_features(df)
        df = self._add_market_features(df)
        
        # Generate predictions if model is fitted
        if self.model is None:
            self.feature_dim = df.shape[1]
            self.model = self._create_model(self.feature_dim)
        if self.is_fitted:
            df = self._add_model_predictions(df)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if not hasattr(df.index, 'dayofweek'):
            return df
        
        # Add time features
        df['day_of_week'] = df.index.dayofweek
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        
        # Add cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # Price momentum
        for window in [5, 10, 20]:
            df[f'roc_{window}'] = df['close'].pct_change(periods=window)
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'std_{window}'] = df['close'].rolling(window=window).std()
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility features
        df['atr'] = self._calculate_atr(df)
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        
        return df
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Price impact
        df['price_impact'] = (df['high'] - df['low']) / df['volume']
        
        # Order flow imbalance (if available)
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        
        # Market depth (if available)
        if 'bid_depth' in df.columns and 'ask_depth' in df.columns:
            df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'])
        
        return df
    
    def _add_model_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add model predictions to features"""
        # Prepare sequences
        sequences = self._prepare_sequences(df)
        
        # Generate predictions
        with torch.no_grad():
            predictions = self.model(sequences)
        
        # Add predictions to dataframe
        df['model_prediction'] = predictions.numpy()
        df['prediction_confidence'] = self._calculate_confidence(predictions)
        
        return df
    
    def _prepare_sequences(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare input sequences for the model"""
        # Select features
        feature_cols = [col for col in df.columns if col not in ['model_prediction', 'prediction_confidence']]
        features = df[feature_cols].values
        
        # Create sequences
        sequences = []
        for i in range(len(features) - self.config.sequence_length + 1):
            sequences.append(features[i:i + self.config.sequence_length])
        
        return torch.FloatTensor(sequences)
    
    def _calculate_confidence(self, predictions: torch.Tensor) -> np.ndarray:
        """Calculate prediction confidence"""
        # For now, use a simple approach based on prediction magnitude
        return np.abs(predictions.numpy())
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate trading signals based on model predictions"""
        if not self.is_fitted:
            return 0.0, 0.0, TradeType.NONE
        
        current = features.iloc[-1]
        
        # Get model prediction and confidence
        prediction = current['model_prediction']
        confidence = current['prediction_confidence']
        
        # Only trade if confidence is high enough
        if confidence < self.config.min_confidence:
            return 0.0, 0.0, TradeType.NONE
        
        # Determine trade type based on prediction
        if prediction > 0:
            return confidence, prediction, TradeType.LONG
        else:
            return confidence, abs(prediction), TradeType.SHORT
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        signal: float, 
        features: pd.DataFrame
    ) -> float:
        """Calculate stop loss level"""
        current = features.iloc[-1]
        atr = current['atr']
        
        if signal > 0:  # Long position
            return entry_price - (atr * self.config.stop_loss_atr)
        else:  # Short position
            return entry_price + (atr * self.config.stop_loss_atr)
    
    def calculate_take_profit(
        self, 
        entry_price: float, 
        stop_loss: float, 
        signal: float, 
        features: pd.DataFrame
    ) -> float:
        """Calculate take profit level"""
        current = features.iloc[-1]
        atr = current['atr']
        
        if signal > 0:  # Long position
            return entry_price + (atr * self.config.take_profit_atr)
        else:  # Short position
            return entry_price - (atr * self.config.take_profit_atr)
    
    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update the model with new data"""
        if self.model is None:
            self.feature_dim = features.shape[1]
            self.model = self._create_model(self.feature_dim)
        # Prepare data
        X = self._prepare_sequences(features)
        y = torch.FloatTensor(target.values[self.config.sequence_length-1:])
        
        # Create datasets
        train_size = int(len(X) * (1 - self.config.validation_split))
        train_dataset = TimeSeriesDataset(X[:train_size], y[:train_size], self.config.sequence_length)
        val_dataset = TimeSeriesDataset(X[train_size:], y[train_size:], self.config.sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        # Setup trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=self.config.early_stopping_patience),
                ModelCheckpoint(monitor='val_loss')
            ]
        )
        
        # Train model
        self.trainer.fit(self.model, train_loader, val_loader)
        
        # Update state
        self.is_fitted = True
        self.last_train_index = features.index[-1]

    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """
        Validate trading signal based on model confidence and market conditions
        
        Args:
            signal: The calculated signal strength
            features: Technical features DataFrame
            
        Returns:
            Boolean indicating if signal passes validation
        """
        # Get base technical validation
        technical_valid = super().validate_technical_signal(signal, features)
        
        # Additional model-specific validation
        if not self.is_fitted:
            return False
            
        # Check model confidence
        current = features.iloc[-1]
        if 'prediction_confidence' not in current:
            return False
            
        confidence_valid = current['prediction_confidence'] >= self.config.min_confidence
        
        return technical_valid and confidence_valid 

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config.atr_period).mean()
        
        return atr 