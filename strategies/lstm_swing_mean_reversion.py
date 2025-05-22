import numpy as np
import pandas as pd
import torch
from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType
from utils import lstm_utils

class LSTMSwingMeanReversionConfig(TechnicalConfig):
    """Configuration for LSTM-based swing mean reversion strategy"""
    model_path: str = None
    sequence_length: int = 14  # Default to 14 days
    feature_list: list = None
    probability_threshold: float = 0.6
    def __post_init__(self):
        super().validate()
        if self.feature_list is None:
            self.feature_list = [
                'price_reversion_speed', 'order_flow_seq', 'rsi_traj', 'volatility_pattern',
                # Add more sequential features as needed
            ]

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all required sequential features for the LSTM model.
        Features are stacked for each rolling window (sequence_length).
        Includes placeholders for on-chain metrics.
        """
        df = self.prepare_base_features(data)
        seq_len = self.config.sequence_length
        features = {}

        # Price reversion speed (z-score of close vs. rolling mean)
        rolling_mean = df['close'].rolling(window=seq_len).mean()
        rolling_std = df['close'].rolling(window=seq_len).std()
        features['price_reversion_speed'] = ((df['close'] - rolling_mean) / rolling_std).fillna(0)

        # Order flow imbalance sequence (z-score)
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            ofi = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-9)
            ofi = (ofi - ofi.rolling(window=seq_len).mean()) / (ofi.rolling(window=seq_len).std() + 1e-9)
        else:
            ofi = pd.Series(0, index=df.index)
        for i in range(seq_len):
            features[f'order_flow_imbalance_seq_{i}'] = ofi.shift(seq_len - 1 - i)

        # Bid/Ask volume sequences (z-score)
        if 'bid_volume' in df.columns:
            bid_vol = (df['bid_volume'] - df['bid_volume'].rolling(window=seq_len).mean()) / (df['bid_volume'].rolling(window=seq_len).std() + 1e-9)
        else:
            bid_vol = pd.Series(0, index=df.index)
        if 'ask_volume' in df.columns:
            ask_vol = (df['ask_volume'] - df['ask_volume'].rolling(window=seq_len).mean()) / (df['ask_volume'].rolling(window=seq_len).std() + 1e-9)
        else:
            ask_vol = pd.Series(0, index=df.index)
        for i in range(seq_len):
            features[f'bid_volume_seq_{i}'] = bid_vol.shift(seq_len - 1 - i)
            features[f'ask_volume_seq_{i}'] = ask_vol.shift(seq_len - 1 - i)

        # RSI sequence and momentum (min-max scale RSI to [0, 1])
        rsi = self.ti.calculate_rsi(df['close'], period=14)
        rsi_scaled = (rsi - 0) / (100 - 0)
        for i in range(seq_len):
            features[f'rsi_seq_{i}'] = rsi_scaled.shift(seq_len - 1 - i)
        rsi_momentum = rsi_scaled.diff()
        for i in range(seq_len):
            features[f'rsi_momentum_seq_{i}'] = rsi_momentum.shift(seq_len - 1 - i)

        # Volatility sequence (z-score of rolling std of returns)
        returns = df['close'].pct_change()
        vol = returns.rolling(window=seq_len).std()
        vol_z = (vol - vol.rolling(window=seq_len).mean()) / (vol.rolling(window=seq_len).std() + 1e-9)
        for i in range(seq_len):
            features[f'volatility_seq_{i}'] = vol_z.shift(seq_len - 1 - i)

        # Volume profile sequence (z-score)
        vol_profile = df['volume'].rolling(window=seq_len).sum()
        vol_profile_z = (vol_profile - vol_profile.rolling(window=seq_len).mean()) / (vol_profile.rolling(window=seq_len).std() + 1e-9)
        for i in range(seq_len):
            features[f'volume_profile_seq_{i}'] = vol_profile_z.shift(seq_len - 1 - i)

        # VWAP deviation sequence (z-score)
        vwap = self.ti.calculate_vwap(df)
        vwap_dev = (df['close'] - vwap) / vwap
        vwap_dev_z = (vwap_dev - vwap_dev.rolling(window=seq_len).mean()) / (vwap_dev.rolling(window=seq_len).std() + 1e-9)
        for i in range(seq_len):
            features[f'vwap_deviation_seq_{i}'] = vwap_dev_z.shift(seq_len - 1 - i)

        # ADX sequence (z-score)
        adx = self.ti.calculate_directional_movement(df['high'], df['low'], df['close'], period=14)
        adx_z = (adx - adx.rolling(window=seq_len).mean()) / (adx.rolling(window=seq_len).std() + 1e-9)
        for i in range(seq_len):
            features[f'adx_seq_{i}'] = adx_z.shift(seq_len - 1 - i)

        # Technical divergence sequence (price vs. volume delta, z-score)
        price_dir = np.sign(df['close'].diff())
        volume_delta = np.sign(df['volume'].diff())
        tech_div = (price_dir != volume_delta).astype(int)
        tech_div_z = (tech_div - tech_div.rolling(window=seq_len).mean()) / (tech_div.rolling(window=seq_len).std() + 1e-9)
        for i in range(seq_len):
            features[f'technical_divergence_seq_{i}'] = tech_div_z.shift(seq_len - 1 - i)

        # Funding rate sequence (z-score, if available)
        if 'funding_rate' in df.columns:
            funding_rate_z = (df['funding_rate'] - df['funding_rate'].rolling(window=seq_len).mean()) / (df['funding_rate'].rolling(window=seq_len).std() + 1e-9)
            for i in range(seq_len):
                features[f'funding_rate_seq_{i}'] = funding_rate_z.shift(seq_len - 1 - i)

        # On-chain metrics (placeholders, z-score)
        # When available, replace zeros with actual on-chain data
        for metric in ['active_addresses', 'tx_volume', 'large_transfers', 'miner_flows']:
            if metric in df.columns:
                metric_z = (df[metric] - df[metric].rolling(window=seq_len).mean()) / (df[metric].rolling(window=seq_len).std() + 1e-9)
            else:
                metric_z = pd.Series(0, index=df.index)
            for i in range(seq_len):
                features[f'{metric}_seq_{i}'] = metric_z.shift(seq_len - 1 - i)

        # Assemble DataFrame
        feat_df = pd.DataFrame(features, index=df.index)
        feat_df = feat_df.fillna(method='ffill').dropna()
        return feat_df

class LSTMSwingMeanReversionStrategy(TechnicalStrategy):
    """
    Swing trading mean reversion strategy using Bidirectional LSTM with attention.
    Features are derived from standard exchange data (OHLCV, order book, volume).
    """
    def __init__(self, config: LSTMSwingMeanReversionConfig = None):
        super().__init__(name="lstm_swing_mean_reversion", config=config or LSTMSwingMeanReversionConfig())
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config.model_path:
            self.load_model(self.config.model_path)

    def train(self, dataloader, criterion, optimizer, num_epochs=10):
        """
        Train the LSTM model using the provided dataloader, criterion, and optimizer.
        Saves the model if model_path is specified in config.
        """
        input_dim = next(iter(dataloader))[0].shape[-1]
        output_dim = 1  # Assuming regression or binary classification
        hidden_dim = 64
        num_layers = 2
        self.model = lstm_utils.LSTMAttentionModel(input_dim, hidden_dim, num_layers, output_dim).to(self.device)
        self.model = lstm_utils.train_lstm(self.model, dataloader, criterion, optimizer, self.device, num_epochs)
        if self.config.model_path:
            self.save_model(self.config.model_path)

    def predict(self, X):
        """
        Predict using the trained LSTM model. Returns predictions and attention weights.
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds, attn = lstm_utils.predict_lstm(self.model, X_tensor, self.device)
        return preds, attn

    def save_model(self, path):
        """
        Save the trained LSTM model to the specified path.
        """
        if self.model is not None:
            lstm_utils.save_lstm(self.model, path)

    def load_model(self, path):
        """
        Load the LSTM model from the specified path.
        """
        input_dim = self.config.sequence_length * len(self.config.feature_list or [])
        output_dim = 1
        hidden_dim = 64
        num_layers = 2
        self.model = lstm_utils.load_lstm(
            lstm_utils.LSTMAttentionModel,
            path,
            input_dim,
            hidden_dim,
            num_layers,
            output_dim
        ).to(self.device)

    def explain(self, X):
        """
        Return attention weights for the given input X.
        X should be a numpy array or DataFrame of shape (batch, seq_len, features) or (seq_len, features).
        Returns attention weights per timestep and a summary (mean attention per timestep).
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            _, attn_weights = self.model(X_tensor.to(self.device))
        attn_weights = attn_weights.cpu().numpy().squeeze()  # (batch, seq_len, 1) or (seq_len,)
        # If batch size is 1, reduce to (seq_len,)
        if attn_weights.ndim == 2:
            attn_weights = attn_weights[0]
        # Return both raw attention and a summary (mean per timestep)
        attn_summary = attn_weights.mean(axis=-1) if attn_weights.ndim > 1 else attn_weights
        return {"attention_weights": attn_weights, "attention_summary": attn_summary}

    def calculate_signals(self, features: pd.DataFrame) -> tuple:
        """
        Generate trading signals using the LSTM model and apply probability thresholding and confirmation logic.
        Entry/exit is based on model prediction, confidence, and secondary indicator confirmation (e.g., ADX, RSI).
        Returns (signal, confidence, trade_type).
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        # Prepare input for LSTM: (batch, seq_len, features)
        X = features.values[-self.config.sequence_length:]
        X = X.reshape(1, self.config.sequence_length, -1)
        preds, attn = self.predict(X)
        pred = preds[0][0]  # Assuming output shape (batch, 1)
        confidence = float(abs(pred))
        threshold = self.config.probability_threshold
        # Model-based signal
        if pred > threshold:
            model_signal = 1  # Long
        elif pred < -threshold:
            model_signal = -1  # Short
        else:
            model_signal = 0  # No trade
        # Secondary confirmation logic
        confirm = True
        # Example: require ADX > 20 for trend strength, or RSI not overbought/oversold
        adx = features.iloc[-1].get('adx_seq_0', None)
        rsi = features.iloc[-1].get('rsi_seq_0', None)
        if adx is not None and not np.isnan(adx):
            confirm = confirm and (adx > 20)
        if rsi is not None and not np.isnan(rsi):
            confirm = confirm and (0.2 < rsi < 0.8)  # RSI scaled to [0,1]
        # Only act if both model and confirmation agree
        signal = model_signal if confirm else 0
        # If confidence is very low, suppress signal
        if confidence < 0.1:
            signal = 0
        return signal, confidence, TradeType.SWING_TRADE

    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        # Use base technical validation and add any additional checks if needed
        return self.validate_technical_signal(signal, features)

    # TODO: Integrate with risk management and backtesting modules
    # TODO: Add unit tests and documentation 