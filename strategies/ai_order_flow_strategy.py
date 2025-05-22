from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from strategies.order_flow_strategy import OrderFlowStrategy, OrderFlowConfig
from utils.lstm_utils import LSTMAttentionModel, train_lstm, predict_lstm, save_lstm, load_lstm
from utils.xgboost_utils import train_xgboost, predict_xgboost, explain_xgboost, save_xgboost, load_xgboost

@dataclass
class AIOrderFlowConfig(OrderFlowConfig):
    """Configuration for AI-powered order flow strategy"""
    # LSTM parameters
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_sequence_length: int = 20
    lstm_probability_threshold: float = 0.7
    
    # XGBoost parameters
    xgboost_probability_threshold: float = 0.7
    xgboost_early_stopping_rounds: int = 10
    
    # Model paths
    lstm_model_path: str = "models/lstm_order_flow.pt"
    xgboost_model_path: str = "models/xgboost_order_flow.json"
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.feature_weights is None:
            self.feature_weights = {
                'day': {
                    'lstm_signal': 0.40,
                    'order_book_imbalance': 0.20,
                    'volume_delta': 0.20,
                    'microprice_deviation': 0.10,
                    'trade_intensity': 0.10
                },
                'swing': {
                    'xgboost_signal': 0.40,
                    'cumulative_delta': 0.20,
                    'liquidity_zones': 0.20,
                    'order_book_trend': 0.10,
                    'volume_profile': 0.10
                }
            }

class AIOrderFlowStrategy(OrderFlowStrategy):
    """
    Enhanced order flow strategy with AI components:
    - LSTM for day trading (high-frequency patterns)
    - XGBoost for swing trading (multi-hour to multi-day patterns)
    """
    
    def __init__(self, config: AIOrderFlowConfig = None):
        super().__init__(config=config or AIOrderFlowConfig())
        self.config: AIOrderFlowConfig = self.config
        
        # Initialize models
        self.lstm_model = None
        self.xgboost_model = None
        
        # Load models if they exist
        self._load_models()
    
    def _load_models(self):
        """Load trained models if they exist"""
        try:
            # Initialize LSTM model
            input_dim = len(self._get_lstm_features())
            self.lstm_model = LSTMAttentionModel(
                input_dim=input_dim,
                hidden_dim=self.config.lstm_hidden_dim,
                num_layers=self.config.lstm_num_layers,
                output_dim=1
            )
            load_lstm(self.lstm_model, self.config.lstm_model_path)
            self.lstm_model.eval()
            
            # Load XGBoost model
            self.xgboost_model = load_xgboost(self.config.xgboost_model_path)
        except:
            # Models don't exist yet, that's okay
            pass
    
    def _get_lstm_features(self) -> List[str]:
        """Get list of features for LSTM model"""
        return [
            'order_book_imbalance',
            'order_book_slope',
            'order_book_trend',
            'volume_delta',
            'normalized_delta_5',
            'normalized_delta_15',
            'normalized_delta_60',
            'microprice_deviation',
            'microprice_momentum',
            'trade_intensity',
            'volume_zscore',
            'vwap_deviation'
        ]
    
    def _get_xgboost_features(self) -> List[str]:
        """Get list of features for XGBoost model"""
        return [
            'cumulative_delta_60',
            'order_book_trend',
            'liquidity_zone_score',
            'volume_profile',
            'volatility_adjusted_spread',
            'price_zscore',
            'vwap_deviation',
            'volume_momentum'
        ]
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for both LSTM and XGBoost models"""
        # Get base features from parent class
        df = super().prepare_features(data)
        
        # Add any additional features needed for AI models
        df = self._add_ai_features(df)
        
        return df
    
    def _add_ai_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to AI models"""
        # Price Z-score for mean reversion
        if 'close' in df.columns:
            df['price_zscore'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
        
        # Liquidity zone score (how close to support/resistance)
        if 'near_support' in df.columns and 'near_resistance' in df.columns:
            df['liquidity_zone_score'] = 0
            df.loc[df['near_support'], 'liquidity_zone_score'] = 1
            df.loc[df['near_resistance'], 'liquidity_zone_score'] = -1
        
        # Volume profile (relative to recent average)
        if 'volume' in df.columns:
            df['volume_profile'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        return df
    
    def _prepare_lstm_data(self, features: pd.DataFrame) -> torch.Tensor:
        """Prepare data for LSTM model"""
        # Get relevant features
        lstm_features = self._get_lstm_features()
        X = features[lstm_features].values
        
        # Create sequences
        sequences = []
        for i in range(len(X) - self.config.lstm_sequence_length + 1):
            sequences.append(X[i:i + self.config.lstm_sequence_length])
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(sequences)
        return X_tensor
    
    def _prepare_xgboost_data(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for XGBoost model"""
        # Get relevant features
        xgboost_features = self._get_xgboost_features()
        return features[xgboost_features]
    
    def train_models(self, features: pd.DataFrame, targets: pd.Series):
        """Train both LSTM and XGBoost models"""
        # Split data into train/validation
        train_size = int(len(features) * 0.8)
        X_train = features.iloc[:train_size]
        y_train = targets.iloc[:train_size]
        X_val = features.iloc[train_size:]
        y_val = targets.iloc[train_size:]
        
        # Train LSTM model
        self._train_lstm(X_train, y_train, X_val, y_val)
        
        # Train XGBoost model
        self._train_xgboost(X_train, y_train, X_val, y_val)
    
    def _train_lstm(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Train LSTM model"""
        # Prepare data
        X_train_tensor = self._prepare_lstm_data(X_train)
        y_train_tensor = torch.FloatTensor(y_train.values[self.config.lstm_sequence_length-1:])
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_dim = len(self._get_lstm_features())
        self.lstm_model = LSTMAttentionModel(
            input_dim=input_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            output_dim=1
        )
        
        # Train model
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters())
        self.lstm_model = train_lstm(
            self.lstm_model,
            train_loader,
            criterion,
            optimizer,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Save model
        save_lstm(self.lstm_model, self.config.lstm_model_path)
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Train XGBoost model"""
        # Prepare data
        X_train_xgb = self._prepare_xgboost_data(X_train)
        X_val_xgb = self._prepare_xgboost_data(X_val)
        
        # Train model
        self.xgboost_model = train_xgboost(
            X_train_xgb,
            y_train,
            X_val=X_val_xgb,
            y_val=y_val,
            early_stopping_rounds=self.config.xgboost_early_stopping_rounds
        )
        
        # Save model
        save_xgboost(self.xgboost_model, self.config.xgboost_model_path)
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate signals using both AI models and traditional indicators"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            return self._calculate_day_trading_signals(current, features)
        else:
            return self._calculate_swing_trading_signals(current, features)
    
    def _calculate_day_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate day trading signals using LSTM and traditional indicators"""
        signals = {}
        
        # Get LSTM prediction
        if self.lstm_model is not None:
            X_lstm = self._prepare_lstm_data(features)
            probabilities, _ = predict_lstm(
                self.lstm_model,
                X_lstm,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            lstm_prob = probabilities[-1]
            
            if lstm_prob >= self.config.lstm_probability_threshold:
                signals['lstm_signal'] = 1
            elif lstm_prob <= (1 - self.config.lstm_probability_threshold):
                signals['lstm_signal'] = -1
            else:
                signals['lstm_signal'] = 0
        else:
            signals['lstm_signal'] = 0
        
        # Get traditional signals from parent class
        base_signals = super()._calculate_day_trading_signals(current, features)
        
        # Combine signals using weights
        weights = self.config.feature_weights['day']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            # Calculate percent agreement with overall signal
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Scale confidence by signal strength and agreement
            confidence = agreement_ratio * min(abs(total_signal) * 1.5, 1.0)
            
            # Boost confidence if near liquidity zone
            if np.sign(total_signal) > 0 and current.get('near_support', False):
                confidence *= 1.2
            elif np.sign(total_signal) < 0 and current.get('near_resistance', False):
                confidence *= 1.2
        else:
            confidence = 0
        
        return total_signal, min(confidence, 1.0), TradeType.DAY_TRADE
    
    def _calculate_swing_trading_signals(
        self, 
        current: pd.Series,
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate swing trading signals using XGBoost and traditional indicators"""
        signals = {}
        
        # Get XGBoost prediction
        if self.xgboost_model is not None:
            X_xgb = self._prepare_xgboost_data(features)
            probabilities, _ = predict_xgboost(
                self.xgboost_model,
                X_xgb,
                probability_threshold=self.config.xgboost_probability_threshold
            )
            xgb_prob = probabilities[-1]
            
            if xgb_prob >= self.config.xgboost_probability_threshold:
                signals['xgboost_signal'] = 1
            elif xgb_prob <= (1 - self.config.xgboost_probability_threshold):
                signals['xgboost_signal'] = -1
            else:
                signals['xgboost_signal'] = 0
        else:
            signals['xgboost_signal'] = 0
        
        # Get traditional signals from parent class
        base_signals = super()._calculate_swing_trading_signals(current, features)
        
        # Combine signals using weights
        weights = self.config.feature_weights['swing']
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if signal_signs:
            # Calculate percent agreement with overall signal
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            
            # Scale confidence by signal strength and agreement
            confidence = agreement_ratio * min(abs(total_signal) * 1.2, 1.0)
        else:
            confidence = 0
        
        return total_signal, min(confidence, 1.0), TradeType.SWING_TRADE
    
    def explain_signal(self, features: pd.DataFrame) -> Dict:
        """Explain the current trading signal using model explainability"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        explanation = {
            'model': 'lstm' if is_intraday else 'xgboost',
            'features': {},
            'confidence': 0.0,
            'reasoning': []
        }
        
        if is_intraday and self.lstm_model is not None:
            # Get LSTM attention weights
            X_lstm = self._prepare_lstm_data(features)
            _, attn_weights = predict_lstm(
                self.lstm_model,
                X_lstm,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            
            # Get feature importances from attention
            feature_importances = attn_weights[-1].squeeze()
            for i, feature in enumerate(self._get_lstm_features()):
                explanation['features'][feature] = float(feature_importances[i])
            
            # Add reasoning based on important features
            top_features = sorted(
                explanation['features'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            for feature, importance in top_features:
                if importance > 0.1:  # Only include significant features
                    explanation['reasoning'].append(
                        f"{feature} is showing strong signal (importance: {importance:.2f})"
                    )
        
        elif not is_intraday and self.xgboost_model is not None:
            # Get XGBoost SHAP values
            X_xgb = self._prepare_xgboost_data(features)
            shap_values, importance_dict = explain_xgboost(
                self.xgboost_model,
                X_xgb.iloc[[-1]],  # Only explain current prediction
                feature_names=self._get_xgboost_features()
            )
            
            # Add feature importances
            explanation['features'] = importance_dict
            
            # Add reasoning based on important features
            top_features = sorted(
                importance_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            for feature, importance in top_features:
                if importance > 0.1:  # Only include significant features
                    explanation['reasoning'].append(
                        f"{feature} is showing strong signal (importance: {importance:.2f})"
                    )
        
        return explanation 