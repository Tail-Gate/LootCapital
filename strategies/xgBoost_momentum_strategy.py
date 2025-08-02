from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType
from XGBoostMean import xgboost_utils
from utils.feature_generator import FeatureGenerator

@dataclass
class MomentumConfig(TechnicalConfig):
    """Configuration for momentum/trend following strategy"""
    # Moving Averages (Swing Trading)
    short_ma_period: int = 5  # 5-day MA
    long_ma_period: int = 10  # 10-day MA
    
    # RSI Parameters
    swing_rsi_period: int = 14  # 14-period RSI for swing trading
    day_rsi_period: int = 14    # 14-period for intraday
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # MACD Parameters
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    
    # ADX Parameters
    adx_period: int = 14
    adx_strong_trend: float = 40.0
    adx_weak_trend: float = 20.0
    
    # ATR Parameters
    swing_atr_multiplier: float = 1.5
    day_atr_multiplier: float = 1.2
    
    # Intraday Parameters
    short_ema_period: int = 15  # 15-minute EMA
    long_ema_period: int = 35   # 35-minute EMA
    
    # Volume Parameters
    min_volume_ratio: float = 1.5
    obi_threshold: float = 0.2  # Order Book Imbalance threshold
    
    # XGBoost Parameters - Updated for three-class classification
    model_path: str = None
    num_classes: int = 3  # 0=short, 1=hold, 2=long
    class_thresholds: Dict[str, float] = None  # Thresholds for each class
    xgboost_early_stopping_rounds: int = 10
    feature_list: list = None
    
    # Prediction Parameters
    lookforward_periods: int = 4  # Number of candlesticks to predict ahead
    price_threshold_pct: float = 0.02  # 2% threshold for significant moves
    min_confidence_threshold: float = 0.5  # Minimum confidence to generate a signal
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
        
        # Set default class thresholds for three-class classification
        if self.class_thresholds is None:
            self.class_thresholds = {
                'short': 0.25,   # Probability threshold for short signal
                'hold': 0.5,     # Probability threshold for hold signal (higher due to 2% threshold)
                'long': 0.25     # Probability threshold for long signal
            }
        
        if self.feature_weights is None:
            self.feature_weights = {
                'swing': {
                    'ma_crossover': 0.3,
                    'rsi': 0.2,
                    'macd': 0.2,
                    'adx': 0.2,
                    'volume': 0.1
                },
                'day': {
                    'ema_crossover': 0.3,
                    'rsi': 0.2,
                    'momentum': 0.2,
                    'vwap': 0.2,
                    'obi': 0.1
                }
            }
        
        if self.feature_list is None:
            self.feature_list = [
                'returns_1', 'returns_2', 'returns_3',
                'rolling_mean', 'rolling_std',
                'roc_1', 'roc_3', 'roc_5', 'roc_10',
                'norm_roc_1', 'norm_roc_3', 'norm_roc_5', 'norm_roc_10',
                'volume_roc_1', 'volume_roc_3', 'volume_roc_5',
                'norm_volume_roc_1', 'norm_volume_roc_3', 'norm_volume_roc_5',
                'historical_vol', 'realized_vol', 'vol_ratio',
                'volatility_percentile', 'volatility_divergence',
                'ma_crossover', 'swing_rsi', 'macd', 'macd_signal',
                'macd_hist', 'adx'
            ]

class MomentumStrategy(TechnicalStrategy):
    def __init__(self, config: MomentumConfig = None):
        super().__init__(name="momentum", config=config or MomentumConfig())
        self.config: MomentumConfig = self.config
        self.model = None
        if self.config.model_path:
            self.load_model(self.config.model_path)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: Dict = None, num_boost_round: int = 100):
        """
        Train the XGBoost model for three-class classification.
        
        Args:
            X_train: Training features
            y_train: Training labels (0=short, 1=hold, 2=long)
            params: XGBoost parameters (defaults to reasonable values if None)
            num_boost_round: Maximum number of training rounds
        """
        if params is None:
            params = {
                'objective': 'multi:softprob',  # Use softprob for probability output
                'eval_metric': 'mlogloss',
                'num_class': self.config.num_classes,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0
            }
        
        # Prepare data
        X = X_train[self.config.feature_list].dropna()
        y = y_train[X.index]
        
        # Ensure labels are integers for multi-class classification
        y = y.astype(int)
        
        # Validate that labels are in the correct range
        unique_labels = y.unique()
        if not all(label in [0, 1, 2] for label in unique_labels):
            raise ValueError(f"Labels must be 0 (short), 1 (hold), or 2 (long). Found: {unique_labels}")
        
        # Simple validation split (80/20)
        split_idx = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if X_tr.empty or X_val.empty or y_tr.empty or y_val.empty:
            raise ValueError("Training or validation set is empty after dropping NaNs. Check feature engineering and data.")
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'validation')]
        
        # Train model
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=self.config.xgboost_early_stopping_rounds
        )
        
        # Save model if path is specified
        if self.config.model_path:
            self.save_model(self.config.model_path)
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained XGBoost model for three-class classification.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (class_probabilities, class_predictions)
            - class_probabilities: Array of shape (n_samples, 3) with probabilities for each class
            - class_predictions: Array of shape (n_samples,) with predicted class labels (0=short, 1=hold, 2=long)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        # Prepare data
        X_xgb = X[self.config.feature_list].dropna()
        
        # Make predictions
        return xgboost_utils.predict_xgboost_multi(
            self.model,
            X_xgb,
            num_classes=self.config.num_classes
        )
    
    def save_model(self, path: str) -> None:
        """Save the trained XGBoost model to file."""
        if self.model is not None:
            xgboost_utils.save_xgboost(self.model, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained XGBoost model from file."""
        self.model = xgboost_utils.load_xgboost(path)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from the trained model."""
        if self.model is not None:
            return xgboost_utils.get_feature_importance(self.model)
        return None
    
    def explain(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate SHAP values and feature importance for model explanation.
        
        Args:
            X: Features to explain
            
        Returns:
            Tuple of (SHAP values, feature importance dictionary)
        """
        if self.model is not None:
            return xgboost_utils.explain_xgboost(
                self.model,
                xgb.DMatrix(X[self.config.feature_list].dropna()),
                feature_names=self.config.feature_list
            )
        return None, None
    
    def calculate_signals(self, features: pd.DataFrame) -> Tuple[float, float, TradeType]:
        """
        Generate trading signals using the XGBoost model for three-class classification.
        Uses the class with the highest probability as the prediction.
        Only generates signals if confidence > 0.5.
        """
        if self.model is None:
            # Fall back to traditional signals if model is not available
            return self.calculate_technical_signals(features)
        
        # Get model prediction
        class_probabilities, class_predictions = self.predict(features)
        latest_probs = class_probabilities[-1]  # Get last prediction probabilities
        latest_pred = class_predictions[-1]     # Get last prediction class (highest probability)
        
        # Calculate confidence (probability of the predicted class)
        confidence = latest_probs[latest_pred]
        
        # Only generate signal if confidence > 0.5
        if confidence < self.config.min_confidence_threshold:
            # Return hold signal if confidence is too low
            signal = 0
            confidence = 0.0
        else:
            # Convert class prediction to signal format
            # 0=short -> -1, 1=hold -> 0, 2=long -> 1
            signal_map = {0: -1, 1: 0, 2: 1}
            signal = signal_map[latest_pred]
        
        # Determine trade type based on data frequency
        trade_type = TradeType.DAY_TRADE if self.is_intraday_data(features) else TradeType.SWING_TRADE
        
        return signal, confidence, trade_type
    
    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        """Validate trading signal with additional checks."""
        if not self.validate_technical_signal(signal, features):
            return False
        
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            validations = []
            
            # Sufficient intraday volume
            if 'volume_ratio' in current:
                validations.append(current['volume_ratio'] > 1.0)
            
            # Not near session VWAP during low-volatility periods
            if 'vwap_ratio' in current and 'volatility_regime' in current:
                validations.append(not (abs(current['vwap_ratio'] - 1) < 0.001 and 
                     current['volatility_regime'] < 0.5))
            
            # Strong enough momentum
            if 'norm_roc_10' in current and 'atr' in current and 'close' in current:
                validations.append(abs(current['norm_roc_10']) > current['atr'] / current['close'])
            
            # If no validations were added, return True
            if not validations:
                return True
                
        else:
            validations = []
            
            # Strong enough trend
            if 'adx' in current:
                validations.append(current['adx'] > self.config.adx_weak_trend)
            
            # Volume confirms trend
            if 'volume_ratio' in current:
                validations.append(current['volume_ratio'] > self.config.min_volume_ratio)
            
            # Not overextended
            if 'ma_crossover' in current and 'atr' in current:
                validations.append(abs(current['ma_crossover']) < current['atr'] * 2)
            
            # If no validations were added, return True
            if not validations:
                return True
        
        return all(validations)
    
    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update the XGBoost model with new data."""
        if self.model is not None:
            # Retrain model with new data
            self.train(features, target)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare momentum/trend specific features using the FeatureGenerator"""
        # Initialize feature generator
        feature_generator = FeatureGenerator()
        
        # Generate features using the feature generator
        df = feature_generator.generate_features(data)
        
        # Add momentum-specific features that might not be in the feature generator
        df = self._add_momentum_specific_features(df, data)
        
        return df
    
    def _add_momentum_specific_features(self, df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-specific features that complement the feature generator"""
        # Calculate lagged returns for 1, 2, and 3 bars
        df['returns_1'] = data['close'].pct_change(periods=1)
        df['returns_2'] = data['close'].pct_change(periods=2)
        df['returns_3'] = data['close'].pct_change(periods=3)
        
        # Calculate rolling statistics
        df['rolling_mean'] = df['returns_1'].rolling(window=20).mean()
        df['rolling_std'] = df['returns_1'].rolling(window=20).std()
        
        # Add ROC for multiple periods (1, 3, 5, 10 bars)
        for period in [1, 3, 5, 10]:
            df[f'roc_{period}'] = data['close'].pct_change(periods=period)
            # Normalize ROC by volatility if available
            if 'atr' in df.columns:
                df[f'norm_roc_{period}'] = df[f'roc_{period}'] / df['atr']
            else:
                df[f'norm_roc_{period}'] = df[f'roc_{period}']
        
        # Add Volume ROC
        df['volume_roc_1'] = data['volume'].pct_change(periods=1)
        df['volume_roc_3'] = data['volume'].pct_change(periods=3)
        df['volume_roc_5'] = data['volume'].pct_change(periods=5)
        
        # Normalize Volume ROC by average volume
        volume_ma = data['volume'].rolling(window=20).mean()
        df['norm_volume_roc_1'] = df['volume_roc_1'] / volume_ma
        df['norm_volume_roc_3'] = df['volume_roc_3'] / volume_ma
        df['norm_volume_roc_5'] = df['volume_roc_5'] / volume_ma
        
        # Volatility Features
        # Historical Volatility (standard deviation of returns)
        df['historical_vol'] = df['returns_1'].rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # Realized Volatility (sum of squared returns)
        df['realized_vol'] = np.sqrt(df['returns_1'].rolling(window=20).apply(lambda x: np.sum(x**2))) * np.sqrt(252)
        
        # Volatility Ratio (short-term vs long-term)
        df['vol_ratio'] = df['historical_vol'] / df['historical_vol'].rolling(window=50).mean()
        
        # Volatility Percentile
        df['volatility_percentile'] = df['historical_vol'].rolling(window=100).rank(pct=True)
        
        # Volatility Divergence (price vs volatility)
        price_roc = data['close'].pct_change(5)
        vol_roc = df['historical_vol'].pct_change(5)
        df['volatility_divergence'] = np.where(
            (price_roc > 0) & (vol_roc < 0), -1,  # Bearish divergence
            np.where(
                (price_roc < 0) & (vol_roc > 0), 1,  # Bullish divergence
                0  # No divergence
            )
        )
        
        # Moving Average Crossover (if not already present)
        if 'ma_crossover' not in df.columns:
            short_ma = data['close'].rolling(window=self.config.short_ma_period).mean()
            long_ma = data['close'].rolling(window=self.config.long_ma_period).mean()
            df['ma_crossover'] = (short_ma - long_ma) / long_ma
        
        # RSI (if not already present)
        if 'swing_rsi' not in df.columns:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config.swing_rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.swing_rsi_period).mean()
            rs = gain / loss
            df['swing_rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (if not already present)
        if 'macd' not in df.columns:
            exp1 = data['close'].ewm(span=self.config.macd_fast_period, adjust=False).mean()
            exp2 = data['close'].ewm(span=self.config.macd_slow_period, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal_period, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ADX (if not already present)
        if 'adx' not in df.columns:
            tr1 = pd.DataFrame(data['high'] - data['low'])
            tr2 = pd.DataFrame(abs(data['high'] - data['close'].shift(1)))
            tr3 = pd.DataFrame(abs(data['low'] - data['close'].shift(1)))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=self.config.adx_period).mean()
            
            up_move = data['high'] - data['high'].shift(1)
            down_move = data['low'].shift(1) - data['low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_di = 100 * pd.Series(plus_dm).rolling(window=self.config.adx_period).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(window=self.config.adx_period).mean() / atr
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=self.config.adx_period).mean()
        
        return df
    
    def is_intraday_data(self, df: pd.DataFrame) -> bool:
        """Check if we're working with intraday data"""
        if len(df) < 2:
            return False
        time_diff = df.index[1] - df.index[0]
        return time_diff.total_seconds() < 24 * 60 * 60
    
    def calculate_technical_signals(
        self, 
        features: pd.DataFrame
    ) -> Tuple[float, float, TradeType]:
        """Calculate momentum/trend signals"""
        current = features.iloc[-1]
        is_intraday = self.is_intraday_data(features)
        
        if is_intraday:
            # Day Trading Signals
            signals = {
                'ema_crossover': np.sign(current['ema_crossover']),
                'rsi': -1 if current['day_rsi'] > self.config.rsi_overbought else 
                        1 if current['day_rsi'] < self.config.rsi_oversold else 0,
                'momentum': np.sign(current['norm_roc_10']),
                'vwap': np.sign(current['close'] - current['vwap']),
                'obi': np.sign(current['obi']) if 'obi' in current else 0
            }
            
            weights = self.config.feature_weights['day']
            trade_type = TradeType.DAY_TRADE
            
        else:
            # Swing Trading Signals
            signals = {
                'ma_crossover': np.sign(current['ma_crossover']),
                'rsi': -1 if current['swing_rsi'] > self.config.rsi_overbought else 
                        1 if current['swing_rsi'] < self.config.rsi_oversold else 0,
                'macd': np.sign(current['macd_hist']),
                'adx': 1 if current['adx'] > self.config.adx_strong_trend else 
                      -1 if current['adx'] < self.config.adx_weak_trend else 0,
                'volume': 1 if current['volume_ratio'] > self.config.min_volume_ratio else -1
            }
            
            weights = self.config.feature_weights['swing']
            trade_type = TradeType.SWING_TRADE
        
        # Calculate weighted signal
        total_signal = sum(signals[k] for k in signals)
        
        # Calculate confidence based on signal agreement
        signal_signs = [np.sign(s) for s in signals.values() if s != 0]
        if signal_signs:
            agreement_ratio = sum(1 for s in signal_signs if s == np.sign(total_signal)) / len(signal_signs)
            confidence = agreement_ratio * abs(total_signal)
        else:
            confidence = 0
        
        return total_signal, confidence, trade_type

    def create_three_class_labels(self, data: pd.DataFrame, lookforward_periods: int = 4, 
                                 threshold_pct: float = 0.02) -> pd.Series:
        """
        Create three-class labels from price data for training.
        
        Args:
            data: DataFrame with price data
            lookforward_periods: Number of periods to look forward for price change (default: 4)
            threshold_pct: Percentage threshold for determining significant moves (default: 2%)
            
        Returns:
            Series with labels: 0=short, 1=hold, 2=long
        """
        # Calculate future price change over multiple periods
        future_price = data['close'].shift(-lookforward_periods)
        price_change_pct = (future_price - data['close']) / data['close']
        
        # Create labels based on price change
        labels = pd.Series(index=data.index, dtype=int)
        
        # Short signal: price decreases by more than threshold
        labels[price_change_pct < -threshold_pct] = 0
        
        # Long signal: price increases by more than threshold  
        labels[price_change_pct > threshold_pct] = 2
        
        # Hold signal: price change is within threshold
        labels[(price_change_pct >= -threshold_pct) & (price_change_pct <= threshold_pct)] = 1
        
        return labels