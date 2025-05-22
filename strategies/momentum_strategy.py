from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType
from utils import xgboost_utils

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
    
    # XGBoost Parameters
    model_path: str = None
    probability_threshold: float = 0.6
    xgboost_early_stopping_rounds: int = 10
    feature_list: list = None
    
    # Feature weights
    feature_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        super().validate()
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
                'order_book_imbalance', 'volume_spike', 'market_depth',
                'depth_imbalance', 'volume_delta', 'cumulative_delta',
                'volume_pressure', 'significant_imbalance',
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
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            params: XGBoost parameters (defaults to reasonable values if None)
            num_boost_round: Maximum number of training rounds
        """
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
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
        Make predictions using the trained XGBoost model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (probabilities, binary predictions)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        # Prepare data
        X_xgb = X[self.config.feature_list].dropna()
        
        # Make predictions
        return xgboost_utils.predict_xgboost(
            self.model,
            X_xgb,
            probability_threshold=self.config.probability_threshold
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
        Generate trading signals using the XGBoost model and apply probability thresholding.
        """
        if self.model is None:
            # Fall back to traditional signals if model is not available
            return self.calculate_technical_signals(features)
        
        # Get model prediction
        probabilities, _ = self.predict(features)
        prob = probabilities[-1]  # Get last prediction
        
        # Generate signal based on probability threshold
        signal = 1 if prob > self.config.probability_threshold else -1 if prob < (1 - self.config.probability_threshold) else 0
        
        # Calculate confidence based on probability distance from threshold
        confidence = abs(prob - 0.5) * 2  # Scale to [0,1]
        
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
            validations = [
                # Sufficient intraday volume
                current['volume_ratio'] > 1.0,
                
                # Not near session VWAP during low-volatility periods
                not (abs(current['vwap_ratio'] - 1) < 0.001 and 
                     current['volatility_regime'] < 0.5),
                
                # Strong enough momentum
                abs(current['norm_roc_10']) > current['atr'] / current['close']
            ]
        else:
            validations = [
                # Strong enough trend
                current['adx'] > self.config.adx_weak_trend,
                
                # Volume confirms trend
                current['volume_ratio'] > self.config.min_volume_ratio,
                
                # Not overextended
                abs(current['ma_crossover']) < current['atr'] * 2
            ]
        
        return all(validations)
    
    def update_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Update the XGBoost model with new data."""
        if self.model is not None:
            # Retrain model with new data
            self.train(features, target)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare momentum/trend specific features"""
        df = self.prepare_base_features(data)
        
        # Calculate lagged returns for 1, 2, and 3 bars
        df['returns_1'] = df['close'].pct_change(periods=1)
        df['returns_2'] = df['close'].pct_change(periods=2)
        df['returns_3'] = df['close'].pct_change(periods=3)
        
        # Calculate rolling statistics
        df['rolling_mean'] = df['returns_1'].rolling(window=20).mean()
        df['rolling_std'] = df['returns_1'].rolling(window=20).std()
        
        # Add ROC for multiple periods (1, 3, 5, 10 bars)
        for period in [1, 3, 5, 10]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period)
            # Normalize ROC by volatility
            df[f'norm_roc_{period}'] = df[f'roc_{period}'] / df['volatility']
        
        # Add Volume ROC
        df['volume_roc_1'] = df['volume'].pct_change(periods=1)
        df['volume_roc_3'] = df['volume'].pct_change(periods=3)
        df['volume_roc_5'] = df['volume'].pct_change(periods=5)
        
        # Normalize Volume ROC by average volume
        volume_ma = df['volume'].rolling(window=20).mean()
        df['norm_volume_roc_1'] = df['volume_roc_1'] / volume_ma
        df['norm_volume_roc_3'] = df['volume_roc_3'] / volume_ma
        df['norm_volume_roc_5'] = df['volume_roc_5'] / volume_ma
        
        # Order Book Features
        if all(col in df.columns for col in ['bid_volume', 'ask_volume']):
            # Order Book Imbalance
            df['order_book_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
            
            # Volume Spike Detection
            df['volume_spike'] = df['volume'] > (df['volume'].rolling(window=20).mean() + 2 * df['volume'].rolling(window=20).std())
            
            # Market Depth Analysis
            df['market_depth'] = df['bid_volume'] + df['ask_volume']
            df['depth_imbalance'] = (df['bid_volume'] - df['ask_volume']) / df['market_depth']
            
            # Cumulative Volume Delta
            df['volume_delta'] = df['bid_volume'] - df['ask_volume']
            df['cumulative_delta'] = df['volume_delta'].cumsum()
            
            # Volume Pressure
            df['volume_pressure'] = df['volume_delta'] * df['returns_1']
            
            # Significant Imbalance Detection
            df['significant_imbalance'] = abs(df['order_book_imbalance']) > self.config.obi_threshold
        else:
            # Placeholder values if order book data isn't available
            df['order_book_imbalance'] = 0
            df['volume_spike'] = False
            df['market_depth'] = df['volume']
            df['depth_imbalance'] = 0
            df['volume_delta'] = 0
            df['cumulative_delta'] = 0
            df['volume_pressure'] = 0
            df['significant_imbalance'] = False
        
        # Volatility Features
        # Historical Volatility (standard deviation of returns)
        df['historical_vol'] = df['returns_1'].rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # Realized Volatility (sum of squared returns)
        df['realized_vol'] = np.sqrt(df['returns_1'].rolling(window=20).apply(lambda x: np.sum(x**2))) * np.sqrt(252)
        
        # Volatility Ratio (short-term vs long-term)
        df['vol_ratio'] = df['historical_vol'] / df['historical_vol'].rolling(window=50).mean()
        
        # Volatility Regime Detection
        df['volatility_regime'] = pd.cut(
            df['vol_ratio'],
            bins=[-np.inf, 0.7, 1.3, np.inf],
            labels=['low', 'normal', 'high']
        )
        
        # Volatility Percentile
        df['volatility_percentile'] = df['historical_vol'].rolling(window=100).rank(pct=True)
        
        # Volatility Divergence (price vs volatility)
        price_roc = df['close'].pct_change(5)
        vol_roc = df['historical_vol'].pct_change(5)
        df['volatility_divergence'] = np.where(
            (price_roc > 0) & (vol_roc < 0), -1,  # Bearish divergence
            np.where(
                (price_roc < 0) & (vol_roc > 0), 1,  # Bullish divergence
                0  # No divergence
            )
        )
        
        # Moving Average Crossover
        short_ma = df['close'].rolling(window=self.config.short_ma_period).mean()
        long_ma = df['close'].rolling(window=self.config.long_ma_period).mean()
        df['ma_crossover'] = (short_ma - long_ma) / long_ma
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.swing_rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.swing_rsi_period).mean()
        rs = gain / loss
        df['swing_rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=self.config.macd_fast_period, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.config.macd_slow_period, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal_period, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ADX
        tr1 = pd.DataFrame(df['high'] - df['low'])
        tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config.adx_period).mean()
        
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=self.config.adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=self.config.adx_period).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=self.config.adx_period).mean()
        
        # Add new feature engineering methods
        df = self._add_feature_interactions(df)
        df = self._add_time_based_features(df)
        df = self._add_statistical_features(df)
        df = self._normalize_features(df)
        
        return df
    
    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions and combinations"""
        # Price-Volume interaction
        df['price_volume_correlation'] = df['returns_1'].rolling(window=20).corr(df['volume_roc_1'])
        
        # Momentum-Volatility interaction
        df['momentum_vol_ratio'] = df['norm_roc_10'] * df['vol_ratio']
        
        # Volume-Volatility interaction
        df['volume_vol_ratio'] = df['norm_volume_roc_5'] * df['vol_ratio']
        
        # Order Book-Volume interaction
        if 'order_book_imbalance' in df.columns:
            df['obi_volume_ratio'] = df['order_book_imbalance'] * df['volume_ratio']
        
        # Price-Momentum interaction
        df['price_momentum'] = df['returns_1'] * df['norm_roc_5']
        
        # Volatility-Momentum interaction
        df['vol_momentum'] = df['vol_ratio'] * df['norm_roc_10']
        
        return df
    
    def _add_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if hasattr(df.index, 'hour'):
            # Time of day features
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            
            # Market session flags
            df['market_open'] = (df['hour'] == 9) | (df['hour'] == 10)  # Early session
            df['lunch_hour'] = (df['hour'] >= 12) & (df['hour'] < 13)
            df['market_close'] = (df['hour'] >= 15) & (df['hour'] < 16)
            
            # Time-based volatility
            df['hourly_vol'] = df.groupby(df.index.hour)['returns_1'].transform('std')
            
            # Time-based volume
            df['hourly_volume'] = df.groupby(df.index.hour)['volume'].transform('mean')
            df['volume_time_ratio'] = df['volume'] / df['hourly_volume']
        
        if hasattr(df.index, 'dayofweek'):
            # Day of week features
            df['day_of_week'] = df.index.dayofweek
            
            # Day-based volatility
            df['daily_vol'] = df.groupby(df.index.dayofweek)['returns_1'].transform('std')
            
            # Day-based volume
            df['daily_volume'] = df.groupby(df.index.dayofweek)['volume'].transform('mean')
            df['volume_day_ratio'] = df['volume'] / df['daily_volume']
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Skewness of returns
        df['returns_skew'] = df['returns_1'].rolling(window=20).skew()
        
        # Kurtosis of returns
        df['returns_kurt'] = df['returns_1'].rolling(window=20).kurt()
        
        # Z-score of returns
        df['returns_zscore'] = (df['returns_1'] - df['returns_1'].rolling(window=20).mean()) / \
                             df['returns_1'].rolling(window=20).std()
        
        # Quantile features
        df['returns_quantile'] = df['returns_1'].rolling(window=20).apply(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop').iloc[-1]
        )
        
        # Autocorrelation of returns
        df['returns_autocorr'] = df['returns_1'].rolling(window=20).apply(
            lambda x: x.autocorr()
        )
        
        # Volume statistics
        df['volume_skew'] = df['volume'].rolling(window=20).skew()
        df['volume_kurt'] = df['volume'].rolling(window=20).kurt()
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / \
                            df['volume'].rolling(window=20).std()
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and scale features"""
        # List of features to normalize
        features_to_normalize = [
            'returns_1', 'returns_2', 'returns_3',
            'volume_roc_1', 'volume_roc_3', 'volume_roc_5',
            'order_book_imbalance', 'depth_imbalance',
            'volume_delta', 'cumulative_delta',
            'price_volume_correlation', 'momentum_vol_ratio',
            'volume_vol_ratio', 'price_momentum', 'vol_momentum',
            'returns_skew', 'returns_kurt', 'returns_zscore',
            'volume_skew', 'volume_kurt', 'volume_zscore'
        ]
        
        # Normalize each feature using rolling z-score
        for feature in features_to_normalize:
            if feature in df.columns:
                mean = df[feature].rolling(window=20).mean()
                std = df[feature].rolling(window=20).std()
                df[f'{feature}_normalized'] = (df[feature] - mean) / std
        
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