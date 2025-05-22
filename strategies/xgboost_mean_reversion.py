import numpy as np
import pandas as pd
from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType
from utils import xgboost_utils

class XGBoostMeanReversionConfig(TechnicalConfig):
    """Configuration for XGBoost-based mean reversion strategy (intraday)"""
    # Model and feature parameters
    model_path: str = None
    probability_threshold: float = 0.6
    feature_list: list = None
    
    def __post_init__(self):
        super().validate()
        if self.feature_list is None:
            self.feature_list = [
                'bollinger_b', 'zscore', 'vwap_deviation', 'rsi', 'order_flow_imbalance',
                'dmi', 'adx', 'volume_profile', 'market_depth_imbalance', 'technical_divergence'
            ]

class XGBoostMeanReversionStrategy(TechnicalStrategy):
    """
    Intraday mean reversion strategy using XGBoost for signal generation.
    Features are derived from standard exchange data (OHLCV, order book, volume).
    """
    def __init__(self, config: XGBoostMeanReversionConfig = None):
        super().__init__(name="xgboost_mean_reversion", config=config or XGBoostMeanReversionConfig())
        self.model = None
        if self.config.model_path:
            self.load_model(self.config.model_path)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all required features for the XGBoost model.
        """
        df = self.prepare_base_features(data)
        ti = self.ti  # TechnicalIndicators instance

        # Bollinger Bands and %B
        upper, middle, lower = ti.calculate_bollinger_bands(df['close'], period=20)
        df['bollinger_b'] = (df['close'] - lower) / (upper - lower)

        # Z-Score
        df['zscore'] = ti.calculate_zscore(df['close'], period=20)

        # VWAP Deviation
        if 'vwap' not in df.columns:
            df['vwap'] = ti.calculate_vwap(df)
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

        # RSI
        df['rsi'] = ti.calculate_rsi(df['close'], period=14)

        # DMI/ADX
        adx = ti.calculate_directional_movement(df['high'], df['low'], df['close'], period=14)
        df['adx'] = adx
        # DMI (Directional Movement Index) - can be approximated as ADX for now, or split into +DI/-DI if needed
        df['dmi'] = adx  # Placeholder, can be replaced with +DI/-DI calculation

        # Order Flow Imbalance (requires bid_volume and ask_volume)
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            df['order_flow_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-9)
        else:
            df['order_flow_imbalance'] = np.nan

        # Volume Profile / Liquidity Zones (proxy: rolling sum of volume at price)
        df['volume_profile'] = df['volume'].rolling(window=20).sum()

        # Market Depth Imbalance (reuse order_flow_imbalance or order_book_imbalance if available)
        if 'order_book_imbalance' in df.columns:
            df['market_depth_imbalance'] = df['order_book_imbalance']
        else:
            df['market_depth_imbalance'] = df['order_flow_imbalance']

        # Technical Divergence (price vs. volume delta divergence)
        if 'price_direction' in df.columns and 'volume_delta' in df.columns:
            df['technical_divergence'] = (np.sign(df['close'].diff()) != np.sign(df['volume_delta'])).astype(int)
        else:
            # Fallback: use price vs. volume z-score divergence
            df['technical_divergence'] = (np.sign(df['close'].diff()) != np.sign(df['volume'].diff())).astype(int)

        # Ensure all features in feature_list are present
        for feat in self.config.feature_list:
            if feat not in df.columns:
                df[feat] = np.nan

        return df

    def train(self, X_train, y_train, params=None, num_boost_round=100):
        self.model = xgboost_utils.train_xgboost(X_train, y_train, params, num_boost_round)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        return xgboost_utils.predict_xgboost(self.model, X)

    def save_model(self, path):
        if self.model is not None:
            xgboost_utils.save_xgboost(self.model, path)

    def load_model(self, path):
        self.model = xgboost_utils.load_xgboost(path)

    def get_feature_importance(self):
        if self.model is not None:
            return xgboost_utils.get_feature_importance(self.model)
        return None

    def explain(self, X):
        if self.model is not None:
            return xgboost_utils.explain_with_shap(self.model, X)
        return None

    def calculate_signals(self, features: pd.DataFrame) -> tuple:
        """
        Generate trading signals using the XGBoost model and apply probability thresholding and indicator confirmation.
        """
        X = features[self.config.feature_list].dropna().values[-1:]
        prob = self.predict(X)[0]
        signal = 1 if prob > self.config.probability_threshold else -1 if prob < (1 - self.config.probability_threshold) else 0
        confidence = abs(prob - 0.5) * 2  # Scale to [0,1]
        # TODO: Add secondary indicator confirmation (e.g., ADX, VWAP, order flow)
        trade_type = TradeType.DAY_TRADE
        return signal, confidence, trade_type

    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        # Use base technical validation and add any additional checks if needed
        return self.validate_technical_signal(signal, features)

    # TODO: Integrate with risk management and backtesting modules
    # TODO: Add unit tests and documentation 