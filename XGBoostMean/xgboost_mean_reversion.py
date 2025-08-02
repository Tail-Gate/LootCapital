import numpy as np
import pandas as pd
import xgboost as xgb
from strategies.technical_strategy import TechnicalStrategy, TechnicalConfig, TradeType
from XGBoostMean import xgboost_utils

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
                'bollinger_b', 'zscore', 'vwap_deviation', 'rsi', 'dmi', 'adx', 'volume_profile', 
                'technical_divergence', 'price_momentum', 'volatility_regime', 'support_resistance', 
                'breakout_intensity', 'ema_12', 'ema_26', 'macd_line', 'macd_signal', 'macd_histogram',
                'sma_20', 'sma_50', 'atr', 'obv', 'momentum', 'volatility', 'price_lag_1', 'price_lag_2',
                'price_lag_5', 'volume_lag_1', 'volume_lag_2', 'price_rolling_mean_10', 'price_rolling_std_10',
                'volume_rolling_mean_10', 'volume_rolling_std_10'
            ]

class XGBoostMeanReversionStrategy(TechnicalStrategy):
    """
    Enhanced intraday mean reversion strategy using XGBoost with hyperparameter optimization.
    """
    
    def __init__(self, config: XGBoostMeanReversionConfig = None):
        super().__init__(name="xgboost_mean_reversion", config=config or XGBoostMeanReversionConfig())
        self.model = None
        self.hyperopt_config = None
        
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

        # Volume Profile / Liquidity Zones (proxy: rolling sum of volume at price)
        df['volume_profile'] = df['volume'].rolling(window=20).sum()

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

    def train_with_hyperopt(self, data: pd.DataFrame, hyperopt_config):
        """Train model using hyperparameter optimization"""
        print("[STRATEGY] Starting hyperparameter optimization training...")
        
        # Import the new classes
        from strategies.xgboost_hyperopt_config import XGBoostHyperoptConfig
        from strategies.xgboost_data_processor import XGBoostDataProcessor
        from strategies.xgboost_hyperopt_trainer import XGBoostHyperoptTrainer
        
        # Create data processor and trainer
        data_processor = XGBoostDataProcessor(hyperopt_config, self.market_data, self.technical_indicators)
        trainer = XGBoostHyperoptTrainer(hyperopt_config, data_processor)
        
        # Prepare features and target
        features = data_processor.prepare_features_enhanced(data)
        y = data_processor.create_target_variable(data)
        
        # Remove NaN values
        valid_mask = ~(features[hyperopt_config.feature_list].isna().any(axis=1))
        features = features[valid_mask]
        y = y[valid_mask]
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data_time_series(
            features[hyperopt_config.feature_list].values, y
        )
        
        # Train model
        training_history = trainer.train_with_smote(X_train, y_train, X_val, y_val)
        
        # Store model and configuration
        self.model = trainer.model
        self.hyperopt_config = hyperopt_config
        
        # Update strategy configuration with best parameters
        self.config.probability_threshold = hyperopt_config.probability_threshold
        self.config.feature_list = hyperopt_config.feature_list
        
        print(f"[STRATEGY] Training completed. Best validation score: {training_history['best_score']:.4f}")
        
        return training_history

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

    def calculate_signals_enhanced(self, features: pd.DataFrame) -> tuple:
        """
        Enhanced signal calculation with confidence scoring and validation.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        # Prepare features
        X = features[self.config.feature_list].dropna().values[-1:]
        
        if len(X) == 0:
            return 0, 0.0, TradeType.DAY_TRADE  # No signal if no valid features
        
        # Get predictions
        probabilities = self.model.predict(xgb.DMatrix(X))
        predicted_class = np.argmax(probabilities[0])
        max_probability = np.max(probabilities[0])
        
        # Calculate signal based on predicted class
        if predicted_class == 0:  # Down
            signal = -1
        elif predicted_class == 2:  # Up
            signal = 1
        else:  # Hold
            signal = 0
        
        # Calculate confidence based on probability and threshold
        confidence = max_probability if max_probability > self.config.probability_threshold else 0.0
        
        # Additional validation using technical indicators
        if self.validate_signal(signal, features):
            trade_type = TradeType.DAY_TRADE
        else:
            signal = 0  # Invalidate signal if technical validation fails
            confidence = 0.0
            trade_type = TradeType.DAY_TRADE
        
        return signal, confidence, trade_type

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

    def get_feature_importance_enhanced(self):
        """Get feature importance with detailed analysis"""
        if self.model is None:
            return None
        
        importance = self.model.get_score(importance_type='gain')
        
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print("[STRATEGY] Feature Importance (Top 10):")
        for i, (feature, score) in enumerate(sorted_importance[:10]):
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        return sorted_importance

    def explain_prediction(self, features: pd.DataFrame):
        """
        Explain individual predictions using SHAP with comprehensive analysis.
        
        Args:
            features: DataFrame containing features for prediction
            
        Returns:
            Dictionary containing comprehensive SHAP explanation including:
            - prediction_details: Class prediction and probabilities
            - feature_contributions: How each feature influenced the prediction
            - top_features: Most important features for this prediction
            - explanation_summary: Human-readable explanation
        """
        if self.model is None:
            print("[EXPLAIN] Model not trained or loaded.")
            return None
        
        # Prepare features for the latest data point
        X = features[self.config.feature_list].dropna().values[-1:]
        
        if len(X) == 0:
            print("[EXPLAIN] No valid features available for explanation.")
            return None
        
        # Get SHAP explanation
        shap_explanation = xgboost_utils.explain_with_shap(self.model, X, self.config.feature_list)
        
        if shap_explanation is None:
            print("[EXPLAIN] Failed to generate SHAP explanation.")
            return None
        
        # Extract prediction details
        prediction_breakdown = shap_explanation['prediction_breakdown'][0]  # Latest sample
        predicted_class = prediction_breakdown['predicted_class']
        prediction_probabilities = prediction_breakdown['prediction_probabilities']
        feature_effects = prediction_breakdown['feature_effects']
        
        # Map class numbers to meaningful labels
        class_labels = {0: 'Down', 1: 'Hold', 2: 'Up'}
        predicted_label = class_labels.get(predicted_class, f'Class_{predicted_class}')
        
        # Get top contributing features (positive and negative)
        positive_features = {k: v for k, v in feature_effects.items() if v > 0}
        negative_features = {k: v for k, v in feature_effects.items() if v < 0}
        
        # Sort by absolute contribution
        top_positive = sorted(positive_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_negative = sorted(negative_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Create human-readable explanation
        explanation_summary = f"Prediction: {predicted_label} (Class {predicted_class})\n"
        explanation_summary += f"Confidence: {prediction_probabilities[predicted_class]:.3f}\n\n"
        
        if top_positive:
            explanation_summary += "Top features supporting this prediction:\n"
            for feature, contribution in top_positive:
                explanation_summary += f"  + {feature}: {contribution:.4f}\n"
        
        if top_negative:
            explanation_summary += "\nTop features opposing this prediction:\n"
            for feature, contribution in top_negative:
                explanation_summary += f"  - {feature}: {contribution:.4f}\n"
        
        # Create comprehensive explanation dictionary
        explanation = {
            'prediction_details': {
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'class_probabilities': {
                    class_labels.get(i, f'Class_{i}'): prob 
                    for i, prob in enumerate(prediction_probabilities)
                },
                'confidence': prediction_probabilities[predicted_class]
            },
            'feature_contributions': feature_effects,
            'top_features': {
                'positive': dict(top_positive),
                'negative': dict(top_negative)
            },
            'explanation_summary': explanation_summary,
            'raw_shap_data': shap_explanation
        }
        
        # Print explanation summary
        print(f"\n[EXPLAIN] SHAP Explanation for Latest Prediction:")
        print(explanation_summary)
        
        return explanation

    def create_shap_plots(self, features: pd.DataFrame, save_path=None):
        """
        Create SHAP visualization plots for the latest prediction.
        
        Args:
            features: DataFrame containing features for prediction
            save_path: Optional path to save plots (if None, uses default location with timestamp)
            
        Returns:
            Dictionary containing plot file paths
        """
        if self.model is None:
            print("[PLOTS] Model not trained or loaded.")
            return None
        
        # Get SHAP explanation first
        explanation = self.explain_prediction(features)
        
        if explanation is None:
            print("[PLOTS] No explanation available for plotting.")
            return None
        
        # Create plots using the utility function
        plots = xgboost_utils.create_shap_plots(explanation['raw_shap_data'], save_path)
        
        if plots:
            print(f"[PLOTS] Created SHAP plots:")
            for plot_type, plot_path in plots.items():
                print(f"  - {plot_type}: {plot_path}")
        
        return plots

    def validate_signal(self, signal: float, features: pd.DataFrame) -> bool:
        # Use base technical validation and add any additional checks if needed
        return self.validate_technical_signal(signal, features)

    # TODO: Integrate with risk management and backtesting modules
    # TODO: Add unit tests and documentation 