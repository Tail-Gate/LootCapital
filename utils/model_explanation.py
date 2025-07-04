import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import matplotlib.pyplot as plt
from pathlib import Path

class ModelExplainer:
    """
    Provides model explanation capabilities using SHAP values.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize the model explainer.
        
        Args:
            model: Trained model (XGBoost, LSTM, etc.)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.logger = logging.getLogger(__name__)
    
    def fit_explainer(self, X: pd.DataFrame, model_type: str = "xgboost") -> None:
        """
        Fit the SHAP explainer to the data.
        
        Args:
            X: Feature DataFrame
            model_type: Type of model ("xgboost", "lstm", etc.)
        """
        try:
            if model_type == "xgboost":
                self.explainer = shap.TreeExplainer(self.model)
            elif model_type in ["lstm", "gru", "transformer"]:
                # For deep learning models, use DeepExplainer
                background = shap.sample(X, 100)  # Use 100 samples for background
                self.explainer = shap.DeepExplainer(self.model, background)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            self.logger.error(f"Error fitting explainer: {str(e)}")
            raise
    
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get SHAP values for the given data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        try:
            shap_values = self.explainer.shap_values(X)
            return shap_values
        except Exception as e:
            self.logger.error(f"Error getting SHAP values: {str(e)}")
            raise
    
    def plot_summary(self, X: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            X: Feature DataFrame
            output_path: Optional path to save the plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        try:
            shap_values = self.get_shap_values(X)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X,
                feature_names=self.feature_names,
                show=False
            )
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting summary: {str(e)}")
            raise
    
    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot SHAP dependence plot for a specific feature.
        
        Args:
            X: Feature DataFrame
            feature: Feature to plot
            output_path: Optional path to save the plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        try:
            shap_values = self.get_shap_values(X)
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature,
                shap_values,
                X,
                feature_names=self.feature_names,
                show=False
            )
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting dependence: {str(e)}")
            raise
    
    def get_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        try:
            shap_values = self.get_shap_values(X)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            importance_dict = dict(zip(self.feature_names, mean_abs_shap))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            raise
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        instance_idx: int,
        output_path: Optional[str] = None
    ) -> None:
        """
        Explain a single prediction using SHAP force plot.
        
        Args:
            X: Feature DataFrame
            instance_idx: Index of the instance to explain
            output_path: Optional path to save the plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        try:
            shap_values = self.get_shap_values(X.iloc[[instance_idx]])
            
            plt.figure(figsize=(10, 6))
            shap.force_plot(
                self.explainer.expected_value,
                shap_values,
                X.iloc[[instance_idx]],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {str(e)}")
            raise 