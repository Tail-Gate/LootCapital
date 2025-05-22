import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime

class DataPreprocessor:
    """
    Handles data preparation for model training, including cleaning, scaling, and validation.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing steps
            cache_dir: Directory to cache preprocessed data and scalers
            version: Version identifier for the preprocessor
        """
        self.config = config or {}
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/preprocessing")
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.scalers: Dict[str, Union[StandardScaler, RobustScaler]] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.feature_stats: Dict[str, Dict] = {}
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def clean_data(
        self,
        data: pd.DataFrame,
        drop_duplicates: bool = True,
        handle_outliers: bool = True,
        outlier_threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Clean the input data by handling duplicates and outliers.
        
        Args:
            data: Input DataFrame
            drop_duplicates: Whether to drop duplicate rows
            handle_outliers: Whether to handle outliers
            outlier_threshold: Number of standard deviations for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        # Drop duplicates if requested
        if drop_duplicates:
            initial_len = len(df)
            df = df.drop_duplicates()
            self.logger.info(f"Dropped {initial_len - len(df)} duplicate rows")
        
        # Handle outliers if requested
        if handle_outliers:
            for col in df.select_dtypes(include=[np.number]).columns:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - outlier_threshold * std
                upper_bound = mean + outlier_threshold * std
                
                # Replace outliers with bounds
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Log outlier handling
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    self.logger.info(f"Handled {outliers} outliers in column {col}")
        
        return df
    
    def scale_features(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        method: str = "standard",
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using the specified method.
        
        Args:
            data: Input DataFrame
            features: List of features to scale (None for all numeric)
            method: Scaling method ("standard" or "robust")
            fit: Whether to fit new scalers or use existing ones
            
        Returns:
            DataFrame with scaled features
        """
        df = data.copy()
        features = features or df.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in features:
            if feature not in df.columns:
                self.logger.warning(f"Feature {feature} not found in data")
                continue
            
            # Get or create scaler
            scaler_key = f"{feature}_{method}"
            if fit or scaler_key not in self.scalers:
                if method == "standard":
                    self.scalers[scaler_key] = StandardScaler()
                elif method == "robust":
                    self.scalers[scaler_key] = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {method}")
                
                # Fit scaler if needed
                if fit:
                    self.scalers[scaler_key].fit(df[[feature]])
            
            # Transform feature
            df[feature] = self.scalers[scaler_key].transform(df[[feature]])
        
        return df
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        strategy: str = "mean",
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: Input DataFrame
            features: List of features to handle (None for all)
            strategy: Imputation strategy ("mean", "median", "most_frequent")
            fit: Whether to fit new imputers or use existing ones
            
        Returns:
            DataFrame with handled missing values
        """
        df = data.copy()
        features = features or df.columns.tolist()
        
        for feature in features:
            if feature not in df.columns:
                self.logger.warning(f"Feature {feature} not found in data")
                continue
            
            # Get or create imputer
            imputer_key = f"{feature}_{strategy}"
            if fit or imputer_key not in self.imputers:
                self.imputers[imputer_key] = SimpleImputer(strategy=strategy)
                
                # Fit imputer if needed
                if fit:
                    self.imputers[imputer_key].fit(df[[feature]])
            
            # Transform feature
            df[feature] = self.imputers[imputer_key].transform(df[[feature]])
        
        return df
    
    def compute_feature_stats(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Compute statistics for each feature.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary of feature statistics
        """
        stats = {}
        for col in data.columns:
            if data[col].dtype in [np.number, np.float64, np.int64]:
                stats[col] = {
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "missing": data[col].isnull().sum(),
                    "unique": data[col].nunique()
                }
            else:
                stats[col] = {
                    "missing": data[col].isnull().sum(),
                    "unique": data[col].nunique(),
                    "most_common": data[col].value_counts().head(1).to_dict()
                }
        
        self.feature_stats = stats
        return stats
    
    def save_state(self, path: Optional[str] = None) -> None:
        """
        Save the preprocessor state (scalers, imputers, stats).
        
        Args:
            path: Path to save state (defaults to cache directory)
        """
        path = Path(path) if path else self.cache_dir / f"preprocessor_state_{self.version}"
        path.mkdir(parents=True, exist_ok=True)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, path / f"scaler_{name}.joblib")
        
        # Save imputers
        for name, imputer in self.imputers.items():
            joblib.dump(imputer, path / f"imputer_{name}.joblib")
        
        # Save feature stats
        with open(path / "feature_stats.json", "w") as f:
            json.dump(self.feature_stats, f, indent=2)
        
        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Saved preprocessor state to {path}")
    
    def load_state(self, path: str) -> None:
        """
        Load the preprocessor state.
        
        Args:
            path: Path to load state from
        """
        path = Path(path)
        
        # Load scalers
        for scaler_file in path.glob("scaler_*.joblib"):
            name = scaler_file.stem.replace("scaler_", "")
            self.scalers[name] = joblib.load(scaler_file)
        
        # Load imputers
        for imputer_file in path.glob("imputer_*.joblib"):
            name = imputer_file.stem.replace("imputer_", "")
            self.imputers[name] = joblib.load(imputer_file)
        
        # Load feature stats
        with open(path / "feature_stats.json", "r") as f:
            self.feature_stats = json.load(f)
        
        # Load config
        with open(path / "config.json", "r") as f:
            self.config = json.load(f)
        
        self.logger.info(f"Loaded preprocessor state from {path}")
    
    def validate_data(
        self,
        data: pd.DataFrame,
        required_features: Optional[List[str]] = None,
        feature_types: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate the input data against requirements.
        
        Args:
            data: Input DataFrame
            required_features: List of required features
            feature_types: Dictionary of feature names and expected types
            
        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []
        
        # Check required features
        if required_features:
            missing_features = set(required_features) - set(data.columns)
            if missing_features:
                errors.append(f"Missing required features: {missing_features}")
        
        # Check feature types
        if feature_types:
            for feature, expected_type in feature_types.items():
                if feature in data.columns:
                    if expected_type == "numeric":
                        if not pd.api.types.is_numeric_dtype(data[feature]):
                            errors.append(f"Feature {feature} should be numeric")
                    elif expected_type == "categorical":
                        if not pd.api.types.is_categorical_dtype(data[feature]):
                            errors.append(f"Feature {feature} should be categorical")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            errors.append(f"Features with missing values: {missing_values[missing_values > 0].to_dict()}")
        
        return len(errors) == 0, errors
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the preprocessor.
        """
        try:
            # Clear scalers
            self.scalers.clear()
            
            # Clear imputers
            self.imputers.clear()
            
            # Clear feature stats
            self.feature_stats.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error during preprocessor cleanup: {str(e)}")
            raise 