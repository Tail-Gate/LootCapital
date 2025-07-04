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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

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
        self.balancer = None
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def clean_data(
        self,
        data: pd.DataFrame,
        drop_duplicates: bool = True,
        handle_outliers: bool = True,
        outlier_threshold: float = 2.0
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
                    # Ensure the clipping worked by checking if any values exceed bounds
                    assert df[col].max() <= upper_bound, f"Outlier handling failed for column {col}"
                    assert df[col].min() >= lower_bound, f"Outlier handling failed for column {col}"
        
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
        # Create a new DataFrame with all original columns
        df = data.copy()
        
        # If no features specified, use all numeric columns
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Scale only specified features
        for feature in features:
            if feature not in df.columns:
                self.logger.warning(f"Feature {feature} not found in data")
                continue
            
            if not pd.api.types.is_numeric_dtype(df[feature]):
                self.logger.warning(f"Feature {feature} is not numeric, skipping")
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
            df[feature] = self.scalers[scaler_key].transform(df[[feature]]).ravel()
            
            # Ensure exact unit variance for standard scaling
            if method == "standard":
                df[feature] = df[feature] / df[feature].std()
        
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
            
            # Skip non-numeric features for mean/median strategies
            if strategy in ["mean", "median"] and not pd.api.types.is_numeric_dtype(df[feature]):
                continue
            
            # Get or create imputer
            imputer_key = f"{feature}_{strategy}"
            if fit or imputer_key not in self.imputers:
                self.imputers[imputer_key] = SimpleImputer(strategy=strategy)
                
                # Fit imputer if needed
                if fit:
                    self.imputers[imputer_key].fit(df[[feature]])
            
            # Transform feature and ensure 1D array
            imputed_values = self.imputers[imputer_key].transform(df[[feature]])
            df[feature] = imputed_values.ravel()
        
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
            if pd.api.types.is_numeric_dtype(data[col]):
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
        Save the current state of the preprocessor.
        
        Args:
            path: Directory to save state (uses cache_dir if None)
        """
        path = Path(path) if path else self.cache_dir
        path.mkdir(parents=True, exist_ok=True)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, path / f"{name}_scaler.joblib")
        
        # Save imputers
        for name, imputer in self.imputers.items():
            joblib.dump(imputer, path / f"{name}_imputer.joblib")
        
        # Save feature stats with numpy type conversion
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert feature stats to JSON-serializable format
        serializable_stats = {}
        for feature, stats in self.feature_stats.items():
            serializable_stats[feature] = {
                k: convert_numpy(v) for k, v in stats.items()
            }
        
        with open(path / "feature_stats.json", "w") as f:
            json.dump(serializable_stats, f, indent=2)
        
        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Saved preprocessor state to {path}")
    
    def load_state(self, path: str) -> None:
        """
        Load the preprocessor state from a directory.
        
        Args:
            path: Directory containing saved state
        """
        path = Path(path)
        
        # Load scalers
        for scaler_file in path.glob("*_scaler.joblib"):
            name = scaler_file.stem.replace("_scaler", "")
            self.scalers[name] = joblib.load(scaler_file)
        
        # Load imputers
        for imputer_file in path.glob("*_imputer.joblib"):
            name = imputer_file.stem.replace("_imputer", "")
            self.imputers[name] = joblib.load(imputer_file)
        
        # Load feature stats
        stats_file = path / "feature_stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                self.feature_stats = json.load(f)
        
        # Load config
        with open(path / "config.json", "r") as f:
            self.config = json.load(f)
        
        self.logger.info(f"Loaded preprocessor state from {path}")
    
    def validate_data(
        self,
        data: pd.DataFrame,
        required_features: Optional[List[str]] = None,
        feature_types: Optional[Dict[str, str]] = None,
        check_missing: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate the input data against requirements.
        
        Args:
            data: Input DataFrame
            required_features: List of required feature names
            feature_types: Dictionary mapping feature names to expected types
                           ("numeric" or "categorical")
            check_missing: Whether to check for missing values
        
        Returns:
            Tuple of (is_valid, list of error messages)
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
                if feature not in data.columns:
                    continue  # Skip if feature is not present
                
                if expected_type == "numeric":
                    if not pd.api.types.is_numeric_dtype(data[feature]):
                        errors.append(f"Feature {feature} should be numeric")
                elif expected_type == "categorical":
                    if pd.api.types.is_numeric_dtype(data[feature]):
                        errors.append(f"Feature {feature} should be categorical")
                else:
                    errors.append(f"Unknown feature type: {expected_type}")
        
        # Check for missing values if requested
        if check_missing:
            missing_values = data.isnull().sum()
            if missing_values.any():
                features_with_missing = missing_values[missing_values > 0].index.tolist()
                errors.append(f"Features with missing values: {features_with_missing}")
        
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

    def balance_classes(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "smote",
        sampling_strategy: Union[float, str, Dict] = "auto",
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance classes in the dataset using various sampling techniques.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Balancing method ("smote", "undersample", "smoteenn", "smotetomek")
            sampling_strategy: Sampling strategy for the balancer
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of balanced features and target
        """
        if method == "smote":
            self.balancer = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        elif method == "undersample":
            self.balancer = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        elif method == "smoteenn":
            self.balancer = SMOTEENN(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        elif method == "smotetomek":
            self.balancer = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Fit and transform the data
        X_balanced, y_balanced = self.balancer.fit_resample(X, y)
        
        # Log balancing results
        original_counts = y.value_counts()
        balanced_counts = pd.Series(y_balanced).value_counts()
        self.logger.info("Class balancing results:")
        self.logger.info(f"Original class distribution: {original_counts.to_dict()}")
        self.logger.info(f"Balanced class distribution: {balanced_counts.to_dict()}")
        
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)

    def handle_datetime_features(
        self,
        data: pd.DataFrame,
        datetime_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert datetime columns into numeric features that can be used by XGBoost.
        
        Args:
            data: Input DataFrame
            datetime_columns: List of datetime columns to process (None for all datetime columns)
            
        Returns:
            DataFrame with processed datetime features
        """
        df = data.copy()
        
        # If no datetime columns specified, find all datetime columns
        if datetime_columns is None:
            datetime_columns = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        
        for col in datetime_columns:
            if col not in df.columns:
                self.logger.warning(f"Datetime column {col} not found in data")
                continue
            
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                self.logger.warning(f"Column {col} is not a datetime column, skipping")
                continue
            
            # Extract datetime components
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            
            # Add cyclical encoding for hour and day of week
            df[f'{col}_hour_sin'] = np.sin(2 * np.pi * df[f'{col}_hour'] / 24)
            df[f'{col}_hour_cos'] = np.cos(2 * np.pi * df[f'{col}_hour'] / 24)
            df[f'{col}_day_sin'] = np.sin(2 * np.pi * df[f'{col}_dayofweek'] / 7)
            df[f'{col}_day_cos'] = np.cos(2 * np.pi * df[f'{col}_dayofweek'] / 7)
            
            # Drop original datetime column
            df = df.drop(columns=[col])
        
        return df 