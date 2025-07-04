import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

class ClassBalancer:
    """
    Handles label distribution analysis and class balancing techniques for trading data.
    Supports both binary and multiclass labels, with various balancing methods.
    """
    def __init__(
        self,
        method: str = 'smote',
        random_state: int = 42,
        sampling_strategy: Optional[Union[float, str, Dict]] = 'auto'
    ):
        """
        Args:
            method: Balancing method ('smote', 'undersample', 'oversample', or 'none')
            random_state: Random seed for reproducibility
            sampling_strategy: Strategy for sampling (see imblearn documentation)
        """
        self.method = method
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self._validate_method()

    def _validate_method(self):
        """Validate the balancing method."""
        valid_methods = ['smote', 'undersample', 'oversample', 'none']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def analyze_distribution(self, labels: pd.Series) -> Dict:
        """
        Analyze the distribution of labels.
        
        Args:
            labels: Series of labels
            
        Returns:
            Dict containing distribution statistics
        """
        counts = Counter(labels)
        total = len(labels)
        
        # Calculate basic statistics
        stats = {
            'counts': counts,
            'total_samples': total,
            'class_ratios': {k: v/total for k, v in counts.items()},
            'imbalance_ratio': max(counts.values()) / min(counts.values()) if len(counts) > 1 else 1.0
        }
        
        # Add additional metrics for binary classification
        if len(counts) == 2:
            minority_class = min(counts.items(), key=lambda x: x[1])[0]
            majority_class = max(counts.items(), key=lambda x: x[1])[0]
            stats.update({
                'minority_class': minority_class,
                'majority_class': majority_class,
                'minority_ratio': counts[minority_class] / total
            })
        
        return stats

    def balance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance the dataset using the specified method.
        
        Args:
            X: Feature DataFrame
            y: Label Series
            
        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        if self.method == 'none':
            return X, y
            
        # Convert to numpy arrays for imblearn
        X_np = X.values
        y_np = y.values
        
        if self.method == 'smote':
            balancer = SMOTE(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy
            )
        elif self.method == 'undersample':
            balancer = RandomUnderSampler(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy
            )
        elif self.method == 'oversample':
            # Implement oversampling using resample
            X_balanced = []
            y_balanced = []
            
            # Get unique classes
            classes = np.unique(y_np)
            
            # Find the majority class size
            majority_size = max(Counter(y_np).values())
            
            for cls in classes:
                # Get samples for this class
                X_cls = X_np[y_np == cls]
                y_cls = y_np[y_np == cls]
                
                if len(X_cls) < majority_size:
                    # Oversample minority class
                    X_resampled, y_resampled = resample(
                        X_cls,
                        y_cls,
                        n_samples=majority_size,
                        random_state=self.random_state
                    )
                else:
                    X_resampled, y_resampled = X_cls, y_cls
                
                X_balanced.append(X_resampled)
                y_balanced.append(y_resampled)
            
            X_balanced = np.vstack(X_balanced)
            y_balanced = np.concatenate(y_balanced)
            
            # Shuffle the balanced dataset
            indices = np.random.permutation(len(X_balanced))
            return pd.DataFrame(X_balanced[indices], columns=X.columns), pd.Series(y_balanced[indices])
        
        # Apply balancing
        X_balanced, y_balanced = balancer.fit_resample(X_np, y_np)
        
        # Convert back to pandas
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)

    def get_balancing_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        balanced_X: Optional[pd.DataFrame] = None,
        balanced_y: Optional[pd.Series] = None
    ) -> Dict:
        """
        Generate a comprehensive report of the balancing process.
        
        Args:
            X: Original feature DataFrame
            y: Original label Series
            balanced_X: Balanced feature DataFrame (if available)
            balanced_y: Balanced label Series (if available)
            
        Returns:
            Dict containing balancing statistics and metrics
        """
        report = {
            'original_distribution': self.analyze_distribution(y),
            'balancing_method': self.method,
            'feature_count': X.shape[1],
            'original_sample_count': len(X)
        }
        
        if balanced_X is not None and balanced_y is not None:
            report.update({
                'balanced_distribution': self.analyze_distribution(balanced_y),
                'balanced_sample_count': len(balanced_X),
                'samples_added': len(balanced_X) - len(X)
            })
        
        return report 