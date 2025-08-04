import xgboost as xgb
from sklearn.metrics import classification_report, precision_recall_fscore_support
import logging
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize, MinMaxScaler
import joblib

class XGBoostHyperoptTrainer:
    """Advanced XGBoost trainer with memory management and comprehensive logging"""
    
    def __init__(self, config, data_processor=None):
        self.config = config
        self.data_processor = data_processor
        self.model = None
        self.best_model = None
        self.best_score = 0.0
        self.scaler = MinMaxScaler()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Training history
        self.train_scores = []
        self.val_scores = []
        self.feature_importance = None
    
    def train_with_smote(self, X_train, y_train, X_val, y_val):
        """Train XGBoost with SMOTE for class balance"""
        print("[TRAINER] Starting XGBoost training with SMOTE...")
        
        # Memory management if data processor is available
        if self.data_processor:
            self.data_processor.manage_memory()
        
        # Scale features
        print("[TRAINER] Scaling features with MinMaxScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Apply ENHANCED minority-focused oversampling
        print("[TRAINER] Applying ENHANCED minority-focused oversampling...")
        
        # Get minority oversampling parameters
        use_minority_oversampling = getattr(self.config, 'use_minority_oversampling', True)
        minority_oversampling_ratio = getattr(self.config, 'minority_oversampling_ratio', 3.0)
        
        if use_minority_oversampling:
            # Enhanced SMOTE with minority focus
            smote = SMOTE(random_state=self.config.random_state, k_neighbors=3, sampling_strategy='auto')
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            # Additional minority oversampling for classes 0 and 2
            minority_classes = [0, 2]  # Down and Up classes
            for minority_class in minority_classes:
                minority_indices = np.where(y_train_balanced == minority_class)[0]
                if len(minority_indices) > 0:
                    # Create additional synthetic samples for minority classes
                    additional_samples = int(len(minority_indices) * (minority_oversampling_ratio - 1))
                    if additional_samples > 0:
                        # Duplicate minority samples with slight noise
                        minority_features = X_train_balanced[minority_indices]
                        minority_labels = y_train_balanced[minority_indices]
                        
                        # Ensure we don't try to create more samples than we have
                        samples_to_use = min(additional_samples, len(minority_features))
                        
                        # Add noise to create additional samples
                        noise = np.random.normal(0, 0.01, (samples_to_use, minority_features.shape[1]))
                        additional_features = minority_features[:samples_to_use] + noise
                        additional_labels = minority_labels[:samples_to_use]
                        
                        # Add to balanced dataset
                        X_train_balanced = np.vstack([X_train_balanced, additional_features])
                        y_train_balanced = np.hstack([y_train_balanced, additional_labels])
            
            print(f"[TRAINER] Enhanced minority oversampling with ratio: {minority_oversampling_ratio}")
        else:
            # Standard SMOTE
            smote = SMOTE(random_state=self.config.random_state, k_neighbors=3)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Log class distribution
        unique_before, counts_before = np.unique(y_train, return_counts=True)
        unique_after, counts_after = np.unique(y_train_balanced, return_counts=True)
        print(f"[TRAINER] Class distribution before oversampling: {dict(zip(unique_before, counts_before))}")
        print(f"[TRAINER] Class distribution after ENHANCED oversampling: {dict(zip(unique_after, counts_after))}")
        
        # Log minority class focus
        minority_0_count = counts_after[0] if 0 in unique_after else 0
        minority_2_count = counts_after[2] if 2 in unique_after else 0
        majority_1_count = counts_after[1] if 1 in unique_after else 0
        
        print(f"[TRAINER] Minority class focus - Class 0 (Down): {minority_0_count}, Class 2 (Up): {minority_2_count}")
        print(f"[TRAINER] Majority class - Class 1 (Hold): {majority_1_count}")
        print(f"[TRAINER] Minority/Majority ratio: {(minority_0_count + minority_2_count) / majority_1_count:.2f}")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_balanced, label=y_train_balanced)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)
        
        # Get XGBoost parameters
        params = self.config.to_xgboost_params()
        
        # Handle focal loss if enabled
        if self.config.use_focal_loss:
            from XGBoostMean.xgboost_utils import focal_loss_objective, focal_loss_eval
            
            # Create wrapper functions with config parameters
            def focal_objective_wrapper(predt, dtrain):
                class_multipliers = [self.config.class_multiplier_0, 
                                   self.config.class_multiplier_1, 
                                   self.config.class_multiplier_2]
                return focal_loss_objective(predt, dtrain, self.config.focal_alpha, 
                                         self.config.focal_gamma, class_multipliers)
            
            def focal_eval_wrapper(predt, dtrain):
                class_multipliers = [self.config.class_multiplier_0, 
                                   self.config.class_multiplier_1, 
                                   self.config.class_multiplier_2]
                return focal_loss_eval(predt, dtrain, self.config.focal_alpha, 
                                    self.config.focal_gamma, class_multipliers)
            
            print(f"[TRAINER] Using ENHANCED focal loss with alpha={self.config.focal_alpha}, gamma={self.config.focal_gamma}")
            print(f"[TRAINER] ENHANCED class multipliers for minority focus:")
            print(f"[TRAINER]   Class 0 (Down): {self.config.class_multiplier_0}x weight - HIGH EMPHASIS")
            print(f"[TRAINER]   Class 1 (Hold): {self.config.class_multiplier_1}x weight - REDUCED (dominates)")
            print(f"[TRAINER]   Class 2 (Up): {self.config.class_multiplier_2}x weight - HIGH EMPHASIS")
            print(f"[TRAINER] Minority/Majority weight ratio: {(self.config.class_multiplier_0 + self.config.class_multiplier_2) / self.config.class_multiplier_1:.1f}x")
            
            # Train with custom objective only (evaluation will use built-in metric)
            evals = [(dtrain, 'train'), (dval, 'val')]
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.config.n_estimators,
                evals=evals,
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose_eval=False,
                obj=focal_objective_wrapper
            )
        else:
            # Train with standard objective
            evals = [(dtrain, 'train'), (dval, 'val')]
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.config.n_estimators,
                evals=evals,
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose_eval=False
            )
        
        # Get best score and feature importance
        self.best_score = self.model.best_score
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        print(f"[TRAINER] Training completed. Best validation score: {self.best_score:.4f}")
        print(f"[TRAINER] Best iteration: {self.model.best_iteration}")
        
        return {
            'best_score': self.best_score,
            'best_iteration': self.model.best_iteration,
            'class_distribution_before': dict(zip(unique_before, counts_before)),
            'class_distribution_after': dict(zip(unique_after, counts_after)),
            'feature_importance': self.feature_importance
        }
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("[TRAINER] Evaluating model...")
        
        # Scale test data using fitted scaler
        X_test_scaled = self.scaler.transform(X_test)
        
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)
        
        # Get predictions
        probabilities = self.model.predict(dtest)
        predictions = np.argmax(probabilities, axis=1)
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, predictions, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        # Calculate log loss correctly for multi-class
        # Convert y_test to one-hot encoding for log loss calculation
        y_test_onehot = label_binarize(y_test, classes=[0, 1, 2])
        
        # Calculate log loss
        log_loss = -np.sum(y_test_onehot * np.log(probabilities + 1e-15)) / len(y_test)
        
        # Create classification report
        class_names = ['Down', 'Hold', 'Up']
        report = classification_report(
            y_test, predictions, 
            target_names=class_names,
            output_dict=True,
            zero_division=0,
            labels=[0, 1, 2]  # Explicitly specify all possible labels
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'log_loss': log_loss,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': y_test,
            'probabilities': probabilities
        }
    
    def save_scaler(self, path):
        """Save the fitted scaler"""
        joblib.dump(self.scaler, path)
        print(f"[TRAINER] Saved scaler to {path}")
    
    def load_scaler(self, path):
        """Load a fitted scaler"""
        self.scaler = joblib.load(path)
        print(f"[TRAINER] Loaded scaler from {path}") 