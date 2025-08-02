import xgboost as xgb
from sklearn.metrics import classification_report, precision_recall_fscore_support
import logging
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize

class XGBoostHyperoptTrainer:
    """Advanced XGBoost trainer with memory management and comprehensive logging"""
    
    def __init__(self, config, data_processor=None):
        self.config = config
        self.data_processor = data_processor
        self.model = None
        self.best_model = None
        self.best_score = 0.0
        
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
        
        # Apply SMOTE to training data only
        print("[TRAINER] Applying SMOTE for class balance...")
        smote = SMOTE(random_state=self.config.random_state, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Log class distribution
        unique_before, counts_before = np.unique(y_train, return_counts=True)
        unique_after, counts_after = np.unique(y_train_balanced, return_counts=True)
        print(f"[TRAINER] Class distribution before SMOTE: {dict(zip(unique_before, counts_before))}")
        print(f"[TRAINER] Class distribution after SMOTE: {dict(zip(unique_after, counts_after))}")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_balanced, label=y_train_balanced)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Get XGBoost parameters
        params = self.config.to_xgboost_params()
        
        # Train with early stopping
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
        
        dtest = xgb.DMatrix(X_test, label=y_test)
        
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