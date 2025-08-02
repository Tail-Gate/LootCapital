import logging
import os
from datetime import datetime
from typing import Optional

def setup_xgboost_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Setup logging configuration for XGBoost module.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file
        log_format: Format string for log messages
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("XGBoost")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_xgboost_logger() -> logging.Logger:
    """
    Get the XGBoost logger instance.
    If not configured, creates a default configuration.
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger("XGBoost")
    
    # If logger has no handlers, setup default configuration
    if not logger.handlers:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/xgboost_{timestamp}.log"
        setup_xgboost_logging(log_file=log_file)
    
    return logger

def log_trial_start(trial_number: int, params: dict) -> None:
    """Log the start of a new trial with parameters"""
    logger = get_xgboost_logger()
    logger.info(f"🚀 NEW XGBOOST TRIAL STARTING - Trial {trial_number}")
    logger.info(f"Parameters: {params}")

def log_trial_complete(trial_number: int, objective_value: float, metrics: dict) -> None:
    """Log the completion of a trial with results"""
    logger = get_xgboost_logger()
    logger.info(f"✅ TRIAL {trial_number} COMPLETED - Objective: {objective_value:.4f}")
    logger.info(f"Metrics: {metrics}")

def log_trial_pruned(trial_number: int, reason: str) -> None:
    """Log when a trial is pruned"""
    logger = get_xgboost_logger()
    logger.warning(f"❌ TRIAL {trial_number} PRUNED - Reason: {reason}")

def log_memory_usage(memory_mb: float) -> None:
    """Log memory usage"""
    logger = get_xgboost_logger()
    logger.debug(f"[MEMORY] CPU Memory usage: {memory_mb:.1f} MB")
    
    if memory_mb > 1000:  # Warning at 1GB
        logger.warning(f"[MEMORY] High memory usage detected: {memory_mb:.1f} MB")

def log_training_progress(epoch: int, train_score: float, val_score: float) -> None:
    """Log training progress"""
    logger = get_xgboost_logger()
    logger.info(f"[TRAINING] Epoch {epoch}: Train={train_score:.4f}, Val={val_score:.4f}")

def log_feature_importance(importance_dict: dict, top_n: int = 10) -> None:
    """Log feature importance"""
    logger = get_xgboost_logger()
    logger.info("[FEATURE IMPORTANCE] Top features:")
    
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_importance[:top_n]):
        logger.info(f"  {i+1}. {feature}: {score:.4f}")

def log_evaluation_results(results: dict) -> None:
    """Log evaluation results"""
    logger = get_xgboost_logger()
    logger.info("[EVALUATION] Model Performance:")
    logger.info(f"  Precision: {results.get('precision', [])}")
    logger.info(f"  Recall: {results.get('recall', [])}")
    logger.info(f"  F1 Score: {results.get('f1', [])}")
    logger.info(f"  Log Loss: {results.get('log_loss', 0):.4f}")

def log_optimization_summary(total_trials: int, best_objective: float, best_params: dict) -> None:
    """Log optimization summary"""
    logger = get_xgboost_logger()
    logger.info("=" * 60)
    logger.info("XGBOOST MEAN REVERSION OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total trials: {total_trials}")
    logger.info(f"Best objective: {best_objective:.4f}")
    logger.info(f"Best parameters: {best_params}")
    logger.info("=" * 60) 