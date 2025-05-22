from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

def calculate_trading_metrics(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate trading performance metrics.
    
    Args:
        predictions: Model predictions (1 for long, -1 for short, 0 for no trade)
        actual_returns: Actual returns for each period
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Dictionary of trading metrics
    """
    # Calculate strategy returns
    strategy_returns = predictions * actual_returns
    
    # Calculate metrics
    total_return = np.prod(1 + strategy_returns) - 1
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    
    # Calculate volatility
    volatility = np.std(strategy_returns) * np.sqrt(252)
    
    # Calculate Sharpe ratio
    excess_returns = strategy_returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    # Calculate Sortino ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252)
    sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (running_max - cumulative_returns) / running_max
    max_drawdown = np.max(drawdowns)
    
    # Calculate win rate
    winning_trades = strategy_returns[strategy_returns > 0]
    total_trades = strategy_returns[strategy_returns != 0]
    win_rate = len(winning_trades) / len(total_trades) if len(total_trades) > 0 else 0
    
    # Calculate profit factor
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

def calculate_statistical_metrics(
    predictions: np.ndarray,
    actual_returns: np.ndarray
) -> Dict[str, float]:
    """
    Calculate statistical performance metrics.
    
    Args:
        predictions: Model predictions (1 for long, -1 for short, 0 for no trade)
        actual_returns: Actual returns for each period
        
    Returns:
        Dictionary of statistical metrics
    """
    # Convert predictions to binary (1 for trade, 0 for no trade)
    trade_signals = np.abs(predictions)
    
    # Calculate accuracy metrics
    accuracy = accuracy_score(trade_signals, np.abs(actual_returns) > 0)
    precision = precision_score(trade_signals, np.abs(actual_returns) > 0, zero_division=0)
    recall = recall_score(trade_signals, np.abs(actual_returns) > 0, zero_division=0)
    f1 = f1_score(trade_signals, np.abs(actual_returns) > 0, zero_division=0)
    
    # Calculate directional accuracy
    directional_accuracy = np.mean(np.sign(predictions) == np.sign(actual_returns))
    
    # Calculate information coefficient (IC)
    ic = np.corrcoef(predictions, actual_returns)[0, 1]
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'directional_accuracy': directional_accuracy,
        'information_coefficient': ic
    }

def perform_walk_forward_analysis(
    model,
    features: pd.DataFrame,
    targets: pd.Series,
    initial_train_size: int,
    step_size: int,
    n_splits: int
) -> Dict[str, List[float]]:
    """
    Perform walk-forward analysis to evaluate model performance over time.
    
    Args:
        model: Trained model with predict method
        features: Feature DataFrame
        targets: Target Series
        initial_train_size: Initial training set size
        step_size: Number of samples to move forward in each step
        n_splits: Number of splits to perform
        
    Returns:
        Dictionary of metrics over time
    """
    # Initialize metrics storage
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'sharpe_ratio': [],
        'sortino_ratio': [],
        'max_drawdown': []
    }
    
    # Perform walk-forward analysis
    for i in range(n_splits):
        # Calculate train and test indices
        train_end = initial_train_size + i * step_size
        test_end = train_end + step_size
        
        if test_end > len(features):
            break
        
        # Split data
        X_train = features.iloc[:train_end]
        y_train = targets.iloc[:train_end]
        X_test = features.iloc[train_end:test_end]
        y_test = targets.iloc[train_end:test_end]
        
        # Train model
        model.train(X_train, y_train)
        
        # Make predictions
        predictions, _ = model.predict(X_test)
        
        # Calculate metrics
        trading_metrics = calculate_trading_metrics(predictions, y_test.values)
        statistical_metrics = calculate_statistical_metrics(predictions, y_test.values)
        
        # Store metrics
        metrics['accuracy'].append(statistical_metrics['accuracy'])
        metrics['precision'].append(statistical_metrics['precision'])
        metrics['recall'].append(statistical_metrics['recall'])
        metrics['f1_score'].append(statistical_metrics['f1_score'])
        metrics['sharpe_ratio'].append(trading_metrics['sharpe_ratio'])
        metrics['sortino_ratio'].append(trading_metrics['sortino_ratio'])
        metrics['max_drawdown'].append(trading_metrics['max_drawdown'])
    
    return metrics

def validate_model(
    model,
    features: pd.DataFrame,
    targets: pd.Series,
    n_splits: int = 5,
    test_size: float = 0.2
) -> Dict[str, Dict[str, float]]:
    """
    Perform comprehensive model validation.
    
    Args:
        model: Trained model with predict method
        features: Feature DataFrame
        targets: Target Series
        n_splits: Number of time series splits
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary of validation results
    """
    # Initialize time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(features) * test_size))
    
    # Initialize metrics storage
    cv_metrics = {
        'trading': [],
        'statistical': []
    }
    
    # Perform cross-validation
    for train_idx, test_idx in tscv.split(features):
        # Split data
        X_train = features.iloc[train_idx]
        y_train = targets.iloc[train_idx]
        X_test = features.iloc[test_idx]
        y_test = targets.iloc[test_idx]
        
        # Train model
        model.train(X_train, y_train)
        
        # Make predictions
        predictions, _ = model.predict(X_test)
        
        # Calculate metrics
        trading_metrics = calculate_trading_metrics(predictions, y_test.values)
        statistical_metrics = calculate_statistical_metrics(predictions, y_test.values)
        
        # Store metrics
        cv_metrics['trading'].append(trading_metrics)
        cv_metrics['statistical'].append(statistical_metrics)
    
    # Calculate average metrics
    avg_trading_metrics = {
        metric: np.mean([m[metric] for m in cv_metrics['trading']])
        for metric in cv_metrics['trading'][0].keys()
    }
    
    avg_statistical_metrics = {
        metric: np.mean([m[metric] for m in cv_metrics['statistical']])
        for metric in cv_metrics['statistical'][0].keys()
    }
    
    # Calculate standard deviations
    std_trading_metrics = {
        metric: np.std([m[metric] for m in cv_metrics['trading']])
        for metric in cv_metrics['trading'][0].keys()
    }
    
    std_statistical_metrics = {
        metric: np.std([m[metric] for m in cv_metrics['statistical']])
        for metric in cv_metrics['statistical'][0].keys()
    }
    
    return {
        'average_metrics': {
            'trading': avg_trading_metrics,
            'statistical': avg_statistical_metrics
        },
        'std_metrics': {
            'trading': std_trading_metrics,
            'statistical': std_statistical_metrics
        }
    } 