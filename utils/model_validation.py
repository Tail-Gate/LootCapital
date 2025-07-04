from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import copy
from itertools import product
import torch

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
    print("\nCalculating trading metrics:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Actual returns shape: {actual_returns.shape}")
    
    # Calculate strategy returns
    strategy_returns = predictions * actual_returns
    print(f"Strategy returns shape: {strategy_returns.shape}")
    print(f"Strategy returns sample: {strategy_returns[:5]}")
    
    # Calculate metrics
    total_return = np.prod(1 + strategy_returns) - 1
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    
    # Calculate volatility
    volatility = np.std(strategy_returns) * np.sqrt(252)
    
    # Calculate excess returns (assuming risk-free rate of 0 for simplicity)
    excess_returns = strategy_returns - risk_free_rate / 252
    print(f"Excess returns mean: {np.mean(excess_returns):.6f}")
    print(f"Excess returns std: {np.std(excess_returns):.6f}")
    
    # Calculate Sharpe ratio
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0
    print(f"Sharpe ratio: {sharpe_ratio:.6f}")
    
    # Calculate Sortino ratio
    downside_returns = excess_returns[excess_returns < 0]
    print(f"Number of downside returns: {len(downside_returns)}")
    print(f"Downside returns sample: {downside_returns[:5] if len(downside_returns) > 0 else 'No downside returns'}")
    
    downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1
    print(f"Downside standard deviation: {downside_std:.6f}")
    
    if downside_std == 0:
        print("Warning: Downside standard deviation is zero, setting Sortino ratio to 0")
        sortino_ratio = 0
    else:
        sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252)
    print(f"Sortino ratio: {sortino_ratio:.6f}")
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (running_max - cumulative_returns) / running_max
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    print(f"Maximum drawdown: {max_drawdown:.6f}")
    
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
    trade_signals = (np.abs(predictions) > 0).astype(int)  # Convert to binary
    actual_trades = (np.abs(actual_returns) > 0).astype(int)  # Convert to binary
    
    # Check if we have any trades
    if np.sum(trade_signals) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'directional_accuracy': 0.0,
            'information_coefficient': 0.0
        }
    
    # Calculate accuracy metrics
    accuracy = accuracy_score(trade_signals, actual_trades)
    precision = precision_score(trade_signals, actual_trades, zero_division=0)
    recall = recall_score(trade_signals, actual_trades, zero_division=0)
    f1 = f1_score(trade_signals, actual_trades, zero_division=0)
    
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
    model_class,
    model_config_base,
    features: pd.DataFrame,
    targets: pd.Series,
    initial_train_size: int,
    step_size: int,
    n_splits_outer: int,
    hyperparameter_grid: Dict,
    n_splits_inner: int = 3
) -> Dict[str, List[float]]:
    """
    Perform nested walk-forward analysis with time series cross-validation.
    """
    print("DEBUG: Starting walk-forward analysis")
    
    # Initialize metrics storage
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'sharpe_ratio': [],
        'sortino_ratio': [],
        'max_drawdown': [],
        'best_params': []
    }
    
    # Calculate split points
    total_size = len(features)
    split_points = [initial_train_size + i * step_size for i in range(n_splits_outer + 1)]
    split_points = [min(x, total_size) for x in split_points]
    
    print(f"\nWalk-forward analysis configuration:")
    print(f"Total data points: {total_size}")
    print(f"Initial train size: {initial_train_size}")
    print(f"Step size: {step_size}")
    print(f"Split points: {split_points}")
    
    # Get sequence length from config
    sequence_length = getattr(model_config_base, 'lstm_sequence_length', 
                            getattr(model_config_base, 'gru_sequence_length', 20))
    
    # Perform outer CV
    for i in range(n_splits_outer):
        print(f"\n--- Outer Fold {i+1}/{n_splits_outer} ---")
        print(f"DEBUG: Starting outer fold {i+1}")
        
        try:
            # Split data
            train_end = split_points[i]
            test_end = split_points[i + 1]
            
            X_train_outer = features.iloc[:train_end]
            y_train_outer = targets.iloc[:train_end]
            X_test_outer = features.iloc[train_end:test_end]
            y_test_outer = targets.iloc[train_end:test_end]
            
            print(f"  Outer Train: {X_train_outer.index[0]} to {X_train_outer.index[-1]} ({len(X_train_outer)} samples)")
            print(f"  Outer Test: {X_test_outer.index[0]} to {X_test_outer.index[-1]} ({len(X_test_outer)} samples)")
            
            # Perform inner loop for hyperparameter optimization
            print("  Performing inner loop for hyperparameter optimization...")
            print("DEBUG: About to start hyperparameter tuning")
            
            try:
                best_params = tune_hyperparameters(
                    model_class=model_class,
                    model_config_base=model_config_base,
                    X_data_for_tuning=X_train_outer,
                    y_data_for_tuning=y_train_outer,
                    hyperparameter_grid=hyperparameter_grid,
                    n_splits_inner=n_splits_inner
                )
                print("DEBUG: Hyperparameter tuning completed")
                print(f"Best parameters found: {best_params}")
            except Exception as e:
                print(f"DEBUG: Error in hyperparameter tuning: {str(e)}")
                print(f"DEBUG: Error details:", e.__class__.__name__)
                import traceback
                print(f"DEBUG: Traceback:\n{traceback.format_exc()}")
                raise
            
            # Train final model with best parameters
            print(f"  Training final model with best parameters: {best_params}")
            print("DEBUG: Starting final model training")
            current_config = copy.deepcopy(model_config_base)
            for param, value in best_params.items():
                setattr(current_config, param, value)
            
            current_model = model_class(current_config)
            current_model.train(X_train_outer, y_train_outer)
            print("DEBUG: Final model training completed")
            
            # Evaluate on test set
            try:
                print("DEBUG: Starting model evaluation")
                # Use strategy-specific validate_model if available
                if hasattr(current_model, 'validate_model'):
                    print("  Using strategy-specific validate_model")
                    metrics_dict = current_model.validate_model(X_test_outer, y_test_outer)
                else:
                    # Fall back to standard evaluation
                    print("  Using standard evaluation")
                    predicted_labels, _ = current_model.predict(X_test_outer)
                    predicted_labels = np.asarray(predicted_labels).flatten()
                    
                    # Get the actual sequence length used in prediction
                    actual_sequence_length = len(X_test_outer) - len(predicted_labels) + 1
                    print(f"  Actual sequence length used: {actual_sequence_length}")
                    
                    # Align targets with predictions using the actual sequence length
                    actual_labels_aligned = y_test_outer.iloc[actual_sequence_length-1:].values
                    
                    print(f"  Prediction shapes:")
                    print(f"    predicted_labels: {predicted_labels.shape}")
                    print(f"    y_test_outer: {y_test_outer.shape}")
                    print(f"    actual_sequence_length: {actual_sequence_length}")
                    print(f"    actual_labels_aligned: {actual_labels_aligned.shape}")
                    
                    if len(predicted_labels) != len(actual_labels_aligned):
                        print(f"  Prediction or alignment mismatch in Outer Fold {i+1}:")
                        print(f"    predicted_labels length: {len(predicted_labels)}")
                        print(f"    actual_labels_aligned length: {len(actual_labels_aligned)}")
                        print(f"    Skipping metrics for this fold.")
                        for key in metrics.keys():
                            if key != 'best_params':
                                metrics[key].append(np.nan)
                        continue
                    
                    trading_metrics = calculate_trading_metrics(predicted_labels, actual_labels_aligned)
                    statistical_metrics = calculate_statistical_metrics(predicted_labels, actual_labels_aligned)
                    metrics_dict = {**trading_metrics, **statistical_metrics}
                
                # Store metrics
                for key in metrics.keys():
                    if key != 'best_params':
                        metrics[key].append(metrics_dict[key])
                metrics['best_params'].append(best_params)
                print("DEBUG: Model evaluation completed")
                
                # Print fold results
                print(f"\nFold {i+1} Results:")
                for metric, value in metrics_dict.items():
                    print(f"  {metric}: {value:.4f}")
                
            except Exception as e:
                print(f"  Error evaluating model in Outer Fold {i+1}: {str(e)}")
                print(f"  Error details:", e.__class__.__name__)
                import traceback
                print(f"  Traceback:\n{traceback.format_exc()}")
                for key in metrics.keys():
                    if key != 'best_params':
                        metrics[key].append(np.nan)
                metrics['best_params'].append(best_params)
            
        except Exception as e:
            print(f"Error in Outer Fold {i+1}: {str(e)}")
            print(f"Error details:", e.__class__.__name__)
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            for key in metrics.keys():
                if key != 'best_params':
                    metrics[key].append(np.nan)
            metrics['best_params'].append(None)
    
    print("DEBUG: Walk-forward analysis completed")
    return metrics


def _prepare_sequence_data_for_prediction(features_df: pd.DataFrame, sequence_length: int) -> torch.Tensor:
    """
    Helper function to prepare sequence data for LSTM prediction.
    This replicates the core logic from LSTMMomentumStrategy._prepare_sequence_data
    for external use.
    """
    feature_array = features_df.values
    X_sequences = []
    if len(feature_array) < sequence_length:
        return torch.FloatTensor([]) # Return empty tensor if not enough data

    for i in range(len(feature_array) - sequence_length + 1):
        X_sequences.append(feature_array[i:(i + sequence_length)])
    
    return torch.FloatTensor(np.array(X_sequences))

def tune_hyperparameters(
    model_class,
    model_config_base,
    X_data_for_tuning: pd.DataFrame,
    y_data_for_tuning: pd.Series,
    hyperparameter_grid: Dict,
    n_splits_inner: int = 3
) -> Dict:
    """
    Perform hyperparameter tuning using nested cross-validation.
    """
    best_score = float('-inf')
    best_params = None
    
    # Get sequence length from config (handle different model types)
    sequence_length = getattr(model_config_base, 'lstm_sequence_length', 
                            getattr(model_config_base, 'gru_sequence_length', 20))
    
    # Check if we have enough data for the inner CV
    if len(X_data_for_tuning) < (sequence_length * (n_splits_inner + 1)):
        print(f"  Not enough data for inner cross-validation (need at least {sequence_length * (n_splits_inner + 1)} samples, have {len(X_data_for_tuning)}).")
        print("  Using all data for a single fold.")
        n_splits_inner = 1
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(hyperparameter_grid.keys(), v)) for v in product(*hyperparameter_grid.values())]
    
    # Evaluate each parameter combination
    for params in param_combinations:
        print(f"  Testing params: {params}")
        
        # Create a copy of the base config and update with current parameters
        temp_config = copy.deepcopy(model_config_base)
        for param, value in params.items():
            setattr(temp_config, param, value)
        
        # Initialize model with current parameters
        current_model = model_class(config=temp_config)
        
        # Perform inner cross-validation
        inner_scores = []
        for i in range(n_splits_inner):
            # Calculate inner train and test indices
            inner_train_end = len(X_data_for_tuning) - (n_splits_inner - i) * (len(X_data_for_tuning) // (n_splits_inner + 1))
            inner_test_start = inner_train_end
            inner_test_end = inner_train_end + (len(X_data_for_tuning) // (n_splits_inner + 1))
            
            # Split data for inner loop
            X_train_inner = X_data_for_tuning.iloc[:inner_train_end]
            y_train_inner = y_data_for_tuning.iloc[:inner_train_end]
            X_test_inner = X_data_for_tuning.iloc[inner_test_start:inner_test_end]
            y_test_inner = y_data_for_tuning.iloc[inner_test_start:inner_test_end]
            
            # Train model
            try:
                current_model.train(X_train_inner, y_train_inner)
            except Exception as e:
                print(f"  Error training model with params {params} in inner fold {i+1}: {e}")
                continue
            
            # Evaluate on test set
            try:
                # Get predictions
                probabilities, _ = current_model.predict(X_test_inner)
                predicted_labels = np.argmax(probabilities, axis=1) - 1  # Convert to -1, 0, 1
                
                # Get the actual sequence length used in prediction
                actual_sequence_length = len(X_test_inner) - len(predicted_labels) + 1
                print(f"  Actual sequence length used: {actual_sequence_length}")
                
                # Align targets with predictions using the actual sequence length
                actual_labels_aligned = y_test_inner.iloc[actual_sequence_length-1:].values
                
                print(f"  Inner fold {i+1} shapes:")
                print(f"    predicted_labels: {predicted_labels.shape}")
                print(f"    y_test_inner: {y_test_inner.shape}")
                print(f"    actual_sequence_length: {actual_sequence_length}")
                print(f"    actual_labels_aligned: {actual_labels_aligned.shape}")
                
                if len(predicted_labels) != len(actual_labels_aligned):
                    print(f"  Prediction or alignment mismatch in inner fold {i+1}. Skipping.")
                    continue
                
                # Calculate metrics
                trading_metrics = calculate_trading_metrics(predicted_labels, actual_labels_aligned)
                statistical_metrics = calculate_statistical_metrics(predicted_labels, actual_labels_aligned)
                
                # Use a combination of metrics as the score
                score = (trading_metrics['sharpe_ratio'] + 
                        trading_metrics['sortino_ratio'] + 
                        statistical_metrics['f1_score']) / 3
                
                inner_scores.append(score)
                
            except Exception as e:
                print(f"  Error evaluating model with params {params} in inner fold {i+1}: {e}")
                continue
        
        # Calculate average score across inner folds
        if inner_scores:
            avg_score = np.mean(inner_scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
    
    if best_params is None:
        print("  No valid parameter combinations found. Using default parameters.")
        best_params = {param: values[0] for param, values in hyperparameter_grid.items()}
    
    return best_params

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

def visualize_walk_forward_results(
    walk_forward_results: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize walk-forward analysis results.
    
    Args:
        walk_forward_results: Dictionary of metrics from walk-forward analysis
        save_path: Optional path to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('default')  # Use default style instead of seaborn
    sns.set_theme()  # This will set seaborn's default theme
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Performance Metrics Over Time
    ax1 = fig.add_subplot(gs[0, :])
    metrics_to_plot = ['balanced_accuracy', 'f1']
    for metric in metrics_to_plot:
        values = walk_forward_results[metric]
        ax1.plot(range(len(values)), values, marker='o', label=metric.replace('_', ' ').title())
    ax1.set_title('Performance Metrics Over Time')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Classification Metrics
    ax2 = fig.add_subplot(gs[1, 0])
    classification_metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [walk_forward_results[metric] for metric in classification_metrics]
    ax2.boxplot(values, labels=[m.replace('_', ' ').title() for m in classification_metrics])
    ax2.set_title('Classification Metrics Distribution')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    
    # 3. Parameter Stability
    ax3 = fig.add_subplot(gs[1, 1])
    best_params = walk_forward_results['best_params']
    param_names = list(best_params[0].keys())
    param_values = {name: [params[name] for params in best_params] for name in param_names}
    
    # Plot parameter values over folds
    for name in param_names:
        values = param_values[name]
        ax3.plot(range(len(values)), values, marker='o', label=name)
    ax3.set_title('Parameter Stability Over Folds')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Parameter Value')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Balanced Accuracy vs F1 Score
    ax4 = fig.add_subplot(gs[2, :])
    balanced_acc = walk_forward_results['balanced_accuracy']
    f1_scores = walk_forward_results['f1']
    ax4.scatter(balanced_acc, f1_scores, c='blue', alpha=0.6)
    ax4.set_title('Balanced Accuracy vs F1 Score')
    ax4.set_xlabel('Balanced Accuracy')
    ax4.set_ylabel('F1 Score')
    ax4.grid(True)
    
    # Add correlation line
    z = np.polyfit(balanced_acc, f1_scores, 1)
    p = np.poly1d(z)
    ax4.plot(balanced_acc, p(balanced_acc), "r--", alpha=0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def visualize_parameter_importance(
    walk_forward_results: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize parameter importance based on performance correlation.
    
    Args:
        walk_forward_results: Dictionary of metrics from walk-forward analysis
        save_path: Optional path to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    # Calculate correlations between parameters and performance
    best_params = walk_forward_results['best_params']
    param_names = list(best_params[0].keys())
    performance = walk_forward_results['balanced_accuracy']  # Use balanced accuracy instead of sharpe ratio
    
    correlations = {}
    for name in param_names:
        param_values = [params[name] for params in best_params]
        correlation = stats.pearsonr(param_values, performance)[0]
        correlations[name] = correlation
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(correlations.keys(), correlations.values())
    plt.title('Parameter Importance (Correlation with Balanced Accuracy)')
    plt.xlabel('Parameter')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def visualize_performance_decomposition(
    walk_forward_results: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize performance decomposition into different components.
    
    Args:
        walk_forward_results: Dictionary of metrics from walk-forward analysis
        save_path: Optional path to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Balanced Accuracy vs F1 Score
    balanced_acc = walk_forward_results['balanced_accuracy']
    f1_scores = walk_forward_results['f1']
    
    ax1.scatter(balanced_acc, f1_scores, c='blue', alpha=0.6)
    ax1.set_title('Balanced Accuracy vs F1 Score')
    ax1.set_xlabel('Balanced Accuracy')
    ax1.set_ylabel('F1 Score')
    ax1.grid(True)
    
    # Add correlation line
    z = np.polyfit(balanced_acc, f1_scores, 1)
    p = np.poly1d(z)
    ax1.plot(balanced_acc, p(balanced_acc), "r--", alpha=0.8)
    
    # 2. Classification Performance
    accuracy = walk_forward_results['accuracy']
    precision = walk_forward_results['precision']
    recall = walk_forward_results['recall']
    f1 = walk_forward_results['f1']
    
    metrics = np.array([accuracy, precision, recall, f1])
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    ax2.boxplot(metrics.T, labels=labels)
    ax2.set_title('Classification Performance Distribution')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    
    plt.show()