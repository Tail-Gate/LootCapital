import optuna
from optuna import TrialPruned
from datetime import datetime, timedelta
import json
import gc
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from XGBoostMean.xgboost_hyperopt_config import XGBoostHyperoptConfig
from XGBoostMean.xgboost_hyperopt_trainer import XGBoostHyperoptTrainer
from XGBoostMean.xgboost_logging import (
    get_xgboost_logger, log_trial_start, log_trial_complete, 
    log_trial_pruned, log_memory_usage, log_evaluation_results
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_trading_metrics_report(study, output_dir="reports"):
    """
    Create a comprehensive trading metrics report focusing on Sharpe ratio, max drawdown, win rate, and profit factor.
    
    Args:
        study: Optuna study object with completed trials
        output_dir: Directory to save the report
    """
    logger = get_xgboost_logger()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract trading metrics from completed trials
    trading_metrics = []
    for trial in study.trials:
        if trial.value is not None:
            # Extract trading metrics from trial user attributes
            if hasattr(trial, 'user_attrs'):
                metrics = {
                    'trial_number': trial.number,
                    'objective_value': trial.value,
                    'sharpe_ratio': trial.user_attrs.get('sharpe_ratio', 0.0),
                    'win_rate': trial.user_attrs.get('win_rate', 0.0),
                    'profit_factor': trial.user_attrs.get('profit_factor', 0.0),
                    'max_drawdown': trial.user_attrs.get('max_drawdown', 0.0),
                    'parameters': trial.params
                }
                trading_metrics.append(metrics)
    
    if not trading_metrics:
        logger.warning("No completed trials with trading metrics found")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(trading_metrics)
    
    # Calculate summary statistics
    summary = {
        'total_trials': len(trading_metrics),
        'best_sharpe_ratio': df['sharpe_ratio'].max(),
        'worst_sharpe_ratio': df['sharpe_ratio'].min(),
        'mean_sharpe_ratio': df['sharpe_ratio'].mean(),
        'std_sharpe_ratio': df['sharpe_ratio'].std(),
        'best_win_rate': df['win_rate'].max(),
        'mean_win_rate': df['win_rate'].mean(),
        'best_profit_factor': df['profit_factor'].max(),
        'mean_profit_factor': df['profit_factor'].mean(),
        'worst_max_drawdown': df['max_drawdown'].min(),
        'mean_max_drawdown': df['max_drawdown'].mean(),
        'best_objective': df['objective_value'].min(),
        'mean_objective': df['objective_value'].mean()
    }
    
    # Find best trial for each metric
    best_sharpe_trial = df.loc[df['sharpe_ratio'].idxmax()]
    best_win_rate_trial = df.loc[df['win_rate'].idxmax()]
    best_profit_factor_trial = df.loc[df['profit_factor'].idxmax()]
    best_objective_trial = df.loc[df['objective_value'].idxmin()]
    
    # Create detailed report
    report = {
        'summary_statistics': summary,
        'best_trials': {
            'best_sharpe_ratio': {
                'trial_number': int(best_sharpe_trial['trial_number']),
                'sharpe_ratio': float(best_sharpe_trial['sharpe_ratio']),
                'win_rate': float(best_sharpe_trial['win_rate']),
                'profit_factor': float(best_sharpe_trial['profit_factor']),
                'max_drawdown': float(best_sharpe_trial['max_drawdown']),
                'objective_value': float(best_sharpe_trial['objective_value'])
            },
            'best_win_rate': {
                'trial_number': int(best_win_rate_trial['trial_number']),
                'sharpe_ratio': float(best_win_rate_trial['sharpe_ratio']),
                'win_rate': float(best_win_rate_trial['win_rate']),
                'profit_factor': float(best_win_rate_trial['profit_factor']),
                'max_drawdown': float(best_win_rate_trial['max_drawdown']),
                'objective_value': float(best_win_rate_trial['objective_value'])
            },
            'best_profit_factor': {
                'trial_number': int(best_profit_factor_trial['trial_number']),
                'sharpe_ratio': float(best_profit_factor_trial['sharpe_ratio']),
                'win_rate': float(best_profit_factor_trial['win_rate']),
                'profit_factor': float(best_profit_factor_trial['profit_factor']),
                'max_drawdown': float(best_profit_factor_trial['max_drawdown']),
                'objective_value': float(best_profit_factor_trial['objective_value'])
            },
            'best_objective': {
                'trial_number': int(best_objective_trial['trial_number']),
                'sharpe_ratio': float(best_objective_trial['sharpe_ratio']),
                'win_rate': float(best_objective_trial['win_rate']),
                'profit_factor': float(best_objective_trial['profit_factor']),
                'max_drawdown': float(best_objective_trial['max_drawdown']),
                'objective_value': float(best_objective_trial['objective_value'])
            }
        },
        'all_trials': trading_metrics
    }
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path(output_dir) / f'trading_metrics_report_{timestamp}.json'
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    logger.info(f"Trading metrics report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("🎯 TRADING METRICS OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"Total Trials: {summary['total_trials']}")
    print(f"\n📊 SHARPE RATIO:")
    print(f"  Best: {summary['best_sharpe_ratio']:.4f} (Trial {best_sharpe_trial['trial_number']})")
    print(f"  Mean: {summary['mean_sharpe_ratio']:.4f} ± {summary['std_sharpe_ratio']:.4f}")
    print(f"  Worst: {summary['worst_sharpe_ratio']:.4f}")
    
    print(f"\n🎯 WIN RATE:")
    print(f"  Best: {summary['best_win_rate']:.4f} (Trial {best_win_rate_trial['trial_number']})")
    print(f"  Mean: {summary['mean_win_rate']:.4f}")
    
    print(f"\n💰 PROFIT FACTOR:")
    print(f"  Best: {summary['best_profit_factor']:.4f} (Trial {best_profit_factor_trial['trial_number']})")
    print(f"  Mean: {summary['mean_profit_factor']:.4f}")
    
    print(f"\n📉 MAX DRAWDOWN:")
    print(f"  Worst: {summary['worst_max_drawdown']:.4f}")
    print(f"  Mean: {summary['mean_max_drawdown']:.4f}")
    
    print(f"\n🏆 BEST OVERALL TRIAL (Lowest Objective):")
    print(f"  Trial {best_objective_trial['trial_number']}:")
    print(f"    Sharpe Ratio: {best_objective_trial['sharpe_ratio']:.4f}")
    print(f"    Win Rate: {best_objective_trial['win_rate']:.4f}")
    print(f"    Profit Factor: {best_objective_trial['profit_factor']:.4f}")
    print(f"    Max Drawdown: {best_objective_trial['max_drawdown']:.4f}")
    print(f"    Objective Value: {best_objective_trial['objective_value']:.4f}")
    print("="*80)
    
    return report

def create_trading_metrics_visualizations(study, output_dir="plots"):
    """
    Create focused visualizations for trading metrics: Sharpe ratio, max drawdown, win rate, and profit factor.
    
    Args:
        study: Optuna study object with completed trials
        output_dir: Directory to save the plots
    """
    logger = get_xgboost_logger()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract trading metrics from completed trials
    trading_metrics = []
    for trial in study.trials:
        if trial.value is not None:
            if hasattr(trial, 'user_attrs'):
                metrics = {
                    'trial_number': trial.number,
                    'objective_value': trial.value,
                    'sharpe_ratio': trial.user_attrs.get('sharpe_ratio', 0.0),
                    'win_rate': trial.user_attrs.get('win_rate', 0.0),
                    'profit_factor': trial.user_attrs.get('profit_factor', 0.0),
                    'max_drawdown': trial.user_attrs.get('max_drawdown', 0.0),
                    'parameters': trial.params
                }
                trading_metrics.append(metrics)
    
    if not trading_metrics:
        logger.warning("No completed trials with trading metrics found")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(trading_metrics)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Sharpe Ratio Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['sharpe_ratio'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(df['sharpe_ratio'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["sharpe_ratio"].mean():.4f}')
    ax1.axvline(df['sharpe_ratio'].max(), color='blue', linestyle='--', 
                label=f'Best: {df["sharpe_ratio"].max():.4f}')
    ax1.set_title('Sharpe Ratio Distribution', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Sharpe Ratio')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Win Rate Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df['win_rate'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(df['win_rate'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["win_rate"].mean():.4f}')
    ax2.axvline(df['win_rate'].max(), color='blue', linestyle='--', 
                label=f'Best: {df["win_rate"].max():.4f}')
    ax2.set_title('Win Rate Distribution', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Win Rate')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Profit Factor Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df['profit_factor'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(df['profit_factor'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["profit_factor"].mean():.4f}')
    ax3.axvline(df['profit_factor'].max(), color='blue', linestyle='--', 
                label=f'Best: {df["profit_factor"].max():.4f}')
    ax3.set_title('Profit Factor Distribution', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Profit Factor')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Max Drawdown Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(df['max_drawdown'], bins=20, alpha=0.7, color='red', edgecolor='black')
    ax4.axvline(df['max_drawdown'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["max_drawdown"].mean():.4f}')
    ax4.axvline(df['max_drawdown'].min(), color='blue', linestyle='--', 
                label=f'Best: {df["max_drawdown"].min():.4f}')
    ax4.set_title('Max Drawdown Distribution', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Max Drawdown')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Sharpe Ratio vs Win Rate
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(df['win_rate'], df['sharpe_ratio'], 
                          c=df['objective_value'], cmap='viridis', alpha=0.7, s=50)
    ax5.set_title('Sharpe Ratio vs Win Rate', fontweight='bold', fontsize=14)
    ax5.set_xlabel('Win Rate')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Objective Value')
    
    # 6. Profit Factor vs Max Drawdown
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(df['max_drawdown'], df['profit_factor'], 
                          c=df['objective_value'], cmap='plasma', alpha=0.7, s=50)
    ax6.set_title('Profit Factor vs Max Drawdown', fontweight='bold', fontsize=14)
    ax6.set_xlabel('Max Drawdown')
    ax6.set_ylabel('Profit Factor')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Objective Value')
    
    # 7. Trading Metrics Correlation Heatmap
    ax7 = fig.add_subplot(gs[2, 0])
    correlation_matrix = df[['sharpe_ratio', 'win_rate', 'profit_factor', 'max_drawdown', 'objective_value']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax7, fmt='.3f')
    ax7.set_title('Trading Metrics Correlation', fontweight='bold', fontsize=14)
    
    # 8. Top 10 Trials by Sharpe Ratio
    ax8 = fig.add_subplot(gs[2, 1])
    top_sharpe = df.nlargest(10, 'sharpe_ratio')
    bars = ax8.barh(range(len(top_sharpe)), top_sharpe['sharpe_ratio'], 
                     color='green', alpha=0.7)
    ax8.set_yticks(range(len(top_sharpe)))
    ax8.set_yticklabels([f'Trial {trial}' for trial in top_sharpe['trial_number']])
    ax8.set_title('Top 10 Trials by Sharpe Ratio', fontweight='bold', fontsize=14)
    ax8.set_xlabel('Sharpe Ratio')
    ax8.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_sharpe['sharpe_ratio'])):
        ax8.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=10)
    
    # 9. Top 10 Trials by Profit Factor
    ax9 = fig.add_subplot(gs[2, 2])
    top_profit = df.nlargest(10, 'profit_factor')
    bars = ax9.barh(range(len(top_profit)), top_profit['profit_factor'], 
                     color='purple', alpha=0.7)
    ax9.set_yticks(range(len(top_profit)))
    ax9.set_yticklabels([f'Trial {trial}' for trial in top_profit['trial_number']])
    ax9.set_title('Top 10 Trials by Profit Factor', fontweight='bold', fontsize=14)
    ax9.set_xlabel('Profit Factor')
    ax9.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_profit['profit_factor'])):
        ax9.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=10)
    
    # Add overall title
    fig.suptitle('XGBoost Trading Metrics Analysis\nSharpe Ratio, Win Rate, Profit Factor, Max Drawdown', 
                 fontsize=16, fontweight='bold')
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = Path(output_dir) / f'trading_metrics_analysis_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Trading metrics visualizations saved to: {plot_path}")
    
    # Create additional focused plots
    create_focused_trading_plots(df, output_dir, timestamp)

def create_focused_trading_plots(df, output_dir, timestamp):
    """
    Create additional focused plots for specific trading metrics analysis.
    
    Args:
        df: DataFrame with trading metrics
        output_dir: Directory to save plots
        timestamp: Timestamp for file naming
    """
    logger = get_xgboost_logger()
    
    # 1. Sharpe Ratio Progression Over Trials
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['trial_number'], df['sharpe_ratio'], 'o-', color='green', alpha=0.7, linewidth=2)
    plt.axhline(df['sharpe_ratio'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["sharpe_ratio"].mean():.4f}')
    plt.title('Sharpe Ratio Progression Over Trials', fontweight='bold', fontsize=14)
    plt.xlabel('Trial Number')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Win Rate Progression Over Trials
    plt.subplot(2, 2, 2)
    plt.plot(df['trial_number'], df['win_rate'], 'o-', color='orange', alpha=0.7, linewidth=2)
    plt.axhline(df['win_rate'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["win_rate"].mean():.4f}')
    plt.title('Win Rate Progression Over Trials', fontweight='bold', fontsize=14)
    plt.xlabel('Trial Number')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Profit Factor Progression Over Trials
    plt.subplot(2, 2, 3)
    plt.plot(df['trial_number'], df['profit_factor'], 'o-', color='purple', alpha=0.7, linewidth=2)
    plt.axhline(df['profit_factor'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["profit_factor"].mean():.4f}')
    plt.title('Profit Factor Progression Over Trials', fontweight='bold', fontsize=14)
    plt.xlabel('Trial Number')
    plt.ylabel('Profit Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Max Drawdown Progression Over Trials
    plt.subplot(2, 2, 4)
    plt.plot(df['trial_number'], df['max_drawdown'], 'o-', color='red', alpha=0.7, linewidth=2)
    plt.axhline(df['max_drawdown'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["max_drawdown"].mean():.4f}')
    plt.title('Max Drawdown Progression Over Trials', fontweight='bold', fontsize=14)
    plt.xlabel('Trial Number')
    plt.ylabel('Max Drawdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save progression plot
    progression_path = Path(output_dir) / f'trading_metrics_progression_{timestamp}.png'
    plt.savefig(progression_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Trading Metrics Comparison (Box Plots)
    plt.figure(figsize=(15, 10))
    
    # Normalize metrics for comparison (0-1 scale) with safety checks
    df_normalized = df.copy()
    for col in ['sharpe_ratio', 'win_rate', 'profit_factor']:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:  # Avoid division by zero
            df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_normalized[col] = 0.5  # Default to middle value if no variation
    
    # Invert max_drawdown so lower is better
    drawdown_min = df['max_drawdown'].min()
    drawdown_max = df['max_drawdown'].max()
    if drawdown_max > drawdown_min:
        df_normalized['max_drawdown_inverted'] = 1 - (df['max_drawdown'] - drawdown_min) / (drawdown_max - drawdown_min)
    else:
        df_normalized['max_drawdown_inverted'] = 0.5  # Default to middle value if no variation
    
    plt.subplot(2, 2, 1)
    metrics_data = [df_normalized['sharpe_ratio'], df_normalized['win_rate'], 
                   df_normalized['profit_factor'], df_normalized['max_drawdown_inverted']]
    plt.boxplot(metrics_data, labels=['Sharpe\nRatio', 'Win\nRate', 'Profit\nFactor', 'Drawdown\n(Inverted)'])
    plt.title('Trading Metrics Distribution Comparison', fontweight='bold', fontsize=14)
    plt.ylabel('Normalized Score (0-1)')
    plt.grid(True, alpha=0.3)
    
    # 3. Best Trials Analysis
    plt.subplot(2, 2, 2)
    best_trials = df.nlargest(5, 'sharpe_ratio')
    x = np.arange(len(best_trials))
    width = 0.2
    
    plt.bar(x - width*1.5, best_trials['sharpe_ratio'], width, label='Sharpe Ratio', color='green', alpha=0.7)
    plt.bar(x - width*0.5, best_trials['win_rate'], width, label='Win Rate', color='orange', alpha=0.7)
    plt.bar(x + width*0.5, best_trials['profit_factor'], width, label='Profit Factor', color='purple', alpha=0.7)
    plt.bar(x + width*1.5, -best_trials['max_drawdown'], width, label='Max Drawdown', color='red', alpha=0.7)
    
    plt.xlabel('Top 5 Trials by Sharpe Ratio')
    plt.ylabel('Metric Value')
    plt.title('Best Trials Metrics Breakdown', fontweight='bold', fontsize=14)
    plt.xticks(x, [f'Trial {trial}' for trial in best_trials['trial_number']])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Parameter Impact Analysis (if we have parameter data)
    plt.subplot(2, 2, 3)
    # This would require extracting specific parameters from trial.params
    # For now, show objective value distribution
    plt.hist(df['objective_value'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(df['objective_value'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["objective_value"].mean():.4f}')
    plt.title('Objective Value Distribution', fontweight='bold', fontsize=14)
    plt.xlabel('Objective Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Trading Metrics Scatter Matrix
    plt.subplot(2, 2, 4)
    # Create a scatter plot of Sharpe vs Profit Factor
    scatter = plt.scatter(df['profit_factor'], df['sharpe_ratio'], 
                          c=df['win_rate'], cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Win Rate')
    plt.xlabel('Profit Factor')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio vs Profit Factor (Colored by Win Rate)', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = Path(output_dir) / f'trading_metrics_comparison_{timestamp}.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Additional trading metrics plots saved:")
    logger.info(f"  Progression plot: {progression_path}")
    logger.info(f"  Comparison plot: {comparison_path}")

def apply_feature_selection(features, method, min_features, max_features, importance_threshold):
    """
    Apply automated feature selection using various methods.
    
    Args:
        features: DataFrame with features
        method: Feature selection method ('importance', 'correlation', 'mutual_info')
        min_features: Minimum number of features to select
        max_features: Maximum number of features to select
        importance_threshold: Threshold for importance-based selection
        
    Returns:
        List of selected feature names
    """
    logger = get_xgboost_logger()
    
    # Remove any features with all NaN values
    features_clean = features.dropna(axis=1, how='all')
    feature_names = features_clean.columns.tolist()
    
    logger.info(f"[FEATURE_SELECTION] Starting {method} selection on {len(feature_names)} features")
    
    if method == 'importance':
        return apply_importance_based_selection(
            features_clean, min_features, max_features, importance_threshold
        )
    elif method == 'correlation':
        return apply_correlation_based_selection(
            features_clean, min_features, max_features
        )
    elif method == 'mutual_info':
        return apply_mutual_info_selection(
            features_clean, min_features, max_features
        )
    else:
        logger.warning(f"[FEATURE_SELECTION] Unknown method '{method}', using all features")
        return feature_names

def apply_importance_based_selection(features, min_features, max_features, importance_threshold):
    """Apply feature selection based on Random Forest importance scores"""
    logger = get_xgboost_logger()
    
    # Create a simple Random Forest to get feature importance
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    # Create synthetic target for importance calculation (we'll use the first few samples)
    # This is a proxy since we don't have the actual target at this stage
    sample_size = min(100, len(features))
    synthetic_target = np.random.choice([0, 1, 2], size=sample_size, p=[0.3, 0.4, 0.3])
    
    # Fit the model
    rf.fit(features.iloc[:sample_size], synthetic_target)
    
    # Get feature importance
    importance_scores = rf.feature_importances_
    feature_names = features.columns.tolist()
    
    # Create importance dictionary
    importance_dict = dict(zip(feature_names, importance_scores))
    
    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Select features based on importance threshold and count constraints
    selected_features = []
    for feature, importance in sorted_features:
        if importance >= importance_threshold and len(selected_features) < max_features:
            selected_features.append(feature)
    
    # Ensure minimum number of features
    if len(selected_features) < min_features:
        # Add more features from the sorted list
        for feature, importance in sorted_features:
            if feature not in selected_features and len(selected_features) < min_features:
                selected_features.append(feature)
    
    logger.info(f"[FEATURE_SELECTION] Importance-based selection: {len(selected_features)} features selected")
    logger.info(f"[FEATURE_SELECTION] Top 5 features: {selected_features[:5]}")
    
    return selected_features

def apply_correlation_based_selection(features, min_features, max_features):
    """Apply feature selection based on correlation analysis"""
    logger = get_xgboost_logger()
    
    # Calculate correlation matrix
    corr_matrix = features.corr().abs()
    
    # Find highly correlated features
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with high correlation (> 0.95)
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    
    # Remove highly correlated features, keeping the one with highest variance
    features_to_remove = set()
    for feature in high_corr_features:
        # Find features highly correlated with this feature
        correlated_features = upper_tri[feature][upper_tri[feature] > 0.95].index.tolist()
        correlated_features.append(feature)
        
        # Keep the feature with highest variance
        variances = [features[feat].var() for feat in correlated_features]
        best_feature = correlated_features[np.argmax(variances)]
        
        # Mark others for removal
        for feat in correlated_features:
            if feat != best_feature:
                features_to_remove.add(feat)
    
    # Get remaining features
    all_features = features.columns.tolist()
    selected_features = [feat for feat in all_features if feat not in features_to_remove]
    
    # Ensure we have the right number of features
    if len(selected_features) > max_features:
        # Keep top features by variance
        variances = [(feat, features[feat].var()) for feat in selected_features]
        variances.sort(key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, _ in variances[:max_features]]
    elif len(selected_features) < min_features:
        # Add back some features if we removed too many
        removed_features = [feat for feat in all_features if feat not in selected_features]
        variances = [(feat, features[feat].var()) for feat in removed_features]
        variances.sort(key=lambda x: x[1], reverse=True)
        
        for feat, _ in variances:
            if len(selected_features) < min_features:
                selected_features.append(feat)
            else:
                break
    
    logger.info(f"[FEATURE_SELECTION] Correlation-based selection: {len(selected_features)} features selected")
    logger.info(f"[FEATURE_SELECTION] Removed {len(features_to_remove)} highly correlated features")
    
    return selected_features

def apply_mutual_info_selection(features, min_features, max_features):
    """Apply feature selection based on mutual information"""
    logger = get_xgboost_logger()
    
    # Create synthetic target for mutual information calculation
    sample_size = min(100, len(features))
    synthetic_target = np.random.choice([0, 1, 2], size=sample_size, p=[0.3, 0.4, 0.3])
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(
        features.iloc[:sample_size], 
        synthetic_target, 
        random_state=42
    )
    
    # Create feature-score pairs
    feature_scores = list(zip(features.columns, mi_scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top features
    selected_features = [feature for feature, score in feature_scores[:max_features]]
    
    # Ensure minimum number of features
    if len(selected_features) < min_features:
        selected_features = [feature for feature, score in feature_scores[:min_features]]
    
    logger.info(f"[FEATURE_SELECTION] Mutual info selection: {len(selected_features)} features selected")
    logger.info(f"[FEATURE_SELECTION] Top 5 features by MI: {selected_features[:5]}")
    
    return selected_features

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for XGBoost mean reversion optimization with automated feature selection"""
    logger = get_xgboost_logger()
    
    # Add trial start logging
    logger.info(f"[TRIAL_START] Starting trial {trial.number}")
    logger.info(f"[TRIAL_START] Trial user attributes at start: {trial.user_attrs}")
    
    # Force memory cleanup
    gc.collect()
    
    try:
        # Define hyperparameter search space (tailored for mean reversion)
        config_dict = {
            # Core XGBoost parameters (aggressive ranges)
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            
            # Mean reversion specific parameters
            'price_threshold': trial.suggest_float('price_threshold', 0.01, 0.05),
            'probability_threshold': trial.suggest_float('probability_threshold', 0.5, 0.8),
            
            # Time-based classification parameters
            'early_window': trial.suggest_int('early_window', 3, 8),
            'late_window': trial.suggest_int('late_window', 8, 12),
            'moderate_threshold_ratio': trial.suggest_float('moderate_threshold_ratio', 0.3, 0.7),
            
            # Feature selection parameters
            'use_feature_selection': trial.suggest_categorical('use_feature_selection', [True]),
            'feature_selection_method': trial.suggest_categorical('feature_selection_method', ['importance', 'correlation', 'mutual_info']),
            'min_features': trial.suggest_int('min_features', 8, 15),
            'max_features': trial.suggest_int('max_features', 20, 26),
            'importance_threshold': trial.suggest_float('importance_threshold', 0.01, 0.1),
            
            # Feature engineering parameters - All 26 features configurable
            
            # Bollinger Bands parameters
            'bollinger_period': trial.suggest_int('bollinger_period', 10, 30),
            'bollinger_std_dev': trial.suggest_float('bollinger_std_dev', 1.5, 3.0),
            
            # RSI parameters
            'rsi_period': trial.suggest_int('rsi_period', 10, 20),
            'swing_rsi_period': trial.suggest_int('swing_rsi_period', 10, 20),
            
            # MACD parameters
            'macd_fast_period': trial.suggest_int('macd_fast_period', 8, 16),
            'macd_slow_period': trial.suggest_int('macd_slow_period', 20, 35),
            'macd_signal_period': trial.suggest_int('macd_signal_period', 7, 12),
            
            # Moving Average parameters
            'short_ma_period': trial.suggest_int('short_ma_period', 3, 10),
            'long_ma_period': trial.suggest_int('long_ma_period', 8, 20),
            
            # Volume parameters
            'volume_ma_period': trial.suggest_int('volume_ma_period', 10, 30),
            'volume_std_period': trial.suggest_int('volume_std_period', 15, 25),
            'volume_surge_period': trial.suggest_int('volume_surge_period', 10, 30),
            
            # ATR parameters
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            
            # ADX parameters
            'adx_period': trial.suggest_int('adx_period', 10, 20),
            
            # Momentum parameters
            'price_momentum_lookback': trial.suggest_int('price_momentum_lookback', 3, 15),
            'volatility_regime_period': trial.suggest_int('volatility_regime_period', 15, 25),
            
            # Support/Resistance parameters
            'support_resistance_period': trial.suggest_int('support_resistance_period', 15, 25),
            
            # Breakout parameters
            'breakout_period': trial.suggest_int('breakout_period', 8, 20),
            
            # Cumulative delta parameters
            'cumulative_delta_period': trial.suggest_int('cumulative_delta_period', 15, 25),
            
            # VWAP parameters
            'vwap_period': trial.suggest_int('vwap_period', 10, 30),
            
            # Returns calculation parameters
            'returns_period': trial.suggest_int('returns_period', 1, 5),
            'log_returns_period': trial.suggest_int('log_returns_period', 1, 5),
            
            # Class imbalance parameters - ENHANCED FOR MINORITY FOCUS
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),

            # Focal loss parameters - MORE AGGRESSIVE FOR MINORITY CLASSES
            'use_focal_loss': trial.suggest_categorical('use_focal_loss', [True]),
            'focal_alpha': trial.suggest_float('focal_alpha', 2.0, 8.0),  # More aggressive alpha range for minority focus
            'focal_gamma': trial.suggest_float('focal_gamma', 2.0, 8.0),  # More aggressive gamma for hard examples

            # Class multiplier parameters - EVEN MORE AGGRESSIVE FOR MINORITY CLASSES
            'class_multiplier_0': trial.suggest_float('class_multiplier_0', 4.0, 12.0),  # Down class - much higher emphasis
            'class_multiplier_1': trial.suggest_float('class_multiplier_1', 1, 4),  # Hold class - even lower weight since it dominates
            'class_multiplier_2': trial.suggest_float('class_multiplier_2', 4.0, 12.0),  # Up class - much higher emphasis

            # NEW: Minority class specific parameters
            'minority_threshold_boost': trial.suggest_float('minority_threshold_boost', 0.1, 0.3),  # Boost threshold for minority classes
            'minority_confidence_boost': trial.suggest_float('minority_confidence_boost', 0.1, 0.4),  # Boost confidence for minority predictions
            'use_minority_oversampling': trial.suggest_categorical('use_minority_oversampling', [True]),  # Additional minority oversampling
            'minority_oversampling_ratio': trial.suggest_float('minority_oversampling_ratio', 2.0, 5.0),  # How much to oversample minorities
        }
        
        # Log trial start
        log_trial_start(trial.number, config_dict)
        
        # Create configuration
        config = XGBoostHyperoptConfig(**config_dict)
        
        # Initialize market data and technical indicators
        from market_analysis.market_data import MarketData
        from market_analysis.technical_indicators import TechnicalIndicators
        from utils.feature_generator import FeatureGenerator
        
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        
        # Initialize feature generator with config
        feature_generator = FeatureGenerator(config=config_dict)
        
        # Load and prepare data (using your existing data loading)
        data_response = market_data.get_data(['ETH/USD'])  # Your existing data loading
        
        # Extract DataFrame from response (handles both single symbol and list of symbols)
        if isinstance(data_response, dict):
            data = data_response['ETH/USD']
        else:
            data = data_response
        
        # Filter data for 2025 May 1 to 2025 July 29 (3 months for better trading signals)
        start_date = '2024-01-01'
        end_date = '2025-08-03'
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        logger.info(f"[DATA] Filtered data from {start_date} to {end_date}")
        logger.info(f"[DATA] Data shape: {data.shape}")
        
        # Generate features using the feature generator with all configurable parameters
        features = feature_generator.generate_features(data)
        
        logger.info(f"[FEATURES] Generated {len(features.columns)} configurable features")
        logger.info(f"[FEATURES] Feature names: {list(features.columns)}")
        
        # Automated Feature Selection
        if config_dict['use_feature_selection']:
            logger.info(f"[FEATURE_SELECTION] Starting automated feature selection using {config_dict['feature_selection_method']}")
            
            # Apply feature selection
            selected_features = apply_feature_selection(
                features, 
                config_dict['feature_selection_method'],
                config_dict['min_features'],
                config_dict['max_features'],
                config_dict['importance_threshold']
            )
            
            logger.info(f"[FEATURE_SELECTION] Selected {len(selected_features)} features: {selected_features}")
            
            # Update features to only include selected ones
            features = features[selected_features]
        else:
            logger.info(f"[FEATURE_SELECTION] Skipping feature selection, using all {len(features.columns)} features")
        
        # Create target variable using event-based trading logic
        # Use configurable price threshold from trial parameters
        price_threshold = config_dict['price_threshold']  # Configurable threshold
        window_size = 15  # 15 candlesticks window
        
        logger.info(f"[TARGET] Using configurable price threshold: {price_threshold:.4f} ({price_threshold*100:.2f}%)")
        
        # Calculate future returns for each candlestick using time-based classification
        future_returns = []
        
        # Time-based classification parameters (configurable)
        early_window = config_dict['early_window']  # Configurable early window
        late_window = config_dict['late_window']    # Configurable late window
        moderate_threshold_ratio = config_dict['moderate_threshold_ratio']  # Configurable moderate threshold ratio
        
        # ENHANCED: Minority-focused parameters
        minority_threshold_boost = config_dict.get('minority_threshold_boost', 0.2)  # Boost threshold for minority classes
        minority_confidence_boost = config_dict.get('minority_confidence_boost', 0.2)  # Boost confidence for minority predictions
        
        logger.info(f"[TARGET] Using ENHANCED minority-focused classification:")
        logger.info(f"[TARGET] Main threshold: {price_threshold:.4f} ({price_threshold*100:.2f}%)")
        logger.info(f"[TARGET] Early window: {early_window} periods, Late window: {late_window} periods")
        logger.info(f"[TARGET] Moderate threshold ratio: {moderate_threshold_ratio:.2f}")
        logger.info(f"[TARGET] Minority threshold boost: {minority_threshold_boost:.2f}")
        logger.info(f"[TARGET] Minority confidence boost: {minority_confidence_boost:.2f}")
        
        for i in range(len(data)):
            if i + window_size >= len(data):
                # For the last few candlesticks, we can't look ahead 15 periods
                future_returns.append(1)  # No signal
            else:
                # Get current price
                current_price = data['close'].iloc[i]
                
                # Calculate early period returns (first 5 periods)
                early_prices = data['close'].iloc[i:i+early_window+1]
                early_max_return = (early_prices.max() - current_price) / current_price
                early_min_return = (early_prices.min() - current_price) / current_price
                
                # Calculate late period returns (periods 6-15)
                late_prices = data['close'].iloc[i+early_window:i+window_size+1]
                late_max_return = (late_prices.max() - current_price) / current_price
                late_min_return = (late_prices.min() - current_price) / current_price
                
                # Calculate full window returns
                future_prices = data['close'].iloc[i:i+window_size+1]
                full_max_return = (future_prices.max() - current_price) / current_price
                full_min_return = (future_prices.min() - current_price) / current_price
                
                # ENHANCED: Minority-focused classification logic
                # More aggressive thresholds for minority classes (0 and 2)
                minority_threshold = price_threshold * (1 - minority_threshold_boost)  # Lower threshold for minorities
                
                # Enhanced classification logic with minority focus
                if early_max_return >= minority_threshold:  # Lower threshold for Up class
                    future_returns.append(2)  # Early Up signal (strong early movement)
                elif early_min_return <= -minority_threshold:  # Lower threshold for Down class
                    future_returns.append(0)  # Early Down signal (strong early movement)
                elif late_max_return >= price_threshold:
                    future_returns.append(1)  # Late Up signal (hold - late movement)
                elif late_min_return <= -price_threshold:
                    future_returns.append(1)  # Late Down signal (hold - late movement)
                elif full_max_return >= price_threshold * moderate_threshold_ratio:  # Moderate movement
                    future_returns.append(1)  # Hold signal (moderate movement)
                elif full_min_return <= -price_threshold * moderate_threshold_ratio:  # Moderate movement
                    future_returns.append(1)  # Hold signal (moderate movement)
                else:
                    future_returns.append(1)  # No significant movement (hold)
        
        # Convert to numpy array
        y = np.array(future_returns)
        
        # Ensure features and target have the same length
        if len(features) != len(y):
            # Trim to the shorter length
            min_length = min(len(features), len(y))
            features = features.iloc[:min_length]
            y = y[:min_length]
        
        logger.info(f"[TARGET] Target variable shape: {y.shape}")
        logger.info(f"[TARGET] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Remove rows with NaN values
        valid_mask = ~(features.isna().any(axis=1))
        features = features[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"[DATA] Final features shape: {features.shape}")
        logger.info(f"[DATA] Final target shape: {y.shape}")
        
        # Check if we have enough data
        if len(features) < 100:
            logger.warning(f"[DATA] Insufficient data points: {len(features)}")
            raise TrialPruned(f"Insufficient data points: {len(features)}")
        
        # Split data (time series aware)
        split_idx = int(len(features) * (1 - config.test_size - config.validation_size))
        val_idx = int(len(features) * (1 - config.test_size))
        
        X_train = features.iloc[:split_idx].values
        y_train = y[:split_idx]
        X_val = features.iloc[split_idx:val_idx].values
        y_val = y[split_idx:val_idx]
        X_test = features.iloc[val_idx:].values
        y_test = y[val_idx:]
        
        # Create trainer
        trainer = XGBoostHyperoptTrainer(config, None)  # No data processor needed
        
        # Train model
        training_history = trainer.train_with_smote(X_train, y_train, X_val, y_val)
        
        # Check for early pruning after training
        if training_history['best_score'] < 0.5:
            reason = f"Poor validation score: {training_history['best_score']:.4f} < 0.5"
            log_trial_pruned(trial.number, reason)
            raise TrialPruned(reason)
        
        # Additional pruning check: if model stopped very early due to poor performance
        if hasattr(trainer.model, 'best_iteration'):
            if trainer.model.best_iteration == 0 and training_history['best_score'] < 0.3:
                reason = f"Model stopped at iteration 0 with poor score: {training_history['best_score']:.4f}"
                log_trial_pruned(trial.number, reason)
                raise TrialPruned(reason)
            elif trainer.model.best_iteration < 3 and training_history['best_score'] < 0.1:
                reason = f"Early stopping at iteration {trainer.model.best_iteration} with very poor score: {training_history['best_score']:.4f}"
                log_trial_pruned(trial.number, reason)
                raise TrialPruned(reason)
        
        # Evaluate on validation set
        evaluation_results = trainer.evaluate(X_val, y_val)
        
        # Log evaluation results
        log_evaluation_results(evaluation_results)
        
        # SHARPE RATIO OBJECTIVE FUNCTION FOR LIVE TRADING SIMULATION
        logger.info("[SHARPE] Starting Sharpe ratio calculation for live trading simulation...")
        
        # Get predictions and probabilities
        import xgboost as xgb
        dtest = xgb.DMatrix(X_val)
        y_pred_proba = trainer.model.predict(dtest, output_margin=False)
        y_pred_raw = trainer.model.predict(dtest, output_margin=False)
        
        # Convert probabilities to class predictions
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Walk-forward validation with 40% holdout
        def calculate_sharpe_ratio_trading_simulation(features_val, y_val, y_pred, y_pred_proba, data_val):
            """Calculate Sharpe ratio through trading simulation"""
            
            # Trading parameters
            position_size = 0.04  # 4% of account balance
            leverage = 50  # 50x leverage
            stop_loss = 0.01  # 0.8% stop loss
            trading_fee = 0.0002  # 0.02% per trade
            max_hold_hours = 24  # Max 24 hours hold time
            
            # Initialize single account with $5000
            account_balance = 5000  # Starting account balance
            available_balance = 5000  # Available balance for new trades
            trades = []
            current_position = None
            entry_time = None
            entry_price = None
            
            # Loss tracking variables
            consecutive_losses = 0
            trading_pause_until = None
            
            # Get price data for validation set
            prices = data_val['close'].values
            timestamps = data_val.index
            
            # Daily Sharpe calculation parameters
            daily_returns = []
            current_day_returns = []
            current_day_start = None
            
            for i in range(len(y_pred)):
                current_time = timestamps[i]
                current_price = prices[i]
                prediction = y_pred[i]
                confidence = np.max(y_pred_proba[i])
                
                # Check if we're in trading pause
                if trading_pause_until is not None and current_time < trading_pause_until:
                    continue  # Skip trading during pause
                elif trading_pause_until is not None and current_time >= trading_pause_until:
                    trading_pause_until = None  # Resume trading
                
                # Daily grouping for Sharpe calculation
                if current_day_start is None:
                    current_day_start = current_time
                elif (current_time - current_day_start).days >= 1:
                    # End of day, calculate daily return
                    if current_day_returns:
                        daily_return = np.sum(current_day_returns)
                        daily_returns.append(daily_return)
                        current_day_returns = []
                    current_day_start = current_time
                
                # Check for stop loss on existing position
                if current_position is not None:
                    hold_hours = (current_time - entry_time).total_seconds() / 3600
                    
                    # Calculate current P&L
                    if current_position == 'long':
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:  # short
                        pnl_pct = (entry_price - current_price) / entry_price
                    
                    # Check stop loss or max hold time
                    if pnl_pct <= -stop_loss or hold_hours >= max_hold_hours:
                        # Calculate position size in USD
                        position_size_usd = available_balance * position_size
                        effective_position = position_size_usd * leverage / account_balance
                        
                        # Close position
                        trade_return = pnl_pct * effective_position - (2 * trading_fee)  # Entry + exit fees
                        trades.append({
                            'type': current_position,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'return': trade_return,
                            'hold_hours': hold_hours,
                            'reason': 'stop_loss' if pnl_pct <= -stop_loss else 'max_hold_time'
                        })
                        
                        # Track consecutive losses
                        if trade_return < 0:
                            consecutive_losses += 1
                            if consecutive_losses >= 3:
                                trading_pause_until = current_time + timedelta(hours=12)
                                consecutive_losses = 0  # Reset after pause
                        else:
                            consecutive_losses = 0  # Reset on win
                        
                        # Add to daily returns
                        current_day_returns.append(trade_return)
                        
                        # Update account balance and available balance
                        account_balance *= (1 + trade_return)
                        # Add back the position size plus the profit/loss
                        position_return_usd = position_size_usd * trade_return
                        available_balance += position_size_usd + position_return_usd
                        current_position = None
                        entry_time = None
                        entry_price = None
                
                # Check for new trading signals (only take signals 0 and 2)
                if prediction in [0, 2] and available_balance > 0:  # Early Down or Early Up
                    # Calculate position size based on available balance
                    position_size_usd = available_balance * position_size
                    
                    # Check if we have enough balance for the trade
                    if position_size_usd >= 20:  # Minimum $20 position
                        # Open new position
                        current_position = 'short' if prediction == 0 else 'long'
                        entry_time = current_time
                        entry_price = current_price
                        
                        # Deduct position size from available balance
                        available_balance -= position_size_usd
                        
                        # Pay entry fee
                        account_balance *= (1 - trading_fee)
            
            # Close any remaining position at the end
            if current_position is not None:
                final_price = prices[-1]
                if current_position == 'long':
                    pnl_pct = (final_price - entry_price) / entry_price
                else:  # short
                    pnl_pct = (entry_price - final_price) / entry_price
                
                # Calculate position size in USD
                position_size_usd = available_balance * position_size
                effective_position = position_size_usd * leverage / account_balance
                
                trade_return = pnl_pct * effective_position - (2 * trading_fee)
                trades.append({
                    'type': current_position,
                    'entry_price': entry_price,
                    'exit_price': final_price,
                    'return': trade_return,
                    'hold_hours': (timestamps[-1] - entry_time).total_seconds() / 3600,
                    'reason': 'end_of_period'
                })
                current_day_returns.append(trade_return)
                
                # Update account balance and available balance
                account_balance *= (1 + trade_return)
                # Add back the position size plus the profit/loss
                position_return_usd = position_size_usd * trade_return
                available_balance += position_size_usd + position_return_usd
            
            # Add final day returns
            if current_day_returns:
                daily_return = np.sum(current_day_returns)
                daily_returns.append(daily_return)
            
            # Calculate Sharpe ratio (daily)
            if len(daily_returns) < 2:
                logger.warning(f"[SHARPE] Insufficient daily returns ({len(daily_returns)}), using hourly returns instead")
                # Fallback to hourly returns if daily returns insufficient
                if len(trade_returns) >= 5:  # Need at least 5 trades
                    hourly_returns = trade_returns  # Use individual trade returns
                    sharpe_ratio = np.mean(hourly_returns) / (np.std(hourly_returns) + 1e-8)
                else:
                    return -10.0, 0.0, 0.0, 0.0, 5000.0  # Penalty for insufficient data, return initial balance
            else:
                daily_returns = np.array(daily_returns)
                sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)
            
            # Calculate additional metrics
            if trades:
                trade_returns = [t['return'] for t in trades]
                profitable_trades = [r for r in trade_returns if r > 0]
                win_rate = len(profitable_trades) / len(trade_returns) if trade_returns else 0
                
                # Profit factor
                gross_profit = sum([r for r in trade_returns if r > 0])
                gross_loss = abs(sum([r for r in trade_returns if r < 0]))
                profit_factor = gross_profit / (gross_loss + 1e-8)
                
                # Max drawdown
                cumulative_returns = np.cumsum(trade_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = cumulative_returns - running_max
                max_drawdown = np.min(drawdown)
            else:
                win_rate = 0.0
                profit_factor = 0.0
                max_drawdown = 0.0
            
            return sharpe_ratio, win_rate, profit_factor, max_drawdown, account_balance
        
        # Calculate Sharpe ratio through trading simulation
        sharpe_ratio, win_rate, profit_factor, max_drawdown, account_balance = calculate_sharpe_ratio_trading_simulation(
            features.iloc[split_idx:val_idx], y_val, y_pred, y_pred_proba, data.iloc[split_idx:val_idx]
        )
        
        # Add detailed logging for trading signal analysis
        trading_signals = np.sum(y_pred != 1)  # Count non-hold signals (0 and 2)
        total_samples = len(y_pred)
        signal_percentage = (trading_signals / total_samples) * 100
        
        logger.info(f"[SHARPE] Trading Signal Analysis:")
        logger.info(f"[SHARPE] Total samples: {total_samples}")
        logger.info(f"[SHARPE] Trading signals (0+2): {trading_signals} ({signal_percentage:.2f}%)")
        logger.info(f"[SHARPE] Hold signals (1): {total_samples - trading_signals} ({100-signal_percentage:.2f}%)")
        logger.info(f"[SHARPE] Signal distribution: Down={np.sum(y_pred==0)}, Hold={np.sum(y_pred==1)}, Up={np.sum(y_pred==2)}")
        
        logger.info(f"[SHARPE] Trading Simulation Results:")
        logger.info(f"[SHARPE] Sharpe Ratio (daily): {sharpe_ratio:.4f}")
        logger.info(f"[SHARPE] Win Rate: {win_rate:.4f}")
        logger.info(f"[SHARPE] Profit Factor: {profit_factor:.4f}")
        logger.info(f"[SHARPE] Max Drawdown: {max_drawdown:.4f}")
        logger.info(f"[SHARPE] Final Account Balance: ${account_balance:.2f}")
        logger.info(f"[SHARPE] Account Return: {((account_balance - 5000) / 5000 * 100):.2f}%")
        
        # SHARPE RATIO OBJECTIVE FUNCTION (lower is better, so we negate Sharpe)
        # Prioritize Sharpe ratio (40%), Profit Factor (20%), Win Rate (35%), Max Drawdown (5%)
        sharpe_penalty = -sharpe_ratio * 0.4  # Negative because we minimize
        profit_factor_penalty = (1 - min(profit_factor, 3.0)) * 0.2  # Cap at 3.0
        win_rate_penalty = (1 - win_rate) * 0.35
        drawdown_penalty = abs(max_drawdown) * 0.05  # Penalize large drawdowns
        
        # Combined Sharpe-based objective
        sharpe_objective = sharpe_penalty + profit_factor_penalty + win_rate_penalty + drawdown_penalty
        
        logger.info(f"[SHARPE] Objective Components:")
        logger.info(f"[SHARPE] Sharpe Penalty: {sharpe_penalty:.4f}")
        logger.info(f"[SHARPE] Profit Factor Penalty: {profit_factor_penalty:.4f}")
        logger.info(f"[SHARPE] Win Rate Penalty: {win_rate_penalty:.4f}")
        logger.info(f"[SHARPE] Drawdown Penalty: {drawdown_penalty:.4f}")
        logger.info(f"[SHARPE] Combined Sharpe Objective: {sharpe_objective:.4f}")
        
        # Store trading metrics in trial user attributes for reporting
        logger.info(f"[USER_ATTR] About to store trading metrics for trial {trial.number}")
        logger.info(f"[USER_ATTR] Sharpe ratio: {sharpe_ratio} (type: {type(sharpe_ratio)})")
        logger.info(f"[USER_ATTR] Win rate: {win_rate} (type: {type(win_rate)})")
        logger.info(f"[USER_ATTR] Profit factor: {profit_factor} (type: {type(profit_factor)})")
        logger.info(f"[USER_ATTR] Max drawdown: {max_drawdown} (type: {type(max_drawdown)})")
        
        try:
            # Convert numpy types to Python types for storage
            sharpe_ratio_py = float(sharpe_ratio) if hasattr(sharpe_ratio, 'item') else sharpe_ratio
            win_rate_py = float(win_rate) if hasattr(win_rate, 'item') else win_rate
            profit_factor_py = float(profit_factor) if hasattr(profit_factor, 'item') else profit_factor
            max_drawdown_py = float(max_drawdown) if hasattr(max_drawdown, 'item') else max_drawdown
            
            logger.info(f"[USER_ATTR] Converted values - Sharpe: {sharpe_ratio_py}, Win Rate: {win_rate_py}, Profit Factor: {profit_factor_py}, Max Drawdown: {max_drawdown_py}")
            
            trial.set_user_attr('sharpe_ratio', sharpe_ratio_py)
            logger.info(f"[USER_ATTR] Successfully stored sharpe_ratio: {sharpe_ratio_py}")
            
            trial.set_user_attr('win_rate', win_rate_py)
            logger.info(f"[USER_ATTR] Successfully stored win_rate: {win_rate_py}")
            
            trial.set_user_attr('profit_factor', profit_factor_py)
            logger.info(f"[USER_ATTR] Successfully stored profit_factor: {profit_factor_py}")
            
            trial.set_user_attr('max_drawdown', max_drawdown_py)
            logger.info(f"[USER_ATTR] Successfully stored max_drawdown: {max_drawdown_py}")
            
            # Verify storage
            logger.info(f"[USER_ATTR] Current trial user attributes: {trial.user_attrs}")
            
        except Exception as e:
            logger.error(f"[USER_ATTR] ERROR storing trading metrics: {e}")
            logger.error(f"[USER_ATTR] Exception type: {type(e)}")
            import traceback
            logger.error(f"[USER_ATTR] Traceback: {traceback.format_exc()}")
        
        # Log trial completion with Sharpe metrics
        metrics = {
            'max_depth': config_dict['max_depth'],
            'learning_rate': config_dict['learning_rate'],
            'price_threshold': config_dict['price_threshold'],
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_objective': sharpe_objective,
            'feature_count': len(features.columns),
            'use_feature_selection': config_dict['use_feature_selection'],
            'feature_selection_method': config_dict['feature_selection_method'] if config_dict['use_feature_selection'] else 'none',
            'use_focal_loss': config_dict['use_focal_loss'],
            'focal_alpha': config_dict['focal_alpha'],
            'focal_gamma': config_dict['focal_gamma'],
            'class_multiplier_0': config_dict['class_multiplier_0'],
            'class_multiplier_1': config_dict['class_multiplier_1'],
            'class_multiplier_2': config_dict['class_multiplier_2']
        }
        log_trial_complete(trial.number, sharpe_objective, metrics)
        
        # Save scaler if trial is successful
        if sharpe_objective < float('inf'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            scaler_path = f'config/xgboost_scaler_trial_{trial.number}_{timestamp}.joblib'
            trainer.save_scaler(scaler_path)
            # Store scaler path in trial user attributes for later retrieval
            logger.info(f"[USER_ATTR] About to store scaler_path: {scaler_path}")
            try:
                trial.set_user_attr('scaler_path', scaler_path)
                logger.info(f"[USER_ATTR] Successfully stored scaler_path: {scaler_path}")
                logger.info(f"[USER_ATTR] Current trial user attributes after scaler: {trial.user_attrs}")
            except Exception as e:
                logger.error(f"[USER_ATTR] ERROR storing scaler_path: {e}")
                import traceback
                logger.error(f"[USER_ATTR] Traceback: {traceback.format_exc()}")
        
        logger.info(f"[OBJECTIVE] XGBoost Sharpe Ratio Trial Results:")
        logger.info(f"  Max Depth: {config_dict['max_depth']}, LR: {config_dict['learning_rate']:.4f}")
        logger.info(f"  Price Threshold: {config_dict['price_threshold']:.4f}")
        logger.info(f"  Focal Loss: {config_dict['use_focal_loss']}, Alpha: {config_dict['focal_alpha']:.4f}, Gamma: {config_dict['focal_gamma']:.4f}")
        logger.info(f"  Class Multipliers: [0: {config_dict['class_multiplier_0']:.4f}, 1: {config_dict['class_multiplier_1']:.4f}, 2: {config_dict['class_multiplier_2']:.4f}]")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"  Win Rate: {win_rate:.4f}")
        logger.info(f"  Profit Factor: {profit_factor:.4f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.4f}")
        logger.info(f"  Sharpe Objective: {sharpe_objective:.4f}")
        
        # Final logging before return
        logger.info(f"[TRIAL_END] Trial {trial.number} completed successfully")
        logger.info(f"[TRIAL_END] Final trial user attributes: {trial.user_attrs}")
        logger.info(f"[TRIAL_END] Objective value: {sharpe_objective:.4f}")
        
        return sharpe_objective
        
    except TrialPruned:
        # Re-raise TrialPruned exceptions
        raise
    except Exception as e:
        logger.error(f"[ERROR] Exception in XGBoost trial {trial.number}: {e}")
        gc.collect()
        return float('inf')

def main():
    """Main function for XGBoost mean reversion hyperparameter optimization"""
    logger = get_xgboost_logger()
    logger.info("==================== XGBOOST MEAN REVERSION HYPEROPT ====================")
    
    # Initialize components (using your existing market data and technical indicators)
    from market_analysis.market_data import MarketData
    from market_analysis.technical_indicators import TechnicalIndicators
    
    market_data = MarketData()
    technical_indicators = TechnicalIndicators()
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        study_name='xgboost_mean_reversion_hyperopt_with_feature_selection',
        storage='sqlite:///xgboost_mean_reversion_hyperopt_with_feature_selection.db',
        load_if_exists=True
    )
    
    # Run optimization with small number of trials for testing
    study.optimize(
        objective,
        n_trials=2500,  # Small number for testing
        timeout=None,
        gc_after_trial=True,
        show_progress_bar=True
    )
    
    # Save results
    best_params = study.best_trial.params
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'config/xgboost_mean_reversion_best_params_test_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Save best scaler
    best_trial = study.best_trial
    if hasattr(best_trial, 'user_attrs') and 'scaler_path' in best_trial.user_attrs:
        best_scaler_path = best_trial.user_attrs['scaler_path']
        import os
        import shutil
        
        # Create best scaler path alongside best params
        best_scaler_dest = f'config/xgboost_best_scaler_{timestamp}.joblib'
        
        # Copy the best scaler to central location
        if os.path.exists(best_scaler_path):
            shutil.copy2(best_scaler_path, best_scaler_dest)
            logger.info(f'Best scaler saved to: {best_scaler_dest}')
            logger.info(f'Best scaler copied from: {best_scaler_path}')
        else:
            logger.warning(f'Best scaler file not found: {best_scaler_path}')
    else:
        logger.warning('No scaler path found in best trial user attributes')
    
    logger.info(f'\nXGBoost Mean Reversion Optimization Summary:')
    logger.info(f'  Total trials: {len(study.trials)}')
    logger.info(f'  Best objective: {study.best_trial.value:.4f}')
    logger.info(f'  Best parameters: {best_params}')
    
    # Generate focused trading metrics reports and visualizations
    logger.info(f'\n🎯 Generating Trading Metrics Reports and Visualizations...')
    
    # Create trading metrics report
    try:
        trading_report = create_trading_metrics_report(study, output_dir="reports")
        logger.info(f'✅ Trading metrics report generated successfully')
    except Exception as e:
        logger.error(f'❌ Error generating trading metrics report: {e}')
    
    # Create trading metrics visualizations
    try:
        create_trading_metrics_visualizations(study, output_dir="plots")
        logger.info(f'✅ Trading metrics visualizations generated successfully')
    except Exception as e:
        logger.error(f'❌ Error generating trading metrics visualizations: {e}')
    
    # Final memory cleanup
    gc.collect()

if __name__ == '__main__':
    main() 