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

# Walk-forward configuration parameters for hyperparameter optimization
WF_TRAIN_WINDOW_DAYS = 120  # Shorter for hyperopt speed
WF_TEST_WINDOW_DAYS = 20    # Shorter for hyperopt speed  
WF_STEP_SIZE_DAYS = 10      # Shorter for hyperopt speed

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
    
    # Log data quality check
    logger.info(f"[VISUALIZATION] 📊 Data quality check:")
    logger.info(f"[VISUALIZATION] ├─ Total trials: {len(df)}")
    logger.info(f"[VISUALIZATION] ├─ Trials with infinite objective: {np.sum(~np.isfinite(df['objective_value']))}")
    logger.info(f"[VISUALIZATION] ├─ Trials with infinite Sharpe: {np.sum(~np.isfinite(df['sharpe_ratio']))}")
    logger.info(f"[VISUALIZATION] ├─ Trials with infinite Win Rate: {np.sum(~np.isfinite(df['win_rate']))}")
    logger.info(f"[VISUALIZATION] ├─ Trials with infinite Profit Factor: {np.sum(~np.isfinite(df['profit_factor']))}")
    logger.info(f"[VISUALIZATION] └─ Trials with infinite Max Drawdown: {np.sum(~np.isfinite(df['max_drawdown']))}")
    
    # Filter out trials with infinite values to prevent visualization errors
    original_count = len(df)
    df = df[np.isfinite(df['objective_value']) & 
            np.isfinite(df['sharpe_ratio']) & 
            np.isfinite(df['win_rate']) & 
            np.isfinite(df['profit_factor']) & 
            np.isfinite(df['max_drawdown'])]
    
    filtered_count = len(df)
    if original_count != filtered_count:
        logger.warning(f"[VISUALIZATION] ⚠️  Filtered out {original_count - filtered_count} trials with infinite values")
        logger.warning(f"[VISUALIZATION] ⚠️  Proceeding with {filtered_count} valid trials")
    
    if len(df) == 0:
        logger.error("[VISUALIZATION] ❌ No valid trials remaining after filtering infinite values")
        return
    
    # Apply bounds to prevent extreme values from affecting visualizations
    df['sharpe_ratio'] = np.clip(df['sharpe_ratio'], -10.0, 10.0)
    df['win_rate'] = np.clip(df['win_rate'], 0.0, 1.0)
    df['profit_factor'] = np.clip(df['profit_factor'], 0.0, 10.0)
    df['max_drawdown'] = np.clip(df['max_drawdown'], -1.0, 0.0)
    df['objective_value'] = np.clip(df['objective_value'], -10.0, 10.0)
    
    logger.info(f"[VISUALIZATION] ✅ Applied bounds to all metrics")
    
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
    
    logger.info(f"✅ Trading metrics visualizations saved to: {plot_path}")
    
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

def apply_feature_selection(features, targets, method, min_features, max_features, importance_threshold):
    """
    Apply automated feature selection using various methods with real targets.
    
    Args:
        features: DataFrame with features
        targets: Real target array corresponding to features
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
    
    # Ensure features and targets have same length
    min_length = min(len(features_clean), len(targets))
    features_clean = features_clean.iloc[:min_length]
    targets_clean = targets[:min_length]
    
    logger.info(f"[FEATURE_SELECTION] Starting {method} selection on {len(feature_names)} features with real targets")
    
    if method == 'importance':
        return apply_importance_based_selection(
            features_clean, targets_clean, min_features, max_features, importance_threshold
        )
    elif method == 'correlation':
        return apply_correlation_based_selection(
            features_clean, min_features, max_features
        )
    elif method == 'mutual_info':
        return apply_mutual_info_selection(
            features_clean, targets_clean, min_features, max_features
        )
    else:
        logger.warning(f"[FEATURE_SELECTION] Unknown method '{method}', using all features")
        return feature_names

def apply_importance_based_selection(features, targets, min_features, max_features, importance_threshold):
    """Apply feature selection based on Random Forest importance scores using real targets"""
    logger = get_xgboost_logger()
    
    # Create a simple Random Forest to get feature importance
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    # Use real targets for importance calculation
    sample_size = min(len(features), len(targets))
    
    # Fit the model with real targets
    rf.fit(features.iloc[:sample_size], targets[:sample_size])
    
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

def apply_mutual_info_selection(features, targets, min_features, max_features):
    """Apply feature selection based on mutual information using real targets"""
    logger = get_xgboost_logger()
    
    # Use real targets for mutual information calculation
    sample_size = min(len(features), len(targets))
    
    # Calculate mutual information scores with real targets
    mi_scores = mutual_info_classif(
        features.iloc[:sample_size], 
        targets[:sample_size], 
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

def create_time_based_target(data: pd.DataFrame, config: XGBoostHyperoptConfig):
    """Create unbiased time-based classification target aligned with trading simulation"""
    
    logger = get_xgboost_logger()
    profit_target_pct = config.price_threshold  # Now represents profit target percentage
    stop_loss_pct = 0.015  # Hardcoded 1.5% stop-loss to match trading simulation
    lookahead_window = getattr(config, 'lookahead_window', 15)  # New configurable parameter
        
    logger.info(f"Creating unbiased time-based target with:")
    logger.info(f"  Profit target: {profit_target_pct:.4f}")
    logger.info(f"  Stop loss: {stop_loss_pct:.4f}")
    logger.info(f"  Lookahead window: {lookahead_window}")
        
    targets = []
        
    for i in range(len(data) - lookahead_window):
        current_price = data['close'].iloc[i]
        future_prices = data['close'].iloc[i+1 : i+1+lookahead_window]
        
        # Calculate profit and stop-loss levels for long positions
        long_profit_level = current_price * (1 + profit_target_pct)
        long_stop_loss_level = current_price * (1 - stop_loss_pct)
        
        # Calculate profit and stop-loss levels for short positions  
        short_profit_level = current_price * (1 - profit_target_pct)
        short_stop_loss_level = current_price * (1 + stop_loss_pct)
        
        # Check for profitable long signal (profit hit before stop-loss)
        long_profit_hit = (future_prices >= long_profit_level).any()
        long_stop_hit = (future_prices <= long_stop_loss_level).any()
        
        # Check for profitable short signal (profit hit before stop-loss)
        short_profit_hit = (future_prices <= short_profit_level).any()
        short_stop_hit = (future_prices >= short_stop_loss_level).any()
        
        if long_profit_hit and not long_stop_hit:
            targets.append(2)  # Class 2: Long Signal - profitable trade
        elif short_profit_hit and not short_stop_hit:
            targets.append(0)  # Class 0: Short Signal - profitable trade
        else:
            targets.append(1)  # Class 1: Hold - no profitable opportunity
    
    # Pad the end with hold signals for remaining data points
    targets.extend([1] * (len(data) - len(targets)))
        
    return np.array(targets)

def get_data_splits(start_date: datetime, end_date: datetime, train_window_days: int, test_window_days: int, step_size_days: int):
    """Generate walk-forward data splits"""
    splits = []
    current_start = start_date
    
    while current_start + timedelta(days=train_window_days + test_window_days) <= end_date:
        # Training period
        train_start = current_start
        train_end = train_start + timedelta(days=train_window_days)
        
        # Testing period
        test_start = train_end
        test_end = test_start + timedelta(days=test_window_days)
        
        splits.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'period_name': f"{train_start.strftime('%Y-%m')}_to_{test_end.strftime('%Y-%m')}"
        })
        
        # Move to next period
        current_start += timedelta(days=step_size_days)
    
    return splits

def prepare_period_data(start_date: datetime, end_date: datetime, config: XGBoostHyperoptConfig):
    """Prepare data for a specific time period"""
    
    logger = get_xgboost_logger()
    logger.info(f"Preparing data for period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize components (using your existing market data and technical indicators)
    from market_analysis.market_data import MarketData
    from utils.feature_generator import FeatureGenerator
    
    market_data = MarketData()
    
    # Load market data
    data_response = market_data.get_data(['ETH/USD'])
    
    # Extract DataFrame from response
    if isinstance(data_response, dict):
        data = data_response['ETH/USD']
    else:
        data = data_response
    
    # Filter data for the specific time range
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    logger.info(f"Data shape: {data.shape}")
    
    # Generate features using FeatureGenerator with config
    feature_generator = FeatureGenerator(config=config.__dict__)
    features = feature_generator.generate_features(data)
    
    logger.info(f"Generated {len(features.columns)} features")
    
    # Create target variable
    y = create_time_based_target(data, config)
    
    # Ensure features and target have the same length
    if len(features) != len(y):
        min_length = min(len(features), len(y))
        features = features.iloc[:min_length]
        y = y[:min_length]
        logger.info(f"Aligned features and target to length: {min_length}")
    
    # Remove rows with NaN values
    valid_mask = ~(features.isna().any(axis=1))
    features = features[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"Final features shape: {features.shape}")
    logger.info(f"Final target shape: {y.shape}")
    
    # Report class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))
    logger.info(f"Class distribution: {class_distribution}")
    
    return features, y, data

def calculate_sharpe_ratio_trading_simulation(features_val, y_val, y_pred, y_pred_proba, data_val, config=None):
    """Calculate Sharpe ratio through trading simulation - same as hyperopt runner"""
    
    logger = get_xgboost_logger()
    
    # Trading parameters (aligned with unbiased target definition)
    position_size = 0.04  # 4% of account balance
    leverage = 50  # 50x leverage
    stop_loss = 0.015  # 1.5% stop loss (hardcoded to match target definition)
    trading_fee = 0.0002  # 0.02% per trade
    max_hold_hours = 24  # Max 24 hours hold time
    
    # Get price_threshold from config for take profit
    price_threshold = getattr(config, 'price_threshold', 0.025) if config else 0.025
    
    logger.info(f"[TRADING_SIM] 🎯 Starting trading simulation with parameters:")
    logger.info(f"[TRADING_SIM] ├─ Position Size: {position_size:.3f} ({position_size*100:.1f}%)")
    logger.info(f"[TRADING_SIM] ├─ Leverage: {leverage}x")
    logger.info(f"[TRADING_SIM] ├─ Stop Loss: {stop_loss:.3f} ({stop_loss*100:.1f}%)")
    logger.info(f"[TRADING_SIM] ├─ Take Profit: {price_threshold:.3f} ({price_threshold*100:.1f}%)")
    logger.info(f"[TRADING_SIM] ├─ Trading Fee: {trading_fee:.4f} ({trading_fee*100:.2f}%)")
    logger.info(f"[TRADING_SIM] ├─ Max Hold Time: {max_hold_hours} hours")
    logger.info(f"[TRADING_SIM] ├─ Data Points: {len(y_pred)}")
    logger.info(f"[TRADING_SIM] ├─ Predictions: {np.bincount(y_pred)}")
    logger.info(f"[TRADING_SIM] └─ Confidence Range: [{np.min(y_pred_proba):.4f}, {np.max(y_pred_proba):.4f}]")
            
    # Initialize single account with $4000
    account_balance = 4000  # Starting account balance
    available_balance = 4000  # Available balance for new trades
    trades = []
    positions = []  # Initialize positions list
            
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
    
    # Track simulation progress
    trades_executed = 0
    positions_opened = 0
    positions_closed = 0
    
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
                logger.debug(f"[TRADING_SIM] 📅 Day {current_day_start.strftime('%Y-%m-%d')}: {len(current_day_returns)} trades, daily return: {daily_return:.6f}")
                current_day_returns = []
            current_day_start = current_time
        
        # Check for stop loss on existing positions (MULTIPLE POSITIONS)
        positions_to_remove = []  # Initialize outside the if block
        if positions:
            for pos_idx, position in enumerate(positions):
                hold_hours = (current_time - position['entry_time']).total_seconds() / 3600
                
                # Calculate current P&L
                if position['type'] == 'long':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:  # short
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                # Check stop loss, take profit, or max hold time
                take_profit = price_threshold  # Use config's price_threshold as take profit
                if pnl_pct <= -stop_loss or pnl_pct >= take_profit or hold_hours >= max_hold_hours:
                    # Calculate P&L in USD terms (corrected logic)
                    pnl_usd = pnl_pct * position['position_size_usd'] * leverage
                    
                    # Calculate fees based on leveraged position value
                    trade_value = position['position_size_usd'] * leverage
                    entry_fee = trade_value * trading_fee
                    exit_fee = trade_value * trading_fee
                    total_fees = entry_fee + exit_fee
                    
                    # Calculate net trade return in USD
                    net_pnl_usd = pnl_usd - total_fees
                    
                    # Calculate trade return as percentage of position size for metrics
                    trade_return = net_pnl_usd / position['position_size_usd']
                    
                    # Log trade details for debugging
                    logger.debug(f"[TRADING_SIM] 💰 Closing position {pos_idx}: {position['type']} @ {position['entry_price']:.4f} → {current_price:.4f}")
                    logger.debug(f"[TRADING_SIM] ├─ P&L %: {pnl_pct:.4f} | P&L USD: {pnl_usd:.2f} | Fees: {total_fees:.2f}")
                    logger.debug(f"[TRADING_SIM] ├─ Net P&L USD: {net_pnl_usd:.2f} | Trade Return: {trade_return:.4f}")
                    logger.debug(f"[TRADING_SIM] ├─ Hold Time: {hold_hours:.1f}h | Reason: {'stop_loss' if pnl_pct <= -stop_loss else 'take_profit' if pnl_pct >= take_profit else 'max_hold_time'}")
                    
                    trades.append({
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'return': trade_return,
                        'hold_hours': hold_hours,
                        'reason': 'stop_loss' if pnl_pct <= -stop_loss else 'take_profit' if pnl_pct >= take_profit else 'max_hold_time'
                    })
                    
                    trades_executed += 1
                    positions_closed += 1
                    
                    # Track consecutive losses
                    if trade_return < 0:
                        consecutive_losses += 1
                        if consecutive_losses >= 3:
                            trading_pause_until = current_time + timedelta(hours=12)
                            consecutive_losses = 0  # Reset after pause
                            logger.debug(f"[TRADING_SIM] ⏸️  Trading pause activated after {consecutive_losses} consecutive losses")
                    else:
                        consecutive_losses = 0  # Reset on win
                    
                    # Add to daily returns
                    current_day_returns.append(trade_return)
                    
                    # Update account balance correctly
                    account_balance += net_pnl_usd
                    # Return position size plus profit/loss to available balance
                    available_balance += position['position_size_usd'] + net_pnl_usd
                    
                    # Mark position for removal
                    positions_to_remove.append(pos_idx)
    
        # Remove closed positions (in reverse order to maintain indices)
        for i in reversed(positions_to_remove):
            positions.pop(i)
        
        # Check for new trading signals (only take signals 0 and 2) with confidence threshold
        confidence_threshold = 0.25  # Lower threshold to allow more trades
        if prediction in [0, 2] and available_balance > 0 and confidence >= confidence_threshold:  # Early Down or Early Up
            # Calculate position size based on available balance
            position_size_usd = available_balance * position_size
            
            # Check if we have enough balance for the trade
            if position_size_usd >= 20:  # Minimum $20 position
                # Open new position (ALLOWING MULTIPLE POSITIONS)
                new_position = {
                    'type': 'short' if prediction == 0 else 'long',
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'position_size_usd': position_size_usd
                }
                
                # Add to positions list (instead of single current_position)
                positions.append(new_position)
                positions_opened += 1
                
                # Deduct position size from available balance
                available_balance -= position_size_usd
                
                # Pay entry fee based on leveraged position value
                trade_value = position_size_usd * leverage
                position_fee = trade_value * trading_fee
                account_balance -= position_fee
                
                logger.debug(f"[TRADING_SIM] 🚀 Opening {new_position['type']} position: ${position_size_usd:.2f} @ {current_price:.4f}")
                logger.debug(f"[TRADING_SIM] ├─ Prediction: {prediction} | Confidence: {confidence:.4f}")
                logger.debug(f"[TRADING_SIM] ├─ Available Balance: ${available_balance:.2f} | Account Balance: ${account_balance:.2f}")
    
            # Close any remaining positions at the end
    if positions:
        final_price = prices[-1]
        logger.debug(f"[TRADING_SIM] 📊 Closing {len(positions)} remaining positions at final price: {final_price:.4f}")
        
        for pos_idx, position in enumerate(positions):
            if position['type'] == 'long':
                pnl_pct = (final_price - position['entry_price']) / position['entry_price']
            else:  # short
                pnl_pct = (position['entry_price'] - final_price) / position['entry_price']
            
            # Calculate P&L in USD terms (corrected logic)
            pnl_usd = pnl_pct * position['position_size_usd'] * leverage
            
            # Calculate fees based on leveraged position value
            trade_value = position['position_size_usd'] * leverage
            entry_fee = trade_value * trading_fee
            exit_fee = trade_value * trading_fee
            total_fees = entry_fee + exit_fee
            
            # Calculate net trade return in USD
            net_pnl_usd = pnl_usd - total_fees
            
            # Calculate trade return as percentage of position size for metrics
            trade_return = net_pnl_usd / position['position_size_usd']
            
            logger.debug(f"[TRADING_SIM] 💰 Final close position {pos_idx}: {position['type']} @ {position['entry_price']:.4f} → {final_price:.4f}")
            logger.debug(f"[TRADING_SIM] ├─ P&L %: {pnl_pct:.4f} | P&L USD: {pnl_usd:.2f} | Trade Return: {trade_return:.4f}")
            
            trades.append({
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'return': trade_return,
                'hold_hours': (timestamps[-1] - position['entry_time']).total_seconds() / 3600,
                'reason': 'end_of_period'
            })
            current_day_returns.append(trade_return)
            trades_executed += 1
            positions_closed += 1
            
            # Update account balance correctly
            account_balance += net_pnl_usd
            # Return position size plus profit/loss to available balance
            available_balance += position['position_size_usd'] + net_pnl_usd
            
    # Add final day returns
    if current_day_returns:
        daily_return = np.sum(current_day_returns)
        daily_returns.append(daily_return)
        logger.debug(f"[TRADING_SIM] 📅 Final day {current_day_start.strftime('%Y-%m-%d')}: {len(current_day_returns)} trades, daily return: {daily_return:.6f}")
            
    # Calculate Sharpe ratio (daily)
    if len(daily_returns) < 2:
        logger.warning(f"[TRADING_SIM] ⚠️  Insufficient daily returns for Sharpe calculation: {len(daily_returns)} < 2")
        return -10.0, 0.0, 0.0, 0.0, []  # Penalty for insufficient data
    
    daily_returns = np.array(daily_returns)
    
    # Log daily returns statistics for debugging
    logger.info(f"[TRADING_SIM] 📊 Daily Returns Analysis:")
    logger.info(f"[TRADING_SIM] ├─ Count: {len(daily_returns)}")
    logger.info(f"[TRADING_SIM] ├─ Mean: {np.mean(daily_returns):.6f}")
    logger.info(f"[TRADING_SIM] ├─ Std: {np.std(daily_returns):.6f}")
    logger.info(f"[TRADING_SIM] ├─ Min: {np.min(daily_returns):.6f}")
    logger.info(f"[TRADING_SIM] ├─ Max: {np.max(daily_returns):.6f}")
    logger.info(f"[TRADING_SIM] ├─ Range: [{np.min(daily_returns):.6f}, {np.max(daily_returns):.6f}]")
    logger.info(f"[TRADING_SIM] └─ Has NaN: {np.any(np.isnan(daily_returns))} | Has Inf: {np.any(np.isinf(daily_returns))}")
    
    # Fix infinity values in Sharpe ratio calculation
    std_returns = np.std(daily_returns)
    if std_returns < 1e-8 or np.isnan(std_returns) or np.isinf(std_returns):
        sharpe_ratio = 0.0  # No volatility or invalid data
        logger.warning(f"[TRADING_SIM] ⚠️  Invalid std_returns for Sharpe: {std_returns:.8f}, setting Sharpe to 0.0")
    else:
        sharpe_ratio = np.mean(daily_returns) / std_returns
        # Bound Sharpe ratio to prevent infinity
        sharpe_ratio = np.clip(sharpe_ratio, -10.0, 10.0)
        logger.info(f"[TRADING_SIM] ✅ Sharpe ratio calculated: {sharpe_ratio:.4f} (clipped from {np.mean(daily_returns) / std_returns:.4f})")

    # Calculate additional metrics
    if trades:
        trade_returns = [t['return'] for t in trades]
        profitable_trades = [r for r in trade_returns if r > 0]
        win_rate = len(profitable_trades) / len(trade_returns) if trade_returns else 0
        
        # Log trade returns statistics
        logger.info(f"[TRADING_SIM] 📊 Trade Returns Analysis:")
        logger.info(f"[TRADING_SIM] ├─ Total Trades: {len(trade_returns)}")
        logger.info(f"[TRADING_SIM] ├─ Profitable Trades: {len(profitable_trades)}")
        logger.info(f"[TRADING_SIM] ├─ Win Rate: {win_rate:.4f}")
        logger.info(f"[TRADING_SIM] ├─ Mean Return: {np.mean(trade_returns):.6f}")
        logger.info(f"[TRADING_SIM] ├─ Std Return: {np.std(trade_returns):.6f}")
        logger.info(f"[TRADING_SIM] ├─ Min Return: {np.min(trade_returns):.6f}")
        logger.info(f"[TRADING_SIM] ├─ Max Return: {np.max(trade_returns):.6f}")
        logger.info(f"[TRADING_SIM] └─ Has NaN: {np.any(np.isnan(trade_returns))} | Has Inf: {np.any(np.isinf(trade_returns))}")
        
        # Profit factor with infinity protection
        gross_profit = sum([r for r in trade_returns if r > 0])
        gross_loss = abs(sum([r for r in trade_returns if r < 0]))
        
        logger.info(f"[TRADING_SIM] 💰 Profit Factor Calculation:")
        logger.info(f"[TRADING_SIM] ├─ Gross Profit: {gross_profit:.6f}")
        logger.info(f"[TRADING_SIM] ├─ Gross Loss: {gross_loss:.6f}")
        
        if gross_loss < 1e-8:
            profit_factor = 10.0 if gross_profit > 0 else 0.0  # Cap at 10.0 for perfect performance
            logger.warning(f"[TRADING_SIM] ⚠️  Gross loss too small: {gross_loss:.8f}, setting profit factor to {profit_factor:.1f}")
        else:
            profit_factor = gross_profit / gross_loss
            # Bound profit factor to prevent infinity
            profit_factor = np.clip(profit_factor, 0.0, 10.0)
            logger.info(f"[TRADING_SIM] ✅ Profit factor calculated: {profit_factor:.4f} (clipped from {gross_profit / gross_loss:.4f})")
        
        # Max drawdown with bounds
        cumulative_returns = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        # Bound max drawdown to prevent extreme negative values
        max_drawdown = np.clip(max_drawdown, -1.0, 0.0)
        
        logger.info(f"[TRADING_SIM] 📉 Max Drawdown Analysis:")
        logger.info(f"[TRADING_SIM] ├─ Raw Max Drawdown: {np.min(drawdown):.6f}")
        logger.info(f"[TRADING_SIM] ├─ Clipped Max Drawdown: {max_drawdown:.6f}")
        logger.info(f"[TRADING_SIM] └─ Cumulative Returns Range: [{np.min(cumulative_returns):.6f}, {np.max(cumulative_returns):.6f}]")
    else:
        win_rate = 0.0
        profit_factor = 0.0
        max_drawdown = 0.0
        logger.warning(f"[TRADING_SIM] ⚠️  No trades executed, setting all metrics to 0.0")
    
    # Final summary logging
    logger.info(f"[TRADING_SIM] 🎯 Trading Simulation Summary:")
    logger.info(f"[TRADING_SIM] ├─ Positions Opened: {positions_opened}")
    logger.info(f"[TRADING_SIM] ├─ Positions Closed: {positions_closed}")
    logger.info(f"[TRADING_SIM] ├─ Trades Executed: {trades_executed}")
    logger.info(f"[TRADING_SIM] ├─ Final Account Balance: ${account_balance:.2f}")
    logger.info(f"[TRADING_SIM] ├─ Final Available Balance: ${available_balance:.2f}")
    logger.info(f"[TRADING_SIM] ├─ Sharpe Ratio: {sharpe_ratio:.4f}")
    logger.info(f"[TRADING_SIM] ├─ Win Rate: {win_rate:.4f}")
    logger.info(f"[TRADING_SIM] ├─ Profit Factor: {profit_factor:.4f}")
    logger.info(f"[TRADING_SIM] └─ Max Drawdown: {max_drawdown:.4f}")
    
    # Final validation check for infinity values
    final_metrics = [sharpe_ratio, win_rate, profit_factor, max_drawdown]
    metric_names = ['Sharpe Ratio', 'Win Rate', 'Profit Factor', 'Max Drawdown']
    
    for i, (metric, name) in enumerate(zip(final_metrics, metric_names)):
        if not np.isfinite(metric):
            logger.error(f"[TRADING_SIM] ❌ CRITICAL: {name} is not finite: {metric}")
        else:
            logger.info(f"[TRADING_SIM] ✅ {name}: {metric:.6f} (finite)")
            
    return sharpe_ratio, win_rate, profit_factor, max_drawdown, trades

def evaluate_hyperparams_walk_forward(trial_params, start_date, end_date):
    """
    Evaluate hyperparameters using walk-forward analysis
    
    Args:
        trial_params: Dictionary of hyperparameters to evaluate
        start_date: Start date for walk-forward evaluation
        end_date: End date for walk-forward evaluation
        
    Returns:
        float: Average Sharpe ratio across all walk-forward splits
    """
    logger = get_xgboost_logger()
    logger.info(f"[WALK_FORWARD] Starting walk-forward evaluation from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate walk-forward splits using the shorter windows for hyperopt
    splits = get_data_splits(
        start_date=start_date,
        end_date=end_date,
        train_window_days=WF_TRAIN_WINDOW_DAYS,
        test_window_days=WF_TEST_WINDOW_DAYS,
        step_size_days=WF_STEP_SIZE_DAYS
    )
    
    logger.info(f"[WALK_FORWARD] Generated {len(splits)} walk-forward splits")
    
    if len(splits) == 0:
        logger.warning(f"[WALK_FORWARD] No valid splits generated, returning penalty")
        return -10.0, 0.0, 0.0, 0.0, []  # Penalty for no data
    
    # Store results for each split
    split_results = []
    
    # Create configuration from trial parameters
    config = XGBoostHyperoptConfig(**trial_params)
    
    # Enhanced progress tracking
    logger.info(f"[WALK_FORWARD_PROGRESS] ═══════════════════════════════════════")
    logger.info(f"[WALK_FORWARD_PROGRESS] Starting Walk-Forward Evaluation")
    logger.info(f"[WALK_FORWARD_PROGRESS] Total Splits: {len(splits)}")
    logger.info(f"[WALK_FORWARD_PROGRESS] Train Window: {WF_TRAIN_WINDOW_DAYS} days")
    logger.info(f"[WALK_FORWARD_PROGRESS] Test Window: {WF_TEST_WINDOW_DAYS} days") 
    logger.info(f"[WALK_FORWARD_PROGRESS] Step Size: {WF_STEP_SIZE_DAYS} days")
    logger.info(f"[WALK_FORWARD_PROGRESS] Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"[WALK_FORWARD_PROGRESS] ═══════════════════════════════════════")
    
    for i, split in enumerate(splits):
        split_start_time = datetime.now()
        progress_pct = ((i + 1) / len(splits)) * 100
        
        logger.info(f"[WALK_FORWARD_PROGRESS] ┌─── Split {i+1}/{len(splits)} ({progress_pct:.1f}%) ───┐")
        logger.info(f"[WALK_FORWARD_PROGRESS] │ Period: {split['period_name']}")
        logger.info(f"[WALK_FORWARD_PROGRESS] │ Train: {split['train_start'].strftime('%Y-%m-%d')} → {split['train_end'].strftime('%Y-%m-%d')}")
        logger.info(f"[WALK_FORWARD_PROGRESS] │ Test:  {split['test_start'].strftime('%Y-%m-%d')} → {split['test_end'].strftime('%Y-%m-%d')}")
        logger.info(f"[WALK_FORWARD_PROGRESS] └───────────────────────────────────────┘")

        try:
            # Prepare training data
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 📊 Preparing training data...")
            X_train, y_train, data_train = prepare_period_data(
                split['train_start'], 
                split['train_end'], 
                config
            )
            
            # Prepare testing data  
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 📊 Preparing testing data...")
            X_test, y_test, data_test = prepare_period_data(
                split['test_start'],
                split['test_end'],
                config
            )
            
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 📈 Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Check if we have enough data for this split
            if len(X_train) < 50 or len(X_test) < 10:
                logger.warning(f"[WALK_FORWARD_PROGRESS] ├─ ⚠️  Insufficient data (train: {len(X_train)}, test: {len(X_test)}), skipping split")
                continue
            
            # Apply feature selection if enabled
            if trial_params.get('use_feature_selection', True):
                logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 🔍 Applying feature selection with real targets...")
                selected_features = apply_feature_selection(
                    X_train,
                    y_train,
                    trial_params.get('feature_selection_method', 'importance'),
                    trial_params.get('min_features', 1),
                    trial_params.get('max_features', 26),
                    trial_params.get('importance_threshold', 0.9)
                )
                
                # Apply feature selection to both train and test sets
                X_train = X_train[selected_features]
                X_test = X_test[selected_features]
                
                logger.info(f"[WALK_FORWARD_PROGRESS] ├─ ✅ Feature selection applied: {len(selected_features)} features")
            
            # Create trainer for this split
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 🤖 Creating XGBoost trainer...")
            trainer = XGBoostHyperoptTrainer(config, None)
            
            # Train model on this split
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 🚀 Training model with SMOTE...")
            train_start_time = datetime.now()
            training_history = trainer.train_with_smote(
                X_train.values, y_train, 
                X_test.values, y_test
            )
            train_duration = (datetime.now() - train_start_time).total_seconds()
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ ✅ Training completed in {train_duration:.1f}s, score: {training_history['best_score']:.4f}")
            
            # Check for poor performance
            if training_history['best_score'] < 0.4:
                logger.warning(f"[WALK_FORWARD_PROGRESS] ├─ ⚠️  Split {i+1} has poor training score: {training_history['best_score']:.4f}, skipping")
                continue
            
            # Get predictions on test set
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 🔮 Generating predictions...")
            import xgboost as xgb
            dtest = xgb.DMatrix(X_test.values)
            y_pred_proba = trainer.model.predict(dtest, output_margin=False)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate trading metrics using the simulation
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 💹 Running trading simulation...")
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 📊 Test data shape: {X_test.shape}")
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 📊 Predictions distribution: {np.bincount(y_pred)}")
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 📊 Confidence range: [{np.min(y_pred_proba):.4f}, {np.max(y_pred_proba):.4f}]")
            
            sharpe_ratio, win_rate, profit_factor, max_drawdown, trades = calculate_sharpe_ratio_trading_simulation(
                X_test, y_test, y_pred, y_pred_proba, data_test, config
            )
            
            # Log immediate results from trading simulation
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ 📊 Trading Simulation Results:")
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ ├─ Sharpe Ratio: {sharpe_ratio:.6f} (finite: {np.isfinite(sharpe_ratio)})")
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ ├─ Win Rate: {win_rate:.6f} (finite: {np.isfinite(win_rate)})")
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ ├─ Profit Factor: {profit_factor:.6f} (finite: {np.isfinite(profit_factor)})")
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ ├─ Max Drawdown: {max_drawdown:.6f} (finite: {np.isfinite(max_drawdown)})")
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ └─ Number of Trades: {len(trades)}")
            
            # Check for infinity values immediately after simulation
            if not all(np.isfinite([sharpe_ratio, win_rate, profit_factor, max_drawdown])):
                logger.error(f"[WALK_FORWARD_PROGRESS] ❌ CRITICAL: Split {i+1} produced non-finite metrics!")
                logger.error(f"[WALK_FORWARD_PROGRESS] ❌ Sharpe: {sharpe_ratio} (finite: {np.isfinite(sharpe_ratio)})")
                logger.error(f"[WALK_FORWARD_PROGRESS] ❌ Win Rate: {win_rate} (finite: {np.isfinite(win_rate)})")
                logger.error(f"[WALK_FORWARD_PROGRESS] ❌ Profit Factor: {profit_factor} (finite: {np.isfinite(profit_factor)})")
                logger.error(f"[WALK_FORWARD_PROGRESS] ❌ Max Drawdown: {max_drawdown} (finite: {np.isfinite(max_drawdown)})")
            
            # Store results for this split
            split_result = {
                'split_name': split['period_name'],
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'num_trades': len(trades),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            split_results.append(split_result)
            
            # Calculate split completion timing
            split_duration = (datetime.now() - split_start_time).total_seconds()
            logger.info(f"[WALK_FORWARD_PROGRESS] ├─ ✅ Split {i+1} completed in {split_duration:.1f}s - Sharpe: {sharpe_ratio:.4f}, Win Rate: {win_rate:.4f}, Trades: {len(trades)}")
            
        except Exception as e:
            split_duration = (datetime.now() - split_start_time).total_seconds()
            logger.error(f"[WALK_FORWARD_PROGRESS] ├─ ❌ Error processing split {i+1} after {split_duration:.1f}s: {e}")
            continue
    
    # Calculate average results across all valid splits
    if len(split_results) == 0:
        logger.warning(f"[WALK_FORWARD_PROGRESS] ⚠️  No valid splits processed, returning penalty")
        logger.info(f"[WALK_FORWARD_PROGRESS] ═══════════════════════════════════════")
        return -10.0, 0.0, 0.0, 0.0, []  # Penalty for no valid results
    
    # Log individual split results for debugging
    logger.info(f"[WALK_FORWARD_PROGRESS] 📊 Individual Split Results:")
    for i, result in enumerate(split_results):
        logger.info(f"[WALK_FORWARD_PROGRESS] ├─ Split {i+1}: Sharpe={result['sharpe_ratio']:.4f}, WR={result['win_rate']:.4f}, PF={result['profit_factor']:.4f}, DD={result['max_drawdown']:.4f}, Trades={result['num_trades']}")
    
    # Calculate average Sharpe ratio (primary metric)
    sharpe_ratios = [result['sharpe_ratio'] for result in split_results]
    avg_sharpe_ratio = np.mean(sharpe_ratios)
    
    # Calculate other averages for logging
    avg_win_rate = np.mean([result['win_rate'] for result in split_results])
    avg_profit_factor = np.mean([result['profit_factor'] for result in split_results])
    avg_max_drawdown = np.mean([result['max_drawdown'] for result in split_results])
    total_trades = sum([result['num_trades'] for result in split_results])
    
    # Log aggregation details
    logger.info(f"[WALK_FORWARD_PROGRESS] 📊 Metric Aggregation Details:")
    logger.info(f"[WALK_FORWARD_PROGRESS] ├─ Sharpe Ratios: {sharpe_ratios}")
    logger.info(f"[WALK_FORWARD_PROGRESS] ├─ Win Rates: {[result['win_rate'] for result in split_results]}")
    logger.info(f"[WALK_FORWARD_PROGRESS] ├─ Profit Factors: {[result['profit_factor'] for result in split_results]}")
    logger.info(f"[WALK_FORWARD_PROGRESS] └─ Max Drawdowns: {[result['max_drawdown'] for result in split_results]}")
    
    # Check for infinity values in individual splits
    infinity_checks = {
        'sharpe_ratios': [not np.isfinite(r) for r in sharpe_ratios],
        'win_rates': [not np.isfinite(r) for r in [result['win_rate'] for result in split_results]],
        'profit_factors': [not np.isfinite(r) for r in [result['profit_factor'] for result in split_results]],
        'max_drawdowns': [not np.isfinite(r) for r in [result['max_drawdown'] for result in split_results]]
    }
    
    logger.info(f"[WALK_FORWARD_PROGRESS] 🔍 Infinity Checks:")
    logger.info(f"[WALK_FORWARD_PROGRESS] ├─ Sharpe Ratios with Inf: {sum(infinity_checks['sharpe_ratios'])}/{len(sharpe_ratios)}")
    logger.info(f"[WALK_FORWARD_PROGRESS] ├─ Win Rates with Inf: {sum(infinity_checks['win_rates'])}/{len(infinity_checks['win_rates'])}")
    logger.info(f"[WALK_FORWARD_PROGRESS] ├─ Profit Factors with Inf: {sum(infinity_checks['profit_factors'])}/{len(infinity_checks['profit_factors'])}")
    logger.info(f"[WALK_FORWARD_PROGRESS] └─ Max Drawdowns with Inf: {sum(infinity_checks['max_drawdowns'])}/{len(infinity_checks['max_drawdowns'])}")
    
    # Log which specific splits have infinity values
    for i, (sharpe_inf, wr_inf, pf_inf, dd_inf) in enumerate(zip(
        infinity_checks['sharpe_ratios'], 
        infinity_checks['win_rates'], 
        infinity_checks['profit_factors'], 
        infinity_checks['max_drawdowns']
    )):
        if any([sharpe_inf, wr_inf, pf_inf, dd_inf]):
            logger.error(f"[WALK_FORWARD_PROGRESS] ❌ Split {i+1} has infinity values: Sharpe={sharpe_inf}, WR={wr_inf}, PF={pf_inf}, DD={dd_inf}")
    
    # Log final averages
    logger.info(f"[WALK_FORWARD_PROGRESS] 📊 Final Averages:")
    logger.info(f"[WALK_FORWARD_PROGRESS] ├─ Avg Sharpe: {avg_sharpe_ratio:.6f} (finite: {np.isfinite(avg_sharpe_ratio)})")
    logger.info(f"[WALK_FORWARD_PROGRESS] ├─ Avg Win Rate: {avg_win_rate:.6f} (finite: {np.isfinite(avg_win_rate)})")
    logger.info(f"[WALK_FORWARD_PROGRESS] ├─ Avg Profit Factor: {avg_profit_factor:.6f} (finite: {np.isfinite(avg_profit_factor)})")
    logger.info(f"[WALK_FORWARD_PROGRESS] └─ Avg Max Drawdown: {avg_max_drawdown:.6f} (finite: {np.isfinite(avg_max_drawdown)})")
    
    # Success rate calculation
    success_rate = (len(split_results) / len(splits)) * 100
    
    logger.info(f"[WALK_FORWARD_PROGRESS] ┌─── 🎯 WALK-FORWARD RESULTS ───┐")
    logger.info(f"[WALK_FORWARD_PROGRESS] │ Success Rate: {success_rate:.1f}% ({len(split_results)}/{len(splits)} splits)")
    logger.info(f"[WALK_FORWARD_PROGRESS] │ Average Sharpe:   {avg_sharpe_ratio:>8.4f}")
    logger.info(f"[WALK_FORWARD_PROGRESS] │ Average Win Rate: {avg_win_rate:>8.4f}")
    logger.info(f"[WALK_FORWARD_PROGRESS] │ Average Profit:   {avg_profit_factor:>8.4f}")
    logger.info(f"[WALK_FORWARD_PROGRESS] │ Average Drawdown: {avg_max_drawdown:>8.4f}")
    logger.info(f"[WALK_FORWARD_PROGRESS] │ Total Trades:     {total_trades:>8d}")
    logger.info(f"[WALK_FORWARD_PROGRESS] │ Sharpe Range: {min(sharpe_ratios):>5.3f} → {max(sharpe_ratios):>5.3f}")
    logger.info(f"[WALK_FORWARD_PROGRESS] └───────────────────────────────┘")
    logger.info(f"[WALK_FORWARD_PROGRESS] ═══════════════════════════════════════")
    
    return avg_sharpe_ratio, avg_win_rate, avg_profit_factor, avg_max_drawdown, split_results

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for XGBoost mean reversion optimization with automated feature selection"""
    logger = get_xgboost_logger()
    
    # Enhanced trial start logging with timing
    trial_start_time = datetime.now()
    logger.info(f"[TRIAL_PROGRESS] ╔══════════════════════════════════════════════════════╗")
    logger.info(f"[TRIAL_PROGRESS] ║                    TRIAL {trial.number:4d}                      ║")
    logger.info(f"[TRIAL_PROGRESS] ║              Walk-Forward Hyperopt Trial            ║")
    logger.info(f"[TRIAL_PROGRESS] ║  Started: {trial_start_time.strftime('%Y-%m-%d %H:%M:%S')}                    ║")
    logger.info(f"[TRIAL_PROGRESS] ╚══════════════════════════════════════════════════════╝")
    
    # Log trial number for easy tracking
    logger.info(f"[TRIAL_{trial.number}] 🎯 Starting trial {trial.number}")
    logger.info(f"[TRIAL_{trial.number}] 📊 Trial {trial.number} - Infinity Debug Mode Active")
    
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
            
            # Mean reversion specific parameters (now aligned with trading simulation)
            'price_threshold': trial.suggest_float('price_threshold', 0.01, 0.05),  # Now represents profit_target_pct
            'probability_threshold': trial.suggest_float('probability_threshold', 0.5, 0.8),
            
            # Unbiased target parameters
            'lookahead_window': trial.suggest_int('lookahead_window', 5, 30),  # Configurable lookahead window
            
            # Feature selection parameters
            'use_feature_selection': trial.suggest_categorical('use_feature_selection', [True]),
            'feature_selection_method': trial.suggest_categorical('feature_selection_method', ['importance', 'correlation', 'mutual_info']),
            'min_features': trial.suggest_int('min_features', 8, 15),
            'max_features': trial.suggest_int('max_features', 20, 26),
            'importance_threshold': trial.suggest_float('importance_threshold', 0.1, 0.7),
            
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
        
        # Call walk-forward evaluation with the hyperparameters
        start_date = datetime(2024, 1, 2)
        end_date = datetime(2025, 8, 3)
        
        logger.info(f"[TRIAL_PROGRESS] 🚀 Starting walk-forward evaluation...")
        logger.info(f"[TRIAL_PROGRESS] 📅 Date range: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        
        wf_start_time = datetime.now()
        avg_sharpe_ratio, avg_win_rate, avg_profit_factor, avg_max_drawdown, split_results = evaluate_hyperparams_walk_forward(
            config_dict, start_date, end_date
        )
        wf_duration = (datetime.now() - wf_start_time).total_seconds()
        
        logger.info(f"[TRIAL_PROGRESS] ✅ Walk-forward evaluation completed in {wf_duration:.1f}s")
        
        # Apply trial pruning based on average metrics
        if avg_sharpe_ratio < -1.0:
            reason = f"Poor average Sharpe ratio: {avg_sharpe_ratio:.4f} < -1.0"
            log_trial_pruned(trial.number, reason)
            raise TrialPruned(reason)
        
        if len(split_results) == 0:
            reason = "No valid walk-forward splits processed"
            log_trial_pruned(trial.number, reason)
            raise TrialPruned(reason)
        
        # Apply the same weighting formula to averaged metrics as the original objective
        # Prioritize Sharpe ratio (40%), Profit Factor (20%), Win Rate (35%), Max Drawdown (5%)
        
        # Log raw metrics before processing
        logger.info(f"[OBJECTIVE] 📊 Raw Walk-Forward Metrics (before bounds):")
        logger.info(f"[OBJECTIVE] ├─ Sharpe Ratio: {avg_sharpe_ratio:.6f} (finite: {np.isfinite(avg_sharpe_ratio)})")
        logger.info(f"[OBJECTIVE] ├─ Profit Factor: {avg_profit_factor:.6f} (finite: {np.isfinite(avg_profit_factor)})")
        logger.info(f"[OBJECTIVE] ├─ Win Rate: {avg_win_rate:.6f} (finite: {np.isfinite(avg_win_rate)})")
        logger.info(f"[OBJECTIVE] └─ Max Drawdown: {avg_max_drawdown:.6f} (finite: {np.isfinite(avg_max_drawdown)})")
        
        # Ensure all metrics are finite and bounded
        avg_sharpe_ratio = np.clip(avg_sharpe_ratio, -10.0, 10.0) if np.isfinite(avg_sharpe_ratio) else 0.0
        avg_profit_factor = np.clip(avg_profit_factor, 0.0, 10.0) if np.isfinite(avg_profit_factor) else 0.0
        avg_win_rate = np.clip(avg_win_rate, 0.0, 1.0) if np.isfinite(avg_win_rate) else 0.0
        avg_max_drawdown = np.clip(avg_max_drawdown, -1.0, 0.0) if np.isfinite(avg_max_drawdown) else 0.0
        
        # Log bounded metrics after processing
        logger.info(f"[OBJECTIVE] 📊 Bounded Metrics (after bounds):")
        logger.info(f"[OBJECTIVE] ├─ Sharpe Ratio: {avg_sharpe_ratio:.6f}")
        logger.info(f"[OBJECTIVE] ├─ Profit Factor: {avg_profit_factor:.6f}")
        logger.info(f"[OBJECTIVE] ├─ Win Rate: {avg_win_rate:.6f}")
        logger.info(f"[OBJECTIVE] └─ Max Drawdown: {avg_max_drawdown:.6f}")
        
        sharpe_penalty = -avg_sharpe_ratio * 0.4  # Negative because we minimize
        profit_factor_penalty = (1 - min(avg_profit_factor, 3.0)) * 0.2  # Cap at 3.0
        win_rate_penalty = (1 - avg_win_rate) * 0.35
        drawdown_penalty = abs(avg_max_drawdown) * 0.05  # Penalize large drawdowns
        
        # Log penalty calculations
        logger.info(f"[OBJECTIVE] 📊 Penalty Calculations:")
        logger.info(f"[OBJECTIVE] ├─ Sharpe Penalty: {sharpe_penalty:.6f} (from {avg_sharpe_ratio:.6f} * 0.4)")
        logger.info(f"[OBJECTIVE] ├─ Profit Factor Penalty: {profit_factor_penalty:.6f} (from {avg_profit_factor:.6f} capped at 3.0)")
        logger.info(f"[OBJECTIVE] ├─ Win Rate Penalty: {win_rate_penalty:.6f} (from {avg_win_rate:.6f})")
        logger.info(f"[OBJECTIVE] └─ Drawdown Penalty: {drawdown_penalty:.6f} (from {avg_max_drawdown:.6f})")
        
        # Combined objective value
        objective_value = sharpe_penalty + profit_factor_penalty + win_rate_penalty + drawdown_penalty
        
        # Log final objective calculation
        logger.info(f"[OBJECTIVE] 📊 Final Objective Calculation:")
        logger.info(f"[OBJECTIVE] ├─ Formula: {sharpe_penalty:.6f} + {profit_factor_penalty:.6f} + {win_rate_penalty:.6f} + {drawdown_penalty:.6f}")
        logger.info(f"[OBJECTIVE] ├─ Result: {objective_value:.6f}")
        logger.info(f"[OBJECTIVE] └─ Finite: {np.isfinite(objective_value)}")
        
        # Final safety check to ensure objective value is finite
        if not np.isfinite(objective_value):
            logger.error(f"[OBJECTIVE] ❌ CRITICAL: Non-finite objective value detected: {objective_value}")
            logger.error(f"[OBJECTIVE] ❌ Components: Sharpe={sharpe_penalty}, PF={profit_factor_penalty}, WR={win_rate_penalty}, DD={drawdown_penalty}")
            logger.error(f"[OBJECTIVE] ❌ Raw metrics: Sharpe={avg_sharpe_ratio}, PF={avg_profit_factor}, WR={avg_win_rate}, DD={avg_max_drawdown}")
            objective_value = 10.0  # Penalty for non-finite objective
            logger.warning(f"[OBJECTIVE] ⚠️  Setting objective to penalty value: {objective_value}")
        
        # Store individual split results in trial user attributes (Option B format)
        logger.info(f"[USER_ATTR] Storing walk-forward results for trial {trial.number}")
        try:
            trial.set_user_attr('split_results', split_results)
            trial.set_user_attr('sharpe_ratio', float(avg_sharpe_ratio))
            trial.set_user_attr('win_rate', float(avg_win_rate))
            trial.set_user_attr('profit_factor', float(avg_profit_factor))
            trial.set_user_attr('max_drawdown', float(avg_max_drawdown))
            trial.set_user_attr('num_splits', len(split_results))
            
            logger.info(f"[USER_ATTR] Successfully stored walk-forward results")
        except Exception as e:
            logger.error(f"[USER_ATTR] Error storing walk-forward results: {e}")
        
        # Log detailed trial results
        total_trades = sum([result['num_trades'] for result in split_results])
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Trial {trial.number} Walk-Forward Results:")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Valid splits: {len(split_results)}")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Average Sharpe Ratio: {avg_sharpe_ratio:.4f}")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Average Win Rate: {avg_win_rate:.4f}")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Average Profit Factor: {avg_profit_factor:.4f}")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Average Max Drawdown: {avg_max_drawdown:.4f}")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Total Trades: {total_trades}")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Objective Components:")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Sharpe Penalty: {sharpe_penalty:.4f}")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Profit Factor Penalty: {profit_factor_penalty:.4f}")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Win Rate Penalty: {win_rate_penalty:.4f}")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Drawdown Penalty: {drawdown_penalty:.4f}")
        logger.info(f"[WALK_FORWARD_OBJECTIVE] Combined Objective: {objective_value:.4f}")
        
        # Log trial completion with walk-forward metrics
        metrics = {
            'avg_sharpe_ratio': avg_sharpe_ratio,
            'avg_win_rate': avg_win_rate,
            'avg_profit_factor': avg_profit_factor,
            'avg_max_drawdown': avg_max_drawdown,
            'objective_value': objective_value,
            'num_splits': len(split_results),
            'total_trades': total_trades,
            'hyperparameters': {
            'max_depth': config_dict['max_depth'],
            'learning_rate': config_dict['learning_rate'],
            'price_threshold': config_dict['price_threshold'],
            'use_feature_selection': config_dict['use_feature_selection'],
                'feature_selection_method': config_dict.get('feature_selection_method', 'none'),
            'focal_alpha': config_dict['focal_alpha'],
                'focal_gamma': config_dict['focal_gamma']
            }
        }
        log_trial_complete(trial.number, objective_value, metrics)
        
        # Calculate total trial duration
        trial_duration = (datetime.now() - trial_start_time).total_seconds()
        trial_duration_min = trial_duration / 60
        
        # Final infinity check for this trial
        final_metrics_check = {
            'objective_value': objective_value,
            'avg_sharpe_ratio': avg_sharpe_ratio,
            'avg_win_rate': avg_win_rate,
            'avg_profit_factor': avg_profit_factor,
            'avg_max_drawdown': avg_max_drawdown
        }
        
        logger.info(f"[TRIAL_{trial.number}] 🎯 Trial {trial.number} Final Infinity Check:")
        for metric_name, metric_value in final_metrics_check.items():
            if np.isfinite(metric_value):
                logger.info(f"[TRIAL_{trial.number}] ✅ {metric_name}: {metric_value:.6f} (finite)")
            else:
                logger.error(f"[TRIAL_{trial.number}] ❌ {metric_name}: {metric_value} (NOT FINITE)")
        
        logger.info(f"[TRIAL_PROGRESS] ╔══════════════════════════════════════════════════════╗")
        logger.info(f"[TRIAL_PROGRESS] ║              TRIAL {trial.number:4d} COMPLETED ✅               ║")
        logger.info(f"[TRIAL_PROGRESS] ║                                                      ║")
        logger.info(f"[TRIAL_PROGRESS] ║  Duration: {trial_duration_min:6.1f} min ({trial_duration:6.1f}s)                 ║")
        logger.info(f"[TRIAL_PROGRESS] ║  Objective: {objective_value:7.4f}                            ║")
        logger.info(f"[TRIAL_PROGRESS] ║  Sharpe: {avg_sharpe_ratio:7.4f} | Win Rate: {avg_win_rate:6.4f}         ║")
        # Calculate total possible splits for display
        total_possible_splits = len(get_data_splits(start_date, end_date, WF_TRAIN_WINDOW_DAYS, WF_TEST_WINDOW_DAYS, WF_STEP_SIZE_DAYS))
        logger.info(f"[TRIAL_PROGRESS] ║  Splits: {len(split_results):2d}/{total_possible_splits:2d} | Trades: {total_trades:6d}             ║")
        logger.info(f"[TRIAL_PROGRESS] ╚══════════════════════════════════════════════════════╝")
        
        return objective_value
        

        
    except TrialPruned:
        # Re-raise TrialPruned exceptions
        raise
    except Exception as e:
        logger.error(f"[TRIAL_{trial.number}] ❌ CRITICAL ERROR in trial {trial.number}: {e}")
        logger.error(f"[TRIAL_{trial.number}] ❌ Exception type: {type(e).__name__}")
        logger.error(f"[TRIAL_{trial.number}] ❌ Exception details: {str(e)}")
        
        # Log additional debugging info
        import traceback
        logger.error(f"[TRIAL_{trial.number}] ❌ Full traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.error(f"[TRIAL_{trial.number}] ❌ {line}")
        
        # Check if this is an infinity-related error
        if 'inf' in str(e).lower() or 'infinity' in str(e).lower() or 'finite' in str(e).lower():
            logger.error(f"[TRIAL_{trial.number}] ❌ This appears to be an infinity-related error!")
        
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
    
    # Enhanced study progress reporting
    study_start_time = datetime.now()
    n_trials = 3500
    logger.info(f"[STUDY_PROGRESS] ╔════════════════════════════════════════════════════════════╗")
    logger.info(f"[STUDY_PROGRESS] ║           WALK-FORWARD HYPEROPT STUDY STARTING           ║")
    logger.info(f"[STUDY_PROGRESS] ║                                                            ║")
    logger.info(f"[STUDY_PROGRESS] ║  Target Trials: {n_trials:4d}                                  ║")
    logger.info(f"[STUDY_PROGRESS] ║  Walk-Forward: 120/20/10 day windows                      ║")
    logger.info(f"[STUDY_PROGRESS] ║  Started: {study_start_time.strftime('%Y-%m-%d %H:%M:%S')}                          ║")
    logger.info(f"[STUDY_PROGRESS] ╚════════════════════════════════════════════════════════════╝")
    
    # Run optimization with enhanced progress tracking
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None,
        gc_after_trial=True,
        show_progress_bar=True
    )
    
    # Study completion summary
    study_duration = (datetime.now() - study_start_time).total_seconds()
    study_duration_hours = study_duration / 3600
    completed_trials = len([t for t in study.trials if t.value is not None])
    
    logger.info(f"[STUDY_PROGRESS] ╔════════════════════════════════════════════════════════════╗")
    logger.info(f"[STUDY_PROGRESS] ║            WALK-FORWARD HYPEROPT STUDY COMPLETED          ║")
    logger.info(f"[STUDY_PROGRESS] ║                                                            ║")
    logger.info(f"[STUDY_PROGRESS] ║  Completed Trials: {completed_trials:4d}/{n_trials:4d}                           ║")
    logger.info(f"[STUDY_PROGRESS] ║  Study Duration: {study_duration_hours:6.1f} hours                        ║")
    logger.info(f"[STUDY_PROGRESS] ║  Best Objective: {study.best_trial.value:8.4f}                       ║")
    logger.info(f"[STUDY_PROGRESS] ║  Avg Time/Trial: {study_duration/max(completed_trials,1):6.1f}s                         ║")
    logger.info(f"[STUDY_PROGRESS] ╚════════════════════════════════════════════════════════════╝")
    
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
        # Check for infinite values in study before visualization
        infinite_trials = []
        for trial in study.trials:
            if trial.value is not None and not np.isfinite(trial.value):
                infinite_trials.append(trial.number)
        
        if infinite_trials:
            logger.warning(f"[VISUALIZATION] ⚠️  Found {len(infinite_trials)} trials with infinite objective values: {infinite_trials}")
            logger.warning(f"[VISUALIZATION] ⚠️  These trials will be filtered out during visualization")
        
        create_trading_metrics_visualizations(study, output_dir="plots")
        logger.info(f'✅ Trading metrics visualizations generated successfully')
    except Exception as e:
        logger.error(f'❌ Error generating trading metrics visualizations: {e}')
        import traceback
        logger.error(f'❌ Full traceback:')
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.error(f'❌ {line}')
    
    # Final memory cleanup
    gc.collect()

if __name__ == '__main__':
    main() 