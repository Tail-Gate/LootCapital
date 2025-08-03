#!/usr/bin/env python3
"""
Standalone script to generate trading metrics reports and visualizations from existing Optuna studies.
Focuses on Sharpe ratio, max drawdown, win rate, and profit factor.
"""

import optuna
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np

def create_trading_metrics_report(study, output_dir="reports"):
    """
    Create a comprehensive trading metrics report focusing on Sharpe ratio, max drawdown, win rate, and profit factor.
    
    Args:
        study: Optuna study object with completed trials
        output_dir: Directory to save the report
    """
    print("📊 Creating Trading Metrics Report...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract trading metrics from completed trials
    trading_metrics = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
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
        print("⚠️  No completed trials with trading metrics found")
        return None
    
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
    
    print(f"✅ Trading metrics report saved to: {report_path}")
    
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
    print("📈 Creating Trading Metrics Visualizations...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract trading metrics from completed trials
    trading_metrics = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
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
        print("⚠️  No completed trials with trading metrics found")
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
    ax1.set_title('📈 Sharpe Ratio Distribution', fontweight='bold', fontsize=14)
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
    ax2.set_title('🎯 Win Rate Distribution', fontweight='bold', fontsize=14)
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
    ax3.set_title('💰 Profit Factor Distribution', fontweight='bold', fontsize=14)
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
    ax4.set_title('📉 Max Drawdown Distribution', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Max Drawdown')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Sharpe Ratio vs Win Rate
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(df['win_rate'], df['sharpe_ratio'], 
                          c=df['objective_value'], cmap='viridis', alpha=0.7, s=50)
    ax5.set_title('📊 Sharpe Ratio vs Win Rate', fontweight='bold', fontsize=14)
    ax5.set_xlabel('Win Rate')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Objective Value')
    
    # 6. Profit Factor vs Max Drawdown
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(df['max_drawdown'], df['profit_factor'], 
                          c=df['objective_value'], cmap='plasma', alpha=0.7, s=50)
    ax6.set_title('💰 Profit Factor vs Max Drawdown', fontweight='bold', fontsize=14)
    ax6.set_xlabel('Max Drawdown')
    ax6.set_ylabel('Profit Factor')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Objective Value')
    
    # 7. Trading Metrics Correlation Heatmap
    ax7 = fig.add_subplot(gs[2, 0])
    correlation_matrix = df[['sharpe_ratio', 'win_rate', 'profit_factor', 'max_drawdown', 'objective_value']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax7, fmt='.3f')
    ax7.set_title('🔄 Trading Metrics Correlation', fontweight='bold', fontsize=14)
    
    # 8. Top 10 Trials by Sharpe Ratio
    ax8 = fig.add_subplot(gs[2, 1])
    top_sharpe = df.nlargest(10, 'sharpe_ratio')
    bars = ax8.barh(range(len(top_sharpe)), top_sharpe['sharpe_ratio'], 
                     color='green', alpha=0.7)
    ax8.set_yticks(range(len(top_sharpe)))
    ax8.set_yticklabels([f'Trial {trial}' for trial in top_sharpe['trial_number']])
    ax8.set_title('🏆 Top 10 Trials by Sharpe Ratio', fontweight='bold', fontsize=14)
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
    ax9.set_title('💰 Top 10 Trials by Profit Factor', fontweight='bold', fontsize=14)
    ax9.set_xlabel('Profit Factor')
    ax9.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_profit['profit_factor'])):
        ax9.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=10)
    
    # Add overall title
    fig.suptitle('🎯 XGBoost Trading Metrics Analysis\nSharpe Ratio, Win Rate, Profit Factor, Max Drawdown', 
                 fontsize=16, fontweight='bold')
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = Path(output_dir) / f'trading_metrics_analysis_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Trading metrics visualizations saved to: {plot_path}")
    
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
    print("📊 Creating Additional Focused Plots...")
    
    # 1. Sharpe Ratio Progression Over Trials
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['trial_number'], df['sharpe_ratio'], 'o-', color='green', alpha=0.7, linewidth=2)
    plt.axhline(df['sharpe_ratio'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["sharpe_ratio"].mean():.4f}')
    plt.title('📈 Sharpe Ratio Progression Over Trials', fontweight='bold', fontsize=14)
    plt.xlabel('Trial Number')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Win Rate Progression Over Trials
    plt.subplot(2, 2, 2)
    plt.plot(df['trial_number'], df['win_rate'], 'o-', color='orange', alpha=0.7, linewidth=2)
    plt.axhline(df['win_rate'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["win_rate"].mean():.4f}')
    plt.title('🎯 Win Rate Progression Over Trials', fontweight='bold', fontsize=14)
    plt.xlabel('Trial Number')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Profit Factor Progression Over Trials
    plt.subplot(2, 2, 3)
    plt.plot(df['trial_number'], df['profit_factor'], 'o-', color='purple', alpha=0.7, linewidth=2)
    plt.axhline(df['profit_factor'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["profit_factor"].mean():.4f}')
    plt.title('💰 Profit Factor Progression Over Trials', fontweight='bold', fontsize=14)
    plt.xlabel('Trial Number')
    plt.ylabel('Profit Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Max Drawdown Progression Over Trials
    plt.subplot(2, 2, 4)
    plt.plot(df['trial_number'], df['max_drawdown'], 'o-', color='red', alpha=0.7, linewidth=2)
    plt.axhline(df['max_drawdown'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["max_drawdown"].mean():.4f}')
    plt.title('📉 Max Drawdown Progression Over Trials', fontweight='bold', fontsize=14)
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
    
    # Normalize metrics for comparison (0-1 scale)
    df_normalized = df.copy()
    for col in ['sharpe_ratio', 'win_rate', 'profit_factor']:
        df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    # Invert max_drawdown so lower is better
    df_normalized['max_drawdown_inverted'] = 1 - (df['max_drawdown'] - df['max_drawdown'].min()) / (df['max_drawdown'].max() - df['max_drawdown'].min())
    
    plt.subplot(2, 2, 1)
    metrics_data = [df_normalized['sharpe_ratio'], df_normalized['win_rate'], 
                   df_normalized['profit_factor'], df_normalized['max_drawdown_inverted']]
    plt.boxplot(metrics_data, labels=['Sharpe\nRatio', 'Win\nRate', 'Profit\nFactor', 'Drawdown\n(Inverted)'])
    plt.title('📊 Trading Metrics Distribution Comparison', fontweight='bold', fontsize=14)
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
    plt.title('🏆 Best Trials Metrics Breakdown', fontweight='bold', fontsize=14)
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
    plt.title('📊 Objective Value Distribution', fontweight='bold', fontsize=14)
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
    plt.title('🎯 Sharpe Ratio vs Profit Factor\n(Colored by Win Rate)', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = Path(output_dir) / f'trading_metrics_comparison_{timestamp}.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Additional trading metrics plots saved:")
    print(f"  Progression plot: {progression_path}")
    print(f"  Comparison plot: {comparison_path}")

def main():
    """Main function to generate trading metrics reports from existing studies"""
    print("🎯 XGBoost Trading Metrics Report Generator")
    print("="*50)
    
    # List available study databases
    study_files = list(Path('.').glob('*.db'))
    
    if not study_files:
        print("❌ No Optuna study databases found in current directory")
        print("Please run the hyperopt first or specify a study database path")
        return
    
    print("📁 Available study databases:")
    for i, file in enumerate(study_files):
        print(f"  {i+1}. {file}")
    
    # For now, use the first XGBoost study found
    xgboost_studies = [f for f in study_files if 'xgboost' in f.name.lower()]
    
    if not xgboost_studies:
        print("❌ No XGBoost study databases found")
        return
    
    study_path = xgboost_studies[0]
    print(f"\n📊 Using study: {study_path}")
    
    try:
        # Load the study
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{study_path}")
        print(f"✅ Loaded study with {len(study.trials)} trials")
        
        # Generate reports and visualizations
        print("\n🎯 Generating Trading Metrics Reports and Visualizations...")
        
        # Create trading metrics report
        try:
            trading_report = create_trading_metrics_report(study, output_dir="reports")
            if trading_report:
                print("✅ Trading metrics report generated successfully")
        except Exception as e:
            print(f"❌ Error generating trading metrics report: {e}")
        
        # Create trading metrics visualizations
        try:
            create_trading_metrics_visualizations(study, output_dir="plots")
            print("✅ Trading metrics visualizations generated successfully")
        except Exception as e:
            print(f"❌ Error generating trading metrics visualizations: {e}")
        
        print("\n🎉 Trading metrics analysis complete!")
        
    except Exception as e:
        print(f"❌ Error loading study: {e}")

if __name__ == '__main__':
    main() 