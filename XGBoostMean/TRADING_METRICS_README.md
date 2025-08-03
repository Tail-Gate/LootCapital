# 🎯 XGBoost Trading Metrics Reporting System

This system focuses on generating comprehensive reports and visualizations for the key trading metrics: **Sharpe ratio**, **max drawdown**, **win rate**, and **profit factor**.

## 📊 Key Features

### Trading Metrics Focus
- **Sharpe Ratio**: Risk-adjusted return measure
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Max Drawdown**: Maximum peak-to-trough decline

### Comprehensive Reporting
- Detailed JSON reports with summary statistics
- Best trials analysis for each metric
- Complete trial data for further analysis

### Rich Visualizations
- Distribution plots for each trading metric
- Progression plots showing metric evolution over trials
- Correlation heatmaps between metrics
- Top trials analysis with detailed breakdowns
- Scatter plots showing relationships between metrics

## 🚀 Usage

### 1. During Hyperopt Optimization

The trading metrics reporting is automatically integrated into the hyperopt runner. When you run:

```bash
python XGBoostMean/xgboost_hyperopt_runner.py
```

The system will:
- Store trading metrics in trial user attributes
- Generate comprehensive reports after optimization
- Create detailed visualizations
- Save everything to `reports/` and `plots/` directories

### 2. Standalone Analysis

For analyzing existing studies, use the standalone script:

```bash
python XGBoostMean/generate_trading_reports.py
```

This will:
- Find available Optuna study databases
- Load the study and extract trading metrics
- Generate reports and visualizations
- Print summary statistics to console

## 📈 Generated Reports

### 1. Trading Metrics Report (`reports/trading_metrics_report_YYYYMMDD_HHMMSS.json`)

Contains:
- **Summary Statistics**: Best, worst, mean, and standard deviation for each metric
- **Best Trials Analysis**: Top performing trials for each metric
- **Complete Trial Data**: All trial results with parameters and metrics

### 2. Main Visualization (`plots/trading_metrics_analysis_YYYYMMDD_HHMMSS.png`)

9-panel visualization including:
- Distribution plots for Sharpe ratio, win rate, profit factor, max drawdown
- Scatter plots showing relationships between metrics
- Correlation heatmap
- Top 10 trials by Sharpe ratio and profit factor

### 3. Progression Plots (`plots/trading_metrics_progression_YYYYMMDD_HHMMSS.png`)

Shows how each metric evolved over trials:
- Sharpe ratio progression
- Win rate progression  
- Profit factor progression
- Max drawdown progression

### 4. Comparison Plots (`plots/trading_metrics_comparison_YYYYMMDD_HHMMSS.png`)

Additional analysis including:
- Normalized metrics comparison
- Best trials breakdown
- Objective value distribution
- Sharpe vs Profit Factor scatter plot

## 🎯 Console Output

The system provides detailed console output showing:

```
🎯 TRADING METRICS OPTIMIZATION SUMMARY
================================================================================
Total Trials: 150

📊 SHARPE RATIO:
  Best: 2.4567 (Trial 89)
  Mean: 1.2345 ± 0.5678
  Worst: -0.1234

🎯 WIN RATE:
  Best: 0.6789 (Trial 45)
  Mean: 0.5432

💰 PROFIT FACTOR:
  Best: 3.4567 (Trial 67)
  Mean: 2.1234

📉 MAX DRAWDOWN:
  Worst: -0.2345
  Mean: -0.1234

🏆 BEST OVERALL TRIAL (Lowest Objective):
  Trial 89:
    Sharpe Ratio: 2.4567
    Win Rate: 0.6543
    Profit Factor: 3.1234
    Max Drawdown: -0.1234
    Objective Value: -1.2345
================================================================================
```

## 📊 Key Insights

### Sharpe Ratio Analysis
- **Target**: > 1.0 (good), > 2.0 (excellent)
- **Focus**: Risk-adjusted returns
- **Interpretation**: Higher is better, indicates consistent positive returns relative to volatility

### Win Rate Analysis
- **Target**: > 0.5 (profitable), > 0.6 (good)
- **Focus**: Trade success frequency
- **Interpretation**: Percentage of profitable trades, higher is generally better

### Profit Factor Analysis
- **Target**: > 1.5 (profitable), > 2.0 (good), > 3.0 (excellent)
- **Focus**: Profit vs loss ratio
- **Interpretation**: Ratio of gross profit to gross loss, higher indicates better risk management

### Max Drawdown Analysis
- **Target**: < -0.2 (acceptable), < -0.1 (good)
- **Focus**: Risk management
- **Interpretation**: Maximum peak-to-trough decline, lower (less negative) is better

## 🔧 Customization

### Modifying Metrics Storage

To add additional trading metrics, modify the objective function in `xgboost_hyperopt_runner.py`:

```python
# Store additional metrics
trial.set_user_attr('your_metric', your_value)
```

### Custom Visualizations

To add custom plots, modify the `create_trading_metrics_visualizations` function:

```python
# Add your custom plot
ax_custom = fig.add_subplot(gs[row, col])
# Your plotting code here
```

## 📁 File Structure

```
XGBoostMean/
├── xgboost_hyperopt_runner.py          # Main hyperopt with trading metrics
├── generate_trading_reports.py          # Standalone reporting script
├── TRADING_METRICS_README.md           # This documentation
├── reports/                            # Generated reports
│   └── trading_metrics_report_*.json
└── plots/                             # Generated visualizations
    ├── trading_metrics_analysis_*.png
    ├── trading_metrics_progression_*.png
    └── trading_metrics_comparison_*.png
```

## 🎯 Best Practices

1. **Run Multiple Trials**: Ensure sufficient data for statistical significance
2. **Monitor Progress**: Check console output during optimization
3. **Analyze Trends**: Use progression plots to identify optimization patterns
4. **Compare Metrics**: Use correlation analysis to understand metric relationships
5. **Focus on Sharpe**: Prioritize Sharpe ratio for risk-adjusted performance

## 🚨 Troubleshooting

### No Trading Metrics Found
- Ensure trials completed successfully
- Check that user attributes are being set correctly
- Verify study database contains completed trials

### Visualization Errors
- Check matplotlib and seaborn installations
- Ensure sufficient memory for large plots
- Verify output directory permissions

### Report Generation Issues
- Check JSON serialization of trial parameters
- Verify file write permissions
- Ensure study database is accessible

## 📈 Next Steps

1. **Run Optimization**: Execute the hyperopt with trading metrics focus
2. **Analyze Results**: Review generated reports and visualizations
3. **Iterate**: Use insights to refine hyperparameter search spaces
4. **Validate**: Test best models on out-of-sample data
5. **Deploy**: Implement best performing configurations

---

**🎯 Focus on what matters: Sharpe ratio, max drawdown, win rate, and profit factor for robust trading system evaluation.** 