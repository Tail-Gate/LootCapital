# Probability Analysis for STGNN Walk-Forward Optimization

## Overview

This document describes the probability analysis functionality implemented to understand why the STGNN model's "Down" class (Class 0) is not being predicted correctly (F1 = 0.0000) and why "No Direction" class (Class 1) precision is low.

## Problem Statement

The STGNN model trained using walk-forward optimization consistently shows:
- **F1 (Down) = 0.0000** - The model is not predicting the "Down" class at all
- **Low "No Direction" precision** - The model struggles with the "No Direction" class
- **Confidence threshold recently adjusted** from 70% to 51%

## Solution Implementation

### 1. Enhanced Walk-Forward Optimization

The `walk_forward_optimization.py` script has been modified to capture and save model prediction probabilities:

**Key Features:**
- Saves raw softmax probabilities for each test period
- Captures true labels and predicted labels
- Creates both numpy arrays (.npy) and CSV files for analysis
- Logs detailed probability statistics during execution

**File Structure:**
```
models/
└── probability_analysis/
    ├── test_probabilities_{period_name}_{timestamp}.npy
    ├── test_true_labels_{period_name}_{timestamp}.npy
    ├── test_predictions_{period_name}_{timestamp}.npy
    └── test_probabilities_{period_name}_{timestamp}.csv
```

**CSV Format:**
```csv
prob_down,prob_no_direction,prob_up,true_label,predicted_label,period_name
0.1234,0.5678,0.3088,0,1,2020-01_to_2020-02
...
```

### 2. Comprehensive Analysis Script

The `analyze_probabilities.py` script provides detailed analysis of model prediction behavior:

**Analysis Features:**
- **Down Class Analysis**: Examines why true Down samples are misclassified
- **No Direction Class Analysis**: Analyzes precision issues
- **Statistical Summaries**: Mean, median, min, max, standard deviation
- **Error Analysis**: Counts of misclassification patterns
- **Confidence Analysis**: Model confidence gaps and distributions

**Visualization Types:**
- **Histograms**: Probability distributions for each class
- **Box Plots**: Statistical summaries of probability distributions
- **Violin Plots**: Detailed distribution shapes
- **Scatter Plots**: Relationships between different probabilities
- **Confusion Matrices**: Overall prediction patterns
- **Combined Analysis**: Cross-period comparisons

## Usage Instructions

### Step 1: Run Walk-Forward Optimization

```bash
# Run with default settings
python scripts/walk_forward_optimization.py

# Run with custom parameters
python scripts/walk_forward_optimization.py \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --train-window-days 365 \
  --test-window-days 60 \
  --step-size-days 30 \
  --price-threshold 0.018
```

This will generate probability files in `models/probability_analysis/`.

### Step 2: Analyze Probabilities

```bash
# Analyze all probability files
python scripts/analyze_probabilities.py

# Analyze specific directory
python scripts/analyze_probabilities.py \
  --probability-dir models/probability_analysis \
  --plots-dir plots/probability_analysis \
  --reports-dir reports/probability_analysis

# Analyze specific file pattern
python scripts/analyze_probabilities.py \
  --file-pattern "*2024*_*.csv"
```

### Step 3: Test the Analysis (Optional)

```bash
# Run test with synthetic data
python scripts/test_probability_analysis.py
```

## Output Structure

### Generated Files

```
plots/probability_analysis/
├── down_class_analysis_{period_name}.png
├── no_direction_class_analysis_{period_name}.png
└── combined_probability_analysis.png

reports/probability_analysis/
└── probability_analysis_report_{timestamp}.json
```

### Analysis Results

The analysis provides comprehensive insights into model behavior:

**Down Class Analysis:**
- Mean probability assigned to Down class for true Down samples
- Frequency of No Direction or Up being predicted instead
- Confidence gaps between correct and incorrect predictions

**No Direction Class Analysis:**
- Mean probability assigned to No Direction class for true No Direction samples
- Frequency of Down or Up being predicted instead
- Precision issues and their root causes

**Combined Analysis:**
- Overall probability distributions across all classes
- Confusion matrices showing prediction patterns
- Model confidence distributions

## Key Metrics Explained

### Error Rates
- **Error rate (No Direction > Down)**: How often the model assigns higher probability to "No Direction" when the true label is "Down"
- **Error rate (Up > Down)**: How often the model assigns higher probability to "Up" when the true label is "Down"
- **Total error rate**: Combined error rate for a class

### Confidence Analysis
- **Mean confidence in correct class**: Average probability assigned to the true class
- **Mean confidence in wrong class**: Average probability assigned to the most likely incorrect class
- **Confidence gap**: Difference between correct and incorrect class probabilities

## Example Analysis Output

```
DOWN CLASS ANALYSIS (True Label = 0):
  Total samples: 200
  Mean probability for Down: 0.2016
  Mean probability for No Direction: 0.5062
  Mean probability for Up: 0.2922
  Error rate (No Direction > Down): 0.7200
  Error rate (Up > Down): 0.5750
  Total error rate: 1.2950

NO DIRECTION CLASS ANALYSIS (True Label = 1):
  Total samples: 300
  Mean probability for Down: 0.4100
  Mean probability for No Direction: 0.1909
  Mean probability for Up: 0.3991
  Error rate (Down > No Direction): 0.6833
  Error rate (Up > No Direction): 0.7067
  Total error rate: 1.3900
```

## Interpretation Guidelines

### High Error Rates (>0.5)
- Indicates systematic misclassification
- Model may be biased toward certain classes
- Consider adjusting class weights or loss function

### Low Confidence Gaps (<0.1)
- Model is uncertain about predictions
- Consider increasing model complexity or training time
- May need more features or different architecture

### Uneven Probability Distributions
- Model may be overconfident in certain classes
- Consider adjusting focal loss parameters
- May indicate data imbalance issues

## Troubleshooting

### No Probability Files Found
- Ensure walk-forward optimization completed successfully
- Check that `models/probability_analysis/` directory exists
- Verify CSV files are present in the directory

### Analysis Errors
- Check that CSV files have the expected column names
- Ensure sufficient samples exist for each class
- Verify matplotlib and seaborn are installed

### Memory Issues
- Reduce the number of periods analyzed at once
- Use smaller file patterns to process subsets
- Consider processing periods individually

## Future Enhancements

### Planned Features
- **Interactive Dashboards**: Web-based visualization interface
- **Real-time Analysis**: Live probability monitoring during training
- **Advanced Metrics**: Calibration plots, reliability diagrams
- **Automated Insights**: AI-powered interpretation of results

### Potential Improvements
- **Cross-validation Analysis**: Probability analysis across different validation folds
- **Feature Importance**: Correlation between features and prediction errors
- **Temporal Analysis**: How prediction patterns change over time
- **Ensemble Analysis**: Comparing probabilities across multiple models

## Technical Details

### Data Format
- **Probabilities**: 3-column numpy arrays (Down, No Direction, Up)
- **Labels**: Integer arrays (0=Down, 1=No Direction, 2=Up)
- **Timestamps**: ISO format for file naming and tracking

### Dependencies
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pathlib (built-in)
- logging (built-in)

### Performance Considerations
- Analysis scales linearly with number of samples
- Visualization generation may take time for large datasets
- Consider processing periods in batches for very large datasets

## Support and Contact

For questions or issues with the probability analysis functionality:
1. Check the logs in `logs/probability_analysis.log`
2. Review the generated reports for detailed error information
3. Run the test script to verify functionality
4. Check the walk-forward optimization logs for data generation issues 