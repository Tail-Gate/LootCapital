#!/usr/bin/env python3
"""
Test script for probability analysis functionality

This script creates synthetic probability data to test the analysis functionality
without needing to run the full walk-forward optimization.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.analyze_probabilities import ProbabilityAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_probability_data():
    """Create synthetic probability data for testing"""
    
    # Create test directory
    test_dir = Path("test_probability_analysis")
    test_dir.mkdir(exist_ok=True)
    
    # Create synthetic data that mimics the expected behavior
    # where Down class is not being predicted correctly
    np.random.seed(42)
    
    # Period 1: Simulate problematic Down class predictions
    n_samples = 1000
    
    # True Down samples (class 0) - model often predicts No Direction or Up
    n_down = 200
    down_probs = np.random.dirichlet([0.2, 0.5, 0.3], n_down)  # Low prob for Down
    down_labels = np.zeros(n_down)
    
    # True No Direction samples (class 1) - model often predicts Down or Up
    n_no_dir = 300
    no_dir_probs = np.random.dirichlet([0.4, 0.2, 0.4], n_no_dir)  # Low prob for No Direction
    no_dir_labels = np.ones(n_no_dir)
    
    # True Up samples (class 2) - model predicts correctly
    n_up = 500
    up_probs = np.random.dirichlet([0.1, 0.2, 0.7], n_up)  # High prob for Up
    up_labels = np.full(n_up, 2)
    
    # Combine all data
    all_probs = np.vstack([down_probs, no_dir_probs, up_probs])
    all_labels = np.concatenate([down_labels, no_dir_labels, up_labels])
    
    # Create DataFrame
    df = pd.DataFrame(all_probs, columns=['prob_down', 'prob_no_direction', 'prob_up'])
    df['true_label'] = all_labels
    df['predicted_label'] = np.argmax(all_probs, axis=1)
    df['period_name'] = 'test_period_1'
    
    # Save to CSV
    csv_path = test_dir / 'test_probabilities_test_period_1_20241219_120000.csv'
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Created synthetic data with {len(df)} samples")
    logger.info(f"True Down samples: {n_down}")
    logger.info(f"True No Direction samples: {n_no_dir}")
    logger.info(f"True Up samples: {n_up}")
    logger.info(f"Saved to: {csv_path}")
    
    return test_dir

def test_probability_analysis():
    """Test the probability analysis functionality"""
    
    logger.info("="*60)
    logger.info("TESTING PROBABILITY ANALYSIS FUNCTIONALITY")
    logger.info("="*60)
    
    try:
        # Create synthetic data
        logger.info("Creating synthetic probability data...")
        test_dir = create_synthetic_probability_data()
        
        # Create analyzer
        logger.info("Creating probability analyzer...")
        analyzer = ProbabilityAnalyzer(
            probability_dir=str(test_dir),
            plots_dir="test_plots",
            reports_dir="test_reports"
        )
        
        # Load probability files
        logger.info("Loading probability files...")
        all_data = analyzer.load_probability_files("*.csv")
        
        if not all_data:
            logger.error("No probability files found")
            return
        
        # Perform analysis
        logger.info("Performing probability analysis...")
        summary_stats = analyzer.generate_summary_report(all_data)
        
        logger.info("="*60)
        logger.info("PROBABILITY ANALYSIS TEST COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("Test results saved to:")
        logger.info(f"  Plots: test_plots/")
        logger.info(f"  Reports: test_reports/")
        
        # Print key findings
        if 'aggregate_down_stats' in summary_stats:
            down_stats = summary_stats['aggregate_down_stats']
            print(f"\nTEST RESULTS - DOWN CLASS:")
            print(f"  Total samples: {down_stats['total_samples']}")
            print(f"  Mean prob_down: {down_stats['mean_prob_down']:.4f}")
            print(f"  Mean prob_no_direction: {down_stats['mean_prob_no_direction']:.4f}")
            print(f"  Mean prob_up: {down_stats['mean_prob_up']:.4f}")
            print(f"  Error rate: {down_stats['total_error_rate']:.4f}")
        
        if 'aggregate_no_direction_stats' in summary_stats:
            no_dir_stats = summary_stats['aggregate_no_direction_stats']
            print(f"\nTEST RESULTS - NO DIRECTION CLASS:")
            print(f"  Total samples: {no_dir_stats['total_samples']}")
            print(f"  Mean prob_down: {no_dir_stats['mean_prob_down']:.4f}")
            print(f"  Mean prob_no_direction: {no_dir_stats['mean_prob_no_direction']:.4f}")
            print(f"  Mean prob_up: {no_dir_stats['mean_prob_up']:.4f}")
            print(f"  Error rate: {no_dir_stats['total_error_rate']:.4f}")
        
        return summary_stats
        
    except Exception as e:
        logger.error(f"Probability analysis test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def cleanup_test_files():
    """Clean up test files"""
    import shutil
    
    test_dirs = ["test_probability_analysis", "test_plots", "test_reports"]
    
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            logger.info(f"Cleaned up: {dir_name}")

if __name__ == "__main__":
    # Run test
    test_probability_analysis()
    
    # Optionally clean up (uncomment to remove test files)
    # cleanup_test_files() 