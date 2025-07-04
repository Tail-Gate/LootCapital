#!/usr/bin/env python3
"""
Probability Analysis for STGNN Walk-Forward Optimization

This script analyzes the raw softmax probabilities output by the STGNN model
on walk-forward optimization test sets to understand prediction behavior,
particularly focusing on why the "Down" class is not being predicted correctly
and why "No Direction" precision is low.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
import argparse
from typing import Dict, List, Tuple, Optional
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/probability_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProbabilityAnalyzer:
    """Analyzer for model prediction probabilities"""
    
    def __init__(self, 
                 probability_dir: str = "models/probability_analysis",
                 plots_dir: str = "plots/probability_analysis",
                 reports_dir: str = "reports/probability_analysis"):
        
        self.probability_dir = Path(probability_dir)
        self.plots_dir = Path(plots_dir)
        self.reports_dir = Path(reports_dir)
        
        # Create output directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Class names for analysis
        self.class_names = ['Down', 'No Direction', 'Up']
        self.class_colors = ['red', 'blue', 'green']
        
        # Results storage
        self.analysis_results = {}
        
    def load_probability_files(self, pattern: str = "*.csv") -> List[Tuple[str, pd.DataFrame]]:
        """Load all probability CSV files from the directory"""
        
        csv_files = list(self.probability_dir.glob(pattern))
        logger.info(f"Found {len(csv_files)} probability files")
        
        loaded_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                period_name = df['period_name'].iloc[0] if 'period_name' in df.columns else csv_file.stem
                loaded_data.append((period_name, df))
                logger.info(f"Loaded {csv_file.name} with {len(df)} samples")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        
        return loaded_data
    
    def analyze_down_class(self, df: pd.DataFrame, period_name: str) -> Dict:
        """Analyze predictions for true Down class (true_label == 0)"""
        
        # Filter for true Down samples
        down_mask = df['true_label'] == 0
        down_samples = df[down_mask]
        
        if len(down_samples) == 0:
            logger.warning(f"No true Down samples found in period {period_name}")
            return {}
        
        logger.info(f"Analyzing {len(down_samples)} true Down samples for period {period_name}")
        
        # Extract probabilities
        prob_down = down_samples['prob_down'].values
        prob_no_direction = down_samples['prob_no_direction'].values
        prob_up = down_samples['prob_up'].values
        
        # Calculate statistics
        stats = {
            'period_name': period_name,
            'num_samples': len(down_samples),
            'prob_down': {
                'mean': np.mean(prob_down),
                'median': np.median(prob_down),
                'min': np.min(prob_down),
                'max': np.max(prob_down),
                'std': np.std(prob_down)
            },
            'prob_no_direction': {
                'mean': np.mean(prob_no_direction),
                'median': np.median(prob_no_direction),
                'min': np.min(prob_no_direction),
                'max': np.max(prob_no_direction),
                'std': np.std(prob_no_direction)
            },
            'prob_up': {
                'mean': np.mean(prob_up),
                'median': np.median(prob_up),
                'min': np.min(prob_up),
                'max': np.max(prob_up),
                'std': np.std(prob_up)
            },
            'prediction_errors': {
                'times_prob_1_higher_than_prob_0': np.sum(prob_no_direction > prob_down),
                'times_prob_2_higher_than_prob_0': np.sum(prob_up > prob_down),
                'times_prob_1_highest': np.sum((prob_no_direction > prob_down) & (prob_no_direction > prob_up)),
                'times_prob_2_highest': np.sum((prob_up > prob_down) & (prob_up > prob_no_direction)),
                'times_prob_0_highest': np.sum((prob_down > prob_no_direction) & (prob_down > prob_up))
            },
            'confidence_analysis': {
                'mean_confidence_in_down': np.mean(prob_down),
                'mean_confidence_in_wrong_class': np.mean(np.maximum(prob_no_direction, prob_up)),
                'confidence_gap': np.mean(prob_down) - np.mean(np.maximum(prob_no_direction, prob_up))
            }
        }
        
        return stats
    
    def analyze_no_direction_class(self, df: pd.DataFrame, period_name: str) -> Dict:
        """Analyze predictions for true No Direction class (true_label == 1)"""
        
        # Filter for true No Direction samples
        no_dir_mask = df['true_label'] == 1
        no_dir_samples = df[no_dir_mask]
        
        if len(no_dir_samples) == 0:
            logger.warning(f"No true No Direction samples found in period {period_name}")
            return {}
        
        logger.info(f"Analyzing {len(no_dir_samples)} true No Direction samples for period {period_name}")
        
        # Extract probabilities
        prob_down = no_dir_samples['prob_down'].values
        prob_no_direction = no_dir_samples['prob_no_direction'].values
        prob_up = no_dir_samples['prob_up'].values
        
        # Calculate statistics
        stats = {
            'period_name': period_name,
            'num_samples': len(no_dir_samples),
            'prob_down': {
                'mean': np.mean(prob_down),
                'median': np.median(prob_down),
                'min': np.min(prob_down),
                'max': np.max(prob_down),
                'std': np.std(prob_down)
            },
            'prob_no_direction': {
                'mean': np.mean(prob_no_direction),
                'median': np.median(prob_no_direction),
                'min': np.min(prob_no_direction),
                'max': np.max(prob_no_direction),
                'std': np.std(prob_no_direction)
            },
            'prob_up': {
                'mean': np.mean(prob_up),
                'median': np.median(prob_up),
                'min': np.min(prob_up),
                'max': np.max(prob_up),
                'std': np.std(prob_up)
            },
            'prediction_errors': {
                'times_prob_0_higher_than_prob_1': np.sum(prob_down > prob_no_direction),
                'times_prob_2_higher_than_prob_1': np.sum(prob_up > prob_no_direction),
                'times_prob_0_highest': np.sum((prob_down > prob_no_direction) & (prob_down > prob_up)),
                'times_prob_2_highest': np.sum((prob_up > prob_no_direction) & (prob_up > prob_down)),
                'times_prob_1_highest': np.sum((prob_no_direction > prob_down) & (prob_no_direction > prob_up))
            },
            'confidence_analysis': {
                'mean_confidence_in_no_direction': np.mean(prob_no_direction),
                'mean_confidence_in_wrong_class': np.mean(np.maximum(prob_down, prob_up)),
                'confidence_gap': np.mean(prob_no_direction) - np.mean(np.maximum(prob_down, prob_up))
            }
        }
        
        return stats
    
    def create_down_class_visualizations(self, df: pd.DataFrame, period_name: str):
        """Create visualizations for Down class analysis"""
        
        # Filter for true Down samples
        down_mask = df['true_label'] == 0
        down_samples = df[down_mask]
        
        if len(down_samples) == 0:
            logger.warning(f"No true Down samples found for visualization in period {period_name}")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Down Class (True Label = 0) Analysis - Period: {period_name}', fontsize=16)
        
        # Plot 1: Histograms of probabilities
        axes[0, 0].hist(down_samples['prob_down'], bins=30, alpha=0.7, label='Down', color='red', edgecolor='black')
        axes[0, 0].hist(down_samples['prob_no_direction'], bins=30, alpha=0.7, label='No Direction', color='blue', edgecolor='black')
        axes[0, 0].hist(down_samples['prob_up'], bins=30, alpha=0.7, label='Up', color='green', edgecolor='black')
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Probability Distributions for True Down Samples')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot of probabilities
        prob_data = [down_samples['prob_down'], down_samples['prob_no_direction'], down_samples['prob_up']]
        axes[0, 1].boxplot(prob_data, labels=['Down', 'No Direction', 'Up'])
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].set_title('Probability Distributions (Box Plot)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Violin plot
        prob_data_long = []
        labels_long = []
        for prob_col, label in [('prob_down', 'Down'), ('prob_no_direction', 'No Direction'), ('prob_up', 'Up')]:
            prob_data_long.extend(down_samples[prob_col].values)
            labels_long.extend([label] * len(down_samples))
        
        violin_data = [down_samples['prob_down'], down_samples['prob_no_direction'], down_samples['prob_up']]
        axes[1, 0].violinplot(violin_data, positions=[0, 1, 2])
        axes[1, 0].set_xticks([0, 1, 2])
        axes[1, 0].set_xticklabels(['Down', 'No Direction', 'Up'])
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].set_title('Probability Distributions (Violin Plot)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Scatter plot of Down vs No Direction probabilities
        axes[1, 1].scatter(down_samples['prob_down'], down_samples['prob_no_direction'], alpha=0.6, color='red')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
        axes[1, 1].set_xlabel('Probability Down')
        axes[1, 1].set_ylabel('Probability No Direction')
        axes[1, 1].set_title('Down vs No Direction Probabilities')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f'down_class_analysis_{period_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Down class visualization to: {plot_path}")
    
    def create_no_direction_class_visualizations(self, df: pd.DataFrame, period_name: str):
        """Create visualizations for No Direction class analysis"""
        
        # Filter for true No Direction samples
        no_dir_mask = df['true_label'] == 1
        no_dir_samples = df[no_dir_mask]
        
        if len(no_dir_samples) == 0:
            logger.warning(f"No true No Direction samples found for visualization in period {period_name}")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'No Direction Class (True Label = 1) Analysis - Period: {period_name}', fontsize=16)
        
        # Plot 1: Histograms of probabilities
        axes[0, 0].hist(no_dir_samples['prob_down'], bins=30, alpha=0.7, label='Down', color='red', edgecolor='black')
        axes[0, 0].hist(no_dir_samples['prob_no_direction'], bins=30, alpha=0.7, label='No Direction', color='blue', edgecolor='black')
        axes[0, 0].hist(no_dir_samples['prob_up'], bins=30, alpha=0.7, label='Up', color='green', edgecolor='black')
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Probability Distributions for True No Direction Samples')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot of probabilities
        prob_data = [no_dir_samples['prob_down'], no_dir_samples['prob_no_direction'], no_dir_samples['prob_up']]
        axes[0, 1].boxplot(prob_data, labels=['Down', 'No Direction', 'Up'])
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].set_title('Probability Distributions (Box Plot)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Violin plot
        violin_data = [no_dir_samples['prob_down'], no_dir_samples['prob_no_direction'], no_dir_samples['prob_up']]
        axes[1, 0].violinplot(violin_data, positions=[0, 1, 2])
        axes[1, 0].set_xticks([0, 1, 2])
        axes[1, 0].set_xticklabels(['Down', 'No Direction', 'Up'])
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].set_title('Probability Distributions (Violin Plot)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Scatter plot of No Direction vs Up probabilities
        axes[1, 1].scatter(no_dir_samples['prob_no_direction'], no_dir_samples['prob_up'], alpha=0.6, color='blue')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
        axes[1, 1].set_xlabel('Probability No Direction')
        axes[1, 1].set_ylabel('Probability Up')
        axes[1, 1].set_title('No Direction vs Up Probabilities')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f'no_direction_class_analysis_{period_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved No Direction class visualization to: {plot_path}")
    
    def create_combined_analysis_visualization(self, all_data: List[Tuple[str, pd.DataFrame]]):
        """Create combined analysis visualization across all periods"""
        
        if not all_data:
            logger.warning("No data available for combined analysis")
            return
        
        # Combine all data
        combined_df = pd.concat([df for _, df in all_data], ignore_index=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Combined Probability Analysis Across All Periods', fontsize=16)
        
        # Plot 1: Overall probability distributions
        axes[0, 0].hist(combined_df['prob_down'], bins=50, alpha=0.7, label='Down', color='red', edgecolor='black', density=True)
        axes[0, 0].hist(combined_df['prob_no_direction'], bins=50, alpha=0.7, label='No Direction', color='blue', edgecolor='black', density=True)
        axes[0, 0].hist(combined_df['prob_up'], bins=50, alpha=0.7, label='Up', color='green', edgecolor='black', density=True)
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Overall Probability Distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Probability distributions by true label
        for i, class_name in enumerate(self.class_names):
            class_mask = combined_df['true_label'] == i
            class_data = combined_df[class_mask]
            if len(class_data) > 0:
                axes[0, 1].hist(class_data['prob_down'], bins=30, alpha=0.5, label=f'True {class_name}', density=True)
        axes[0, 1].set_xlabel('Probability Down')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Down Probability by True Label')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: No Direction probability by true label
        for i, class_name in enumerate(self.class_names):
            class_mask = combined_df['true_label'] == i
            class_data = combined_df[class_mask]
            if len(class_data) > 0:
                axes[0, 2].hist(class_data['prob_no_direction'], bins=30, alpha=0.5, label=f'True {class_name}', density=True)
        axes[0, 2].set_xlabel('Probability No Direction')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('No Direction Probability by True Label')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Up probability by true label
        for i, class_name in enumerate(self.class_names):
            class_mask = combined_df['true_label'] == i
            class_data = combined_df[class_mask]
            if len(class_data) > 0:
                axes[1, 0].hist(class_data['prob_up'], bins=30, alpha=0.5, label=f'True {class_name}', density=True)
        axes[1, 0].set_xlabel('Probability Up')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Up Probability by True Label')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Confusion matrix of highest probability
        highest_prob = np.argmax(combined_df[['prob_down', 'prob_no_direction', 'prob_up']].values, axis=1)
        confusion_matrix = np.zeros((3, 3))
        for true_label in range(3):
            for pred_label in range(3):
                confusion_matrix[true_label, pred_label] = np.sum((combined_df['true_label'] == true_label) & (highest_prob == pred_label))
        
        im = axes[1, 1].imshow(confusion_matrix, cmap='Blues', aspect='auto')
        axes[1, 1].set_xticks(range(3))
        axes[1, 1].set_yticks(range(3))
        axes[1, 1].set_xticklabels(['Down', 'No Direction', 'Up'])
        axes[1, 1].set_yticklabels(['Down', 'No Direction', 'Up'])
        axes[1, 1].set_xlabel('Predicted (Highest Probability)')
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_title('Confusion Matrix (Highest Probability)')
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = axes[1, 1].text(j, i, f'{int(confusion_matrix[i, j])}', 
                                     ha="center", va="center", color="black" if confusion_matrix[i, j] < confusion_matrix.max()/2 else "white")
        
        # Plot 6: Confidence distribution
        max_probs = np.max(combined_df[['prob_down', 'prob_no_direction', 'prob_up']].values, axis=1)
        axes[1, 2].hist(max_probs, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 2].set_xlabel('Maximum Probability (Confidence)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Distribution of Model Confidence')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f'combined_probability_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved combined analysis visualization to: {plot_path}")
    
    def generate_summary_report(self, all_data: List[Tuple[str, pd.DataFrame]]):
        """Generate comprehensive summary report"""
        
        logger.info("Generating comprehensive probability analysis report...")
        
        # Initialize summary statistics
        summary_stats = {
            'total_periods': len(all_data),
            'total_samples': sum(len(df) for _, df in all_data),
            'down_class_stats': [],
            'no_direction_class_stats': [],
            'overall_stats': {}
        }
        
        # Analyze each period
        for period_name, df in all_data:
            logger.info(f"Analyzing period: {period_name}")
            
            # Analyze Down class
            down_stats = self.analyze_down_class(df, period_name)
            if down_stats:
                summary_stats['down_class_stats'].append(down_stats)
                self.create_down_class_visualizations(df, period_name)
            
            # Analyze No Direction class
            no_dir_stats = self.analyze_no_direction_class(df, period_name)
            if no_dir_stats:
                summary_stats['no_direction_class_stats'].append(no_dir_stats)
                self.create_no_direction_class_visualizations(df, period_name)
        
        # Create combined visualization
        self.create_combined_analysis_visualization(all_data)
        
        # Calculate aggregate statistics
        if summary_stats['down_class_stats']:
            self._calculate_aggregate_down_stats(summary_stats)
        
        if summary_stats['no_direction_class_stats']:
            self._calculate_aggregate_no_direction_stats(summary_stats)
        
        # Print summary report
        self._print_summary_report(summary_stats)
        
        # Save detailed report
        self._save_detailed_report(summary_stats)
        
        return summary_stats
    
    def _calculate_aggregate_down_stats(self, summary_stats: Dict):
        """Calculate aggregate statistics for Down class"""
        
        down_stats = summary_stats['down_class_stats']
        
        # Aggregate probability statistics
        prob_down_means = [stats['prob_down']['mean'] for stats in down_stats]
        prob_no_dir_means = [stats['prob_no_direction']['mean'] for stats in down_stats]
        prob_up_means = [stats['prob_up']['mean'] for stats in down_stats]
        
        # Aggregate error statistics
        total_samples = sum(stats['num_samples'] for stats in down_stats)
        total_prob_1_higher = sum(stats['prediction_errors']['times_prob_1_higher_than_prob_0'] for stats in down_stats)
        total_prob_2_higher = sum(stats['prediction_errors']['times_prob_2_higher_than_prob_0'] for stats in down_stats)
        
        summary_stats['aggregate_down_stats'] = {
            'total_samples': total_samples,
            'mean_prob_down': np.mean(prob_down_means),
            'mean_prob_no_direction': np.mean(prob_no_dir_means),
            'mean_prob_up': np.mean(prob_up_means),
            'error_rate_prob_1_higher': total_prob_1_higher / total_samples if total_samples > 0 else 0,
            'error_rate_prob_2_higher': total_prob_2_higher / total_samples if total_samples > 0 else 0,
            'total_error_rate': (total_prob_1_higher + total_prob_2_higher) / total_samples if total_samples > 0 else 0
        }
    
    def _calculate_aggregate_no_direction_stats(self, summary_stats: Dict):
        """Calculate aggregate statistics for No Direction class"""
        
        no_dir_stats = summary_stats['no_direction_class_stats']
        
        # Aggregate probability statistics
        prob_down_means = [stats['prob_down']['mean'] for stats in no_dir_stats]
        prob_no_dir_means = [stats['prob_no_direction']['mean'] for stats in no_dir_stats]
        prob_up_means = [stats['prob_up']['mean'] for stats in no_dir_stats]
        
        # Aggregate error statistics
        total_samples = sum(stats['num_samples'] for stats in no_dir_stats)
        total_prob_0_higher = sum(stats['prediction_errors']['times_prob_0_higher_than_prob_1'] for stats in no_dir_stats)
        total_prob_2_higher = sum(stats['prediction_errors']['times_prob_2_higher_than_prob_1'] for stats in no_dir_stats)
        
        summary_stats['aggregate_no_direction_stats'] = {
            'total_samples': total_samples,
            'mean_prob_down': np.mean(prob_down_means),
            'mean_prob_no_direction': np.mean(prob_no_dir_means),
            'mean_prob_up': np.mean(prob_up_means),
            'error_rate_prob_0_higher': total_prob_0_higher / total_samples if total_samples > 0 else 0,
            'error_rate_prob_2_higher': total_prob_2_higher / total_samples if total_samples > 0 else 0,
            'total_error_rate': (total_prob_0_higher + total_prob_2_higher) / total_samples if total_samples > 0 else 0
        }
    
    def _print_summary_report(self, summary_stats: Dict):
        """Print comprehensive summary report"""
        
        print("\n" + "="*80)
        print("PROBABILITY ANALYSIS SUMMARY REPORT")
        print("="*80)
        print(f"Total periods analyzed: {summary_stats['total_periods']}")
        print(f"Total samples: {summary_stats['total_samples']}")
        
        # Down class analysis
        if 'aggregate_down_stats' in summary_stats:
            down_stats = summary_stats['aggregate_down_stats']
            print(f"\nDOWN CLASS ANALYSIS (True Label = 0):")
            print(f"  Total samples: {down_stats['total_samples']}")
            print(f"  Mean probability for Down: {down_stats['mean_prob_down']:.4f}")
            print(f"  Mean probability for No Direction: {down_stats['mean_prob_no_direction']:.4f}")
            print(f"  Mean probability for Up: {down_stats['mean_prob_up']:.4f}")
            print(f"  Error rate (No Direction > Down): {down_stats['error_rate_prob_1_higher']:.4f}")
            print(f"  Error rate (Up > Down): {down_stats['error_rate_prob_2_higher']:.4f}")
            print(f"  Total error rate: {down_stats['total_error_rate']:.4f}")
        
        # No Direction class analysis
        if 'aggregate_no_direction_stats' in summary_stats:
            no_dir_stats = summary_stats['aggregate_no_direction_stats']
            print(f"\nNO DIRECTION CLASS ANALYSIS (True Label = 1):")
            print(f"  Total samples: {no_dir_stats['total_samples']}")
            print(f"  Mean probability for Down: {no_dir_stats['mean_prob_down']:.4f}")
            print(f"  Mean probability for No Direction: {no_dir_stats['mean_prob_no_direction']:.4f}")
            print(f"  Mean probability for Up: {no_dir_stats['mean_prob_up']:.4f}")
            print(f"  Error rate (Down > No Direction): {no_dir_stats['error_rate_prob_0_higher']:.4f}")
            print(f"  Error rate (Up > No Direction): {no_dir_stats['error_rate_prob_2_higher']:.4f}")
            print(f"  Total error rate: {no_dir_stats['total_error_rate']:.4f}")
        
        print("\n" + "="*80)
    
    def _save_detailed_report(self, summary_stats: Dict):
        """Save detailed report to file"""
        
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert summary stats
        serializable_stats = convert_numpy_types(summary_stats)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f'probability_analysis_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(serializable_stats, f, indent=4, default=str)
        
        logger.info(f"Detailed report saved to: {report_path}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze STGNN Model Prediction Probabilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all probability files in default directory
  python scripts/analyze_probabilities.py

  # Analyze specific probability directory
  python scripts/analyze_probabilities.py \\
    --probability-dir models/probability_analysis \\
    --plots-dir plots/probability_analysis \\
    --reports-dir reports/probability_analysis

  # Analyze specific file pattern
  python scripts/analyze_probabilities.py \\
    --file-pattern "*2024*_*.csv"
        """
    )
    
    parser.add_argument(
        '--probability-dir',
        type=str,
        default='models/probability_analysis',
        help='Directory containing probability CSV files (default: models/probability_analysis)'
    )
    
    parser.add_argument(
        '--plots-dir',
        type=str,
        default='plots/probability_analysis',
        help='Directory to save plots (default: plots/probability_analysis)'
    )
    
    parser.add_argument(
        '--reports-dir',
        type=str,
        default='reports/probability_analysis',
        help='Directory to save reports (default: reports/probability_analysis)'
    )
    
    parser.add_argument(
        '--file-pattern',
        type=str,
        default='*.csv',
        help='File pattern to match probability files (default: *.csv)'
    )
    
    return parser.parse_args()

def main():
    """Main analysis function"""
    
    # Parse arguments
    args = parse_arguments()
    
    logger.info("="*80)
    logger.info("STARTING PROBABILITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Probability directory: {args.probability_dir}")
    logger.info(f"Plots directory: {args.plots_dir}")
    logger.info(f"Reports directory: {args.reports_dir}")
    logger.info(f"File pattern: {args.file_pattern}")
    
    try:
        # Create analyzer
        analyzer = ProbabilityAnalyzer(
            probability_dir=args.probability_dir,
            plots_dir=args.plots_dir,
            reports_dir=args.reports_dir
        )
        
        # Load probability files
        logger.info("Loading probability files...")
        all_data = analyzer.load_probability_files(args.file_pattern)
        
        if not all_data:
            logger.error("No probability files found. Please run walk-forward optimization first.")
            return
        
        # Perform analysis
        logger.info("Performing probability analysis...")
        summary_stats = analyzer.generate_summary_report(all_data)
        
        logger.info("="*80)
        logger.info("PROBABILITY ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Analysis results saved to:")
        logger.info(f"  Plots: {args.plots_dir}")
        logger.info(f"  Reports: {args.reports_dir}")
        
        return summary_stats
        
    except Exception as e:
        logger.error(f"Probability analysis failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 