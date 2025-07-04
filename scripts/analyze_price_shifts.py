#!/usr/bin/env python3
"""
Price Shift Distribution Analysis for Ethereum Futures

This script analyzes the distribution of price shifts for different thresholds
to understand the class imbalance problem in the trading model.
Focuses on Ethereum futures data and analyzes moves >= threshold in either direction.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_ethereum_data(start_date: datetime, end_date: datetime):
    """Load Ethereum futures data for analysis"""
    
    print(f"Loading Ethereum futures data from {start_date} to {end_date}")
    
    # Load ETH-USDT-SWAP data directly
    data_path = Path("data/historical/ETH-USDT-SWAP_ohlcv_15m.csv")
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return None
    
    try:
        # Load the data
        data = pd.read_csv(data_path)
        
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Filter by date range
        data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Loaded {len(data)} records for ETH-USDT-SWAP")
        print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_price_shifts(data: pd.DataFrame, prediction_horizon: int = 15):
    """Calculate price shifts using event-based analysis - detect any significant move within the window"""
    
    if len(data) < prediction_horizon + 1:
        print(f"Warning: Insufficient data for {prediction_horizon}-period analysis")
        return None
    
    print(f"Performing event-based analysis: detecting moves >= threshold within {prediction_horizon}-period windows")
    
    # Initialize results arrays
    n_samples = len(data) - prediction_horizon
    max_positive_moves = np.zeros(n_samples)
    max_negative_moves = np.zeros(n_samples)
    event_occurred = np.zeros(n_samples, dtype=bool)
    
    # For each starting candle, analyze the next prediction_horizon candles
    for i in range(n_samples):
        start_price = data['close'].iloc[i]
        window_prices = data['close'].iloc[i+1:i+1+prediction_horizon]
        
        # Calculate all price changes within the window
        price_changes = (window_prices - start_price) / start_price
        
        # Find maximum positive and negative moves
        max_positive_moves[i] = price_changes.max()
        max_negative_moves[i] = price_changes.min()
        
        # Check if any significant move occurred (we'll apply thresholds later)
        event_occurred[i] = True  # All windows have some price movement
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'max_positive_move': max_positive_moves,
        'max_negative_move': max_negative_moves,
        'event_occurred': event_occurred
    })
    
    print(f"Analyzed {n_samples} price windows")
    print(f"Sample of results:")
    print(f"  Max positive move range: {max_positive_moves.min():.4f} to {max_positive_moves.max():.4f}")
    print(f"  Max negative move range: {max_negative_moves.min():.4f} to {max_negative_moves.max():.4f}")
    
    return results_df

def analyze_threshold_distribution(price_shifts_df: pd.DataFrame, thresholds: list):
    """Analyze distribution for different thresholds using event-based analysis"""
    
    results = {}
    
    for threshold in thresholds:
        # Count occurrences for each class using event-based analysis
        # Long opportunities: maximum positive move >= threshold within the window
        long_opportunities = (price_shifts_df['max_positive_move'] >= threshold).sum()
        
        # Short opportunities: maximum negative move <= -threshold within the window
        short_opportunities = (price_shifts_df['max_negative_move'] <= -threshold).sum()
        
        # Both opportunities: either long OR short opportunity occurred
        both_opportunities = ((price_shifts_df['max_positive_move'] >= threshold) | 
                             (price_shifts_df['max_negative_move'] <= -threshold)).sum()
        
        # No significant moves: neither long nor short opportunity occurred
        no_significant_moves = ((price_shifts_df['max_positive_move'] < threshold) & 
                               (price_shifts_df['max_negative_move'] > -threshold)).sum()
        
        total = len(price_shifts_df)
        
        results[threshold] = {
            'threshold': threshold,
            'total_samples': total,
            'long_opportunities': long_opportunities,
            'short_opportunities': short_opportunities,
            'both_opportunities': both_opportunities,
            'no_significant_moves': no_significant_moves,
            'long_percentage': long_opportunities / total * 100,
            'short_percentage': short_opportunities / total * 100,
            'both_percentage': both_opportunities / total * 100,
            'no_moves_percentage': no_significant_moves / total * 100,
            'total_trading_opportunities': both_opportunities
        }
    
    return results

def print_analysis_results(results: dict):
    """Print analysis results in a formatted way"""
    
    print(f"\n{'='*100}")
    print(f"ETHEREUM FUTURES PRICE SHIFT ANALYSIS")
    print(f"{'='*100}")
    print(f"Analysis: Price moves >= threshold in either direction (Long/Short opportunities)")
    print(f"Prediction Horizon: 15 periods ahead")
    
    # Create header
    header = f"{'Threshold':<10} {'Total':<8} {'Long':<8} {'Short':<8} {'No Move':<10} {'Long%':<8} {'Short%':<8} {'No%':<8} {'Trade%':<10}"
    print(header)
    print("-" * len(header))
    
    # Print results for each threshold
    for threshold, data in results.items():
        row = (f"{threshold*100:>6.1f}%+ "
               f"{data['total_samples']:<8} "
               f"{data['long_opportunities']:<8} "
               f"{data['short_opportunities']:<8} "
               f"{data['no_significant_moves']:<10} "
               f"{data['long_percentage']:<8.2f} "
               f"{data['short_percentage']:<8.2f} "
               f"{data['no_moves_percentage']:<8.2f} "
               f"{data['both_percentage']:<10.2f}")
        print(row)
    
    print("-" * len(header))
    
    # Print summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"{'='*50}")
    for threshold, data in results.items():
        print(f"\nThreshold {threshold*100:.1f}%+ moves:")
        print(f"  Total trading opportunities: {data['total_trading_opportunities']:,} out of {data['total_samples']:,} samples")
        print(f"  Long opportunities: {data['long_opportunities']:,} ({data['long_percentage']:.2f}%)")
        print(f"  Short opportunities: {data['short_opportunities']:,} ({data['short_percentage']:.2f}%)")
        print(f"  No significant moves: {data['no_significant_moves']:,} ({data['no_moves_percentage']:.2f}%)")
        
        # Calculate expected trades per day (assuming 15-minute data = 96 periods per day)
        periods_per_day = 96
        trades_per_day = data['total_trading_opportunities'] / (data['total_samples'] / periods_per_day)
        print(f"  Expected trades per day: {trades_per_day:.2f}")

def create_visualizations(results: dict, output_dir: str = "plots"):
    """Create visualizations for the analysis"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    thresholds = list(results.keys())
    long_percentages = [results[t]['long_percentage'] for t in thresholds]
    short_percentages = [results[t]['short_percentage'] for t in thresholds]
    no_moves_percentages = [results[t]['no_moves_percentage'] for t in thresholds]
    both_percentages = [results[t]['both_percentage'] for t in thresholds]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ethereum Futures Price Shift Analysis\n(Moves >= Threshold in Either Direction)', fontsize=16)
    
    # Plot 1: Class distribution by threshold
    x = [f"{t*100:.1f}%+" for t in thresholds]
    width = 0.25
    
    ax1.bar([i - width for i in range(len(x))], long_percentages, width, label='Long Opportunities', color='green', alpha=0.7)
    ax1.bar([i for i in range(len(x))], short_percentages, width, label='Short Opportunities', color='red', alpha=0.7)
    ax1.bar([i + width for i in range(len(x))], no_moves_percentages, width, label='No Significant Moves', color='gray', alpha=0.7)
    
    ax1.set_xlabel('Price Threshold')
    ax1.set_ylabel('Percentage of Samples')
    ax1.set_title('Trading Opportunities by Threshold')
    ax1.set_xticks(range(len(x)))
    ax1.set_xticklabels(x)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total trading opportunities vs threshold
    ax2.plot(thresholds, both_percentages, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Price Threshold')
    ax2.set_ylabel('Trading Opportunities (%)')
    ax2.set_title('Total Trading Opportunities vs Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(thresholds)
    ax2.set_xticklabels([f"{t*100:.1f}%+" for t in thresholds])
    
    # Plot 3: Long vs Short opportunities
    ax3.plot(thresholds, long_percentages, 'go-', linewidth=2, markersize=8, label='Long Opportunities')
    ax3.plot(thresholds, short_percentages, 'ro-', linewidth=2, markersize=8, label='Short Opportunities')
    ax3.set_xlabel('Price Threshold')
    ax3.set_ylabel('Percentage of Samples')
    ax3.set_title('Long vs Short Opportunities')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(thresholds)
    ax3.set_xticklabels([f"{t*100:.1f}%+" for t in thresholds])
    
    # Plot 4: No significant moves percentage
    ax4.plot(thresholds, no_moves_percentages, 'ko-', linewidth=2, markersize=8)
    ax4.set_xlabel('Price Threshold')
    ax4.set_ylabel('No Significant Moves (%)')
    ax4.set_title('Periods with No Significant Moves')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(thresholds)
    ax4.set_xticklabels([f"{t*100:.1f}%+" for t in thresholds])
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_path / f"ethereum_price_shift_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()

def generate_summary_report(results: dict, output_dir: str = "reports"):
    """Generate a summary report of the analysis"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path / f"ethereum_price_shift_analysis_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("ETHEREUM FUTURES PRICE SHIFT DISTRIBUTION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prediction Horizon: 15 periods\n")
        f.write(f"Analysis: Price moves >= threshold in either direction\n\n")
        
        for threshold, data in results.items():
            f.write(f"Threshold {threshold*100:.1f}%+ moves:\n")
            f.write(f"  Total samples: {data['total_samples']:,}\n")
            f.write(f"  Long opportunities: {data['long_opportunities']:,} ({data['long_percentage']:.2f}%)\n")
            f.write(f"  Short opportunities: {data['short_opportunities']:,} ({data['short_percentage']:.2f}%)\n")
            f.write(f"  No significant moves: {data['no_significant_moves']:,} ({data['no_moves_percentage']:.2f}%)\n")
            f.write(f"  Total trading opportunities: {data['total_trading_opportunities']:,} ({data['total_trading_opportunities'] / data['total_samples'] * 100:.2f}%)\n\n")
        
        # Add recommendations
        f.write("\nTRADING STRATEGY RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. For 0.5%+ threshold: High frequency trading opportunities\n")
        f.write("   - Many signals but may include noise\n")
        f.write("   - Suitable for scalping strategies\n\n")
        f.write("2. For 1.0%+ threshold: Balanced frequency and quality\n")
        f.write("   - Good balance of signal frequency and quality\n")
        f.write("   - Recommended for day trading\n\n")
        f.write("3. For 1.5%+ threshold: Higher quality signals\n")
        f.write("   - Fewer but higher quality signals\n")
        f.write("   - Suitable for swing trading\n\n")
        f.write("4. For 2.0%+ threshold: Low frequency, high quality\n")
        f.write("   - Very few signals but likely high quality\n")
        f.write("   - May be too conservative for active trading\n\n")
        
        f.write("RECOMMENDED THRESHOLD: 1.0% or 1.5% for balanced approach\n")
    
    print(f"Report saved to: {report_path}")

def main():
    """Main analysis function"""
    
    parser = argparse.ArgumentParser(description='Analyze Ethereum futures price shift distributions')
    parser.add_argument('--start-date', type=str, default='2020-01-01', 
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-05-29', 
                       help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--thresholds', nargs='+', type=float,
                       default=[0.005, 0.01, 0.015, 0.0175,0.02],
                       help='Price thresholds to analyze (as decimals)')
    parser.add_argument('--prediction-horizon', type=int, default=15,
                       help='Prediction horizon in periods')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--reports-dir', type=str, default='reports',
                       help='Output directory for reports')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    print(f"Ethereum Futures Price Shift Distribution Analysis")
    print(f"Period: {start_date} to {end_date}")
    print(f"Prediction Horizon: {args.prediction_horizon} periods")
    print(f"Thresholds: {[f'{t*100:.1f}%+' for t in args.thresholds]}")
    print(f"Analysis: Price moves >= threshold in either direction (Long/Short opportunities)")
    
    # Load Ethereum data
    data = load_ethereum_data(start_date, end_date)
    
    if data is None:
        print("No data loaded. Exiting.")
        return
    
    # Calculate price shifts
    price_shifts = calculate_price_shifts(data, args.prediction_horizon)
    
    if price_shifts is None:
        print("Insufficient data for analysis. Exiting.")
        return
    
    # Analyze thresholds
    results = analyze_threshold_distribution(price_shifts, args.thresholds)
    
    # Print results
    print_analysis_results(results)
    
    # Create visualizations
    if not args.no_plots:
        create_visualizations(results, args.output_dir)
    
    # Generate summary report
    generate_summary_report(results, args.reports_dir)
    
    # Print final recommendations
    print(f"\n{'='*100}")
    print("FINAL RECOMMENDATIONS FOR TRADING STRATEGY")
    print(f"{'='*100}")
    
    # Find the threshold with best balance
    best_threshold = None
    best_balance = 0
    
    for threshold, data in results.items():
        # Calculate balance score (signal percentage * quality factor)
        signal_pct = data['total_trading_opportunities'] / data['total_samples']
        quality_factor = threshold * 100  # Higher threshold = higher quality
        balance_score = signal_pct * quality_factor
        
        if balance_score > best_balance:
            best_balance = balance_score
            best_threshold = threshold
    
    if best_threshold:
        best_data = results[best_threshold]
        print(f"\nRECOMMENDED THRESHOLD: {best_threshold*100:.1f}%+")
        print(f"  Trading opportunities: {best_data['total_trading_opportunities'] / best_data['total_samples'] * 100:.2f}% of periods")
        print(f"  Long opportunities: {best_data['long_opportunities'] / best_data['total_samples'] * 100:.2f}%")
        print(f"  Short opportunities: {best_data['short_opportunities'] / best_data['total_samples'] * 100:.2f}%")
        print(f"  Expected trades per day: {best_data['total_trading_opportunities'] / (best_data['total_samples'] / 96):.2f}")
    
    print(f"\nThis analysis shows the distribution of price moves >= threshold in either direction.")
    print(f"Use this information to set appropriate thresholds for your trading strategy.")

if __name__ == "__main__":
    main() 