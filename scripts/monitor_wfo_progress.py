#!/usr/bin/env python3
"""
Monitor Walk-Forward Optimization Progress

This script monitors the progress of the walk-forward optimization
and provides real-time updates on completion status.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_wfo_progress():
    """Check the progress of walk-forward optimization"""
    
    print("Walk-Forward Optimization Progress Monitor")
    print("=" * 50)
    
    # Check for log files
    log_files = glob.glob("logs/walk_forward_optimization_*.log")
    if not log_files:
        print("No walk-forward optimization logs found")
        return
    
    # Get the most recent log file
    latest_log = max(log_files, key=os.path.getctime)
    print(f"Monitoring log file: {latest_log}")
    
    # Check for model files
    model_files = glob.glob("models/wfo_stgnn_*.pt")
    print(f"Models completed: {len(model_files)}")
    
    # Check for report files
    report_files = glob.glob("reports/walk_forward_report_*.json")
    print(f"Reports generated: {len(report_files)}")
    
    # Check for plot files
    plot_files = glob.glob("plots/walk_forward_results_*.png")
    print(f"Plots generated: {len(plot_files)}")
    
    # Read the latest log file
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            
        if lines:
            print(f"\nLatest log entries:")
            print("-" * 30)
            
            # Show last 10 lines
            for line in lines[-10:]:
                line = line.strip()
                if line:
                    print(line)
                    
            # Check for completion
            if any("WALK-FORWARD OPTIMIZATION COMPLETED" in line for line in lines):
                print(f"\n✅ Walk-forward optimization completed!")
                
                # Try to find summary statistics
                for line in lines:
                    if "Mean test accuracy:" in line:
                        print(f"📊 {line.strip()}")
                    elif "Mean F1 (Up):" in line:
                        print(f"📈 {line.strip()}")
                    elif "Mean F1 (Down):" in line:
                        print(f"📉 {line.strip()}")
                    elif "Total time:" in line:
                        print(f"⏱️  {line.strip()}")
                        
            elif any("WALK-FORWARD OPTIMIZATION FAILED" in line for line in lines):
                print(f"\n❌ Walk-forward optimization failed!")
                
                # Show error details
                for line in lines:
                    if "Error:" in line:
                        print(f"🔴 {line.strip()}")
                        
            else:
                print(f"\n🔄 Walk-forward optimization is still running...")
                
    except Exception as e:
        print(f"Error reading log file: {e}")

def check_expected_periods():
    """Calculate expected number of periods"""
    
    print(f"\nExpected Walk-Forward Periods:")
    print("-" * 30)
    
    # Parameters from the command
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    train_window_days = 180
    test_window_days = 30
    step_size_days = 15
    
    total_days = (end_date - start_date).days
    required_days_per_period = train_window_days + test_window_days
    max_periods = (total_days - required_days_per_period) // step_size_days + 1
    
    print(f"Data period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total days: {total_days}")
    print(f"Training window: {train_window_days} days")
    print(f"Testing window: {test_window_days} days")
    print(f"Step size: {step_size_days} days")
    print(f"Expected periods: {max_periods}")
    
    # Estimate completion time
    avg_time_per_period = 10  # minutes (estimate)
    total_estimated_time = max_periods * avg_time_per_period
    print(f"Estimated total time: {total_estimated_time} minutes ({total_estimated_time/60:.1f} hours)")

def check_recent_results():
    """Check recent results if available"""
    
    report_files = glob.glob("reports/walk_forward_report_*.json")
    if report_files:
        latest_report = max(report_files, key=os.path.getctime)
        
        try:
            with open(latest_report, 'r') as f:
                report = json.load(f)
                
            print(f"\nLatest Results Summary:")
            print("-" * 30)
            print(f"Total periods: {report.get('total_periods', 'N/A')}")
            print(f"Mean test accuracy: {report.get('mean_test_accuracy', 'N/A')}")
            print(f"Mean F1 (Up): {report.get('mean_f1_up', 'N/A')}")
            print(f"Mean F1 (Down): {report.get('mean_f1_down', 'N/A')}")
            print(f"Best period: {report.get('best_period', 'N/A')}")
            print(f"Best accuracy: {report.get('best_accuracy', 'N/A')}")
            
        except Exception as e:
            print(f"Error reading report: {e}")

def main():
    """Main monitoring function"""
    
    while True:
        # Clear screen (optional)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"Walk-Forward Optimization Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        check_expected_periods()
        check_wfo_progress()
        check_recent_results()
        
        print(f"\nPress Ctrl+C to stop monitoring")
        print("Refreshing in 30 seconds...")
        
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print(f"\nMonitoring stopped by user")
            break

if __name__ == "__main__":
    main() 