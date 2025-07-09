#!/usr/bin/env python3
"""
Monitor AI Platform quota and automatically submit walk-forward optimization job
when quota becomes available.
"""

import subprocess
import time
import logging
import sys
from datetime import datetime, timedelta
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quota_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuotaMonitor:
    def __init__(self, check_interval_minutes=5):
        self.check_interval = check_interval_minutes * 60  # Convert to seconds
        self.job_command = [
            'python', 'scripts/walk_forward_optimization.py',
            '--start-date=2020-01-01',
            '--end-date=2023-12-31',
            '--train-window-days=365',
            '--test-window-days=60',
            '--step-size-days=30',
            '--price-threshold=0.018',
            '--scaler-type=minmax',
            '--output-dir=models',
            '--reports-dir=reports',
            '--plots-dir=plots',
            '--logs-dir=logs',
            '--log-level=INFO'
        ]
        
    def check_quota(self):
        """Check if the system is ready to run the walk-forward optimization"""
        try:
            # Try to run a minimal test to check if the system is ready
            test_command = [
                'python', 'scripts/walk_forward_optimization.py',
                '--help'
            ]
            
            result = subprocess.run(test_command, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✅ System is ready! Test command executed successfully.")
                return True
            else:
                logger.warning(f"System not ready: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("Timeout checking system readiness, assuming ready")
            return True
        except Exception as e:
            logger.error(f"Error checking system readiness: {e}")
            return False
    
    def submit_job(self):
        """Run the actual walk-forward optimization job"""
        try:
            logger.info("🚀 Running walk-forward optimization job...")
            logger.info(f"Command: {' '.join(self.job_command)}")
            
            result = subprocess.run(self.job_command, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("✅ Job completed successfully!")
                logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Failed to run job: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout running job")
            return False
        except Exception as e:
            logger.error(f"❌ Error running job: {e}")
            return False
    
    def monitor_and_submit(self):
        """Monitor system readiness and run job when ready"""
        logger.info("🔍 Starting system monitoring...")
        logger.info(f"Checking every {self.check_interval/60:.1f} minutes")
        logger.info("Press Ctrl+C to stop monitoring")
        
        check_count = 0
        start_time = datetime.now()
        
        while True:
            try:
                check_count += 1
                current_time = datetime.now()
                elapsed = current_time - start_time
                
                logger.info(f"Check #{check_count} at {current_time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {elapsed})")
                
                if self.check_quota():
                    logger.info("🎉 System is ready! Running job...")
                    if self.submit_job():
                        logger.info("🎊 Job completed successfully! Monitoring complete.")
                        break
                    else:
                        logger.warning("Job execution failed, continuing to monitor...")
                
                logger.info(f"⏰ Waiting {self.check_interval/60:.1f} minutes until next check...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in monitoring loop: {e}")
                logger.info("Continuing to monitor...")
                time.sleep(self.check_interval)

def main():
    """Main function"""
    print("="*80)
    print("System Monitor & Job Runner")
    print("="*80)
    print("This script will monitor system readiness and automatically run")
    print("the walk-forward optimization job when the system is ready.")
    print()
    print("Job Details:")
    print("- Local execution with Python")
    print("- Walk-forward optimization for STGNN model")
    print("- Prediction Horizon: 15 (already configured)")
    print("- Output: Local directories for models, reports, plots, logs")
    print()
    
    # Create logs directory
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Start monitoring
    monitor = QuotaMonitor(check_interval_minutes=5)
    monitor.monitor_and_submit()

if __name__ == "__main__":
    main() 