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
            'gcloud', 'ai', 'custom-jobs', 'create',
            '--region=us-central1',
            '--display-name=STGNN Walk-Forward Optimization Horizon15',
            '--worker-pool-spec=machine-type=n1-standard-8,replica-count=1,accelerator-type=NVIDIA_TESLA_V100,accelerator-count=1,container-image-uri=us-central1-docker.pkg.dev/delta-crane-464102-a1/lootcapital-repo/lootcapital-wfo:latest',
            '--args=--start-date=2020-01-01,--end-date=2023-12-31,--train-window-days=365,--test-window-days=60,--step-size-days=30,--price-threshold=0.018,--scaler-type=minmax,--output-dir=gs://lootcapital-models,--reports-dir=gs://lootcapital-reports,--plots-dir=gs://lootcapital-plots,--logs-dir=gs://lootcapital-logs,--log-level=INFO'
        ]
        
    def check_quota(self):
        """Check if AI Platform quota is available by attempting to create a test job"""
        try:
            # Try to create a minimal test job to check quota
            test_command = [
                'gcloud', 'ai', 'custom-jobs', 'create',
                '--region=us-central1',
                '--display-name=quota-test',
                '--worker-pool-spec=machine-type=e2-standard-4,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/delta-crane-464102-a1/lootcapital-repo/lootcapital-wfo:latest',
                '--args=--help'
            ]
            
            result = subprocess.run(test_command, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✅ Quota is available! Test job created successfully.")
                return True
            elif "RESOURCE_EXHAUSTED" in result.stderr:
                logger.info("❌ Quota still exhausted. Waiting...")
                return False
            else:
                logger.warning(f"Unexpected error checking quota: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("Timeout checking quota, assuming available")
            return True
        except Exception as e:
            logger.error(f"Error checking quota: {e}")
            return False
    
    def submit_job(self):
        """Submit the actual walk-forward optimization job"""
        try:
            logger.info("🚀 Submitting walk-forward optimization job...")
            logger.info(f"Command: {' '.join(self.job_command)}")
            
            result = subprocess.run(self.job_command, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("✅ Job submitted successfully!")
                logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Failed to submit job: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout submitting job")
            return False
        except Exception as e:
            logger.error(f"❌ Error submitting job: {e}")
            return False
    
    def monitor_and_submit(self):
        """Monitor quota and submit job when available"""
        logger.info("🔍 Starting quota monitoring...")
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
                    logger.info("🎉 Quota is available! Submitting job...")
                    if self.submit_job():
                        logger.info("🎊 Job submitted successfully! Monitoring complete.")
                        break
                    else:
                        logger.warning("Job submission failed, continuing to monitor...")
                
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
    print("AI Platform Quota Monitor & Job Submitter")
    print("="*80)
    print("This script will monitor AI Platform quota and automatically submit")
    print("the walk-forward optimization job when quota becomes available.")
    print()
    print("Job Details:")
    print("- Machine: n1-standard-8 with NVIDIA_TESLA_V100 GPU")
    print("- Region: us-central1")
    print("- Prediction Horizon: 15 (already configured)")
    print("- Output: GCS buckets for models, reports, plots, logs")
    print()
    
    # Create logs directory
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Start monitoring
    monitor = QuotaMonitor(check_interval_minutes=5)
    monitor.monitor_and_submit()

if __name__ == "__main__":
    main() 