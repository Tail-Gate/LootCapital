import asyncio
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_management.historical_data_collector import HistoricalDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('historical_data_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    # Set specific start and end times
    end_time = datetime.now()
    start_time = datetime(2020, 1, 1)  # January 1st, 2020
    
    # Initialize collector with OKX
    collector = HistoricalDataCollector(
        exchange_id='okx',  # Using OKX
        symbol='ETH-USDT-SWAP',   # Perpetual futures symbol format for OKX
        interval='15m',
        data_dir='data/historical',
        max_retries=5,
        retry_delay=10,
        batch_size=1000,  # OKX allows up to 1000 candles per request
        max_workers=4
    )
    
    try:
        # Load existing data first
        existing_data = collector.load_data()
        if not existing_data.empty:
            logger.info(f"Loaded {len(existing_data)} existing candles")
            collector.ohlcv_data = existing_data
            
            # Find the earliest date in existing data
            earliest_date = existing_data.index.min()
            if earliest_date > start_time:
                logger.info(f"Existing data starts from {earliest_date}, will collect data from {start_time} to {earliest_date}")
                # Collect missing data from start_time to earliest_date
                await collector.collect_historical_data(start_time, earliest_date)
            else:
                logger.info("Existing data already covers the requested time range")
        else:
            # No existing data, collect everything
            logger.info(f"No existing data found. Starting collection from {start_time} to {end_time}")
            await collector.collect_historical_data(start_time, end_time)
        
        # Validate collected data
        validation_results = collector.validate_historical_data()
        logger.info(f"Data validation results: {validation_results}")
        
        if all(validation_results.values()):
            logger.info("Historical data collection completed successfully!")
        else:
            logger.warning("Historical data collection completed with some issues. Check validation results.")
        
    except Exception as e:
        logger.error(f"Error in historical data collection: {str(e)}")
        raise
    finally:
        # Save final progress
        collector.save_progress()

if __name__ == "__main__":
    asyncio.run(main()) 