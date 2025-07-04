# Docker Setup for LootCapital Walk-Forward Optimization

This document explains how to use Docker to run the walk-forward optimization script in a containerized environment with support for command-line arguments.

## Prerequisites

- Docker installed and running
- Docker Compose installed
- At least 4GB of available RAM
- At least 10GB of available disk space

## Quick Start

### Option 1: Using the provided script (Recommended)

```bash
# Make the script executable (if not already done)
chmod +x run_docker.sh

# Build and run the walk-forward optimization with default settings
./run_docker.sh run

# Run with custom arguments
./run_docker.sh args --start-date 2020-01-01 --end-date 2023-12-31

# Run with custom parameters
./run_docker.sh args --train-window-days 180 --test-window-days 30 --step-size-days 15

# Run with Google Cloud Storage paths (for Vertex AI)
./run_docker.sh args --output-dir gs://my-bucket/models --reports-dir gs://my-bucket/reports

# Or run in detached mode (runs in background)
./run_docker.sh detached
```

### Option 2: Using Docker Compose directly

```bash
# Build and run the container with default settings
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Option 3: Using Docker directly with custom arguments

```bash
# Build the image
docker build -t lootcapital-wfo .

# Run with default settings
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/plots:/app/plots \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/lightning_logs:/app/lightning_logs \
  -v $(pwd)/profiling_results:/app/profiling_results \
  lootcapital-wfo

# Run with custom arguments
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/plots:/app/plots \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/lightning_logs:/app/lightning_logs \
  -v $(pwd)/profiling_results:/app/profiling_results \
  lootcapital-wfo python scripts/walk_forward_optimization.py \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --train-window-days 180 \
    --test-window-days 30 \
    --output-dir /app/models \
    --reports-dir /app/reports
```

## Script Commands

The `run_docker.sh` script provides several useful commands:

```bash
./run_docker.sh build      # Build the Docker image only
./run_docker.sh run        # Build and run the container with default settings
./run_docker.sh args       # Build and run the container with custom arguments
./run_docker.sh detached   # Build and run the container in detached mode
./run_docker.sh logs       # Show container logs
./run_docker.sh stop       # Stop the container
./run_docker.sh cleanup    # Stop and remove all containers, images, and volumes
./run_docker.sh help       # Show help message with examples
```

## Command-Line Arguments

The walk-forward optimization script supports the following command-line arguments:

### Data and Time Range
- `--start-date YYYY-MM-DD`: Start date for walk-forward optimization (default: 5 years ago)
- `--end-date YYYY-MM-DD`: End date for walk-forward optimization (default: today)

### Output Directories
- `--output-dir PATH`: Directory to save trained models (default: models)
- `--reports-dir PATH`: Directory to save reports (default: reports)
- `--plots-dir PATH`: Directory to save plots (default: plots)
- `--logs-dir PATH`: Directory to save logs (default: logs)

### Walk-Forward Parameters
- `--train-window-days N`: Training window size in days (default: 365)
- `--test-window-days N`: Testing window size in days (default: 60)
- `--step-size-days N`: Step size between periods in days (default: 30)
- `--price-threshold FLOAT`: Price threshold for classification (default: 0.018)

### Model Parameters
- `--scaler-type TYPE`: Type of scaler to use: minmax or standard (default: minmax)

### Logging
- `--log-level LEVEL`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--quiet`: Suppress console output (only log to file)

## Vertex AI Integration

For running on Google Cloud Vertex AI, you can use Google Cloud Storage paths:

```bash
# Example for Vertex AI with GCS paths
./run_docker.sh args \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output-dir gs://my-bucket/models \
  --reports-dir gs://my-bucket/reports \
  --plots-dir gs://my-bucket/plots \
  --logs-dir gs://my-bucket/logs \
  --train-window-days 180 \
  --test-window-days 30 \
  --step-size-days 15
```

### Vertex AI Custom Job Configuration

When creating a Vertex AI custom job, you can specify the command and arguments:

```yaml
# Example Vertex AI custom job configuration
containerSpec:
  imageUri: gcr.io/your-project/lootcapital-wfo:latest
  command: ["python"]
  args:
    - "scripts/walk_forward_optimization.py"
    - "--start-date"
    - "2020-01-01"
    - "--end-date"
    - "2023-12-31"
    - "--output-dir"
    - "gs://my-bucket/models"
    - "--reports-dir"
    - "gs://my-bucket/reports"
    - "--plots-dir"
    - "gs://my-bucket/plots"
    - "--logs-dir"
    - "gs://my-bucket/logs"
    - "--train-window-days"
    - "180"
    - "--test-window-days"
    - "30"
    - "--step-size-days"
    - "15"
    - "--log-level"
    - "INFO"
```

## Directory Structure

The Docker setup mounts the following directories for data persistence:

```
./data/                    # Input data files
./models/                  # Trained models and checkpoints
./reports/                 # Walk-forward optimization reports
./plots/                   # Generated visualizations
./logs/                    # Application logs
./cache/                   # Feature cache and temporary files
./lightning_logs/          # PyTorch Lightning logs
./profiling_results/       # Performance profiling results
```

## Configuration

### Environment Variables

The following environment variables are set in the container:

- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered
- `PYTHONPATH=/app`: Sets the Python path to include the application directory
- `CUDA_VISIBLE_DEVICES=""`: Disables GPU usage (for containerized environment)

### Resource Limits

The docker-compose.yml file includes resource limits:

- Memory: 8GB maximum, 4GB reserved
- CPU: 4 cores maximum, 2 cores reserved

You can adjust these limits in the `docker-compose.yml` file based on your system capabilities.

## Examples

### Basic Usage
```bash
# Run with default settings (5 years of data, 1-year training window)
./run_docker.sh run
```

### Custom Time Range
```bash
# Run with specific date range
./run_docker.sh args --start-date 2020-01-01 --end-date 2023-12-31
```

### Custom Parameters
```bash
# Run with shorter training windows for faster execution
./run_docker.sh args \
  --train-window-days 180 \
  --test-window-days 30 \
  --step-size-days 15 \
  --price-threshold 0.02
```

### Production Settings
```bash
# Run with production-optimized settings
./run_docker.sh args \
  --train-window-days 365 \
  --test-window-days 60 \
  --step-size-days 30 \
  --price-threshold 0.018 \
  --scaler-type standard \
  --log-level INFO
```

### Vertex AI Deployment
```bash
# Run with GCS paths for Vertex AI
./run_docker.sh args \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output-dir gs://my-project-bucket/models \
  --reports-dir gs://my-project-bucket/reports \
  --plots-dir gs://my-project-bucket/plots \
  --logs-dir gs://my-project-bucket/logs \
  --train-window-days 180 \
  --test-window-days 30 \
  --step-size-days 15 \
  --log-level INFO
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce the batch size in the configuration
   - Increase Docker memory limits
   - Use smaller training windows

2. **Build Failures**
   - Ensure Docker has enough disk space
   - Check internet connection for package downloads
   - Clear Docker cache: `docker system prune -a`

3. **Permission Issues**
   - Ensure the script is executable: `chmod +x run_docker.sh`
   - Check file permissions in mounted directories

4. **GCS Path Issues**
   - Ensure proper authentication for Google Cloud Storage
   - Check bucket permissions and existence
   - Verify gcsfs is properly installed

### Logs and Debugging

```bash
# View real-time logs
docker-compose logs -f

# View logs from a specific time
docker-compose logs --since="2024-01-01T00:00:00"

# Execute commands in the running container
docker-compose exec lootcapital-wfo bash

# Check container resource usage
docker stats lootcapital-walk-forward-optimization

# Run with debug logging
./run_docker.sh args --log-level DEBUG
```

## Performance Optimization

### For Production Use

1. **Use GPU Support** (if available):
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

2. **Optimize Memory Usage**:
   - Reduce batch sizes
   - Use smaller sequence lengths
   - Enable gradient checkpointing

3. **Use Volume Mounts for Large Data**:
   - Mount data directories as volumes
   - Use external storage for large datasets

### For Development

1. **Enable Hot Reloading**:
   ```yaml
   # Mount source code for development
   volumes:
     - .:/app
   ```

2. **Use Development Dependencies**:
   - Install additional debugging tools
   - Enable verbose logging

## Security Considerations

1. **Run as Non-Root User** (recommended for production):
   ```dockerfile
   # Add to Dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

2. **Limit Container Capabilities**:
   ```yaml
   # In docker-compose.yml
   security_opt:
     - no-new-privileges:true
   ```

3. **Use Secrets for Sensitive Data**:
   ```yaml
   # Mount secrets
   secrets:
     - api_key
   ```

## Monitoring and Maintenance

### Health Checks

Add health checks to monitor the application:

```yaml
# In docker-compose.yml
healthcheck:
  test: ["CMD", "python", "-c", "import torch; print('OK')"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Backup Strategy

1. **Regular Backups**:
   - Backup model files: `./models/`
   - Backup reports: `./reports/`
   - Backup configuration files

2. **Version Control**:
   - Keep configuration files in version control
   - Tag Docker images with versions

## Support

For issues related to:

- **Docker Setup**: Check this README and Docker documentation
- **Walk-Forward Optimization**: Check the main project documentation
- **Model Training**: Check the training script documentation
- **Vertex AI Integration**: Check Google Cloud documentation

## License

This Docker setup is part of the LootCapital project and follows the same license terms. 