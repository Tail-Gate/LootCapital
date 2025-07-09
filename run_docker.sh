#!/bin/bash

# Script to build and run the LootCapital Walk-Forward Optimization Docker container

set -e  # Exit on any error

echo "🐳 LootCapital Walk-Forward Optimization Docker Setup"
echo "=================================================="

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker and try again."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to build the Docker image
build_image() {
    echo "🔨 Building Docker image..."
    docker build -t lootcapital-wfo .
    echo "✅ Docker image built successfully"
}

# Function to run the container
run_container() {
    echo "🚀 Starting walk-forward optimization..."
    
    # Create necessary directories if they don't exist
    mkdir -p data models reports plots logs cache lightning_logs profiling_results
    
    # Run the container with default settings
    docker-compose up --build
    
    echo "✅ Container finished running"
}

# Function to run with custom arguments
run_with_args() {
    echo "🚀 Starting walk-forward optimization with custom arguments..."
    echo "Arguments: $@"
    
    # Create necessary directories if they don't exist
    mkdir -p data models reports plots logs cache lightning_logs profiling_results
    
    # Build the image first
    build_image
    
    # Run the container with custom arguments
    docker run -it --rm \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/reports:/app/reports \
        -v $(pwd)/plots:/app/plots \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/cache:/app/cache \
        -v $(pwd)/lightning_logs:/app/lightning_logs \
        -v $(pwd)/profiling_results:/app/profiling_results \
        lootcapital-wfo python scripts/walk_forward_optimization.py "$@"
    
    echo "✅ Container finished running"
}

# Function to run in detached mode
run_detached() {
    echo "🚀 Starting walk-forward optimization in detached mode..."
    
    # Create necessary directories if they don't exist
    mkdir -p data models reports plots logs cache lightning_logs profiling_results
    
    # Run the container in detached mode
    docker-compose up -d --build
    
    echo "✅ Container started in detached mode"
    echo "📋 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
}

# Function to show logs
show_logs() {
    echo "📋 Showing container logs..."
    docker-compose logs -f
}

# Function to stop the container
stop_container() {
    echo "🛑 Stopping container..."
    docker-compose down
    echo "✅ Container stopped"
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up Docker resources..."
    docker-compose down --rmi all --volumes --remove-orphans
    echo "✅ Cleanup completed"
}

# Function to show help with examples
show_help() {
    echo "Usage: $0 [command] [arguments...]"
    echo ""
    echo "Commands:"
    echo "  build     - Build the Docker image only"
    echo "  run       - Build and run the container with default settings"
    echo "  args      - Build and run the container with custom arguments"
    echo "  detached  - Build and run the container in detached mode"
    echo "  logs      - Show container logs"
    echo "  stop      - Stop the container"
    echo "  cleanup   - Stop and remove all containers, images, and volumes"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings"
    echo "  $0 run"
    echo ""
    echo "  # Run with custom time range"
    echo "  $0 args --start-date 2020-01-01 --end-date 2023-12-31"
    echo ""
    echo "  # Run with custom parameters"
    echo "  $0 args --train-window-days 180 --test-window-days 30 --step-size-days 15"
    echo ""
    echo "  # Run with custom output directories"
    echo "  $0 args --output-dir /path/to/models --reports-dir /path/to/reports"
    echo ""
    echo "  # Run with custom output directories"
    echo "  $0 args --output-dir /path/to/models --reports-dir /path/to/reports"
    echo ""
    echo "Available Arguments:"
    echo "  --start-date YYYY-MM-DD     Start date (default: 5 years ago)"
    echo "  --end-date YYYY-MM-DD       End date (default: today)"
    echo "  --output-dir PATH           Models output directory (default: models)"
    echo "  --reports-dir PATH          Reports output directory (default: reports)"
    echo "  --plots-dir PATH            Plots output directory (default: plots)"
    echo "  --logs-dir PATH             Logs output directory (default: logs)"
    echo "  --train-window-days N       Training window in days (default: 365)"
    echo "  --test-window-days N        Testing window in days (default: 60)"
    echo "  --step-size-days N          Step size in days (default: 30)"
    echo "  --price-threshold FLOAT     Price threshold (default: 0.018)"
    echo "  --scaler-type TYPE          Scaler type: minmax or standard (default: minmax)"
    echo "  --log-level LEVEL           Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
    echo "  --quiet                     Suppress console output"
}

# Main script logic
case "${1:-run}" in
    "build")
        check_docker
        build_image
        ;;
    "run")
        check_docker
        build_image
        run_container
        ;;
    "args")
        check_docker
        shift  # Remove 'args' from arguments
        run_with_args "$@"
        ;;
    "detached")
        check_docker
        build_image
        run_detached
        ;;
    "logs")
        show_logs
        ;;
    "stop")
        stop_container
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 