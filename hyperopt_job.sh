#!/bin/bash
#
# SLURM job script for STGNN Hyperparameter Optimization (Memory Optimized)
#
#SBATCH --job-name=train_stgnn_improved    # Job name that appears in squeue
#SBATCH --mail-type=END,FAIL             # Email notifications for END and FAIL
#SBATCH --mail-user=eoobon24@stlawu.edu  # Your St. Lawrence email address
#SBATCH --mem=120gb                      # Maximum memory (node has 128GB)
#SBATCH --time=48:00:00                  #  time for thorough optimization
#SBATCH --output=hyperopt_log_%j.txt     # Standard output file, %j is replaced by job number
#SBATCH --cpus-per-task=32               # Use ALL 32 CPUs on the node
#SBATCH --gres=gpu:2                     # 2 GPUs requested
#SBATCH -n 1                             # Request 1 nodes for the job
#SBATCH --exclusive                      # Request exclusive access to the node

echo "STARTED JOB AT: $(date)"

# --- Step 1: Clean the environment to prevent conflicts ---
module purge

# Unset ALL common Python and Conda related environment variables
unset PYTHONHOME
unset PYTHONPATH
unset CONDA_EXE
unset _CE_M
unset _CE_CONDA
unset CONDA_PYTHON_EXE

# --- Step 2: Load essential system modules in the correct order ---
module load cuda

# --- Step 3: Activate your personal Conda environment ---
source ~/.bashrc
conda activate stgnn_train

# Check if conda activation was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Conda environment 'stgnn_train' activation failed! Please check your Conda setup."
    exit 1
fi

echo "--- Environment Check within SLURM job (after Conda activation) ---"
echo "Current directory: $(pwd)"
echo "Which Python: $(which python)"
echo "Conda environment list (should show 'stgnn_train' as active):"
conda env list
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH: $PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# --- Step 4: Verify PyTorch CUDA availability *within this job's environment* ---
echo -n "PyTorch CUDA available: "
python -c "import torch; print(torch.cuda.is_available())"

echo -n "PyTorch CUDA device count: "
python -c "import torch; print(torch.cuda.device_count())"

echo -n "PyTorch CUDA version: "
python -c "import torch; print(torch.version.cuda)"

echo -n "CUDA_HOME: "
python -c "import os; print(os.environ.get('CUDA_HOME') or 'N/A')"

echo -n "NVCC_PATH: "
python -c "import os; print(os.environ.get('NVCC_PATH') or 'N/A')"

# --- Step 5: Prepare your Python project environment ---
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary output directories if they don't exist
mkdir -p logs
mkdir -p models
mkdir -p reports
mkdir -p plots
mkdir -p config
echo "Ensured output directories exist."

# --- Step 6: Set memory optimization environment variables ---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32


# --- Step 7.5: Determine resume point for walk-forward optimization ---
RESUME_ARGS="--start-date 2024-05-01"

# --- Step 8: Run your Python script with memory monitoring ---
echo "Starting memory-optimized walk_forward_optimization script..."
echo "Memory optimization settings:"
echo "  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "  CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "  NUMEXPR_NUM_THREADS: $NUMEXPR_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
echo ""
echo "Starting memory-optimized walk_forward_optimization script..."
echo "Resource utilization strategy:"
echo "  - Using GPU training for maximum speed"
echo "  - Using all 32 cores for parallel processing"
echo "  - Using 2 GPUs"
echo "  - Using 120GB memory (maximum available)"

python scripts/walk_forward_optimization.py $RESUME_ARGS

echo "JOB FINISHED AT: $(date)" 