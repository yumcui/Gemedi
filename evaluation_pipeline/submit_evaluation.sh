#!/bin/bash
#SBATCH -J eval-notes
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o slurm-eval-%j.out

echo "============================================================"
echo "Medical Notes Evaluation Job"
echo "============================================================"
echo "  Time: $(date)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo "  Args: $@"
echo "============================================================"
echo ""

# Change to evaluation pipeline directory FIRST
# Use SLURM_SUBMIT_DIR which points to where sbatch was called
if [ ! -z "$SLURM_SUBMIT_DIR" ]; then
    # Running in SLURM
    echo "SLURM submit directory: $SLURM_SUBMIT_DIR"
    cd "$SLURM_SUBMIT_DIR"
else
    # Running locally, use script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "Script directory: $SCRIPT_DIR"
    cd "$SCRIPT_DIR"
fi
echo "Working directory: $(pwd)"
echo ""

# Check for Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY is not set"
    echo "Please set it before submitting:"
    echo "  export GEMINI_API_KEY='your-api-key'"
    echo "  sbatch submit_evaluation.sh <args>"
    exit 1
fi

# Set Ollama environment variable for CCV-hosted models
export OLLAMA_MODELS=/oscar/data/shared/ollama_models

# Load Ollama module
echo "Loading Ollama module..."
module load ollama
if [ $? -ne 0 ]; then
    echo "Warning: Failed to load Ollama module, clinical consistency will be skipped"
else
    echo "✓ Ollama module loaded"
    
    # Start Ollama server in background
    echo "Starting Ollama server..."
    export OLLAMA_HOST=127.0.0.1:11434
    ollama serve > ollama_server_$SLURM_JOB_ID.log 2>&1 &
    OLLAMA_PID=$!
    echo "  Ollama server PID: $OLLAMA_PID"
    
    # Wait for server to start
    echo "  Waiting for Ollama server to start..."
    sleep 10
    
    # Test if server is running
    if ollama list > /dev/null 2>&1; then
        echo "✓ Ollama server is running"
    else
        echo "⚠ Ollama server failed to start, clinical consistency will be skipped"
        kill $OLLAMA_PID 2>/dev/null || true
    fi
fi
echo ""

# Activate Python environment
source /users/$USER/pytorch.venv/bin/activate

# Run evaluation with all passed arguments
echo "Running: python3 evaluate_medical_notes.py $@"
echo ""

python3 evaluate_medical_notes.py "$@"

EXIT_CODE=$?

# Cleanup: stop Ollama server if running
if [ ! -z "$OLLAMA_PID" ]; then
    echo ""
    echo "Stopping Ollama server (PID: $OLLAMA_PID)..."
    kill $OLLAMA_PID 2>/dev/null || true
    sleep 2
fi

echo ""
echo "============================================================"
echo "Evaluation completed with exit code: $EXIT_CODE"
echo "  Time: $(date)"
echo "============================================================"

exit $EXIT_CODE

