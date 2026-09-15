#!/bin/bash
#SBATCH --job-name="odd_iw_overlap"
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_koch
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --account=invest
#SBATCH --output=/storage/homefs/tf24s166/IW_dis2/disagree_discrep_IW/logs/odd_iw_overlap-%j.log
#SBATCH --error=/storage/homefs/tf24s166/IW_dis2/disagree_discrep_IW/logs/odd_iw_overlap-%j.log

set -euo pipefail

# --- Load environment ---
module load Anaconda3
module load CUDA/12.2.0
module load Workspace_Home
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-performance_metrics}"

# --- Paths ---
PROJECT_ROOT="/storage/homefs/tf24s166/IW_dis2/disagree_discrep_IW"
FEATS_DIR="/storage/homefs/tf24s166/IW_dis2/disagree_discrep_IW/data/features"
RESULTS_DIR="/storage/homefs/tf24s166/IW_dis2/disagree_discrep_IW/results/iw_hybrids"


# Slurm usually exposes the allocated GPU as cuda:0 inside the job.
GPU_ID="${GPU_ID:-0}"

mkdir -p "$RESULTS_DIR" "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

echo "Starting ODD + IW overlap job at $(date)"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Features dir: $FEATS_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Conda env: ${CONDA_ENV:-performance_metrics}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Extra arguments: $*"

srun python -m src.eval.iw_hybrid \
  --feats_dir "$FEATS_DIR" \
  --results_dir "$RESULTS_DIR" \
  --include_dis2_reference

echo "Job finished at $(date)"

