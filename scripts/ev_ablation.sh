#!/bin/bash
#SBATCH --partition=gpu_h100          # GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --job-name=ev-ablation-448    # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --mem=128G                    # Memory per node
#SBATCH --time=24:00:00               # Walltime
#SBATCH --output=logs/out_%j.txt      # Standard output
#SBATCH --error=logs/err_%j.txt       # Standard error
# Compatibility launcher. Resolve against the repository working directory under Slurm.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then root="$SLURM_SUBMIT_DIR"; else root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; fi
exec bash "$root/models/subspacead/scripts/ev_ablation.sh" "$@"
