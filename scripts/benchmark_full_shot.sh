#!/bin/bash
#SBATCH --partition=gpu_h100     # GPU partition
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --job-name=full-shot     # job name
#SBATCH --ntasks=1               # number of tasks
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --mem=128G               # memory per node
#SBATCH --time=24:00:00          # walltime
#SBATCH --output=logs/out_%j.txt # standard output
#SBATCH --error=logs/err_%j.txt  # standard error
# Compatibility launcher. Resolve against the repository working directory under Slurm.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then root="$SLURM_SUBMIT_DIR"; else root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; fi
exec bash "$root/models/subspacead/scripts/benchmark_full_shot.sh" "$@"
