#!/bin/bash -l
#SBATCH --job-name=ackley
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
set -euo pipefail
hostname; date
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTHONUNBUFFERED=1
source .venv/bin/activate
which python && python --version
python experiments/ackley.py "$@"
