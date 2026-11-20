#!/bin/bash
#SBATCH --job-name=heldout_residual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768GB   
#SBATCH --time=12:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/heldout_%j.log

# Held-out residual-energy curve (leave-one-member-out cross-validation).
# Rebuilds the POD basis M times (one per held-out member) and projects the
# held-out member onto each — the family-closure diagnostic. Heavier than a
# single POD build, hence the longer walltime.

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

precompile_julia

run_jl offline/run_heldout_residual.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
