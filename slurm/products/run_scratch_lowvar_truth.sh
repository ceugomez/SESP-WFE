#!/bin/bash
#SBATCH --job-name=scratch_lv_truth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/scratch_lv_truth_%j.log

# Scratchpad: lowvar nature run as truth, 3 nearest-direction highvar members as the basis.
# Mem: 3-member cache (~118G) + truth_U (~39G) + chunked transients.
set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh
precompile_julia
run_jl products/scratch_lowvar_truth_highvar_basis.jl
echo "Job $SLURM_JOB_ID finished at $(date)"
