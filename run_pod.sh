#!/bin/bash
#SBATCH --job-name=pod_ensemble_allsteps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=/home/cego6160/workspace/prediction/log/pod_%j.log

# POD basis construction for FastEddy wind-direction ensemble.
# Builds a reduced-order basis via the method of snapshots (Sirovich 1987)
# and projects all ensemble members + truth onto it.
#
# Mode and L are set in run_pod.jl.
# :timemean  — one time-mean snapshot/member → N=10, ~2 GB peak, a few minutes
# :all       — all post-spinup snapshots    → N≈530, ~95 GB peak, ~30 min
#
# Adjust --mem and --time for :all mode:
#   #SBATCH --mem=256G
#   #SBATCH --time=4:00:00

set -euo pipefail

mkdir -p /home/cego6160/workspace/prediction/log

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK   Mem: $(grep MemTotal /proc/meminfo | awk '{print $2/1048576 " GB"}')"

julia --startup-file=no \
      --threads=$SLURM_CPUS_PER_TASK \
      /home/cego6160/workspace/prediction/run_pod.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
