#!/bin/bash
#SBATCH --job-name=pod_ensemble_allsteps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=612G
#SBATCH --time=6:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/pod_%j.log

# POD basis construction for FastEddy wind-direction ensemble.
# Builds a reduced-order basis via the method of snapshots (Sirovich 1987)
# and projects all ensemble members + truth onto it.


set -euo pipefail

mkdir -p /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK   Mem: $(grep MemTotal /proc/meminfo | awk '{print $2/1048576 " GB"}')"

# precompile packages with a single thread to avoid multi-threaded precompile memory spike
julia --startup-file=no --threads=1 -e \
  'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'

julia --startup-file=no \
      --threads=$SLURM_CPUS_PER_TASK \
      /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/scripts/run_pod.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
