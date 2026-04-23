#!/bin/bash
# Submit Monte Carlo sweep experiments as a SLURM dependency chain.
# Each job starts only after the previous one completes.
#
# Usage:
#   bash prediction/scripts/submit_sweep.sh
#
# After all jobs complete, analyze results with:
#   julia prediction/scripts/run_montecarlo.jl --analyze-all prediction/output/sweep/

set -euo pipefail

SCRIPT="/home/cego6160/workspace/prediction/scripts/run_montecarlo.jl"
OUTDIR="/home/cego6160/workspace/prediction/output/sweep"
LOGDIR="/home/cego6160/workspace/prediction/output/sweep/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

N_TRIALS=100
BETA=0.1
Q_DEFAULT=0.01

PREV_JID=""

submit_job() {
    local name="$1"
    local args="$2"

    local dep_flag=""
    if [[ -n "$PREV_JID" ]]; then
        dep_flag="--dependency=afterany:${PREV_JID}"
    fi

    PREV_JID=$(sbatch --parsable $dep_flag <<EOF
#!/bin/bash
#SBATCH --job-name=mc_${name}
#SBATCH --output=${LOGDIR}/${name}_%j.out
#SBATCH --error=${LOGDIR}/${name}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "Job: ${name}"
echo "Args: ${args}"
echo "Start: \$(date)"

julia ${SCRIPT} --n_trials ${N_TRIALS} --beta ${BETA} ${args}

echo "End: \$(date)"
EOF
    )
    echo "Submitted: ${name}  (job ${PREV_JID})"
}

# ══════════════════════════════════════════════════════════════════════════════
# Sweep 1: sensor count, random sampling
# ══════════════════════════════════════════════════════════════════════════════
for n in 1 2 3 5 10 15 25; do
    name="random_n${n}_q${Q_DEFAULT}"
    submit_job "$name" \
        "--n_sensors $n --sensor_mode random --q_scale $Q_DEFAULT --outpath ${OUTDIR}/${name}.jld2"
done

# ══════════════════════════════════════════════════════════════════════════════
# Sweep 2: sensor count, flight-track sampling
# ══════════════════════════════════════════════════════════════════════════════
for n in 1 2 3 5 10 15 25; do
    name="flighttrack_n${n}_q${Q_DEFAULT}"
    submit_job "$name" \
        "--n_sensors $n --sensor_mode flighttrack --q_scale $Q_DEFAULT --outpath ${OUTDIR}/${name}.jld2"
done

# ══════════════════════════════════════════════════════════════════════════════
# Sweep 3: Q sweep, 2 sensors, random
# ══════════════════════════════════════════════════════════════════════════════
for q in 0.001 0.005 0.01 0.025 0.05 0.1 0.25; do
    name="random_n2_q${q}"
    submit_job "$name" \
        "--n_sensors 2 --sensor_mode random --q_scale $q --outpath ${OUTDIR}/${name}.jld2"
done

echo ""
echo "All jobs submitted as a chain. Monitor with: squeue -u \$USER"
echo "After completion, analyze with:"
echo "  julia ${SCRIPT} --analyze-all ${OUTDIR}/"
