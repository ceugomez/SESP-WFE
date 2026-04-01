"""
generate_ensemble.py
--------------------
Generates N FastEddy ensemble members for a wind-direction ensemble over complex
terrain.  For each member this script creates:

  runs/member_XX/
    FE_ensemble_XX.in      namelist with per-member U_g / V_g
    log/                   SLURM log directory
    output/                FastEddy output directory
    runFEmemberXX.sh       SLURM batch script

It also writes runs/ensemble_fp/ensemble_config.json summarising every member's
parameter set for post-processing and provenance.

Ensemble design
---------------
The ensemble perturbs only the geostrophic wind DIRECTION while keeping wind
SPEED constant.  This captures the main source of terrain-induced forecast
divergence from an NWP ensemble.

  Wind direction  ~ N(mu_dir, sigma_dir^2)   [meteorological convention, degrees]
  Wind speed      = spd_ms                    [constant across members]

Meteorological convention: direction = angle FROM which wind is blowing.
  270 deg = westerly (flow moving eastward), U_g > 0, V_g = 0

Geostrophic component formula:
  U_g = -spd * sin(met_dir_rad)   (positive = eastward)
  V_g = -spd * cos(met_dir_rad)   (positive = northward)

  Check: met_dir=270 deg  =>  U_g = -10*sin(270) = +10  V_g = -10*cos(270) = 0  (westerly) OK
         met_dir=  0 deg  =>  U_g = 0              V_g = -10*cos(0)  = -10  (northerly) OK

Usage
-----
  python generate_ensemble.py [--n_members N] [--mu_dir DEG] [--sigma_dir DEG]
                               [--spd M_S] [--seed SEED]
"""

import argparse
import json
import math
import os
import shutil
import stat

import numpy as np

from generate_boundary_planes import generate_boundary_planes, generate_ic

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_MEMBERS = 10
DEFAULT_MU_DIR    = 270.0   # degrees, meteorological; westerly flow
DEFAULT_SIGMA_DIR =  20.0   # degrees; realistic NWP ensemble spread (~24 h horizon, complex terrain)
DEFAULT_SPD       =  10.0   # m/s; matches base-state NBL case
DEFAULT_SEED      =  42

# SLURM / FastEddy binary settings — edit here if the cluster config changes
FE_BINARY     = "/opt/FastEddy-model/SRC/FEMAIN/FastEddy"
SLURM_NODES   = 1
SLURM_NTASKS  = 1
SLURM_CPUS    = 2
SLURM_GPU     = "gpu:1"
SLURM_MEM     = "64G"
SLURM_TIME    = "1-00:00:00"
SLURM_ENV_SRC = "~/workspace/fasteddy_env.sh"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--n_members",  type=int,   default=DEFAULT_N_MEMBERS)
parser.add_argument("--mu_dir",     type=float, default=DEFAULT_MU_DIR,
                    help="Mean wind direction (met. degrees, 270=westerly)")
parser.add_argument("--sigma_dir",  type=float, default=DEFAULT_SIGMA_DIR,
                    help="Std-dev of wind direction (degrees)")
parser.add_argument("--spd",        type=float, default=DEFAULT_SPD,
                    help="Geostrophic wind speed, constant across ensemble (m/s)")
parser.add_argument("--seed",       type=int,   default=DEFAULT_SEED,
                    help="NumPy random seed for reproducibility")
parser.add_argument("--truth_dir",  type=float, default=None,
                    help="Wind direction (met. degrees) for the held-out truth member. "
                         "If omitted (default), the truth direction is drawn as an "
                         "additional independent sample from the same N(mu_dir, sigma_dir) "
                         "distribution, keeping it out-of-sample. Supply an explicit value "
                         "only to pin the truth to a specific direction.")
parser.add_argument("--runs_dir",   type=str,   default=None,
                    help="Root directory for member output (default: workspace/runs/). "
                         "Set to a new path (e.g. ../runs_lad) to keep old runs intact.")
parser.add_argument("--ic_template",  type=str,   default=None,
                    help="Path to an existing FastEddy output .nc file to use as the "
                         "grid-coordinate template when generating per-member IC files "
                         "(needed for hydroBCs=1 boundary-plane loading). "
                         "Example: runs_lad/member_truth/output/FE_Membertruth.0")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir    = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)       # …/workspace/
runs_dir      = (os.path.abspath(args.runs_dir)
                 if args.runs_dir is not None
                 else os.path.join(workspace_dir, "runs"))
topo_file     = os.path.join(workspace_dir, "runs", "ensemble_fp", "setup", "terrain.bin")
template_file = os.path.join(workspace_dir, "runs", "ensemble_fp", "ensemble_mean_member_1gpu.in")

for path, label in [(template_file, "Template"), (topo_file, "Terrain file")]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")

with open(template_file) as f:
    template_text = f.read()

# Verify all placeholders are present in the template
required_placeholders = [
    "OUTPATH_PLACEHOLDER", "MEMBER_PLACEHOLDER",
    "TOPOFILE_PLACEHOLDER", "UG_PLACEHOLDER", "VG_PLACEHOLDER",
    "BNDYS_PLACEHOLDER", "INPATH_PLACEHOLDER", "INFILE_PLACEHOLDER",
]
missing = [p for p in required_placeholders if p not in template_text]
if missing:
    raise ValueError(
        f"Template is missing placeholder(s): {missing}\n"
        f"Check {template_file}"
    )

# ---------------------------------------------------------------------------
# Sample wind directions
# ---------------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
if args.truth_dir is not None:
    # Pinned truth direction: draw only the prior members.
    dirs      = rng.normal(args.mu_dir, args.sigma_dir, args.n_members)
    truth_dir = args.truth_dir
else:
    # Draw n_members + 1 directions in one call so the truth member is a
    # genuine independent sample from the same distribution (not the mean).
    all_dirs  = rng.normal(args.mu_dir, args.sigma_dir, args.n_members + 1)
    dirs      = all_dirs[:args.n_members]
    truth_dir = float(all_dirs[-1])

print(f"Ensemble: {args.n_members} prior members  +  1 held-out truth member")
print(f"  Wind direction  : N(mu={args.mu_dir:.1f} deg, sigma={args.sigma_dir:.1f} deg)")
print(f"  Wind speed      : {args.spd:.1f} m/s")
print(f"  Random seed     : {args.seed}")
print(f"  Prior dirs      : {' '.join(f'{d:.1f} deg' for d in dirs)}")
print(f"  Truth dir       : {truth_dir:.1f} deg"
      + (" (pinned)" if args.truth_dir is not None else " (sampled)"))
print()

# ---------------------------------------------------------------------------
# Per-member file generation
# ---------------------------------------------------------------------------
config_members = []

for i, met_dir in enumerate(dirs):
    tag = f"{i:02d}"
    rad = math.radians(met_dir)
    ug  = -args.spd * math.sin(rad)
    vg  = -args.spd * math.cos(rad)

    member_dir = os.path.join(runs_dir, f"member_{tag}")
    output_dir = os.path.join(member_dir, "output")
    log_dir    = os.path.join(member_dir, "log")
    icbc_dir   = os.path.join(member_dir, "ICBC")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)

    # -- Boundary planes (LAD inflow) -------------------------------------
    bndys_base = generate_boundary_planes(icbc_dir, "FE_Bndys", ug, vg)

    # -- Initial condition file (required for hydroBCs=1 BC loading) ------
    ic_base = f"FE_Member{tag}"
    if args.ic_template is not None:
        generate_ic(args.ic_template, output_dir, f"{ic_base}.0", ug, vg)
        inpath_val = output_dir + "/"
        infile_val = f"{ic_base}.0"
    else:
        inpath_val = ""
        infile_val = ""

    # -- Namelist ----------------------------------------------------------
    text = (template_text
            .replace("OUTPATH_PLACEHOLDER",  output_dir + "/")
            .replace("MEMBER_PLACEHOLDER",   tag)
            .replace("TOPOFILE_PLACEHOLDER",  topo_file)
            .replace("UG_PLACEHOLDER",        f"{ug:.6f}")
            .replace("VG_PLACEHOLDER",        f"{vg:.6f}")
            .replace("BNDYS_PLACEHOLDER",     bndys_base)
            .replace("INPATH_PLACEHOLDER",    inpath_val)
            .replace("INFILE_PLACEHOLDER",    infile_val))

    in_file = os.path.join(member_dir, f"FE_ensemble_{tag}.in")
    with open(in_file, "w") as f:
        f.write(text)

    # -- Run script --------------------------------------------------------
    run_script = os.path.join(member_dir, f"runFEmember{tag}.sh")
    slurm_log  = os.path.join(log_dir, "FastEddy_%j.log")
    script_text = f"""\
#!/bin/bash
#SBATCH --job-name=FE_member_{tag}
#SBATCH --nodes={SLURM_NODES}
#SBATCH --ntasks={SLURM_NTASKS}
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --gres={SLURM_GPU}
#SBATCH --mem={SLURM_MEM}
#SBATCH --time={SLURM_TIME}
#SBATCH --output={slurm_log}

source {SLURM_ENV_SRC}

export BASEDIR=/opt/FastEddy-model
export SRCDIR=$BASEDIR/SRC/FEMAIN
export RUNDIR={member_dir}

srun --mpi=pmi2 -n 1 $SRCDIR/FastEddy $RUNDIR/FE_ensemble_{tag}.in
"""
    with open(run_script, "w") as f:
        f.write(script_text)
    os.chmod(run_script, os.stat(run_script).st_mode | stat.S_IXUSR | stat.S_IXGRP)

    config_members.append({
        "id":          tag,
        "met_dir_deg": round(met_dir, 4),
        "U_g_ms":      round(ug, 6),
        "V_g_ms":      round(vg, 6),
        "namelist":    in_file,
        "run_script":  run_script,
        "output_dir":  output_dir,
        "log_dir":     log_dir,
    })

    print(f"  member_{tag} [prior]:  dir={met_dir:7.3f} deg  "
          f"U_g={ug:+8.4f} m/s  V_g={vg:+8.4f} m/s")

# ---------------------------------------------------------------------------
# Write ensemble config file
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Truth member (held-out, not in the POD basis)
# ---------------------------------------------------------------------------
truth_rad = math.radians(truth_dir)
truth_ug  = -args.spd * math.sin(truth_rad)
truth_vg  = -args.spd * math.cos(truth_rad)

truth_dir_path = os.path.join(runs_dir, "member_truth")
truth_output   = os.path.join(truth_dir_path, "output")
truth_log      = os.path.join(truth_dir_path, "log")
truth_icbc_dir = os.path.join(truth_dir_path, "ICBC")

os.makedirs(truth_output, exist_ok=True)
os.makedirs(truth_log,    exist_ok=True)

truth_bndys_base = generate_boundary_planes(truth_icbc_dir, "FE_Bndys",
                                             truth_ug, truth_vg)

if args.ic_template is not None:
    generate_ic(args.ic_template, truth_output, "FE_Membertruth.0",
                truth_ug, truth_vg)
    truth_inpath_val = truth_output + "/"
    truth_infile_val = "FE_Membertruth.0"
else:
    truth_inpath_val = ""
    truth_infile_val = ""
truth_text = (template_text
              .replace("OUTPATH_PLACEHOLDER",  truth_output + "/")
              .replace("MEMBER_PLACEHOLDER",   "truth")
              .replace("TOPOFILE_PLACEHOLDER",  topo_file)
              .replace("UG_PLACEHOLDER",        f"{truth_ug:.6f}")
              .replace("VG_PLACEHOLDER",        f"{truth_vg:.6f}")
              .replace("BNDYS_PLACEHOLDER",     truth_bndys_base)
              .replace("INPATH_PLACEHOLDER",    truth_inpath_val)
              .replace("INFILE_PLACEHOLDER",    truth_infile_val))

truth_in_file = os.path.join(truth_dir_path, "FE_ensemble_truth.in")
with open(truth_in_file, "w") as f:
    f.write(truth_text)

truth_run_script = os.path.join(truth_dir_path, "runFEmembertruth.sh")
truth_slurm_log  = os.path.join(truth_log, "FastEddy_%j.log")
truth_script_text = f"""\
#!/bin/bash
#SBATCH --job-name=FE_member_truth
#SBATCH --nodes={SLURM_NODES}
#SBATCH --ntasks={SLURM_NTASKS}
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --gres={SLURM_GPU}
#SBATCH --mem={SLURM_MEM}
#SBATCH --time={SLURM_TIME}
#SBATCH --output={truth_slurm_log}

source {SLURM_ENV_SRC}

export BASEDIR=/opt/FastEddy-model
export SRCDIR=$BASEDIR/SRC/FEMAIN
export RUNDIR={truth_dir_path}

srun --mpi=pmi2 -n 1 $SRCDIR/FastEddy $RUNDIR/FE_ensemble_truth.in
"""
with open(truth_run_script, "w") as f:
    f.write(truth_script_text)
os.chmod(truth_run_script, os.stat(truth_run_script).st_mode | stat.S_IXUSR | stat.S_IXGRP)

truth_member_config = {
    "id":          "truth",
    "met_dir_deg": round(truth_dir, 4),
    "U_g_ms":      round(truth_ug, 6),
    "V_g_ms":      round(truth_vg, 6),
    "namelist":    truth_in_file,
    "run_script":  truth_run_script,
    "output_dir":  truth_output,
    "log_dir":     truth_log,
}

print()
print(f"  member_truth [TRUTH]:  dir={truth_dir:7.3f} deg  "
      f"U_g={truth_ug:+8.4f} m/s  V_g={truth_vg:+8.4f} m/s")

config = {
    "ensemble": {
        "n_members":    args.n_members,
        "mu_dir_deg":   args.mu_dir,
        "sigma_dir_deg": args.sigma_dir,
        "spd_ms":       args.spd,
        "seed":         args.seed,
        "truth_dir_deg": truth_dir,
        "template":     template_file,
        "topo_file":    topo_file,
    },
    "truth_member":  truth_member_config,
    "prior_members": config_members,
}

os.makedirs(runs_dir, exist_ok=True)
config_file = os.path.join(runs_dir, "ensemble_config.json")
with open(config_file, "w") as f:
    json.dump(config, f, indent=2)

print()
print(f"Config written : {config_file}")
print()
print("Truth / prior split:")
print(f"  Truth  (held-out, sample measurements from) : member_truth  dir={truth_dir:.1f} deg")
print(f"  Prior  (POD basis + GMF prior)              : "
      + ", ".join(f"member_{m['id']}" for m in config_members))
print()
print("To submit all members + truth:")
print()
print(f"  for i in $(seq -f \"%02g\" 0 {args.n_members - 1}); do")
print(f"    sbatch {runs_dir}/member_${{i}}/runFEmember${{i}}.sh")
print( "  done")
print(f"  sbatch {truth_run_script}")
