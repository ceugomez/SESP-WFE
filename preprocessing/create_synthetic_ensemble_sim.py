# create_synthetic_ensemble_sim.py
# cgf cego6160@colorado.edu 5.26.26
#
# sampler for FastEddy ensemble IC generation.
# Samples (wind speed, wind direction) from N(mu, sigma)
# writes per-member namelists and a master ensemble_config.json.
#
# Sampling axes:
#   spd   [m/s]  : geostrophic wind speed     ~ N(mu_spd,  sigma_spd)
#   dir   [deg]  : meteorological wind dir    ~ N(mu_dir,  sigma_dir)
#
# (U_g, V_g) are derived from (spd, dir)

import json
import os
import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these before running
# ---------------------------------------------------------------------------

# Ensemble parameters
N_MEMBERS    = 20          # number of prior ensemble members
N_TRUTH      = 1           # always 1 truth member drawn independently
SEED         = 7291        # RNG seed for reproducibility

# Prior mean state (idealized westerly, neutral)
MU_SPD_MS    = 10.0        # mean geostrophic wind speed [m/s]
MU_DIR_DEG   = 270.0       # mean met direction [deg]; 270 = westerly
SIGMA_SPD_MS = 1.5         # std dev of speed [m/s]
SIGMA_DIR_DEG = 20.0       # std dev of direction [deg]

# Truth member offset from mean (can be zero for a clean test)
TRUTH_DIR_DEG = 265.0      # fix truth dir, or set to None to sample independently
TRUTH_SPD_MS  = 10.0       # fix truth speed, or set to None to sample independently

# Paths
BASE_RUN_DIR  = "/home/cego6160/workspace/runs_synthetic"
TEMPLATE_FILE = "/home/cego6160/workspace/prediction/preprocessing/FE_template_synthetic.in"
TOPO_FILE     = "/home/cego6160/workspace/runs_synthetic/setup/terrain_idealized.bin"
CONFIG_OUT    = os.path.join(BASE_RUN_DIR, "ensemble_config.json")

# ---------------------------------------------------------------------------
# SAMPLER
# ---------------------------------------------------------------------------

def gaussian_lhs(n: int, mu: np.ndarray, sigma: np.ndarray, seed: int) -> np.ndarray:
    """
    Latin Hypercube sample mapped through Gaussian quantile function.
    Returns (n, d) array where d = len(mu).
    Steps:
      1. LHS gives n stratified uniform samples in [0,1]^d
      2. norm.ppf maps each to standard normal
      3. Scale by sigma and shift by mu
    """
    d = len(mu)
    sampler = LatinHypercube(d=d, seed=seed)
    u = sampler.random(n)           # (n, d) uniform in [0,1]
    z = norm.ppf(u)                 # (n, d) standard normal via quantile fn
    return mu + sigma * z           # (n, d) scaled samples


def dir_spd_to_geostrophic(dir_deg: float, spd_ms: float):
    """
    Convert meteorological wind direction + speed to geostrophic (U_g, V_g).
    Met convention: direction FROM which wind blows.
      U_g = -spd * sin(dir_rad)
      V_g = -spd * cos(dir_rad)
    """
    dir_rad = np.deg2rad(dir_deg)
    U_g = -spd_ms * np.sin(dir_rad)
    V_g = -spd_ms * np.cos(dir_rad)
    return round(U_g, 6), round(V_g, 6)


# ---------------------------------------------------------------------------
# RUN SCRIPT WRITER
# ---------------------------------------------------------------------------

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=FE_member_{member_id}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output={log_dir}/FastEddy_%j.log

source ~/workspace/fasteddy_env.sh

export BASEDIR=/opt/FastEddy-model
export SRCDIR=$BASEDIR/SRC/FEMAIN
export RUNDIR={member_dir}

srun --mpi=pmi2 -n 1 $SRCDIR/FastEddy {nml_path}
"""

def write_run_script(member_dir: str, member_id: str, nml_path: str, log_dir: str) -> str:
    """Write SLURM run script for a member. Returns path."""
    script = SLURM_TEMPLATE.format(
        member_id=member_id,
        log_dir=log_dir,
        member_dir=member_dir,
        nml_path=nml_path,
    )
    script_path = os.path.join(member_dir, f"runFEmember{member_id}.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


# ---------------------------------------------------------------------------
# NAMELIST WRITER
# ---------------------------------------------------------------------------

def write_namelist(template_path: str, member_dir: str, member_id: str,
                   U_g: float, V_g: float) -> str:
    """Fill template placeholders and write member namelist. Returns path."""
    with open(template_path, "r") as f:
        text = f.read()

    out_dir    = os.path.join(member_dir, "output")
    log_dir    = os.path.join(member_dir, "log")
    infile     = os.path.join(member_dir, f"FE_ensemble_{member_id}_IC.nc")
    bndys_base = os.path.join(member_dir, f"FE_ensemble_{member_id}_bndys")
    nml_path   = os.path.join(member_dir, f"FE_ensemble_{member_id}.in")

    replacements = {
        "INPATH_PLACEHOLDER":   "",   # cold start — no IC file needed
        "INFILE_PLACEHOLDER":   "",
        "OUTPATH_PLACEHOLDER":  out_dir,
        "MEMBER_PLACEHOLDER":   member_id,
        "TOPOFILE_PLACEHOLDER": TOPO_FILE,
        "BNDYS_PLACEHOLDER":    bndys_base,
        "UG_PLACEHOLDER":       f"{U_g:.6f}",
        "VG_PLACEHOLDER":       f"{V_g:.6f}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    with open(nml_path, "w") as f:
        f.write(text)
    return nml_path


def make_member_dict(member_id: str, dir_deg: float, spd_ms: float,
                     U_g: float, V_g: float, member_dir: str, nml_path: str) -> dict:
    return {
        "id":          member_id,
        "met_dir_deg": round(dir_deg, 4),
        "spd_ms":      round(spd_ms, 4),
        "U_g_ms":      U_g,
        "V_g_ms":      V_g,
        "namelist":    nml_path,
        "run_script":  os.path.join(member_dir, f"runFEmember{member_id}.sh"),
        "output_dir":  os.path.join(member_dir, "output"),
        "log_dir":     os.path.join(member_dir, "log"),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)

    # --- Sample prior members via Gaussian LHS ---
    mu    = np.array([MU_SPD_MS,    MU_DIR_DEG])
    sigma = np.array([SIGMA_SPD_MS, SIGMA_DIR_DEG])
    samples = gaussian_lhs(N_MEMBERS, mu, sigma, seed=SEED)   # (N, 2): [spd, dir]

    prior_members = []
    for i, (spd, dir_deg) in enumerate(samples):
        member_id  = f"{i:02d}"
        member_dir = os.path.join(BASE_RUN_DIR, f"member_{member_id}")
        U_g, V_g   = dir_spd_to_geostrophic(dir_deg, spd)
        nml_path   = write_namelist(TEMPLATE_FILE, member_dir, member_id, U_g, V_g)
        log_dir    = os.path.join(member_dir, "log")
        write_run_script(member_dir, member_id, nml_path, log_dir)
        prior_members.append(make_member_dict(member_id, dir_deg, spd, U_g, V_g, member_dir, nml_path))
        print(f"  member {member_id}: dir={dir_deg:.2f}°  spd={spd:.2f} m/s  "
              f"U_g={U_g:.3f}  V_g={V_g:.3f}")

    # --- Truth member ---
    t_dir = TRUTH_DIR_DEG if TRUTH_DIR_DEG is not None else float(rng.normal(MU_DIR_DEG, SIGMA_DIR_DEG))
    t_spd = TRUTH_SPD_MS  if TRUTH_SPD_MS  is not None else float(rng.normal(MU_SPD_MS,  SIGMA_SPD_MS))
    t_Ug, t_Vg = dir_spd_to_geostrophic(t_dir, t_spd)
    truth_dir   = os.path.join(BASE_RUN_DIR, "member_truth")
    truth_nml   = write_namelist(TEMPLATE_FILE, truth_dir, "truth", t_Ug, t_Vg)
    truth_log   = os.path.join(truth_dir, "log")
    write_run_script(truth_dir, "truth", truth_nml, truth_log)
    truth_member = make_member_dict("truth", t_dir, t_spd, t_Ug, t_Vg, truth_dir, truth_nml)
    print(f"  truth:       dir={t_dir:.2f}°  spd={t_spd:.2f} m/s  "
          f"U_g={t_Ug:.3f}  V_g={t_Vg:.3f}")

    # --- Write ensemble_config.json ---
    config = {
        "ensemble": {
            "n_members":     N_MEMBERS,
            "mu_dir_deg":    MU_DIR_DEG,
            "sigma_dir_deg": SIGMA_DIR_DEG,
            "mu_spd_ms":     MU_SPD_MS,
            "sigma_spd_ms":  SIGMA_SPD_MS,
            "seed":          SEED,
            "sampler":       "gaussian_lhs",
        },
        "truth_member":  truth_member,
        "prior_members": prior_members,
    }
    os.makedirs(BASE_RUN_DIR, exist_ok=True)
    with open(CONFIG_OUT, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nWrote {N_MEMBERS} prior members + truth → {CONFIG_OUT}")


if __name__ == "__main__":
    main()
