"""sesp_paths.py — output layout for the viz scripts; mirrors scripts/paths.jl.

Every viz script imports its default locations from here, so Python and Julia agree on where
results live. Select the ensemble with the SESP_ENSEMBLE environment variable (default
ensemble-highvar); outputs are namespaced by its tag (ensemble-highvar -> highvar).

    from sesp_paths import KF_DIR, FIELDS_DIR, figure_dir
"""
import os
import pathlib

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]      # SESP-WFE/
REPO_ROOT = PKG_ROOT.parent                                  # ensemble_runs_paper/

ENSEMBLE_NAME = os.environ.get("SESP_ENSEMBLE", "ensemble-highvar")
ENSEMBLE_TAG = ENSEMBLE_NAME.removeprefix("ensemble-")
ENSEMBLE_DIR = REPO_ROOT / ENSEMBLE_NAME
CONFIG_PATH = ENSEMBLE_DIR / "ensemble_config.json"

BASES_DIR = PKG_ROOT / f"Bases-{ENSEMBLE_TAG}"
RESULTS_DIR = PKG_ROOT / "results" / ENSEMBLE_TAG
KF_DIR = RESULTS_DIR / "kf"
COV_DIR = KF_DIR / "cov_analysis"
FIELDS_DIR = RESULTS_DIR / "fields"
HELDOUT_DIR = RESULTS_DIR / "heldout"
REPR_DIR = RESULTS_DIR / "repr"
FIGURE_DIR = PKG_ROOT / "figure" / ENSEMBLE_TAG


def figure_dir(sub: str = "") -> pathlib.Path:
    """FIGURE_DIR (or a subfolder of it), created if missing."""
    d = FIGURE_DIR / sub if sub else FIGURE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d
