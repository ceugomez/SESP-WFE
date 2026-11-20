# SESP-WFE — ensemble-subspace wind-field estimation

Offline: an LES ensemble sampled over the forcing distribution supplies a POD basis, a prior,
a random-walk process-noise model, and a representativeness-error model. Online: a rank-r
Kalman filter in the POD coefficient space, with no flow model in the loop.

## Layout

```
src/            Julia library (no top-level side effects)
  fasteddy_io.jl   FastEddy output + ensemble config, analysis-domain mask, state layout [u;v;w]
  pod.jl           snapshot POD, LOO fold bases from one Gram matrix, closure curve, cov maps
  statmodel.jl     prior μ0/Σ0 and process noise Q from the basis coefficients
  measurement.jl   synthetic observations, measurement operator H
  reperror.jl      representativeness-error model R_rep (LOO residual + spatial kernel)
  kalman.jl        THE filter (random walk in the subspace), Riccati/observability diagnostics
  kf_adapters.jl   glue to kalman.jl + saving filter histories; including it loads the library
  experiment.jl    shared OSSE setup: spin-up, margin, noise, seed, ranks, sensors, PD guard
  regions.jl       scoring regions (currently only "full"; see below)
  provenance.jl    git/host/time attributes stamped on result files
scripts/        Julia drivers, one per product, grouped by pipeline stage
  paths.jl         filesystem layout (every driver includes it first)
  offline/         bases + companion models, closure and Q diagnostics
  online/          filter experiments and baselines
  products/        NetCDF field products for the plotters
slurm/          sbatch launchers, mirroring scripts/; all source slurm/common.sh
viz/            Python figure scripts; default paths from viz/sesp_paths.py
preprocessing/  ensemble sampling, FastEddy namelist template, terrain generation
test/           julia --startup-file=no test/<file>.jl
```

Outputs are namespaced by ensemble (`SESP_ENSEMBLE`, default `ensemble-highvar`, tag `highvar`;
**export `SESP_ENSEMBLE=ensemble-lowvar` before every lowvar sbatch/julia/python call**):
`Bases-<tag>/`, `results/<tag>/{kf,fields,heldout,repr}/`, `figure/<tag>/`. Logs go to `log/`.

## Pipeline

| Step | Launcher (`sbatch slurm/...`) | Driver (`scripts/...`) | Output |
|---|---|---|---|
| 1. Ensemble | — (`python preprocessing/create_synthetic_ensemble_sim.py`, then each member's run script) | — | `../<ensemble>/member_*/output/` |
| 2. Bases + models | `offline/run_pod_all_bases.sh [folds]` | `offline/run_pod_all_bases.jl` | `Bases-<tag>/truth_<id>_{basis,models}.jld2`, `rep_error_truth_<id>.jld2` |
| 3. Closure | `offline/run_heldout_residual.sh`, `offline/run_repr_error.sh` | `offline/*` | `results/<tag>/heldout/`, `results/<tag>/repr/` |
| 4. Q check | `offline/run_q_diagnostics.sh` | `offline/run_q_diagnostics.jl` | `results/<tag>/kf/q_diag_*` |
| 5. Filter, LOO folds | `online/run_fold_sweep_dense.sh` (dense array), `online/run_fold_sweep.sh` (rake) | `online/run_kf_rank_sweep{,_dense}.jl` + `online/run_theta_baseline{,_dense}.jl` | `results/<tag>/kf/kf_rank_sweep*`, `theta_baseline*` |
| 6. Twin (inverse crime) | `online/run_twin_sweep_dense.sh` | `online/run_kf_twin_dense.jl` | `results/<tag>/kf/kf_twin_dense_*` |
| 7. Moving sensor | `online/run_kf_transect_sweep.sh` | `online/run_kf_transect_sweep.jl` | `results/<tag>/kf/kf_transect_sweep_*` |
| 8. Covariance analysis | `online/run_cov_rep_arms.sh` | `online/analyze_filter_covariance.jl` | `results/<tag>/kf/cov_analysis/` |
| 9. Single run + fields | `online/run_simple_KF_all_bases.sh`, then `products/run_kf_field_*.sh` | `online/run_simple_KF_assimilation.jl`, `products/make_kf_field_*.jl` | `results/<tag>/kf/kf_result_*`, `results/<tag>/fields/` |
| 10. Figures | — | `python viz/paper_figures.py` (+ `viz/plot_*.py`) | `figure/<tag>/` |

Folds: `truth` = the nature run against the all-member basis; `00`–`11` = leave-one-out, member k
as truth against the basis of the other 11. Every online driver seeds `NOISE_SEED` before
sampling, so experiments on the same fold see identical measurement noise (paired comparisons).

## Conventions and known limits

- Shared setup lives in `src/experiment.jl` and `src/regions.jl`; do not redefine sensors,
  regions, spin-up or margin inside a driver. `test/test_experiment_setup.jl` pins them.
- **Scoring is full-domain only.** The wake scoring box (and the wake/ambient closure split
  and time-resolved wake dump) were removed 2026-11-05 to be redesigned for the new
  ensemble; the removed code is in `.removed_code_2026-11-05/`. Add the revised region in
  `src/regions.jl` and take it by name in the drivers.
- Uncentered POD with the plain Euclidean norm over the masked [u;v;w] state (no volume or
  per-component weighting).
- Q is the 6 s increment covariance; strided runs scale it linearly (`OBS_STRIDE`), which is
  exact only for uncorrelated increments (see `src/statmodel.jl`).
- R_rep is indefinite; the dense lattice must stay at ≥ ~400 m spacing (`check_R_pd`).
- Never pass `basis_dir=BASES_DIR` to `build_rep_error` for a fold basis (see NESTING in
  `src/reperror.jl`).
