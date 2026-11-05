# Quantum Oracle Value Readout

Reproducibility code and numerical data accompanying **Quantum Oracle Value
Readout: Coherence, Complementarity, and Fragility**, by Milad Ghadimi,
Hesam Soltanpanahi, and Vahid Salari.

This repository contains the numerical study used in the revised H1 manuscript.
The reference quadratic coefficient for the `d=8`, `Phi=0.50` scan is
`c = 7.465353134198574`. The original computational scripts, reference matrices
and unrounded results are included. The manuscript itself is distributed
separately.

## Install

Use Python 3.12. The saved run used Python 3.12.14; its package versions are
pinned in `requirements.txt` and also recorded in `numerics/results.json`.
No LaTeX installation is needed to run the Python code or generate the plots.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Quick check using saved data

```bash
python reproduce_all.py --from-saved
```

This recomputes objectives and feasibility checks from the stored matrices,
checks the mesh curvature and query-channel kernel identities, and regenerates
the numerical tables and three plots. It does not rerun the SDP solvers or the
Monte Carlo simulations.

## Full reproduction

```bash
python reproduce_all.py
```

This reruns discrimination and robustness SDPs, mesh simulations and the
mode-dependent-loss study, then validates the new output and regenerates all
numerical tables and plots. It replaces the generated files in this working
copy. Keep a separate copy or use Git if you want to preserve the bundled
reference outputs. The runner sets BLAS thread counts to one in its child
processes. Runtime depends on the machine and numerical libraries.

The individual commands are:

```bash
OPENBLAS_NUM_THREADS=1 python numerics/reproduce.py
python numerics/validate.py
python numerics/make_tables.py
python numerics/plot_saved_scan.py
python figures_source/build_analytic_figures.py
```

## Files and manuscript outputs

| File | Role |
| --- | --- |
| `numerics/reproduce.py` | Full numerical pipeline: SDPs, simulations, loss study and Fig. 4 |
| `numerics/mesh_core.py` | Explicit triangular mesh, analytic tangent generators, correlations and loss coefficients |
| `numerics/validate.py` | Independent checks of saved matrices, mesh curvature and query-channel identities |
| `numerics/make_tables.py` | Numerical tables and LaTeX macros generated from unrounded results |
| `numerics/plot_saved_scan.py` | Fig. 4 from saved simulation statistics |
| `figures_source/build_analytic_figures.py` | Analytic capability map and tradeoff plot |
| `numerics/results.json` | Unrounded results, seed, sampling settings, package versions and solver diagnostics |
| `numerics/certificates.npz` | Input states, raw and repaired discrimination solutions, and robustness solutions |
| `numerics/mesh_data.npz` | Analyzer phases, target matrices, mesh settings, tangent generators and correlations |
| `numerics/validation.json` | Saved-matrix validation results |
| `numerics/table_mixed.tex` | Table I: mixed-state discrimination and robustness |
| `numerics/table_profiles.tex` | Table II: variance profiles |
| `numerics/table_mesh.tex` | Table III: direct, full, profile, off-diagonal scalar and isotropic comparisons |
| `numerics/table_boundary.tex` | Table IV: boundary scan |
| `numerics/table_diagnostics.tex` | Solver feasibility and objective diagnostics |
| `numerics/numeric_values.tex` | Numerical macros used by the manuscript |
| `oracle_capability_map.pdf` | Fig. 2: analytic phase-value region |
| `sigma_scan_expectation.pdf` | Fig. 4: coherent-error scale scan |
| `tradeoff.pdf` | Appendix tradeoff figure |

The circuit/photonic illustrations (Figs. 1 and 3) are explanatory manuscript
assets and are not outputs of this numerical study. They are not included here.
The `.tex` table fragments use REVTeX commands and manuscript cross-references;
they are inputs to the manuscript rather than standalone documents.

## Numerical conventions

- Base random seed: `20261102`; each experiment uses the explicit offsets in
  `reproduce.py`. The three generic mixed states use `seed + 400`.
- Table III uses 3,000 independent Gaussian vectors and their sign reversals.
  Each antithetic pair average is one independent sample. All state rows at a
  given dimension share those vectors.
- Fig. 4 uses a separate set of 25,000 Gaussian vectors and their sign reversals,
  shared across noise strengths.
- The same nontrivial analyzer at each dimension is used throughout the mesh
  comparisons. The identity configuration and parameter subsets are explicitly
  separate settings.
- The full, profile and scalar coefficients are analytic second-moment
  predictions. Direct finite-error statistics are Monte Carlo estimates.
- The full correlated quadratic coefficient is the reference model. On the
  upper phase-value boundary the profile and off-diagonal scalar coefficients
  coincide; this does not imply equality with the full coefficient.
- Clarabel independently solves the discrimination primal and dual and the
  robustness primal and dual. SCS cross-checks the three generic discrimination
  primals. Requested tolerances, solver status and measured feasibility
  residuals are distinguished in the stored results.
- Repaired primal/dual pairs are floating-point certificates, not
  interval-arithmetic proofs. Platform-dependent numerical differences and
  `optimal_inaccurate` statuses are possible; inspect measured residuals.

## Verification of this archive

On 2026-11-05, the pinned dependencies were installed in a fresh environment
and both workflows were executed in an isolated copy. The full rerun reproduced
all fields of `results.json` except elapsed time (version metadata excluded from
the comparison), and all six generated LaTeX files matched byte-for-byte. The
186 stored discrimination certificates passed validation. Reference outputs in
this archive were retained. See `REPRODUCIBILITY_CHECK.json` for the environment
and check summary.

## Citation and release metadata

Author metadata is supplied in `CITATION.cff`. Add the actual repository URL,
release tag and archived DOI when they exist. No DOI is assigned by this
package. The authors can select a reuse license for the public release.

`CHECKSUMS.sha256` identifies the files in the bundled reference snapshot.
Regenerating outputs can change those checksums, including PDF metadata.
