# understandInDetail.md

Comprehensive technical documentation for the hrJS/WannChi project.
**Last updated: 2026-03-26** (260324update)

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [gp_bo.f90 — Gaussian Process Bayesian Optimization](#2-gp_bof90--gaussian-process-bayesian-optimization)
3. [cma_es.f90 — CMA-ES Optimizer](#3-cma_esf90--cma-es-optimizer)
4. [classical_mc.f90 — Classical Heisenberg Monte Carlo](#4-classical_mcf90--classical-heisenberg-monte-carlo)
5. [transp_calc.f90 — Transport Calculations](#5-transp_calcf90--transport-calculations)
6. [wannlog.f90 — Timing and Logging](#6-wannlogf90--timing-and-logging)
7. [wanneff_JS.f90 — Main Workflow](#7-wanneff_jsf90--main-workflow)
8. [Downfolding Architecture](#8-downfolding-architecture)
9. [Test Documentation](#9-test-documentation)
10. [Build Process](#10-build-process)
11. [Q&A](#11-qa)
12. [Appendix: Fortran Syntax Reference](#12-appendix-fortran-syntax-reference)

---

## 1. Module Overview

### File Locations

| Module | File | Purpose |
|--------|------|---------|
| GP Bayesian Optimization | `modules/gp_bo.f90` | RBF kernel, Expected Improvement acquisition |
| CMA-ES Optimizer | `modules/cma_es.f90` | Covariance Matrix Adaptation Evolution Strategy |
| Classical MC | `modules/classical_mc.f90` | Heisenberg spin simulation via Metropolis |
| Transport | `modules/transp_calc.f90` | AHC (σ_xy) and DC conductivity (σ_xx) |
| Logging | `modules/wannlog.f90` | Wall-clock timing, message logging |
| Main Workflow | `src/wanneff_JS.f90` | Downfolding + BO/MC + transport pipeline |

### Module Dependency Graph

```
constants ← linalgwrap ← gp_bo
constants ← cma_es
constants ← classical_mc
constants ← linalgwrap ← wanndata ← transp_calc
para (or para_serial) ← wanndata
```

---

## 2. gp_bo.f90 — Gaussian Process Bayesian Optimization

**Location**: `modules/gp_bo.f90`
**Purpose**: Bayesian Optimization using Gaussian Processes with RBF kernel and Expected Improvement acquisition

### 2.1 TYPE gp_model

```fortran
TYPE gp_model
  integer :: n_train           ! number of observations so far
  integer :: n_params          ! dimension of parameter space
  real(dp), allocatable :: x_train(:,:)  ! (n_params, n_train)
  real(dp), allocatable :: y_train(:)    ! objective values (n_train)
  real(dp) :: length_scale     ! RBF kernel length scale l
  real(dp) :: signal_var       ! sigma_f^2 (signal variance)
  real(dp) :: noise_var        ! sigma_n^2 (jitter/noise)
  real(dp), allocatable :: K_inv(:,:)  ! (K+sigma_n^2*I)^{-1}
  real(dp), allocatable :: alpha(:)    ! K_inv * y_train
END TYPE gp_model
```

**Key design decisions**:
- `K_inv` and `alpha` are recomputed from scratch after each `gp_update` (not incrementally updated via Sherman-Morrison)
- NaN checks after `invmat` to detect ill-conditioned matrices — reverts `n_train` on failure
- Diagonal jittering `K + sigma_n^2*I` ensures positive definiteness

### 2.2 gp_init

```fortran
SUBROUTINE gp_init(gp, n_params, ls, sv, nv)
  TYPE(gp_model), intent(out) :: gp
  integer,  intent(in) :: n_params
  real(dp), intent(in) :: ls, sv, nv
```

**Inputs**:
- `n_params`: dimensionality of optimization problem
- `ls`: initial length scale (typically 0.5 × parameter range)
- `sv`: signal variance (typically 1.0)
- `nv`: noise variance (default 1e-2 for eigenvalue problems)

**Action**: Initializes `gp_model` with zeros/empty arrays. Does NOT allocate `x_train` etc. — those are allocated lazily in `gp_update`.

**Algorithm**: Simple assignment to type components.

### 2.3 gp_finalize

```fortran
SUBROUTINE gp_finalize(gp)
  TYPE(gp_model), intent(inout) :: gp
```

**Action**: Deallocates all allocated arrays (`x_train`, `y_train`, `K_inv`, `alpha`) and resets `n_train = 0`.

### 2.4 rbf_kernel

```fortran
FUNCTION rbf_kernel(x1, x2, n, ls, sv) RESULT(k)
  integer,  intent(in) :: n
  real(dp), dimension(n), intent(in) :: x1, x2
  real(dp), intent(in) :: ls, sv
  real(dp) :: k
```

**Formula**:
```
k(x1, x2) = sv * exp( -||x1 - x2||^2 / (2 * ls^2) )
```

**Inputs**:
- `x1, x2`: parameter vectors of length `n`
- `ls`: length scale (controls how far apart points can be before correlation drops)
- `sv`: signal variance (vertical scale of function variation)

**Algorithm**: Computes squared Euclidean distance, applies exponential decay.

### 2.5 gp_update

```fortran
SUBROUTINE gp_update(gp, x_new, y_new)
  TYPE(gp_model), intent(inout) :: gp
  real(dp), dimension(gp%n_params), intent(in) :: x_new
  real(dp), intent(in) :: y_new
```

**Purpose**: Add a new observation (x_new, y_new) to the GP model.

**Algorithm**:
1. If first observation (`n=0`): allocate all arrays with size 1
2. If subsequent: copy existing data to temp arrays → reallocate to `n+1` → restore old data
3. Append new observation to `x_train(:, n+1)` and `y_train(n+1)`
4. Build full `(n+1)×(n+1)` covariance matrix `K` using `rbf_kernel`
5. Add diagonal jitter: `K(ii,ii) += noise_var`
6. Additional diagonal floor: if `K(ii,ii) < noise_var`, set to `10*noise_var`
7. Compute `K_inv = K^{-1}` via `invmat` (LU decomposition)
8. Check for NaN in `K_inv` — if found, revert `n_train` and return (keep old state)
9. Compute `alpha = K_inv * y_train`
10. Check for NaN in `alpha` — if found, revert and return

**Key bug fixed**: The original implementation incorrectly used `K_inv` data when copying to temporary arrays for reallocation. Fixed to use `x_train(:, 1:n)` and `alpha(1:n)`.

### 2.6 gp_predict

```fortran
SUBROUTINE gp_predict(gp, x_test, mu_out, sigma_out)
  TYPE(gp_model), intent(in) :: gp
  real(dp), dimension(gp%n_params), intent(in) :: x_test
  real(dp), intent(out) :: mu_out, sigma_out
```

**Purpose**: Predict mean and variance at an untested point.

**Algorithm**:
1. Compute self-kernel: `k_ss = signal_var` (since `k(x,x) = sv * exp(0) = sv`)
2. Compute `k_star(ii) = rbf_kernel(x_test, x_train(:, ii))` for all `ii = 1..n_train`
3. GP mean: `mu = k_star · alpha` (dot product)
4. GP variance: `sigma^2 = k_ss - k_star · K_inv · k_star`
5. Apply floors: if `sigma^2 <= 0`, set to `1e-8`; if `sigma^2 > sv`, cap at `sv`

**Fallback**: If `mu` is NaN, set to `signal_var` (large positive value to discourage exploration).

### 2.7 expected_improvement

```fortran
FUNCTION expected_improvement(mu, sigma, y_best) RESULT(ei)
  real(dp), intent(in) :: mu, sigma, y_best
  real(dp) :: ei
```

**Purpose**: Expected Improvement acquisition function for Bayesian optimization.

**Formula**:
```
z = (y_best - mu) / sigma
EI = (y_best - mu) * Phi(z) + sigma * phi(z)
```
where `Phi` is the standard normal CDF and `phi` is the standard normal PDF.

**Algorithm**:
1. Handle NaN inputs: return 0 if any input is NaN
2. If `sigma < 1e-10`: return `max(y_best - mu, 0)` (deterministic limit)
3. Compute `z = (y_best - mu) / sigma`
4. Guard against overflow: if `|z| > 50`:
   - If `z > 0`: return 0 (EI → 0 for very large positive z)
   - If `z < 0`: compute `(y_best - mu) + sigma * phi(z)` (negative z case)
5. Otherwise: compute full EI formula
6. Final guard: if EI is NaN or negative, return 0

**Physical meaning**: EI balances exploitation (points near known good regions) vs exploration (points with high uncertainty). `y_best` is the current best observed value.

### 2.8 normal_cdf / normal_pdf

```fortran
FUNCTION normal_cdf(x) RESULT(p)
  ! Phi(x) = 0.5 * erfc(-x/sqrt(2))

FUNCTION normal_pdf(x) RESULT(phi)
  ! phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
```

Standard normal CDF and PDF using Fortran's `erfc` and `exp`.

### 2.9 latin_hypercube

```fortran
SUBROUTINE latin_hypercube(samples, n_samples, n_params, bounds)
  integer, intent(in) :: n_samples, n_params
  real(dp), dimension(2, n_params), intent(in) :: bounds
  real(dp), dimension(n_params, n_samples), intent(out) :: samples
```

**Purpose**: Generate space-filling Latin Hypercube samples for initial GP training.

**Algorithm**:
1. Create grid: `grid(i, j) = (j-1)/(n_samples-1)` for dimension `i`
2. Shuffle each dimension independently using Fisher-Yates (in-place swap)
3. Map to bounds with jitter: `samples(i,j) = bounds(1,i) + (grid(i,j) + (u-0.5)/n_samples) * (bounds(2,i)-bounds(1,i))`
4. Clip to bounds

**Property**: Each row and each column has exactly one sample — ensures good coverage of each dimension.

### 2.10 gp_log_marginal_likelihood

```fortran
SUBROUTINE gp_log_marginal_likelihood(lml, gp)
  TYPE(gp_model), intent(inout) :: gp
  real(dp), intent(out) :: lml
```

**Purpose**: Compute log marginal likelihood of GP hyperparameters.

**Formula**:
```
LML = -0.5 * y^T * K^{-1} * y - 0.5 * log|K| - n/2 * log(2*pi)
```

**Algorithm**:
1. Rebuild K matrix with current `length_scale`, `signal_var`, `noise_var`
2. Compute `K_inv` via `invmat`
3. Compute log determinant: `log|K| = -2 * sum(log(diag(K_inv)))` (Cholesky-like)
4. Compute data fit term: `0.5 * y^T * K_inv * y`
5. Combine into LML

**Use**: Called by `gp_optimize_ls` to find optimal length scale.

### 2.11 gp_optimize_ls

```fortran
SUBROUTINE gp_optimize_ls(gp, bounds, n_params)
  TYPE(gp_model), intent(inout) :: gp
  integer, intent(in) :: n_params
  real(dp), dimension(2, n_params), intent(in) :: bounds
```

**Purpose**: Optimize GP length scale via golden-section search on marginal likelihood.

**Algorithm**:
1. Compute average parameter range: `avg_range = mean(bounds(2,:) - bounds(1,:))`
2. Set search bounds: `ls_lo = 0.01*avg_range`, `ls_hi = 5.0*avg_range`
3. Golden-section iteration (15 steps):
   - Evaluate LML at `ls_mid1 = ls_hi - phi*(ls_hi-ls_lo)` and `ls_mid2 = ls_lo + phi*(ls_hi-ls_lo)`
   - Shrink interval toward better LML
4. Set optimal length scale to midpoint of final interval
5. Rebuild `K_inv` and `alpha` with optimal length scale

**Note**: Uses golden ratio `phi = 0.618...` for the search.

### 2.12 refine_ei

```fortroutine
SUBROUTINE refine_ei(x_out, x0, gp, y_best, n_params, bounds, n_steps)
  TYPE(gp_model), intent(in) :: gp
  integer, intent(in) :: n_params, n_steps
  real(dp), dimension(n_params), intent(in) :: x0
  real(dp), intent(in) :: y_best
  real(dp), dimension(2, n_params), intent(in) :: bounds
  real(dp), dimension(n_params), intent(out) :: x_out
```

**Purpose**: Refine initial candidate point via gradient ascent on Expected Improvement.

**Algorithm** (20 steps by default):
1. Set step size: `avg_range * 0.02` (2% of parameter range)
2. For each step:
   - For each dimension `ip`:
     - Finite difference: `ei_p = EI(mu(x_p), sigma(x_p), y_best)` where `x_p` is perturbed by `+h`
     - `ei_m` for perturbation by `-h`
     - Gradient: `(ei_p - ei_m) / (x_p(ip) - x_m(ip))`
   - Update: `x_cur += step_size * gradient`
   - Clip to bounds

**Purpose**: Exploits local structure of EI surface around candidate points.

### 2.13 bayesian_optimize (Main Entry Point)

```fortran
SUBROUTINE bayesian_optimize(objective_func, bounds, n_params, result, n_iter)
  interface
    function objective_func(params, n) result(val)
      use constants, only : dp
      integer,  intent(in) :: n
      real(dp), dimension(n), intent(in) :: params
      real(dp) :: val
    end function
  end interface
  integer,  intent(in)  :: n_params, n_iter
  real(dp), dimension(2, n_params), intent(in)  :: bounds
  real(dp), dimension(n_params), intent(out) :: result
```

**Purpose**: Main Bayesian Optimization loop with multi-start EI refinement.

**Algorithm**:

**Phase 1 — Latin Hypercube Initialization** (`n_init = max(5, n_params)`):
1. Generate `n_init` space-filling samples
2. Evaluate objective at each
3. Update GP with each observation
4. Track best point

**Phase 2 — Bayesian Optimization Loop** (`n_iter` iterations):
1. Sample `n_cand = max(2000, 50*n_params)` random candidates
2. For each candidate: compute GP predictive mean/variance via `gp_predict`
3. Compute EI at each candidate via `expected_improvement`
4. Identify top-K (`K_REFINE = 5`) candidates by EI
5. Refine each top-K candidate via `refine_ei` (gradient ascent on EI, 20 steps)
6. Select best refined candidate
7. Evaluate objective at best candidate
8. Update GP with new observation
9. Every 10 iterations (for `n_params > 10`): re-optimize length scale via `gp_optimize_ls`
10. Track overall best

**Output**: Returns best parameters found in `result`.

**Key parameters**:
- `noise_var = 1e-2` (higher than default 1e-4 for eigenvalue-based objectives)
- `n_cand` scales with dimensionality: for 4 params → 2000 candidates; for 20 params → 1000 candidates
- Length scale re-optimization every 10 iterations for high-dimensional problems

---

## 3. cma_es.f90 — CMA-ES Optimizer

**Location**: `modules/cma_es.f90`
**Purpose**: Covariance Matrix Adaptation Evolution Strategy for high-dimensional optimization

### 3.1 Overview

CMA-ES is a population-based evolutionary algorithm that adapts both the step size and covariance matrix during optimization. This implementation is a simplified **axis-parallel** version (covariance matrix is diagonal only).

**Applicability**:
- High-dimensional problems (`n_params >= 20`)
- Non-convex, multi-modal functions
- Black-box optimization without gradients
- When GP-BO is too slow or fails to converge

**NOT suitable for**:
- Very low dimensional problems (`n_params < 10`)
- Discrete/categorical parameters
- When sample efficiency is critical (population-based = many evaluations)

### 3.2 cmaes_optimize

```fortran
SUBROUTINE cmaes_optimize(objective_func, bounds, n_params, result, n_iter)
  interface
    function objective_func(params, n) result(val)
      use constants, only : dp
      integer,  intent(in) :: n
      real(dp), dimension(n), intent(in) :: params
      real(dp) :: val
    end function
  end interface
  integer,  intent(in)  :: n_params, n_iter
  real(dp), dimension(2, n_params), intent(in)  :: bounds
  real(dp), dimension(n_params), intent(out) :: result
```

**Algorithm**:

**Initialization**:
1. Set `lambda` (population size):
   - `n_params <= 20`: `lambda = max(20, 4 + floor(3*log(n_params)))`
   - `20 < n_params <= 100`: `lambda = max(50, n_params/2)`
   - `n_params > 100`: `lambda = max(100, n_params)`
2. `mu = lambda/2` (number of parents)
3. Mean: center of bounds
4. Step size `sigma`: 30% of parameter range
5. Covariance matrix: diagonal with `sigma^2` entries
6. Evolution paths `pc`, `ps`: initialized to zero

**CMA-ES Parameters**:
```fortran
cc   = 4.0 / (n_params + 4.0)
c1   = 2.0 / ((n_params + 1.3)^2 + mu)
cmu  = min(1 - c1, 2*(mu - 2 + 1/mu) / ((n_params + 2)^2 + mu))
damps = 1 + 2*max(0, sqrt(mu-1) - 1) + cc
chiN = sqrt(n_params) * (1 - 1/(4*n_params) + 1/(21*n_params^2))
```

**Main Loop** (for `n_iter` iterations):
1. **Sample**: Generate `lambda` offspring by adding Gaussian noise: `x_k = mean + sigma * N(0, I)`
2. **Evaluate**: Compute objective for all offspring
3. **Sort**: Sort by fitness (ascending — minimize)
4. **Update mean**: `mean = (1/mu) * sum(x_1 .. x_mu)` (weighted average of top mu)
5. **Update evolution paths**:
   - `pc = (1-cc)*pc + sqrt(cc*(2-cc)) * (mean_new - mean_old) / sigma`
   - `ps = (1-1/damps)*ps + sqrt(cc*(2-cc)) * sqrt(mu) * (mean_new - mean_old) / sigma`
6. **Update covariance**: `cov = (1-c1-cmu)*cov + c1*(pc*pc^T + (1-1/(4n))*cov) + cmu*(1/mu)*outer_diag_sum`
7. **Update step size**: `sigma = sigma * exp((||ps|| - chiN) / (sqrt(n)*damps))`
8. **Clip sigma**: maintain within [0.01*range, 0.5*range]
9. **Check convergence**: if `f_best < 1e-8`, exit

**Convergence**: Typically requires 100-1000×`n_params` evaluations for smooth problems.

---

## 4. classical_mc.f90 — Classical Heisenberg Monte Carlo

**Location**: `modules/classical_mc.f90`
**Purpose**: Classical Heisenberg spin simulation on f-site sublattice using Metropolis algorithm

### 4.1 TYPE mc_lattice

```fortran
TYPE mc_lattice
  integer :: n_sites           ! total spins in supercell
  integer :: n_neighbors_max   ! max neighbors per site (default 12)
  integer, allocatable :: neighbor_list(:,:)  ! (n_neighbors_max, n_sites)
  integer, allocatable :: n_nn(:)             ! actual neighbor count per site
  real(dp), allocatable :: spin(:,:)          ! (3, n_sites) unit spin vectors
  real(dp) :: J_mc            ! exchange coupling (>0 = ferromagnetic)
  real(dp) :: S_mag           ! spin magnitude |S|
END TYPE mc_lattice
```

### 4.2 mc_init

```fortran
SUBROUTINE mc_init(mc, n_sites, J_mc_in, S_mag_in)
  TYPE(mc_lattice), intent(out) :: mc
  integer,  intent(in) :: n_sites
  real(dp), intent(in) :: J_mc_in, S_mag_in
```

**Inputs**:
- `n_sites`: total number of spins (`NX * NY * NZ * n_f_sites`)
- `J_mc_in`: Heisenberg exchange coupling (eV)
- `S_mag_in`: spin magnitude

**Action**:
1. Sets `n_sites`, `J_mc`, `S_mag`
2. Sets `n_neighbors_max = 12` (enough for FCC/HCP)
3. Allocates `spin`, `neighbor_list`, `n_nn`
4. Initializes `n_nn = 0`, `neighbor_list = 0`
5. Randomizes spins on unit sphere via `mc_random_spin`

### 4.3 mc_random_spin

```fortran
SUBROUTINE mc_random_spin(spin)
  real(dp), dimension(3), intent(out) :: spin
```

**Purpose**: Generate uniform random unit vector on S² via Marsaglia method.

**Algorithm**:
1. Pick `u, v` uniform in `[-1, 1]` with `s = u² + v² < 1`
2. Compute: `spin = (2u*sqrt(1-s), 2v*sqrt(1-s), 1-2s)`

**Property**: Distribution is uniform over the unit sphere (not clustered at poles like naive `r*sin(theta)` methods).

### 4.4 mc_build_neighbors

```fortran
SUBROUTINE mc_build_neighbors(mc, frac_pos, n_uc_sites, avec, nx, ny, nz, cutoff)
  TYPE(mc_lattice), intent(inout) :: mc
  integer, intent(in)  :: n_uc_sites, nx, ny, nz
  real(dp), dimension(3, n_uc_sites), intent(in) :: frac_pos
  real(dp), dimension(3, 3), intent(in)  :: avec
  real(dp), intent(in) :: cutoff
```

**Purpose**: Build periodic neighbor list for supercell.

**Algorithm**:
1. Site indexing: `idx = ((ic*ny + ib)*nx + ia) * n_uc_sites + isite`
2. For each site `i`:
   - Compute Cartesian position `ri_cart = avec * (frac_pos(:,isite) + [ia, ib, ic])`
   - Loop over neighbor unit cells with PBC: `±2` shell in each direction
   - For each candidate neighbor site `j`:
     - Compute Cartesian distance
     - If `dist < cutoff`: add to neighbor list if space available
3. PBC wrapping: `mod(ic+dc+2*nz, nz)`

**Cutoff selection**: Typically `1.5 * dist_nn` where `dist_nn` is nearest-neighbor distance.

### 4.5 mc_local_energy

```fortran
FUNCTION mc_local_energy(mc, isite) RESULT(E)
  TYPE(mc_lattice), intent(in) :: mc
  integer, intent(in) :: isite
  real(dp) :: E
```

**Formula**:
```
E = -J_mc * S_mag² * sum_{j in NN(i)} spin_i · spin_j
```

**Algorithm**: Sum dot products with all neighbors, multiply by `-J_mc * S_mag²`.

### 4.6 mc_sweep

```fortran
SUBROUTINE mc_sweep(mc, temperature, n_accepted)
  TYPE(mc_lattice), intent(inout) :: mc
  real(dp), intent(in)  :: temperature
  integer,  intent(out) :: n_accepted
```

**Purpose**: One Monte Carlo sweep = `n_sites` attempted single-spin updates.

**Algorithm** (for each spin):
1. Select random site: `isite = floor(uniform * n_sites) + 1`
2. Generate new spin via `mc_random_spin`
3. Compute `dE` from neighbor interactions only (efficient: only recompute changed terms):
   ```
   dE = -J_mc * S_mag² * sum_{j in NN} (spin_new - spin_old) · spin_j
   ```
4. Metropolis acceptance:
   - If `dE <= 0`: accept
   - If `T > 1e-12`: accept with probability `exp(-dE/T)`
   - Otherwise: reject (zero temperature)

### 4.7 mc_thermalize

```fortroutine
SUBROUTINE mc_thermalize(mc, temperature, n_therm)
  TYPE(mc_lattice), intent(inout) :: mc
  real(dp), intent(in) :: temperature
  integer,  intent(in) :: n_therm
```

**Purpose**: Equilibrate system before measurement.

**Algorithm**: Call `mc_sweep` `n_therm` times (default: 5000 sweeps).

### 4.8 mc_measure_magnetization

```fortran
SUBROUTINE mc_measure_magnetization(mc, mvec_out)
  TYPE(mc_lattice), intent(in) :: mc
  real(dp), dimension(3), intent(out) :: mvec_out
```

**Formula**:
```
mvec = (1/N) * sum_i spin_i
```

**Output**: Fractional magnetization vector in `[-1, 1]³`. `|mvec|` is in `[0, 1]`.

**Note**: `spin` vectors are unit vectors (`|spin_i| = 1`), so `|mvec|` represents the ferromagnetic order parameter.

### 4.9 classical_mc_run (Main Entry Point)

```fortran
SUBROUTINE classical_mc_run(J_mc_in, S_mag_in, frac_pos, n_f_sites, avec, &
                             T_start, T_step, T_end, mvec_vs_T, n_temps, &
                             mc_supercell_in)
```

**Purpose**: Temperature sweep from `T_start` to `T_end`.

**Supercell auto-detection**:
- If `mc_supercell_in(i) > 0`: use user-specified `NX, NY, NZ`
- Otherwise: if `|a_i| > 2 * min(|a_j|, |a_k|)`: set `N_i = 1` (vacuum direction)
- Else: set `N_i = 10` (standard thermodynamic limit)

**Algorithm**:
1. Determine supercell dimensions
2. Compute total sites: `n_total = NX * NY * NZ * n_f_sites`
3. Initialize lattice: `mc_init`
4. Build neighbor list with `cutoff = 1.5 * dist_nn`
5. For each temperature:
   - If `T < 1e-12`: perfect ferromagnetic order (all spins aligned along z)
   - Else: thermalize (`N_THERM = 5000` sweeps), then measure (`N_MEAS = 10000`, every 10 sweeps)
6. Return `mvec_vs_T(3, n_temps)`

**Output magnetization**: `S_eff_vec = S_mag * mvec_vs_T` (actual moment per site)

---

## 5. transp_calc.f90 — Transport Calculations

**Location**: `modules/transp_calc.f90`
**Purpose**: Compute anomalous Hall conductivity (σ_xy) and longitudinal DC conductivity (σ_xx)

### 5.1 calc_velocity

```fortran
SUBROUTINE calc_velocity(v_alpha, ham, kvec, alpha)
  TYPE(wannham), intent(in) :: ham
  real(dp), dimension(3), intent(in) :: kvec
  integer, intent(in) :: alpha   ! 1=x, 2=y, 3=z
  complex(dp), dimension(ham%norb, ham%norb), intent(out) :: v_alpha
```

**Purpose**: Compute velocity matrix `v_alpha(k)` analytically from Wannier Hamiltonian (no numerical k-derivatives).

**Formula** (from Bloch sum derivative):
```
v_alpha(io, jo) = sum_R [i * 2pi * Rtilde_alpha * exp(i*2pi*k.R) / w(R)]
                   * conj(phase_io) * phase_jo * hr(io, jo, R)
where:
  Rtilde_alpha = R_alpha + tau_jo_alpha - tau_io_alpha
  phase_io = exp(i*2pi*k.tau_io)
```

**Algorithm**:
1. Compute orbital phase factors: `phase(io) = exp(i*2pi*k.tau_io)` for all orbitals
2. For each R-vector:
   - Compute `rdotk = 2pi * k.R`
   - Factor: `fact = exp(i*rdotk) / w(R)`
   - For each orbital pair `(io, jo)`:
     - Compute `Rtilde_alpha = rvec(alpha, ir) + tau(jo) - tau(io)`
     - `v_alpha(io,jo) += i * 2pi * Rtilde_alpha * fact * conj(phase(io)) * phase(jo) * hr(io,jo,R)`

**Significance**: This is the exact derivative `dH(k)/dk_alpha` of the Bloch sum formula, avoiding numerical differentiation artifacts.

### 5.2 calc_berry_curvature

```fortran
SUBROUTINE calc_berry_curvature(omega_n, eigvec, vx, vy, eig, norb)
  complex(dp), dimension(norb, norb), intent(in) :: eigvec, vx, vy
  real(dp), dimension(norb), intent(in) :: eig
  real(dp), dimension(norb), intent(out) :: omega_n
```

**Purpose**: Compute Berry curvature per band from velocity matrices and eigenvectors.

**Formula**:
```
Omega_n^{xy}(k) = -2 * Im sum_{m≠n} Vx_{nm} * Vy_{mn} / (E_n - E_m)²
```

where `Vx_{nm} = <n|v_x|m>` in the eigenbasis.

**Algorithm**:
1. Transform orbital velocities to eigenbasis via ZGEMM:
   ```
   vx_band = eigvec^H . vx_orb . eigvec
   vy_band = eigvec^H . vy_orb . eigvec
   ```
2. For each band `n`:
   - For each other band `m`:
     - Skip if `|E_n - E_m| < eps6` (near-degenerate)
     - `Omega_n += -2 * Im(Vx_{nm} * Vy_{mn}) / (E_n - E_m)²`

### 5.3 fermi_func

```fortran
SUBROUTINE fermi_func(f, eig, mu, temperature, norb)
  real(dp), dimension(norb), intent(out) :: f
```

**Formula**: Fermi-Dirac distribution
```
f(E) = 1 / (exp((E - mu) / T) + 1)
```

**Algorithm**:
- If `T < eps6`: step function — `1` if `E <= mu`, else `0`
- Else: standard Fermi-Dirac with overflow guards (`x > 500` → 0, `x < -500` → 1)

### 5.4 calc_sigma_xy

```fortran
SUBROUTINE calc_sigma_xy(sigma_xy, ham, kvec_all, kwt_all, nk, mu_chem, temperature)
  TYPE(wannham), intent(in) :: ham
  real(dp), intent(out) :: sigma_xy
```

**Purpose**: Compute anomalous Hall conductivity via Berry curvature sum.

**Formula**:
```
sigma_xy = -(e²/h) * (1/N_k) * sum_k sum_n f_n(k) * Omega_n^{xy}(k)
```

**Algorithm**:
1. For each k-point:
   - `calc_hk`: get `H(k)`
   - `eigen`: diagonalize → eigenvalues `eig`, eigenvectors (stored in `hk`)
   - `calc_velocity`: get `v_x(k)` and `v_y(k)`
   - `calc_berry_curvature`: get `Omega_n` for each band
   - `fermi_func`: get occupations `f_n`
   - Accumulate: `sigma_acc += sum(f_n * Omega_n) * kwt`
2. Apply prefactor: `sigma_xy = -sigma_acc / (2pi)²`

**Note**: Factor `1/(2pi)²` arises from converting fractional k integration to physical units. For 2D systems with `nk3=1`, the result is in units of `e²/h`.

### 5.5 calc_sigma_xx

```fortran
SUBROUTINE calc_sigma_xx(sigma_xx, ham, kvec_all, kwt_all, nk, mu_chem, temperature, broadening)
```

**Purpose**: Longitudinal DC conductivity via Kubo-Greenwood formula.

**Formula**:
```
sigma_xx = (1/N_k) * sum_k Tr[v_x · G(k, mu+i*eta) · v_x · G(k, mu+i*eta)]
```

where `G(k, z) = (z - H(k))^{-1}` is the retarded Green's function.

**Algorithm**:
1. `w_cmplx = mu + i * broadening`
2. For each k-point:
   - `calc_hk`: get `H(k)`
   - `calc_g0`: compute `G(k, mu+i*eta) = (mu+i*eta - H(k))^{-1}`
   - `calc_velocity`: get `v_x(k)`
   - Kubo bubble via ZGEMM:
     ```
     tmp1 = v_x . G
     tmp2 = tmp1 . v_x = v_x . G . v_x
     tmp1 = tmp2 . G = v_x . G . v_x . G
     ```
   - Accumulate: `sigma_acc += Tr(tmp1) * kwt`
3. `sigma_xx = -imag(sigma_acc) / (2pi)²`

**Physical meaning**: `broadening = eta` corresponds to `ℏ/(2*tau)` where `tau` is the scattering time. The Kubo formula naturally includes both Drude (intra-band) and inter-band contributions through the full Green's function.

---

## 6. wannlog.f90 — Timing and Logging

**Location**: `modules/wannlog.f90`
**Purpose**: Lightweight wall-clock timing and message logging for Fortran programs

### 6.1 log_init

```fortran
SUBROUTINE log_init()
```

**Action**:
1. Reset `wl_n_timers = 0`, `wl_n_messages = 0`
2. Record program start time via `cpu_time(wl_wall_start)`
3. Print initialization banner to stdout

**Note**: Uses `cpu_time` (wall-clock) not `system_clock` (may wrap).

### 6.2 log_start

```fortran
SUBROUTINE log_start(label)
  character(len=*), intent(in) :: label
```

**Purpose**: Start (or restart) a named timer.

**Algorithm**:
1. Get current CPU time
2. Search for existing timer with matching label
3. If found: reset start time, increment call count
4. If not found and slots available: create new timer entry

### 6.3 log_stop

```fortran
SUBROUTINE log_stop(label)
  character(len=*), intent(in) :: label
```

**Purpose**: Stop a named timer and accumulate elapsed time.

**Algorithm**:
1. Get current CPU time
2. Find matching timer
3. Accumulate `elapsed += (t_now - start_time)`
4. Guard against clock wraps: if `dt < 0`, set `dt = 0`

### 6.4 log_msg

```fortran
SUBROUTINE log_msg(msg)
  character(len=*), intent(in) :: msg
```

**Action**:
1. Append message to buffer (up to 200 messages)
2. Echo immediately to stdout with `  [log] ` prefix

### 6.5 log_print_summary

```fortran
SUBROUTINE log_print_summary()
```

**Output**: Formatted table to stdout:
```
============================================================
  wannlog: timing summary
============================================================
  Total CPU time:  XXX.XXX s

  Stage                                    Calls     CPU (s)  %Total
  ---------------------------------------------------------------
  downfolding                                   1     XX.XXX   XX.X
  bayesian_optimize                             1    XXX.XXX   XX.X
  transport_calc                                1     XX.XXX   XX.X
  classical_mc                                  1    XXX.XXX   XX.X
  ---------------------------------------------------------------
```

---

## 7. wanneff_JS.f90 — Main Workflow

**Location**: `src/wanneff_JS.f90`
**Purpose**: End-to-end pipeline: downfolding → J·S optimization → MC → transport

### 7.1 Module-Level Globals (for Bayesian callback)

```fortran
integer :: g_norb_bare, g_norb_cc, g_nkirr, g_n_jrpt
complex(dp), allocatable :: g_hk_eff_cc(:,:,:)  ! (norb_cc, norb_cc, nkirr)
TYPE(wannham), pointer :: g_ham_bare => null()
real(dp), allocatable :: g_kvec(:,:)            ! (3, nkirr)
real(dp), allocatable :: g_rvec_J(:,:)           ! (3, n_jrpt)
integer :: g_eff_mode  ! 1=scalar, 2=J_TENSOR, 3=J_S_TENSOR
```

**Purpose**: These module-level pointers are set in the main program and accessed by `js_objective` (the Bayesian callback) which has a fixed `f(params, n) -> val` interface.

### 7.2 downfold_rspace

```fortran
SUBROUTINE downfold_rspace(ham_eff_cc, ham_full, cc_idx, ff_idx, n_cc, n_ff)
```

**Purpose**: R-space Schur complement downfolding.

**Formula**:
```
H_eff_CC(R) = H_CC(R) - H_CF(R) * H_FF(R)^{-1} * H_FC(R)
```

**Algorithm**:
1. Allocate output `ham_eff_cc` with `n_cc` orbitals and same R-grid as `ham_full`
2. Copy tau positions for CC orbitals only
3. For each R-point:
   - Extract 4 blocks: `H_CC` (n_cc×n_cc), `H_CF` (n_cc×n_ff), `H_FC` (n_ff×n_cc), `H_FF` (n_ff×n_ff)
   - If `maxval(|H_FF|) < eps6`: no FF hopping → `H_eff = H_CC` (skip inversion)
   - Else: invert `H_FF` → compute `H_CF * H_FF^{-1} * H_FC` → subtract from `H_CC`
4. Store result in `ham_eff_cc%hr`

**Significance**: This is the **block matrix Schur complement** — exactly equivalent to integrating out FF degrees of freedom. Unlike the k-space approach which uses `H_eff = -[G_CC]^{-1}` and can produce ghost states, the Schur complement directly gives the CC block of the inverse.

**⚠️ Known limitation**: The sign of the Schur complement term depends on whether FF states are filled (below Fermi) or empty (above Fermi). Current implementation always uses `-H_CF * H_FF^{-1} * H_FC`. For systems with FF states above Fermi (e.g., Kondo systems), the sign should be `+`.

### 7.3 js_objective (Bayesian callback)

```fortran
SUBROUTINE js_objective(params, n_params, val)
```

**Purpose**: Compute L2 eigenvalue mismatch for Bayesian optimization.

**Objective function**:
```
L(J, S) = (1/N_k) sum_k ||eig(H_bare(k) + H_JS(k)) - eig(H_eff_CC(k))||²
```

**Algorithm**:
1. Unpack parameters based on `g_eff_mode`:
   - Mode 1: `(J_0, S_x, S_y, S_z)` → 4 params
   - Mode 2: `(J_R(1:n), S_x, S_y, S_z)` → n+3 params
   - Mode 3: `(J_R, S_x(1:n), S_y(1:n), S_z(1:n))` → 4n params
2. For each k-point:
   - Compute `H_bare(k)` via `calc_hk`
   - Add `H_JS(k)` via `add_js_coupling_kspace` (scalar) or `add_js_coupling_tensor_kspace` (tensor)
   - Diagonalize both: `eig_trial = eig(H_bare + H_JS)`, `eig_eff = eig(H_eff_CC)`
   - Accumulate squared eigenvalue differences
3. Average over k-points

### 7.4 add_js_coupling / add_js_coupling_kspace

```fortran
SUBROUTINE add_js_coupling(ham_out, norb_bare, J_val, Svec, irpt)
  ! R-space: adds J*(S.sigma)/2 at specific R-point

SUBROUTINE add_js_coupling_kspace(hk, norb, J_val, Svec)
  ! k-space: adds J*(S.sigma)/2 to diagonal elements
```

**Spinor exchange matrix** (for `n_c = norb/2` spatial orbitals):
```
hr(io,     io,     ir) += J * S_z / 2       (spin-up diagonal)
hr(io+n_c, io+n_c, ir) -= J * S_z / 2       (spin-down diagonal)
hr(io,     io+n_c, ir) += J * (S_x - i*S_y) / 2  (off-diagonal)
hr(io+n_c, io,     ir) += J * (S_x + i*S_y) / 2  (off-diagonal)
```

### 7.5 add_js_coupling_tensor_kspace

```fortran
SUBROUTINE add_js_coupling_tensor_kspace(hk, norb, jeff_R, S_R, n_jrpt, rvec_J, kvec)
```

**Purpose**: Add tensor JS coupling where both J(R) and S(R) vary with R-vector.

**Formula**:
```
H_JS(k) = sum_R J(R) * exp(i*k*R) * (S(R).sigma) / 2
```

**Algorithm**:
1. For each R-vector:
   - Compute `Jk_R = J(R) * exp(i*k*R)`
   - Accumulate `Jsz += Jk_R * S_z(R) / 2`, etc.
2. Add combined coupling to Hamiltonian

### 7.6 Main Program Flow

```
1. read_input → &EFFJS namelist
2. read_ham(ham, seed) → full system
3. read_posfile(seed.pos) → set ham%tau
4. wannham_shift_ef(ham, mu)
5. read_ham(ham_bare, seedbare) → conduction-only
6. read_posfile(seedbare.pos) → get nbasis/xat
7. Assign ham_bare%tau from xat
8. read_kmesh(IBZKPT)
9. Determine CC/FF orbital indices
10. R-space downfolding: downfold_rspace
11. calc_hk for all k-points → g_hk_eff_cc
12. If eff_js:
    a. Set up module globals for callback
    b. bayesian_optimize or cmaes_optimize → J_opt, S_opt
    c. Write seed_JS.output
    d. Construct ham_out = ham_bare + JS coupling at T=0
    e. Write seedbare_hr_0K.dat
    f. calc_sigma_xy + calc_sigma_xx at T=0
    g. Read QPOINTS → write seed_JS_result.dat
13. If eff_mc:
    a. classical_mc_run → mvec_vs_T
    b. For each temperature:
       - Reconstruct ham_out with T-dependent S_eff
       - Write seedbare_hr_TK.dat
       - Compute transport → seed_transport_vs_T.dat
14. Cleanup + log_print_summary
```

---

## 8. Downfolding Architecture

### 8.1 Physical Model

```
H_seed = [ H_CC  H_CF ]
         [ H_FC  H_FF ]    (full system with f-electrons)

H_eff_CC = H_CC - H_CF * H_FF^{-1} * H_FC   (Schur complement)
H_bare   = Wannier90 HR of seedbare (conduction-only)
```

**Key assumption**: FF orbitals are at different atomic sites than CC orbitals, so the CC-FF coupling `H_CF` is hopping-like.

### 8.2 R-space vs K-space Downfolding

**R-space (used here)**:
- Computes `H_eff_CC(R)` directly via Schur complement
- Inverse FT to get `H_eff_CC(k)` for eigenvalue comparison
- Numerically more stable for localized Wannier functions

**K-space (deprecated)**:
- Computes `G_full(k,0) = (0 - H_seed(k))^{-1}`
- Extracts CC block: `H_eff_CC(k) = -[G_CC(k,0)]^{-1}`
- Prone to ghost states when FF energies are near Fermi

### 8.3 ⚠️ Known Limitation: Energy Window Sign Dependency

The Schur complement sign depends on the energy window:

| FF states | Sign | Physical meaning |
|-----------|------|-----------------|
| Below Fermi (filled) | `-H_CF * H_FF^{-1} * H_FC` | Electrons virtual-hop into empty FF states |
| Above Fermi (empty) | `+H_CF * H_FF^{-1} * H_FC` | Virtual hopping from filled FF states |

**Current behavior**: Always uses negative sign regardless of energy window.

**Detection**: Check `H_FF` diagonal at R=0 (on-site energies). If diagonal is positive (above Fermi), flip the sign.

**Test case**: `kagome_f_spinor_test.py` places f-orbitals at `EF + 0.4 eV` → requires positive sign.

---

## 9. Test Documentation

### 9.1 kagome_f_spinor_test.py (Test 6)

**Location**: `tests/kagome_f_spinor_test.py`
**Purpose**: End-to-end pipeline test — Python I/O only, all computation in Fortran

**Lattice**:
- a1 = (1, 0, 0), a2 = (0.5, √3/2, 0), a3 = (0, 0, 10)
- 4 kagome sites + f-site decoration at site 4
- seed: 10 spinor orbitals (4 sites × [1,1,1,2] basis)
- seedbare: 8 spinor orbitals (4 sites × [1,1,1,1] basis)

**Test variants**:
1. `full_pipeline_e2e`: scalar J, full workflow
2. `cubic_pipeline`: simple cubic lattice variant
3. `tensor_pipeline_*`: tensor J mode
4. `kagome3_pipeline`: Heisenberg antiferromagnet
5. `js_tensor_pipeline_*`: J_S_TENSOR mode

**Usage**:
```bash
cd tests
source /Users/ykxu/Projects/hrJS/hrJS/bin/activate
python3 kagome_f_spinor_test.py
```

**Output**: Test results in `test_tsf0.5_output/` directory with:
- `test6_report.md`: summary of all test results
- Band comparison plots
- Transport vs temperature plots

### 9.2 algo_test.py (Tests 1-5)

**Location**: `tests/algo_test.py`
**Purpose**: Pure Python algorithm validation — no Fortran needed

| Test | Name | Validates |
|------|------|-----------|
| 1 | Kagome reference | Wannier interpolation + band structure |
| 2 | Bayesian 8D | GP-BO on 8D Ackley function |
| 3 | MC square lattice | Heisenberg MC magnetization |
| 4 | Berry curvature | AHC calculation |
| 5 | Fortran pipeline | (skipped if no Fortran) |

---

## 10. Build Process

### 10.1 make.sys.laptop (gfortran + Accelerate)

**Location**: `build_laptop/Makefile`
**Config**: `make.sys.laptop`

```makefile
F90 = gfortran
F90FLAGS = -O2 -framework Accelerate -fpp
LAPACKLIBS = -framework Accelerate
```

**Key features**:
- No MPI (`para_serial.f90` provides stubs)
- gfortran with `-fpp` (Fortran preprocessor)
- Accelerate framework provides LAPACK/BLAS

### 10.2 para_serial.f90

**Location**: `modules/para_serial.f90`
**Purpose**: Serial stub for MPI parallel utilities

Provides empty/no-op versions of:
- `init_para`, `finalize_para`
- `distribute_calc` (sets `first_idx=1, last_idx=n`)
- `para_merge_cmplx`, `para_sync_logical`, `para_sync0`

### 10.3 Build Commands

```bash
# Laptop build
cd build_laptop
make wanneff_js.x wannband.x
cp wanneff_js.x wannband.x ../src/

# Fortran unit tests
cd tests
make test_all

# Module compilation
cd modules
make mod.a
```

---

## 11. Q&A

### Q1: What is n_jrpt?

`n_jrpt` is the number of R-vectors (real-space lattice vectors) used for the J(R) tensor in tensor modes (eff_mode=2 or 3).

- In scalar mode (eff_mode=1): `n_jrpt = 0`, only J(R=0) is used
- In tensor mode: `n_jrpt` equals the number of R-vectors in the Wannier Hamiltonian (or custom grid if `J_R_range` is specified)

Each R-vector has an associated J(R) value (and in mode 3, an associated S(R) vector).

### Q2: How does the Wigner-Seitz cell division work?

The Wigner-Seitz (WS) cell division determines which lattice points `R` belong to which periodic image of a neighboring cell.

**Algorithm in `find_ws_rvectors`**:
1. Loop over candidate R-vectors in a box `[-nr1:nr1] × [-nr2:nr2] × [-nr3:nr3]`
2. For each candidate `R`, compute distances to all 125 periodic images `R + n1*a1 + n2*a2 + n3*a3` (with `ni ∈ {-2,-1,0,1,2}`)
3. Find the minimum distance
4. If the original R has the minimum distance to itself (i.e., it's the "home" cell), include it with weight = number of equivalent R-vectors

**Use in wanndata.f90**:
- `rvec`: fractional coordinates of R-vectors in the WS cell
- `weight`: number of equivalent R-vectors (for proper normalization of Fourier transform)

---

## 12. Appendix: Fortran Syntax Reference

### 12.1 TYPE Definition

```fortran
TYPE :: type_name
  integer :: field1
  real(dp), allocatable :: array(:,:)
END TYPE type_name
```

### 12.2 Interface Block (callback)

```fortran
interface
  function objective_func(params, n) result(val)
    use constants, only : dp
    integer, intent(in) :: n
    real(dp), dimension(n), intent(in) :: params
    real(dp) :: val
  end function
end interface
```

### 12.3 Intent Specifiers

| Intent | Meaning |
|--------|---------|
| `intent(in)` | Input — not modified |
| `intent(out)` | Output — will be written to |
| `intent(inout)` | Both — may be modified |

### 12.4 Array Slice

```fortran
real(dp), dimension(10) :: a
a(1:5)        ! First 5 elements
a(2:10:2)    ! Elements 2,4,6,8,10 (step 2)
a(:)         ! All elements
```

### 12.5 Module vs Program

- **MODULE**: Collection of types, constants, procedures; `CONTAINS` marks procedure definitions
- **PROGRAM**: Executable unit; `CALL` procedures, `USE` modules
- **Internal procedures**: Procedures defined after `CONTAINS` inside a module/program; can access module-level variables

### 12.6 ZGEMM (Complex Matrix Multiply)

```fortran
call zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
! C = alpha * op(A) * op(B) + beta * C
! op(X) = X (no transpose), X^T (transpose), X^H (conjugate transpose)
```

**Common usage**:
```fortran
! vx_band = eigvec^H . vx_orb . eigvec
call zgemm('N', 'N', norb, norb, norb, zone, vx, norb, eigvec, norb, zzero, tmp, norb)
call zgemm('C', 'N', norb, norb, norb, zone, eigvec, norb, tmp, norb, zzero, vx_band, norb)
```

### 12.7 Random Number Generation

```fortran
real(dp) :: u
call random_number(u)  ! u in [0, 1)
u = 2.0_dp * u - 1.0_dp  ! map to [-1, 1)
```

### 12.8 Array Construction

```fortran
real(dp), dimension(3) :: vec
vec = [1.0_dp, 2.0_dp, 3.0_dp]  ! 1D array constructor
! Or with explicit bounds:
real(dp), dimension(0:2) :: vec
```

### 12.9 Deallocate with Check

```fortran
if (allocated(arr)) deallocate(arr)
if (associated(ptr)) nullify(ptr)
```

### 12.10 Merge (Conditional Assignment)

```fortran
! If condition is true, use tval; otherwise use fval
NX = merge(1, N_DEFAULT, len_a1 > 2.0_dp * len_min)
```

---

## References

1. **Gaussian Process BO**: Mockus (1989), Brochu et al. (arXiv:1012.2599), Bergstra & Bengio (JMLR 2012)
2. **CMA-ES**: Hansen & Ostermeier (Evolutionary Computation 2001), Hansen et al. (JMLR 2019)
3. **AHC/Berry curvature**: Wang et al. (PRB 74, 195118, 2006), Yao et al. (PRL 92, 037204, 2004), Xiao et al. (RMP 82, 1959, 2010)
4. **Kubo-Greenwood**: Kubo (J. Phys. Soc. Jpn. 12, 570, 1957), Bastin et al. (J. Phys. Chem. Solids 32, 1811, 1971)
5. **Monte Carlo**: Metropolis et al. (J. Chem. Phys. 21, 1087, 1953), Janke (Springer 1996)
6. **Downfolding**: Kotliar et al. (RMP 78, 865, 2006), Coleman (Introduction to Many-Body Physics, 2015)
