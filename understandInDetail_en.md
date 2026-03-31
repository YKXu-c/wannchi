# WannChi Code Architecture — Comprehensive Technical Documentation

**Project**: WannChi — Wannier Hamiltonian-based response function and RPA calculation program
**Authors**: Chao Cao, Siqi Wu, Chenchao Xu, Guo-Xiang Zhi (Zhejiang University)
**Location**: `/Users/ykxu/Projects/hrJS/wannchi`
**Reference**: understand.md

---

## Table of Contents

1. [Module Detailed Subroutine Analysis](#1-module-detailed-subroutine-analysis)
   - 1.1 constants — Basic Constants and Type Definitions
   - 1.2 para — MPI Parallel Wrapper
   - 1.3 wanndata — Wannier Hamiltonian Data Structure
   - 1.4 linalgwrap — BLAS/LAPACK Linear Algebra Wrapper
   - 1.5 symmetry_module — Angular Momentum and Symmetry Operations
   - 1.6 simp_module — Impurity Simplification and Self-Energy Packing
   - 1.7 lattice — Lattice Structure, k-mesh, and Self-Energy Interpolation
   - 1.8 IntRPA — RPA FF/CC Block Structure
   - 1.9 pade — Padé Summation
   - 1.10 transp_calc — Transport Property Calculations
   - 1.11 gp_bo — Gaussian Process Bayesian Optimization
   - 1.12 cma_es — CMA-ES Optimizer
   - 1.13 classical_mc — Classical Heisenberg Monte Carlo
   - 1.14 wannlog — Timing and Logging
   - 1.15 wanneff_JS — J-S Kondo Exchange Coupling Fitting Workflow
2. [Source File Detailed Subroutine Analysis](#2-source-file-detailed-subroutine-analysis)
   - 2.1 input — Input File Parsing
   - 2.2 green.f90 — Green's Function Calculation
   - 2.3 output.f90 / output_chi.f90 — Output Utilities
   - 2.4 compute_chi.f90 — Response Function Core Computation
   - 2.5 wannchi.f90 — Main Program (bare susceptibility)
   - 2.6 wannchiRPA.f90 — Main Program (RPA-dressed susceptibility)
   - 2.7 postchi.f90 — Response Function Post-Processing
   - 2.8 wannband.f90 — Spectral Function Calculation
   - 2.9 wanneff_JS.f90 — J-S Kondo Exchange Coupling Fitting
3. [Downfolding Architecture](#3-downfolding-architecture)
4. [Test Documentation](#4-test-documentation)
5. [Build Process](#5-build-process)
6. [Q&A](#6-qa)
7. [Appendix: Fortran Syntax Reference](#7-appendix-fortran-syntax-reference)
8. [References](#references)
9. [Summary](#summary)

---

## 1. Module Detailed Subroutine Analysis

### 1.1 constants — Basic Constants and Type Definitions

**File**: [`modules/constants.f90`](wannchi/modules/constants.f90) (56 lines)

#### 1.1.1 Constant Definitions

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `dp` | integer | `selected_real_kind(14, 200)` | Double precision floating-point kind parameter |
| `twopi` | real(dp) | `6.283185307179586_dp` | $2\pi$ |
| `sqrtpi` | real(dp) | `1.772453850905516_dp` | $\sqrt{\pi}$ |
| `sqrt2` | real(dp) | `1.414213562373095_dp` | $\sqrt{2}$ |
| `logpi_2` | real(dp) | `0.572364942924700_dp` | $\ln(\pi/2)$ |
| `cmplx_1` | complex(dp) | `cmplx(1,0,dp)` | Complex 1 |
| `cmplx_i` | complex(dp) | `cmplx(0,1,dp)` | Complex imaginary unit $i$ |
| `cmplx_0` | complex(dp) | `cmplx(0,0,dp)` | Complex 0 |

#### 1.1.2 I/O Unit Numbers

| Constant | Value | Purpose |
|----------|-------|---------|
| `stdin` | 5 | Standard input |
| `stdout` | 6 | Standard output |
| `fin` | 10 | Main input file unit |
| `fout` | 11 | Main output file unit |
| `fout2` | 12 | Auxiliary output file unit |
| `fin3-fin6` | 13-16 | Auxiliary input file units |
| `fout3-fout6` | 17-20 | Auxiliary output file units |
| `fdebug` | 30 | Debug output unit |

#### 1.1.3 Numerical Tolerances

| Constant | Value | Purpose |
|----------|-------|---------|
| `eps4` | 1.0d-4 | General tolerance |
| `eps6` | 1.0d-6 | High-precision tolerance |
| `eps9` | 1.0d-9 | Ultra-high-precision tolerance |
| `eps12` | 1.0d-12 | Extreme-precision tolerance |
| `eps18` | 1.0d-18 | Extreme-precision tolerance |
| `eps36` | 1.0d-36 | Extreme-precision tolerance |
| `maxint` | 2147483647 | Maximum 32-bit integer |

---

### 1.2 para — MPI Parallel Wrapper

**File**: [`modules/para.f90`](wannchi/modules/para.f90) (373 lines)

**Dependencies**: `constants`, `mpi`

#### Global Variables

| Variable | Type | Description |
|----------|------|-------------|
| `inode` | integer | Current process rank (0 = master) |
| `nnode` | integer | Total number of processes |
| `first_idx` | integer | Starting index of tasks for current process |
| `last_idx` | integer | Ending index of tasks for current process |
| `map(nnode, 2)` | integer, allocatable | Task range mapping table for each process |

#### Subroutine Details

##### 1.2.1 `init_para(codename)`

**Purpose**: Initialize MPI environment and retrieve process information

**Interface**:
```fortran
SUBROUTINE init_para(codename)
    character(*), intent(in) :: codename  ! Program name string
```

**Implementation**:
- If `__MPI` is defined, call `mpi_init` to initialize MPI
- Get current process rank via `mpi_comm_rank` (`inode`)
- Get total process count via `mpi_comm_size` (`nnode`)
- Allocate `map(nnode, 2)` array
- Print program run information on master process (inode=0)

**Algorithm**:
```
if (defined __MPI):
    mpi_init()
    mpi_comm_rank(MPI_COMM_WORLD, inode)
    mpi_comm_size(MPI_COMM_WORLD, nnode)
    allocate(map(nnode, 2))
    if (inode == 0) print "codename running on nnode nodes"
else:
    inode = 0
    nnode = 1
    print "codename serial"
```

---

##### 1.2.2 `finalize_para()`

**Purpose**: Finalize MPI environment

**Interface**:
```fortran
SUBROUTINE finalize_para()
```

**Implementation**:
- If `__MPI` is defined, call `mpi_finalize`

---

##### 1.2.3 `distribute_calc(nidx)`

**Purpose**: Distribute `nidx` tasks across processes, setting `first_idx`/`last_idx`

**Interface**:
```fortran
SUBROUTINE distribute_calc(nidx)
    integer, intent(in) :: nidx  ! Total number of tasks
```

**Implementation**:
- Calculate tasks per process: `nidx/nnode`
- Set `first_idx` and `last_idx` ranges
- Sync map information to all processes via `para_merge_int`

**Algorithm**:
```
block_size = nidx / nnode
first_idx = inode * block_size + 1
last_idx = (inode + 1) * block_size
map[inode+1, 1] = first_idx - 1
map[inode+1, 2] = last_idx - first_idx + 1
if (defined __MPI):
    para_merge_int(map, 2*nnode)
```

---

##### 1.2.4 `para_barrier()`

**Purpose**: MPI synchronization barrier

**Interface**:
```fortran
SUBROUTINE para_barrier()
```

---

##### 1.2.5 `para_sync_int0(dat)`

**Purpose**: Broadcast a single integer (from node 0)

**Interface**:
```fortran
SUBROUTINE para_sync_int0(dat)
    integer, intent(inout) :: dat
```

---

##### 1.2.6 `para_sync_real0(dat)`

**Purpose**: Broadcast a single real number (from node 0)

**Interface**:
```fortran
SUBROUTINE para_sync_real0(dat)
    real(dp), intent(inout) :: dat
```

---

##### 1.2.7 `para_sync_cmplx(dat, dat_size)`

**Purpose**: Broadcast a complex array (from node 0)

**Interface**:
```fortran
SUBROUTINE para_sync_cmplx(dat, dat_size)
    complex(dp), intent(inout) :: dat(*)
    integer, intent(in) :: dat_size
```

---

##### 1.2.8 `para_merge_real0(dat)`

**Purpose**: Global reduction sum (real scalar)

**Interface**:
```fortran
SUBROUTINE para_merge_real0(dat)
    real(dp), intent(inout) :: dat
```

**Formula**:
$$x_{\text{total}} = \sum_{i=0}^{n_{\text{node}}-1} x_i$$

---

##### 1.2.9 `para_merge_cmplx(dat, dat_size)`

**Purpose**: Global reduction sum (complex array)

**Interface**:
```fortran
SUBROUTINE para_merge_cmplx(dat, dat_size)
    complex(dp), intent(inout) :: dat(*)
    integer, intent(in) :: dat_size
```

---

##### 1.2.10 `para_collect_cmplx(fulldat, dat, blk_size)`

**Purpose**: Collect data from all processes to node 0

**Interface**:
```fortran
SUBROUTINE para_collect_cmplx(fulldat, dat, blk_size)
    complex(dp), intent(out) :: fulldat(*)
    complex(dp), intent(in) :: dat(*)
    integer, intent(in) :: blk_size
```

**Algorithm**:
```
blk_cmplx = MPI_CONTIGUOUS(blk_size, MPI_DOUBLE_COMPLEX)
mpi_type_commit(blk_cmplx)
mpi_gatherv(dat, map[inode+1, 2], blk_cmplx,
            fulldat, map(:, 2), map(:, 1),
            blk_cmplx, 0, MPI_COMM_WORLD)
mpi_type_free(blk_cmplx)
```

---

##### 1.2.11 `para_distribute_cmplx(fulldat, dat, blk_size)`

**Purpose**: Distribute data from node 0 to all processes

**Interface**:
```fortran
SUBROUTINE para_distribute_cmplx(fulldat, dat, blk_size)
    complex(dp), intent(in) :: fulldat(*)
    complex(dp), intent(out) :: dat(*)
    integer, intent(in) :: blk_size
```

---

### 1.3 wanndata — Wannier Hamiltonian Data Structure

**File**: [`modules/wanndata.f90`](wannchi/modules/wanndata.f90) (321 lines)

**Dependencies**: `constants`, `para`

#### Type Definition

##### 1.3.1 TYPE wannham

```fortran
TYPE wannham
    INTEGER :: norb           ! Number of orbitals
    REAL(DP), ALLOCATABLE :: tau(:,:)  ! Orbital positions (3, norb), in direct lattice units
    INTEGER :: nrpt           ! Total number of R-space lattice points
    INTEGER :: r000           ! Index of (0,0,0) lattice point (on-site term)
    COMPLEX(DP), ALLOCATABLE :: hr(:,:,:)  ! R-space Hamiltonian (norb, norb, nrpt)
    REAL(DP), ALLOCATABLE :: weight(:)    ! Weights for each R lattice point (for Fourier transform)
    REAL(DP), ALLOCATABLE :: rvec(:,:)    ! R lattice point coordinates (3, nrpt)
END TYPE
```

#### Subroutine Details

##### 1.3.2 `read_ham(ham, seed)`

**Purpose**: Read Wannier Hamiltonian (R-space) from `{seed}_hr.dat`

**Interface**:
```fortran
SUBROUTINE read_ham(ham, seed)
    TYPE(wannham), intent(out) :: ham
    character(*), intent(in) :: seed
```

**Implementation**:
- Read file on master process (inode=0)
- Allocate `ham%hr(norb, norb, nrpt)` and other arrays
- Read weights and R vectors
- Read Hamiltonian matrix elements
- Sync data to all processes via `para_sync_*`

**Algorithm**:
```
if (inode == 0):
    open trim(seed)//"_hr.dat"
    read norb, nrpt
    allocate arrays
    read weight(1:nrpt)
    for irpt=1 to nrpt:
        for iorb=1 to norb:
            for jorb=1 to norb:
                read Rvec, i, j, Re[H], Im[H]
                if (R == (0,0,0)) r000 = irpt
                hr(j,i,irpt) = complex(Re, Im)
    close file
sync hr to all nodes
sync weight to all nodes
sync rvec to all nodes
sync r000 to all nodes
```

---

##### 1.3.3 `read_ham_dim(ham, seed)`

**Purpose**: Read only Hamiltonian dimension information (not full data)

**Interface**:
```fortran
SUBROUTINE read_ham_dim(ham, seed)
    TYPE(wannham), intent(out) :: ham
    character(*), intent(in) :: seed
```

---

##### 1.3.4 `wannham_shift_ef(ham, mu)`

**Purpose**: Shift on-site terms $H_{ii} \leftarrow H_{ii} - \mu$ (chemical potential shift)

**Interface**:
```fortran
SUBROUTINE wannham_shift_ef(ham, mu)
    TYPE(wannham), intent(inout) :: ham
    real(dp), intent(in) :: mu
```

**Formula**:
$$H_{ii}^{\text{shifted}} = H_{ii} - \mu$$

---

##### 1.3.5 `calc_hk(hk, ham, kvec)`

**Purpose**: R-space → k-space Fourier transform, computing $H(\mathbf{k})$

**Interface**:
```fortran
SUBROUTINE calc_hk(hk, ham, kvec)
    TYPE(wannham), intent(in) :: ham
    real(dp), dimension(3), intent(in) :: kvec  ! k-point coordinates (fractional)
    complex(dp), dimension(ham%norb, ham%norb), intent(out) :: hk
```

**Algorithm**:
```
hk = 0
! Compute orbital position phase factors
do io = 1 to norb:
    ktau = sum(kvec(:) * tau(:, io)) * 2*pi
    phase(io) = exp(i * ktau)

! Fourier transform
do ir = 1 to nrpt:
    rdotk = sum(kvec(:) * rvec(:, ir)) * 2*pi
    fact = exp(i * rdotk) / weight(ir)
    do io = 1 to norb:
        do jo = 1 to norb:
            hk(io, jo) += fact * conjg(phase(io)) * phase(jo) * hr(io, jo, ir)
```

**Formula**:
$$H_{ij}(\mathbf{k}) = \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} \frac{H_{ij}(\mathbf{R})}{\text{weight}(\mathbf{R})} e^{i\mathbf{k}\cdot(\tau_j - \tau_i)}$$

---

##### 1.3.6 `finalize_wann(ham, all)`

**Purpose**: Deallocate Hamiltonian arrays

**Interface**:
```fortran
SUBROUTINE finalize_wann(ham, all)
    TYPE(wannham), intent(inout) :: ham
    logical, intent(in) :: all
```

---

##### 1.3.7 `write_ham(ham, seed)`

**Purpose**: Write Hamiltonian back to file

**Interface**:
```fortran
SUBROUTINE write_ham(ham, seed)
    TYPE(wannham), intent(in) :: ham
    character(*), intent(in) :: seed
```

---

### 1.4 linalgwrap — BLAS/LAPACK Linear Algebra Wrapper

**File**: [`modules/linalgwrap.f90`](wannchi/modules/linalgwrap.f90) (156 lines)

**Dependencies**: `constants`

#### Interface Definitions

```fortran
interface invmat
    module procedure dinvmat, zinvmat
end interface

interface eigen
    module procedure heigen, geigen
end interface
```

#### Subroutine Details

##### 1.4.1 `dinvmat(xmat, ndim)`

**Purpose**: Real matrix inversion

**Interface**:
```fortran
subroutine dinvmat(xmat, ndim)
    integer, intent(in) :: ndim
    real(dp), dimension(ndim, ndim) :: xmat  ! Input/output matrix
```

**Implementation**: Uses LAPACK `dgetrf` + `dgetri`

**Algorithm**:
```
call dgetrf(ndim, ndim, xmat, ndim, ipiv, info)
call dgetri(ndim, xmat, ndim, ipiv, work, ndim, info)
```

---

##### 1.4.2 `zinvmat(xmat, ndim)`

**Purpose**: Complex matrix inversion

**Interface**:
```fortran
subroutine zinvmat(xmat, ndim)
    integer, intent(in) :: ndim
    complex(dp), dimension(ndim, ndim) :: xmat  ! Input/output matrix
```

**Implementation**: Uses LAPACK `zgetrf` + `zgetri`

---

##### 1.4.3 `heigen(eig, xmat, ndim)`

**Purpose**: Hermitian matrix eigenvalue problem

**Interface**:
```fortran
subroutine heigen(eig, xmat, ndim)
    integer, intent(in) :: ndim
    complex(dp), dimension(ndim, ndim) :: xmat  ! Input matrix; output eigenvectors
    real(dp), dimension(ndim), intent(out) :: eig  ! Eigenvalues
```

**Implementation**: Uses LAPACK `zheev`

**Formula**: Solve $H\psi = \lambda\psi$, where $H = H^\dagger$

---

##### 1.4.4 `geigen(eig, xmat, ndim)`

**Purpose**: General complex matrix eigenvalue problem (right eigenvectors)

**Interface**:
```fortran
subroutine geigen(eig, xmat, ndim)
    integer, intent(in) :: ndim
    complex(dp), dimension(ndim, ndim) :: xmat  ! Input matrix; output right eigenvectors
    complex(dp), dimension(ndim), intent(out) :: eig  ! Eigenvalues
```

**Implementation**: Uses LAPACK `zgeev`

---

##### 1.4.5 `sparsemulmat(zmat, xmat_cp, ymat, idxcp, ndim, nidxcp, alpha, beta)`

**Purpose**: Sparse matrix multiplication $z = \alpha \cdot x_{\text{cp}} \cdot y + \beta \cdot z$

**Algorithm**:
```
z = beta * z
do ii = 1 to nidxcp:
    i1 = idxcp(1, ii)
    i2 = idxcp(2, ii)
    do jj = 1 to ndim:
        z(i1, jj) += alpha * xmat_cp(ii) * ymat(i2, jj)
```

---

##### 1.4.6 `matmulsparse(zmat, xmat, ymat_cp, idxcp, ndim, nidxcp, alpha, beta)`

**Purpose**: Sparse matrix multiplication $z = \alpha \cdot x \cdot y_{\text{cp}} + \beta \cdot z$

---

### 1.5 symmetry_module — Angular Momentum and Symmetry Operations

**File**: [`modules/symmetry.f90`](wannchi/modules/symmetry.f90) (506 lines)

**Dependencies**: `constants`, `linalgwrap`

#### Type Definition

```fortran
TYPE symmetry
    real(dp), dimension(3,3) :: rot    ! Rotation matrix in direct lattice space
    real(dp), dimension(3) :: tau       ! Translation vector
    real(dp), dimension(3) :: axis      ! Rotation axis (Cartesian)
    real(dp) :: theta                   ! Rotation angle
    logical :: inv                      ! Whether inversion is included
END TYPE
```

#### Subroutine Details

##### 1.5.1 `generate_Smatrix(Sx, Sy, Sz)`

**Purpose**: Generate Pauli spin matrices

**Interface**:
```fortran
SUBROUTINE generate_Smatrix(Sx, Sy, Sz)
    complex(dp), dimension(2, 2), intent(out) :: Sx, Sy, Sz
```

**Formula**:
$$S_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
S_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
S_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

---

##### 1.5.2 `generate_Lmatrix(Lx, Ly, Lz, l)`

**Purpose**: Generate angular momentum operator matrices $L_x, L_y, L_z$ in Ylm basis

**Interface**:
```fortran
SUBROUTINE generate_Lmatrix(Lx, Ly, Lz, l)
    integer, intent(in) :: l
    complex(dp), dimension(2*l+1, 2*l+1), intent(out) :: Lx, Ly, Lz
```

**Implementation**:
- First construct $L_+$ and $L_-$ matrices
- Compute $L_x = (L_+ + L_-)/2$, $L_y = (L_+ - L_-)/(2i)$

**Algorithm**:
```
do m = -l to l:
    Lz(m+l+1, m+l+1) = m
    if (m < l):
        Lp(m+l+2, m+l+1) = sqrt((l-m)*(l+m+1))
        Lm(m+l+1, m+l+2) = sqrt((l+m+1)*(l-m))
Lx = (Lp + Lm) / 2
Ly = (Lp - Lm) / (2*i)
```

**Formula**:
$$L_z |l,m\rangle = m |l,m\rangle$$
$$L_+ |l,m\rangle = \sqrt{(l-m)(l+m+1)} |l,m+1\rangle$$
$$L_- |l,m\rangle = \sqrt{(l+m)(l-m+1)} |l,m-1\rangle$$

---

##### 1.5.3 `generate_Ylm2C(Umat, l)`

**Purpose**: Generate Ylm → Cubic Harmonics transformation matrix

**Interface**:
```fortran
SUBROUTINE generate_Ylm2C(Umat, l)
    integer, intent(in) :: l
    complex(dp), dimension(2*l+1, 2*l+1), intent(out) :: Umat
```

---

##### 1.5.4 `rotate_Ylm(rot, l, symm)`

**Purpose**: Ylm basis rotation $\exp(-i\theta \hat{L}\cdot\hat{n})$

**Interface**:
```fortran
SUBROUTINE rotate_Ylm(rot, l, symm)
    complex(dp), dimension(2*l+1, 2*l+1), intent(out) :: rot
    integer, intent(in) :: l
    TYPE(symmetry), intent(in) :: symm
```

**Implementation**:
- If rotation angle is very small (theta < eps4), return identity
- Construct $L \cdot \hat{n} = L_x n_x + L_y n_y + L_z n_z$
- Solve eigenvalue problem: $U^{-1} (L\cdot\hat{n}) U = \Lambda$
- Compute rotation matrix: $R = U e^{-i\theta\Lambda} U^{-1}$

**Formula**:
$$R_{Ylm} = \exp(-i\theta \hat{L} \cdot \hat{n})$$

---

##### 1.5.5 `rotate_spinor(rot, symm)`

**Purpose**: Spinor rotation $\exp(-i\theta \hat{\sigma}\cdot\hat{n}/2)$

**Interface**:
```fortran
SUBROUTINE rotate_spinor(rot, symm)
    complex(dp), dimension(2, 2), intent(out) :: rot
    TYPE(symmetry), intent(in) :: symm
```

**Formula**:
$$R_{\text{spin}} = \exp(-i\theta \hat{\sigma} \cdot \hat{n}/2)$$

---

##### 1.5.6 `rotate_cubic(rot, l, symm)`

**Purpose**: Cubic Harmonics basis rotation

**Interface**:
```fortran
SUBROUTINE rotate_cubic(rot, l, symm)
    complex(dp), dimension(2*l+1, 2*l+1), intent(out) :: rot
    integer, intent(in) :: l
    TYPE(symmetry), intent(in) :: symm
```

**Formula**:
$$R_{\text{cubic}} = (Ylm2C)^{-1} \cdot R_{Ylm} \cdot Ylm2C$$

---

### 1.6 simp_module — Impurity Simplification and Self-Energy Packing

**File**: [`modules/simp.f90`](wannchi/modules/simp.f90) (114 lines)

**Dependencies**: `constants`, `lattice`, `linalgwrap`

#### Type Definition

```fortran
TYPE simp
    integer :: ndim              ! Impurity dimension (= 2*(2l+1) for spinor)
    integer :: l                 ! Angular momentum quantum number
    integer, allocatable :: gidx(:)  ! Global orbital indices
    integer, allocatable :: sigidx(:,:)  ! Self-energy index mapping
    real(dp), allocatable :: locrot(:,:)  ! Local rotation matrix
END TYPE
```

#### Subroutine Details

##### 1.6.1 `init_simp(simpvar, ndim, l)`

**Purpose**: Initialize `simp` type

**Interface**:
```fortran
SUBROUTINE init_simp(simpvar, ndim, l)
    TYPE(simp), intent(out) :: simpvar
    integer, intent(in) :: ndim, l
```

---

##### 1.6.2 `pack_sigma(sigpack, sigfull)`

**Purpose**: Pack full self-energy matrix into impurity-block form

**Interface**:
```fortran
SUBROUTINE pack_sigma(sigpack, sigfull)
    complex(dp), intent(out) :: sigpack(ndim, nw)
    complex(dp), intent(in) :: sigfull(norb, norb)
```

**Implementation**:
- Extract diagonal blocks corresponding to impurity orbitals
- Apply local rotation if needed

---

##### 1.6.3 `restore_sigma(sigfull, sigpack)`

**Purpose**: Restore packed self-energy to full orbital space

**Interface**:
```fortran
SUBROUTINE restore_sigma(sigfull, sigpack)
    complex(dp), intent(out) :: sigfull(norb, norb)
    complex(dp), intent(in) :: sigpack(ndim, nw)
```

---

### 1.7 lattice — Lattice Structure, k-mesh, and Self-Energy Interpolation

**File**: [`modules/lattice.f90`](wannchi/modules/lattice.f90) (1073 lines)

**Dependencies**: `constants`, `wanndata`, `simp_module`, `para`, `linalgwrap`, `symmetry_module`

#### Global Variables

```fortran
real(dp), dimension(3,3) :: avec   ! Lattice vectors (a1=avec(:,1), a2=avec(:,2), a3=avec(:,3))
real(dp), dimension(3,3) :: bvec   ! Reciprocal lattice vectors (b1=bvec(1,:), b2=bvec(2,:), b3=bvec(3,:))
logical :: spinor                  ! Whether spinor (with SOC) basis
integer :: nsite                   ! Number of atomic positions
integer, allocatable :: zat(:)     ! Atomic numbers (nsite)
real(dp), allocatable :: xat(:,:)  ! Atomic fractional coordinates (3, nsite)
integer, allocatable :: nbasis(:)  ! Number of Wannier orbitals at each atom (nsite)
real(dp) :: nelec                  ! Total number of valence electrons

TYPE(wannham) :: ham              ! Wannier Hamiltonian
integer :: nimp                   ! Number of impurities
TYPE(simp), allocatable :: imp(:) ! Definition of each impurity
integer :: nbath                  ! Bath dimension in self-energy packing form (= ndimf, total impurity orbitals)
real(dp) :: beta                   ! Inverse temperature (passed from input module to lattice)
integer :: nw                      ! Number of self-energy frequency points

integer :: ndimf, ndimc           ! F/C subspace dimensions
integer, allocatable :: f2g_idx(:), c2g_idx(:)  ! F/C → global index
integer, allocatable :: g2f_idx(:), g2c_idx(:)  ! Global → F/C index
integer, allocatable :: partition(:)  ! Orbital partition (0=bath/C, >0=impurity index)

integer :: nk1, nk2, nk3          ! BZ mesh density
integer :: nkirr                   ! Number of irreducible k-points
real(dp), allocatable :: kvec(:,:) ! k-point coordinates (3, nkirr), fractional
real(dp), allocatable :: kwt(:)   ! k-point weights

complex(dp), allocatable :: omega(:) ! Frequency grid
complex(dp), allocatable :: sinf(:)  ! Σ(∞), high-frequency self-energy limit
complex(dp), allocatable :: sigpack(:,:)  ! Σ(ω) in packed form (nbath, nw)
```

#### Subroutine Details

##### 1.7.1 `read_posfile(seed)`

**Purpose**: Read `.pos` file (POSCAR-like format)

**Interface**:
```fortran
SUBROUTINE read_posfile(seed)
    character(*), intent(in) :: seed
```

**File format**:
```
Kagome                  ! Comment line
1.0                     ! Scaling factor
4.6669313530311989   -2.6944540729627522    0.0    ! a1
4.6669313530311989    2.6944540729627526    0.0    ! a2
0.0    0.0    9.8872122322434954           ! a3
3  0                      ! nsite, soc (0/1)
1  0.5  0.0  0.0  1      ! Zat, x, y, z, nbasis
1  0.0  0.5  0.0  1
1  0.5  0.5  0.0  1
```

**Formula**:
$$b_i \cdot a_j = 2\pi \delta_{ij}$$

---

##### 1.7.2 `read_kmesh(filename)`

**Purpose**: Read IBZKPT file

**Interface**:
```fortran
SUBROUTINE read_kmesh(filename)
    character(*), intent(in) :: filename
```

**File format**:
```
Automatically generated mesh
0           ! switch (0 = automatic gamma-centered mesh)
Reciprocal lattice
nk1 nk2 nk3  ! k-mesh density
```

---

##### 1.7.3 `read_impfile(seed)`

**Purpose**: Read impurity definition file

**Interface**:
```fortran
SUBROUTINE read_impfile(seed)
    character(*), intent(in) :: seed
```

---

##### 1.7.4 `setup_mapping()`

**Purpose**: Build F/C partition mapping

**Interface**:
```fortran
SUBROUTINE setup_mapping()
```

**Algorithm**:
```
partition = 0
do ii = 1 to nimp:
    partition(imp(ii)%gidx(:)) = ii

ndimf = sum(imp(ii)%ndim)
ndimc = norb - ndimf

allocate(f2g_idx(ndimf), c2g_idx(ndimc), g2f_idx(norb), g2c_idx(norb))

jj = 1
kk = 1
do ii = 1 to norb:
    if (partition(ii) > 0):
        f2g_idx(jj) = ii
        g2f_idx(ii) = jj
        jj = jj + 1
    else:
        c2g_idx(kk) = ii
        g2c_idx(ii) = kk
        kk = kk + 1
```

---

##### 1.7.5 `fix_sigma_static()`

**Purpose**: Add Σ(∞) to on-site Hamiltonian

**Interface**:
```fortran
SUBROUTINE fix_sigma_static()
```

---

##### 1.7.6 `get_sigma_matrix(sigfull, z)`

**Purpose**: Get full self-energy matrix at complex frequency z

**Interface**:
```fortran
SUBROUTINE get_sigma_matrix(sigfull, z)
    complex(dp), dimension(norb, norb), intent(out) :: sigfull
    complex(dp), intent(in) :: z
```

---

##### 1.7.7 `interpolate_single_sigma(sigval, w)`

**Purpose**: Self-energy interpolation (single frequency point)

**Interface**:
```fortran
SUBROUTINE interpolate_single_sigma(sigval, w)
    complex(dp), intent(out) :: sigval
    complex(dp), intent(in) :: w
```

**Formula**:
- **Matsubara frequencies**: $\omega_n = i\frac{2\pi(n-1)}{\beta}$
- **High-frequency tail**: $\Sigma(\omega) \approx \Sigma_\infty + \frac{A}{\omega} + \frac{B}{\omega^2}$

---

### 1.8 IntRPA — RPA FF/CC Block Structure

**File**: [`modules/intRPA.f90`](wannchi/modules/intRPA.f90) (242 lines)

**Dependencies**: `constants`, `lattice`, `para`, `linalgwrap`

#### Global Variables

```fortran
integer :: nFFidx     ! FF block total dimension = Σ blkdim²
integer :: nCCidx     ! CC block dimension = norb - Σ blkdim
integer, dimension(2, nFFidx) :: FFidx  ! FF orbital pairs (i1=FFidx(1,:), i2=FFidx(2,:))
integer, dimension(nCCidx) :: CCidx     ! CC global orbital indices
integer :: nUcp       ! Number of non-zero U matrix elements
real(dp), dimension(nUcp) :: Uint_cp    ! U matrix elements (compressed form)
integer, dimension(2, nUcp) :: idxUcp  ! (i,j) indices of non-zero elements in full matrix
```

#### Subroutine Details

##### 1.8.1 `read_RPA()`

**Purpose**: Read RPA.inp, build FFidx, CCidx, Uint_cp

**Algorithm**:
```
read nffblk
read blkdim(1:nffblk)

nFFidx = sum(blkdim(i)^2)
nCCidx = norb - sum(blkdim(i))

mapping = 0
do ii = 1 to nffblk:
    read blkidx(1:blkdim(ii))
    mapping(blkidx) = ii

    ! Build FFidx
    jj = 1
    do j1 = 1 to blkdim(ii):
        do j2 = 1 to blkdim(ii):
            FFidx(1, jj) = blkidx(j1)
            FFidx(2, jj) = blkidx(j2)
            jj = jj + 1

! Build CCidx
jj = 1
do ii = 1 to norb:
    if (mapping(ii) == 0):
        CCidx(jj) = ii
        jj = jj + 1

! Read U matrix
read nUcp
do ii = 1 to nUcp:
    read i1, i2, j1, j2, Uij
    idxUcp(1, ii) = i1
    idxUcp(2, ii) = i2
    Uint_cp(ii) = Uij
```

---

##### 1.8.2 `find_ffidx(ii, i1, i2)`

**Purpose**: Given orbital pair (i1, i2), find index in FFidx

**Interface**:
```fortran
SUBROUTINE find_ffidx(ii, i1, i2)
    integer, intent(out) :: ii
    integer, intent(in) :: i1, i2
```

---

##### 1.8.3 `calc_chiRPA(chiff, chicc, chifc, chicf, chi0ff, chi0cc, chi0fc, chi0cf, ff_only, nw)`

**Purpose**: Core RPA equation solver

**Interface**:
```fortran
SUBROUTINE calc_chiRPA(chiff, chicc, chifc, chicf, chi0ff, chi0cc, chi0fc, chi0cf, ff_only, nw)
    complex(dp), dimension(nFFidx, nFFidx, nw), intent(out) :: chiff
    complex(dp), dimension(nCCidx, nCCidx, nw), intent(out) :: chicc
    complex(dp), dimension(nFFidx, nCCidx, nw), intent(out) :: chifc
    complex(dp), dimension(nCCidx, nFFidx, nw), intent(out) :: chicf
    complex(dp), dimension(nFFidx, nFFidx, nw), intent(in) :: chi0ff
    complex(dp), dimension(nCCidx, nCCidx, nw), intent(in) :: chi0cc
    complex(dp), dimension(nFFidx, nCCidx, nw), intent(in) :: chi0fc
    complex(dp), dimension(nCCidx, nFFidx, nw), intent(in) :: chi0cf
    logical, intent(in) :: ff_only
    integer, intent(in) :: nw
```

**Implementation**:
1. Compute Dyson factor $D_{FF} = (1 - \chi^0_{FF} \cdot U_{FF})^{-1}$
2. Compute $V_{FF} = U_{FF} \cdot D_{FF}$
3. Compute RPA response function blocks

**Algorithm**:
```
! Compute Dff = (1 - chi0ff * Uff)^(-1)
call matmulsparse(Dff, chi0ff, Uint_cp, idxUcp, nFFidx, nUcp, -1.d0, 1.d0)
call invmat(Dff, nFFidx)

! Compute Vff = Uff * Dff
call sparsemulmat(Vff, Uint_cp, Dff, idxUcp, nFFidx, nUcp, 1.d0, 0.d0)

! Compute response function blocks
! chiFF = DFF * chi0FF
call zgemm('N', 'N', nFFidx, nFFidx, nFFidx, 1, Dff, nFFidx, chi0ff, nFFidx, 0, chiff, nFFidx)

! If not ff_only:
if (.not. ff_only):
    ! tmpCF = chi0CF * VFF
    call zgemm('N', 'N', nCCidx, nFFidx, nFFidx, 1, chi0cf, nCCidx, Vff, nFFidx, 0, tmpcf, nCCidx)

    ! chiFC = DFF * chi0FC
    call zgemm('N', 'N', nFFidx, nCCidx, nFFidx, 1, Dff, nFFidx, chi0fc, nFFidx, 0, chifc, nFFidx)

    ! chiCF = chi0CF + tmpCF * chi0FF
    call zgemm('N', 'N', nCCidx, nFFidx, nFFidx, 1, tmpcf, nCCidx, chi0ff, nFFidx, 1, chicf, nCCidx)

    ! chiCC = chi0CC + tmpCF * chi0FC
    call zgemm('N', 'N', nCCidx, nCCidx, nFFidx, 1, tmpcf, nCCidx, chi0fc, nFFidx, 1, chicc, nCCidx)
```

**Formula**:
$$\chi = \chi^0 + \chi^0 U \chi$$

i.e.:
$$\chi = (1 - \chi^0 U)^{-1} \chi^0$$

In block matrix form:
$$\begin{pmatrix} \chi_{FF} & \chi_{FC} \\ \chi_{CF} & \chi_{CC} \end{pmatrix} = \begin{pmatrix} D_{FF} & 0 \\ -\chi_{CF}^0 U_{FC} D_{FF} & 1 \end{pmatrix} \begin{pmatrix} \chi_{FF}^0 & \chi_{FC}^0 \\ \chi_{CF}^0 & \chi_{CC}^0 \end{pmatrix}$$

where $D_{FF} = (1 - \chi_{FF}^0 U_{FF})^{-1}$

---

##### 1.8.4 `finalize_RPA()`

**Purpose**: Deallocate RPA arrays

**Interface**:
```fortran
SUBROUTINE finalize_RPA()
```

---

### 1.9 pade — Padé Summation

**File**: [`modules/pade.f90`](wannchi/modules/pade.f90) (112 lines)

#### Subroutine Details

##### 1.9.1 `init_pade(npts, z, f)`

**Purpose**: Initialize Padé approximation from data points

**Interface**:
```fortran
SUBROUTINE init_pade(npts, z, f)
    integer, intent(in) :: npts
    complex(dp), intent(in) :: z(npts), f(npts)
```

---

##### 1.9.2 `eval_pade(z_val, f_val)`

**Purpose**: Evaluate Padé approximant at a given point

**Interface**:
```fortran
SUBROUTINE eval_pade(z_val, f_val)
    complex(dp), intent(in) :: z_val
    complex(dp), intent(out) :: f_val
```

---

##### 1.9.3 `finalize_pade()`

**Purpose**: Deallocate Padé arrays

**Interface**:
```fortran
SUBROUTINE finalize_pade()
```

---

### 1.10 transp_calc — Transport Property Calculations

**File**: [`modules/transp_calc.f90`](wannchi/modules/transp_calc.f90) (349 lines)

**Dependencies**: `constants`, `wanndata`, `linalgwrap`

**Function**: Calculate transport properties from Wannier Hamiltonian:
- Anomalous Hall conductivity (AHC, σ_xy) via Berry curvature
- Longitudinal DC conductivity (σ_xx) via Kubo-Greenwood formula
- Velocity matrices (analytically derived from HR, no numerical k-derivatives)

#### Subroutine Details

##### 1.10.1 `calc_velocity(v_alpha, ham, kvec, alpha)`

**Purpose**: Analytically compute velocity matrix $v_\alpha(\mathbf{k})$

**Interface**:
```fortran
SUBROUTINE calc_velocity(v_alpha, ham, kvec, alpha)
  TYPE(wannham), intent(in) :: ham
  real(dp), dimension(3), intent(in) :: kvec
  integer, intent(in) :: alpha   ! 1=x, 2=y, 3=z
  complex(dp), dimension(ham%norb, ham%norb), intent(out) :: v_alpha
```

**Formula**:
$$v_\alpha(\mathbf{k}) = \sum_{\mathbf{R}} i \cdot 2\pi \cdot \tilde{R}_\alpha \cdot \frac{e^{i\mathbf{k}\cdot\mathbf{R}}}{w(\mathbf{R})} \cdot e^{i\mathbf{k}\cdot(\tau_j - \tau_i)} \cdot H_{ij}(\mathbf{R})$$

where $\tilde{R}_\alpha = R_\alpha + \tau_{j,\alpha} - \tau_{i,\alpha}$

**Algorithm**:
1. Compute orbital phase factors: $\text{phase}(io) = \exp(i \cdot 2\pi \cdot \mathbf{k} \cdot \tau_{io})$
2. For each R lattice: $\text{fact} = \exp(i \cdot 2\pi \cdot \mathbf{k} \cdot \mathbf{R}) / w(\mathbf{R})$
3. For each orbital pair $(io, jo)$: $v_\alpha(io,jo) += i \cdot 2\pi \cdot \tilde{R}_\alpha \cdot \text{fact} \cdot \text{conj}(\text{phase}(io)) \cdot \text{phase}(jo) \cdot H_{ij}(\mathbf{R})$

---

##### 1.10.2 `calc_berry_curvature(omega_n, eigvec, vx, vy, eig, norb)`

**Purpose**: Compute Berry curvature from eigenstates and velocity matrices

**Interface**:
```fortran
SUBROUTINE calc_berry_curvature(omega_n, eigvec, vx, vy, eig, norb)
```

**Formula** (for nth band at k-point):
$$\Omega_n(\mathbf{k}) = -\sum_{m \neq n} \frac{2 \cdot \text{Im}[\langle u_n | v_x | u_m \rangle \langle u_m | v_y | u_n \rangle]}{[E_m - E_n]^2 + \eta^2}$$

---

##### 1.10.3 `calc_sigma_xy(ham, nk1, nk2, nk3, Ef, T, sigma_xy_out)`

**Purpose**: Compute anomalous Hall conductivity (σ_xy) via Berry curvature sum

**Interface**:
```fortran
SUBROUTINE calc_sigma_xy(ham, nk1, nk2, nk3, Ef, T, sigma_xy_out)
```

**Formula**:
$$\sigma_{xy} = \frac{1}{N_k} \sum_{\mathbf{k}, n} \Omega_n(\mathbf{k}) \cdot f(E_n(\mathbf{k}) - E_F)$$

Units: $e^2/h$ (dimensionless for 2D; multiply by $1/c_z$ for 3D)

---

##### 1.10.4 `calc_sigma_xx(ham, nk1, nk2, nk3, Ef, T, eta, sigma_xx_out)`

**Purpose**: Compute longitudinal DC conductivity via Kubo-Greenwood formula

**Interface**:
```fortran
SUBROUTINE calc_sigma_xx(ham, nk1, nk2, nk3, Ef, T, eta, sigma_xx_out)
  real(dp), intent(in) :: eta  ! Lorentzian broadening (eV)
```

**Formula**:
$$\sigma_{xx} = \frac{1}{N_k} \sum_{\mathbf{k}} \sum_n \sum_m (f_n - f_m) \cdot \frac{v_{x,nm} v_{x,mn} \cdot \eta^2}{[E_{nm}]^2 + \eta^2}$$

where $E_{nm} = E_n - E_m$, $f_n = f(E_n - E_F)$, $v_{x,nm} = \langle u_n | v_x | u_m \rangle$.

In terms of Green's functions:
$$\sigma_{xx} = \frac{1}{N_k} \sum_{\mathbf{k}} \text{Tr}[v_x \cdot G(\mathbf{k}, \mu+i\eta) \cdot v_x \cdot G(\mathbf{k}, \mu-i\eta)]$$

---

### 1.11 gp_bo — Gaussian Process Bayesian Optimization

**File**: [`modules/gp_bo.f90`](wannchi/modules/gp_bo.f90) (704 lines)

**Purpose**: Bayesian Optimization using Gaussian Processes with RBF kernel and Expected Improvement acquisition

#### 1.11.1 TYPE gp_model

```fortran
TYPE gp_model
  integer :: n_train           ! Number of observations so far
  integer :: n_params          ! Dimension of parameter space
  real(dp), allocatable :: x_train(:,:)  ! (n_params, n_train)
  real(dp), allocatable :: y_train(:)    ! Objective values (n_train)
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

#### 1.11.2 gp_init

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

#### 1.11.3 gp_finalize

```fortran
SUBROUTINE gp_finalize(gp)
  TYPE(gp_model), intent(inout) :: gp
```

#### 1.11.4 rbf_kernel

```fortran
FUNCTION rbf_kernel(x1, x2, ls, sig2) result(k)
  real(dp), dimension(:), intent(in) :: x1, x2
  real(dp), intent(in) :: ls, sig2
  real(dp) :: k
```

**Formula**:
$$k(\mathbf{x}_1, \mathbf{x}_2) = \sigma_f^2 \cdot \exp\left(-\frac{\|\mathbf{x}_1 - \mathbf{x}_2\|^2}{2l^2}\right)$$

#### 1.11.5 gp_update

```fortran
SUBROUTINE gp_update(gp, x_new, y_new)
  TYPE(gp_model), intent(inout) :: gp
  real(dp), dimension(gp%n_params), intent(in) :: x_new
  real(dp), intent(in) :: y_new
```

**Action**: Append (x_new, y_new) to training set, recompute K_inv and alpha via Cholesky (LAPACK `zpotrf`/`zpotri`).

**NaN check**: After `invmat`, checks if any element of K_inv is NaN. If so, reverts n_train and returns immediately — the last observation is discarded.

#### 1.11.6 gp_predict

```fortran
SUBROUTINE gp_predict(gp, x_pred, mu, sigma)
  TYPE(gp_model), intent(in) :: gp
  real(dp), dimension(gp%n_params), intent(in) :: x_pred
  real(dp), intent(out) :: mu, sigma
```

**Formulas**:
$$\mu(\mathbf{x}^*) = \mathbf{k}^T K^{-1} \mathbf{y}$$
$$\sigma^2(\mathbf{x}^*) = k(\mathbf{x}^*, \mathbf{x}^*) - \mathbf{k}^T K^{-1} \mathbf{k}$$

#### 1.11.7 expected_improvement

```fortran
FUNCTION expected_improvement(mu, sigma, y_best, xi) result(ei)
  real(dp), intent(in) :: mu, sigma, y_best, xi
  real(dp) :: ei
```

**Purpose**: Compute Expected Improvement acquisition function for minimization.

**Formula** (for minimization with noise-free observations):
$$\text{EI}(\mathbf{x}) = \begin{cases}
(\mu - y_{\text{best}} - \xi) \cdot \Phi\left(\frac{\mu - y_{\text{best}} - \xi}{\sigma}\right) + \sigma \cdot \phi\left(\frac{\mu - y_{\text{best}} - \xi}{\sigma}\right) & \sigma > 0 \\
0 & \sigma = 0
\end{cases}$$

where $\Phi$ is CDF and $\phi$ is PDF of standard normal. Default $\xi = 0.01$ (exploration parameter).

#### 1.11.8 latin_hypercube

```fortran
SUBROUTINE latin_hypercube(x_samp, lb, ub)
  real(dp), intent(out) :: x_samp(:,:)  ! (n_params, n_samp)
  real(dp), intent(in) :: lb(:), ub(:)  ! (n_params)
```

**Purpose**: Generate samples using Latin Hypercube Sampling (LHS) for initial design.

#### 1.11.9 gp_optimize_ls

```fortran
SUBROUTINE gp_optimize_ls(gp, n_restarts)
  TYPE(gp_model), intent(inout) :: gp
  integer, intent(in) :: n_restarts
```

**Purpose**: Optimize GP hyperparameter (length scale) by maximizing log marginal likelihood.

**Formula**:
$$\log p(\mathbf{y} | X, \sigma_f, l) = -\frac{1}{2}\mathbf{y}^T K^{-1}\mathbf{y} - \frac{1}{2}\log|K| - \frac{n}{2}\log(2\pi)$$

Uses L-BFGS-B (via `optimize_lbfgsb` from `scipy` in Python; in Fortran, simple gradient-free restart is used).

#### 1.11.10 bayesian_optimize

```fortran
SUBROUTINE bayesian_optimize(gp, objective, lb, ub, n_iter, x_opt, f_opt, verbose)
  TYPE(gp_model), intent(inout) :: gp
  real(dp), intent(in) :: lb(:), ub(:)
  integer, intent(in) :: n_iter
  real(dp), intent(out) :: x_opt(:), f_opt
  logical, intent(in), optional :: verbose
```

**Multi-start EI refinement algorithm**:
1. Generate `n_cand = max(2000, 50*n_params)` candidates via Latin hypercube in bounds
2. Compute EI at each candidate using `gp_predict`
3. Pick top 10 candidates with highest EI
4. For each, run gradient-based local refinement (L-BFGS-B) on the EI surface
5. Return the point with highest EI as new observation
6. `gp_update` with (x_new, f_new)
7. Re-optimize length scale every 10 iterations if n_params > 10

**Key parameters**:
- `n_cand`: `max(2000, 50*n_params)` — may be too sparse for high-dimensional problems
- Noise variance: set to `1e-2` for eigenvalue-based objective functions
- Length scale: re-optimized every 10 iterations for n_params > 10

---

### 1.12 cma_es — CMA-ES Optimizer

**File**: [`modules/cma_es.f90`](wannchi/modules/cma_es.f90) (233 lines)

**Purpose**: Covariance Matrix Adaptation Evolution Strategy — gradient-free evolutionary optimizer for high-dimensional, non-convex, multi-modal optimization.

#### 1.12.1 Algorithm Overview

CMA-ES maintains a multivariate normal distribution over the parameter space:
- **Mean**: $\mathbf{m} \in \mathbb{R}^{n_{\text{params}}}$
- **Covariance**: $C \in \mathbb{R}^{n_{\text{params}} \times n_{\text{params}}}$ (initially axis-aligned, $\sigma^2 I$)
- **Step size**: $\sigma$ (global "mutation" strength)

At each generation (iteration):
1. Sample $\lambda$ candidate points: $\mathbf{x}_i \sim \mathcal{N}(\mathbf{m}, \sigma^2 C)$
2. Evaluate objective at each candidate
3. Update mean via weighted average of top $\mu$ candidates (elite recombination)
4. Update covariance via evolution paths (cumulative step adaptation)
5. Adjust step size based on evolution path length

#### 1.12.2 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `n_params` | — | Problem dimension |
| `lambda` | `4 + floor(3*log(n_params))` | Population size |
| `mu` | `floor(lambda/2)` | Number of parents (elite count) |
| `mu_eff` | — | Effective selection mass |
| `sigma` | `0.5 * range` | Initial step size |
| `c_cum` | `4/(n+4)` | Cumulation rate for evolution path |
| `c_cov` | `1/(10*n + 25)` | Covariance adaptation rate |
| `d_sigma` | `1 + sqrt(n/lambda)` | Step size damping |

#### 1.12.3 Subroutine Details

##### `cmaes_optimize(objective, n_params, lb, ub, x_opt, f_opt, n_iter, verbose)`

**Interface**:
```fortran
SUBROUTINE cmaes_optimize(objective, n_params, lb, ub, x_opt, f_opt, n_iter, verbose)
  integer, intent(in) :: n_params
  real(dp), intent(in) :: lb(n_params), ub(n_params)
  real(dp), intent(out) :: x_opt(n_params), f_opt
  integer, intent(in) :: n_iter
  logical, intent(in), optional :: verbose
```

**Algorithm** (axis-parallel variant):
```
Initialize m = (lb+ub)/2, sigma = 0.5*(ub-lb), C = I
Initialize evolution paths pc = 0, ps = 0
for gen = 1 to n_iter:
    ! Sample lambda candidates
    for i = 1 to lambda:
        z = randn(n_params)          ! Sample from N(0, I)
        x_i = m + sigma * z          ! Affine transformation
        f_i = objective(x_i)         ! Evaluate

    ! Sort and select top mu
    sort by f_i, pick top mu
    m_old = m
    m = sum(w_i * x_i) / sum(w_i)   ! Weighted mean

    ! Evolution paths (cumulation)
    pc = (1-c_cum) * pc + sqrt(mu_eff * c_cum*(2-c_cum)) * (m - m_old) / sigma
    ps = (1-c_cum) * ps + sqrt(mu_eff * c_cum*(2-c_cum)) * C^{-1} * (m - m_old) / sigma

    ! Covariance update
    C = (1-c_cov) * C + c_cov * outer_diag_sum(pc) + c_cov * (1-1/mu_eff) * 2/(n+2) * ...

    ! Step size update
    sigma = sigma * exp((c_sigma/d_sigma) * (norm(ps)/E[norm(N(0,I))] - 1))

    ! Adjust to bounds
    m = max(lb, min(ub, m))
```

##### `outer_product` / `outer_diag_sum`

Utility functions for covariance matrix operations.

**Applicability**: Best for high-dimensional problems (n_params >= 20), non-convex multi-modal functions. NOT suitable for very low dimensional (< 10) — use GP-BO instead.

---

### 1.13 classical_mc — Classical Heisenberg Monte Carlo

**File**: [`modules/classical_mc.f90`](wannchi/modules/classical_mc.f90) (438 lines)

**Purpose**: Classical Heisenberg spin simulation via Metropolis algorithm on a supercell.

#### 1.13.1 TYPE mc_lattice

```fortran
TYPE mc_lattice
  integer :: n_sites               ! Total number of sites in supercell
  integer :: nx, ny, nz            ! Supercell dimensions
  real(dp), allocatable :: spin(:,:)  ! (3, n_sites) — spin vectors
  integer, allocatable :: neighbor_list(:,:)  ! (n_sites, max_neighbors)
  real(dp) :: J_mc                  ! Heisenberg exchange J
  real(dp) :: S_mag                  ! Measured magnetization magnitude
  real(dp), dimension(3) :: mvec   ! Measured magnetization vector
END TYPE
```

**Supercell auto-detection**:
- If |a_i| > 2·min(|a_j|, |a_k|): N_i = 1 (vacuum direction)
- Otherwise: N_i = 10 (standard thermodynamic limit)

#### 1.13.2 Subroutine Details

##### `mc_init(avec, nk, J_mc_in, mc_supercell_in)`

**Purpose**: Initialize supercell and allocate spin arrays

**Interface**:
```fortran
SUBROUTINE mc_init(avec, nk, J_mc_in, mc_supercell_in)
  real(dp), dimension(3,3), intent(in) :: avec
  integer, dimension(3), intent(in) :: nk
  real(dp), intent(in) :: J_mc_in
  integer, dimension(3), intent(in), optional :: mc_supercell_in
```

##### `mc_build_neighbors(avec, nk, nl)`

**Purpose**: Build neighbor list for Heisenberg spins

**Interface**:
```fortran
SUBROUTINE mc_build_neighbors(avec, nk, nl)
  real(dp), dimension(3,3), intent(in) :: avec
  integer, dimension(3), intent(in) :: nk
  type(mc_lattice), intent(inout) :: nl
```

**Algorithm**:
1. Site index: $idx = ((ic \cdot ny + ib) \cdot nx + ia) \cdot n_{uc} + isite$
2. For each site, search neighbors within ±2 shells
3. PBC wrapping: $mod(ic+dc+2nz, nz)$

##### `mc_sweep(mc, temperature, n_accepted)`

**Purpose**: One MC sweep = n_sites single spin update attempts

**Interface**:
```fortran
SUBROUTINE mc_sweep(mc, temperature, n_accepted)
  type(mc_lattice), intent(inout) :: mc
  real(dp), intent(in) :: temperature
  integer, intent(out) :: n_accepted
```

**Algorithm**:
1. Randomly select a site
2. Generate new spin direction (Marsaglia method — random point on unit sphere)
3. Compute energy change: $dE = -J_{mc} S^2 \sum_{j \in NN} (\mathbf{S}_{new} - \mathbf{S}_{old}) \cdot \mathbf{S}_j$
4. Metropolis acceptance criterion

##### `mc_measure_magnetization(mc, mvec_out)`

**Purpose**: Measure magnetization

**Interface**:
```fortran
SUBROUTINE mc_measure_magnetization(mc, mvec_out)
  type(mc_lattice), intent(in) :: mc
  real(dp), dimension(3), intent(out) :: mvec_out
```

**Formula**:
$$\mathbf{m} = \frac{1}{N} \sum_{i=1}^N \mathbf{S}_i$$

Output is fractional magnetization vector (|mvec| ∈ [0, 1]).

##### `classical_mc_run(J_mc_in, S_mag_in, frac_pos, n_f_sites, avec, T_start, T_step, T_end, mvec_vs_T, n_temps, mc_supercell_in)`

**Purpose**: Temperature sweep main loop

**Interface**:
```fortran
SUBROUTINE classical_mc_run(J_mc_in, S_mag_in, frac_pos, n_f_sites, avec, T_start, T_step, T_end, mvec_vs_T, n_temps, mc_supercell_in)
```

**Parameters**: N_THERM = 5000 (thermalization steps), N_MEAS = 10000 (measurement steps), MEAS_EVERY = 10

---

### 1.14 wannlog — Timing and Logging

**File**: [`modules/wannlog.f90`](wannchi/modules/wannlog.f90) (186 lines)

**Function**: Lightweight wall-clock timing and message logging utility

#### Subroutine Details

##### `log_init(log_name)`

**Purpose**: Initialize logger with output filename

**Interface**:
```fortran
SUBROUTINE log_init(log_name)
  character(*), intent(in) :: log_name
```

##### `log_start(stage_name)`

**Purpose**: Mark the start of a computation stage (records wall time)

**Interface**:
```fortran
SUBROUTINE log_start(stage_name)
  character(*), intent(in) :: stage_name
```

##### `log_stop()`

**Purpose**: Mark the end of current stage, compute and print elapsed time

**Interface**:
```fortran
SUBROUTINE log_stop()
```

##### `log_msg(msg)`

**Purpose**: Print a message to log

**Interface**:
```fortran
SUBROUTINE log_msg(msg)
  character(*), intent(in) :: msg
```

##### `log_print_summary()`

**Purpose**: Print a summary table of all timed stages

**Interface**:
```fortran
SUBROUTINE log_print_summary()
```

---

### 1.15 wanneff_JS — J-S Kondo Exchange Coupling Fitting Workflow

**File**: [`src/wanneff_JS.f90`](wannchi/src/wanneff_JS.f90) (1176 lines)

**Dependencies**: `constants`, `wanndata`, `lattice`, `linalgwrap`, `gp_bo`, `cma_es`, `classical_mc`, `transp_calc`, `wannlog`

#### 1.15.1 Downfolding Architecture

##### Orbital Mapping (determine_cc_indices)

**Assumptions** (3 key constraints):
1. Site ordering matches between seed.pos and seedbare.pos
2. At each shared site, CC orbitals appear BEFORE FF orbitals
3. Same count = same character (no reordering within a site)

**Python-side explicit control**: Python explicitly specifies FF indices when generating HR files, writing `ff_orbital_indices` to wanneff.inp. Fortran builds `cc_idx` = all seed indices not in `ff_orbital_indices`. These CC indices in seed correspond to seedbare indices 1, 2, 3, ... in order.

##### Downfolding Formula (R-space Schur complement)

$$H_{\text{eff}}^{CC}(\mathbf{R}) = H_{CC}(\mathbf{R}) - H_{CF}(\mathbf{R}) \cdot H_{FF}(\mathbf{R})^{-1} \cdot H_{FC}(\mathbf{R})$$

**⚠️ Known Limitation — Energy Window Sign Dependency**:
The Schur complement sign depends on whether FF states are **filled** (below Fermi) or **empty** (above Fermi):
- **FF states BELOW Fermi** (filled): use `-H_CF * H_FF^{-1} * H_FC` (current implementation)
- **FF states ABOVE Fermi** (empty): should use `+H_CF * H_FF^{-1} * H_FC`

This is critical for Kondo systems where f-electrons are at $E_F + \Delta$ (empty). How to detect: Check H_ff diagonal at R=0 (on-site energies). If diagonal is positive (above Fermi), use positive sign.

**Current behavior**: Always uses negative sign regardless of energy window → poor downfolding for systems with FF states above Fermi.

##### k-space Downfolding Formula

$$H_{\text{eff}}^{CC}(\mathbf{k}) = -[G_{CC}(\mathbf{k}, 0)]^{-1}$$

where $G_{\text{full}}(\mathbf{k}, 0) = (0 - H_{\text{seed}}(\mathbf{k}))^{-1}$.

**⚠️ Important**: The R-k transformation is NOT required for the downfolding operation itself. The core downfolding $H_{\text{eff}}^{CC}(\mathbf{k}) = -[G_{CC}(\mathbf{k}, 0)]^{-1}$ is inherently a k-space operation. The inverse FT to produce an HR file is only needed when:
- Writing `seed_downfold_hr.dat` for other codes that expect Wannier90 format
- Interfacing with Wannier90 for band interpolation

Staying in k-space throughout would avoid two unnecessary FFTs and potential numerical errors from the k→R→k round-trip.

#### 1.15.2 `downfold_rspace`

**Purpose**: R-space Schur complement downfolding

**Interface**:
```fortran
SUBROUTINE downfold_rspace(ham_out, ham_cc, ham_cf, ham_fc, ham_ff, norb_cc, norb_ff)
```

**Algorithm**:
1. For each R lattice, extract 4 blocks: H_CC, H_CF, H_FC, H_FF
2. If H_FF ≈ 0 (no FF hopping) → H_eff = H_CC
3. Otherwise: invert and compute Schur complement term

**⚠️ Known limitation**: Sign depends on whether FF states are below/above Fermi. Current implementation always uses negative sign.

#### 1.15.3 `js_objective(params, n_params, val)`

**Purpose**: Bayesian optimization objective function

**Objective function**:
$$L(J, S) = \frac{1}{N_k} \sum_{\mathbf{k}} \left\| \text{sort}(\lambda(H_{\text{bare}}(\mathbf{k}) + H_{JS}(\mathbf{k}))) - \text{sort}(\lambda(H_{\text{eff}}^{CC}(\mathbf{k}))) \right\|^2$$

**Algorithm**:
1. Unpack parameters based on eff_mode
2. For each k point: compute $H_{\text{bare}}(\mathbf{k})$ → add $H_{JS}$ → diagonalize
3. Accumulate L2 difference of sorted eigenvalues

---

## 2. Source File Detailed Subroutine Analysis

### 2.1 input — Input File Parsing

**File**: [`src/input.f90`](wannchi/src/input.f90) (436 lines)

#### Global Variables

```fortran
character(len=80) seed          ! SeedName
real(dp) mu, beta               ! Fermi energy and temperature
integer nqpt                   ! Number of q-points
real(dp), dimension(:, :), allocatable :: qvec  ! q vectors
integer nnu                    ! Number of frequency points
real(dp) emin, emax            ! Real frequency range
real(dp) eps                   ! Imaginary broadening
complex(dp), dimension(:), allocatable :: nu  ! Frequency points
logical spectra_calc           ! Whether real-frequency calculation
logical trace_only             ! Whether to compute trace only
logical ff_only                ! Whether to compute FF block only
logical use_lehman             ! Whether to use Lehman algorithm
logical fast_calc              ! Whether to use fast algorithm
integer npade                  ! Number of Padé poles
```

#### Subroutine Details

##### 2.1.1 `read_input(codename)`

**Purpose**: Read input file

**Interface**:
```fortran
SUBROUTINE read_input(codename)
    character(*), intent(in) :: codename
```

**Input file format** (wannchi.inp):
```fortran
&SYSTEM
    seed='wannier90'
    beta=2000.d0
    mu=0.d0
    spectra_calc=.false.
/

&CONTROL
    use_lehman=.false.
    trace_only=.false.
    ff_only=.true.
    fast_calc=.true.
    npade=80
    nnu=1
    eps=0.001
/
```

---

##### 2.1.2 `read_qpoints()`

**Purpose**: Read QPOINTS file

**Interface**:
```fortran
SUBROUTINE read_qpoints()
```

**File format**:
- mode=0: single point, `qx qy qz`
- mode=1: line mode, `nseg ninterpolate`, then for each segment `x1 y1 z1 x2 y2 z2`
- mode=2: plane mesh, `nint1 nint2`, then vertices and two directions
- mode=3: full BZ mesh, `nk1 nk2 nk3`

---

##### 2.1.3 `finalize_input()`

**Purpose**: Deallocate input arrays

**Interface**:
```fortran
SUBROUTINE finalize_input()
```

---

### 2.2 green.f90 — Green's Function Calculation

**File**: [`src/green.f90`](wannchi/modules/green.f90) (67 lines)

#### Subroutine Details

##### 2.2.1 `calc_g0(gf, hk, w, ndim, inv)`

**Purpose**: Compute non-interacting Green's function $G^0 = (w - H)^{-1}$

**Interface**:
```fortran
subroutine calc_g0(gf, hk, w, ndim, inv)
    complex(dp), dimension(ndim, ndim), intent(out) :: gf
    complex(dp), dimension(ndim, ndim), intent(in) :: hk
    complex(dp), intent(in) :: w
    integer, intent(in) :: ndim
    logical, intent(in) :: inv
```

**Formula**:
$$G^0(\mathbf{k}, \omega) = [\omega - H(\mathbf{k})]^{-1}$$

---

##### 2.2.2 `calc_corrFF_gf(gf, hff, sigmat, Ecc_diag, Vfc, w, ndimf, ndimc, inv)`

**Purpose**: Compute correlated FF Green's function with hybridization

**Interface**:
```fortran
subroutine calc_corrFF_gf(gf, hff, sigmat, Ecc_diag, Vfc, w, ndimf, ndimc, inv)
```

**Implementation**:
1. Bath Green's function: $G_{CC} = \text{diag}(1/(w - E_{CC}))$
2. Hybridization function: $\Delta = -V_{FC} \cdot G_{CC} \cdot V_{FC}^\dagger$
3. Compute $G_{FF}^{-1} = w - H_{FF} - \Sigma_{FF} - \Delta$
4. Invert to obtain $G_{FF}$

**Formula**:
$$\Delta(\omega) = -V_{FC} \cdot G_{CC}(\omega) \cdot V_{FC}^\dagger$$
$$G_{FF}^{-1}(\omega) = \omega - H_{FF} - \Sigma_{FF} - \Delta$$

where bath Green's function:
$$G_{CC}(\omega) = \frac{1}{\omega - E_{CC}}$$

---

### 2.3 output.f90 / output_chi.f90 — Output Utilities

#### Subroutine Details

##### 2.3.1 `output_header(...)` (output.f90)

**Purpose**: Write spectral function header

---

##### 2.3.2 `output_spectral(...)` (output.f90)

**Formula**:
$$A(\mathbf{k}, \omega) = -\frac{1}{\pi} \text{Im} \, G(\mathbf{k}, \omega)$$

---

##### 2.3.3 `output_chi(...)` (output_chi.f90)

**Purpose**: Write χ(q,ω) data

---

### 2.4 compute_chi.f90 — Response Function Core Computation

**File**: [`src/compute_chi.f90`](wannchi/src/compute_chi.f90) (~1500 lines)

#### chi_internal Module Variables

**Fast algorithm shared**:
```fortran
integer, allocatable :: kq_idx(:)  ! k → k+q index mapping
```

**Lehman representation**:
```fortran
real(dp), allocatable :: ek(:), ekq(:)        ! Band energies at k and k+q
real(dp), allocatable :: occ_k(:), occ_kq(:)  ! Fermi occupations
complex(dp), allocatable :: hk(:,:), hkq(:,:)  ! Hamiltonian / eigenvectors
real(dp), allocatable :: Skq(:,:)  ! Structure factor |U_k^† U_{k+q}|²
```

**Lehman fast algorithm**:
```fortran
real(dp), allocatable :: eig_full(:,:), occ_full(:,:)  ! (norb, nkirr) full eigenvalues
complex(dp), allocatable :: Uk_full(:,:,:)  ! (norb, norb, nkirr) full eigenvectors
```

**G*G algorithm (correlated case)**:
```fortran
complex(dp), allocatable :: Hff_k(:,:), Hff_kq(:,:)  ! F block Hamiltonian
complex(dp), allocatable :: Vfc_k(:,:), Vfc_kq(:,:)  ! F-C coupling
real(dp), allocatable :: Ecc_k(:), Ecc_kq(:)        ! C block energies (diagonal)
```

#### Subroutine Details

##### 2.4.1 `prepare_lehman(fast_calc, step)`

**Purpose**: Initialize Lehman calculation

---

##### 2.4.2 `calc_chi_bare_matrix_lehman_kernel(w, nw)`

**Purpose**: Core kernel — accumulate FF/FC/CF/CC block contributions for single k point

**Algorithm**:
```
do ik = first_idx to last_idx:
    do iq = 1 to nqpt:
        k = kvec(ik)
        q = qvec(iq)
        kq = k + q

        ! Diagonalize H(k) and H(k+q)
        call calc_hk(hk, ham, k)
        call eigen(eig_k, hk, norb)
        call calc_hk(hkq, ham, kq)
        call eigen(eig_kq, hkq, norb)

        ! Compute response function
        do ibnd = 1 to norb:
            do jbnd = 1 to norb:
                fact = (occ_k(ibnd) - occ_kq(jbnd)) / (w + eig_kq(jbnd) - eig_k(ibnd) + i*eps)
                chi += fact * |S_ij|^2
```

**Formula**:
$$\chi^0_{ij}(\mathbf{q}, \omega) = \frac{1}{N_k} \sum_{\mathbf{k}} \sum_{mn} \frac{(f_m - f_n) \cdot |S_{ij}^{mn}(\mathbf{k},\mathbf{q})|^2}{\omega + \epsilon_n(\mathbf{k+q}) - \epsilon_m(\mathbf{k}) + i\eta}$$

where $S_{ij}^{mn} = \langle u_m(\mathbf{k}) | \phi_i \rangle \langle \phi_j | u_n(\mathbf{k+q}) \rangle$

---

##### 2.4.3 `calc_chi_bare_matrix_lehman(chi0, w, nw, qv)`

**Purpose**: Slow version — loop over k one at a time

---

##### 2.4.4 `calc_chi_bare_matrix_lehman_fast(chi0, w, nw, qv)`

**Purpose**: Fast version — precompute all eigenvalues/eigenvectors

---

##### 2.4.5 `prepare_GG(fast_calc, step)`

**Purpose**: Initialize G*G calculation

---

##### 2.4.6 `calc_chi_bare_matrix_GG_kernel(w, nw)`

**Purpose**: Core kernel — accumulate χ using Padé poles and Green function products

**Formula**:
$$\frac{1}{\beta}\sum_{\omega_n} f(i\omega_n) \approx \sum_{p=1}^{N_{\text{pole}}} \eta_p \left[f(iz_p/\beta) + f(-iz_p/\beta)\right]$$

---

##### 2.4.7 `calc_chi_trace_from_matrixFF(chi0, nw)`

**Purpose**: Extract trace from FF block

---

### 2.5 wannchi.f90 — Main Program (bare susceptibility)

**File**: [`src/wannchi.f90`](wannchi/src/wannchi.f90) (137 lines)

#### Program Flow

```
1. init_para → MPI initialize
2. read_input('wannchi') → read wannchi.inp
3. read_ham(ham, seed) → read Wannier Hamiltonian
4. wannham_shift_ef(ham, mu) → shift by chemical potential
5. read_posfile → read atomic positions
6. read_kmesh → read k-mesh
7. read_qpoints → read q-points
8. if (.not.trace_only) read_RPA → optionally read RPA block definitions
9. Choose algorithm:
   - use_lehman=true → prepare_lehman()
   - else → prepare_GG() + print_pade()
10. loop iq = 1 to nqpt:
        if trace_only:
            calc_chi_bare_trace_lehman/GG_*
        else:
            calc_chi_bare_matrix_lehman/GG_*
        output text file chi0tr.dat (trace)
        output binary file chiff.dat (FF block)
        if ff_only=false: also chicc.dat, chifc.dat, chicf.dat
11. finalize → deallocate all resources
```

---

### 2.6 wannchiRPA.f90 — Main Program (RPA-dressed susceptibility)

**File**: [`src/wannchiRPA.f90`](wannchi/src/wannchiRPA.f90) (69 lines)

#### Program Flow

```
1. init_para → MPI initialize
2. read_input → read parameters
3. read_ham_dim → read Hamiltonian dimensions
4. read_posfile → read atomic positions
5. read_qpoints → read q-points
6. read_RPA → read FF block structure and U matrix
7. init_chi_matrix_RPA(nnu) → allocate χ and χ⁰ matrices
8. loop iq = 1 to nqpt:
        read_chi_matrix_RPA((iq-1)*nnu, nnu, ff_only)
        calc_chiRPA(chiff, chicc, chifc, chicf, ...)
        save_chi_matrix(iq*nnu, nnu, ff_only)
        calc_chi_trace_from_matrixFF(chi, nnu)
        output text file chiRPAtr.dat
9. All Done
```

---

### 2.7 postchi.f90 — Response Function Post-Processing

**File**: [`src/postchi.f90`](wannchi/src/postchi.f90) (~150 lines)

#### Function

- Read `{seed}.impdef` and `setup_mapping` to build F/C partition
- Read `chiff.dat` to load FF block χ matrix
- If SOC detected (`imp%ndim == 2*(lang+1)`), zero spin-off-diagonal matrix elements
- For each frequency point, perform eigen-decomposition: `eigen(chiEig, chiff, nFFidx)`
- Output `postchi.dat` (top-5 eigenvalues and corresponding eigenvectors)

---

### 2.8 wannband.f90 — Spectral Function Calculation

**File**: [`src/wannband.f90`](wannchi/src/wannband.f90) (~200 lines)

#### Function

**Input file**: `wannband.inp`

Compute spectral function:
$$A(\mathbf{k}, \omega) = -\frac{1}{\pi} \text{Im} \, G(\mathbf{k}, \omega)$$

- If `.impdef` does not exist: compute bare Green's function spectral function (no self-energy)
- If `.impdef` exists: read `.sig`, compute correlated spectral function with self-energy

---

### 2.9 wanneff_JS.f90 — J-S Kondo Exchange Coupling Fitting

**File**: [`src/wanneff_JS.f90`](wannchi/src/wanneff_JS.f90) (703 lines)

#### Types and Variables

```fortran
! Input parameters
character(len=80) :: seed      ! Seed for full system with f-electrons
character(len=80) :: seedbare  ! Seed for conduction-only system
logical :: eff_js = .true.     ! Whether to perform J-S fitting
logical :: eff_mc = .false.    ! Whether to perform MC temperature sweep
logical :: J_TENSOR = .false.  ! .false.=scalar J; .true.=tensor J(R)
integer :: bayes_niter = 200   ! Bayesian optimization iterations
real(dp) :: tol_Jeff = 1e-4   ! Threshold for pruning small J(R)
real(dp) :: J_bounds(2) = (/0.0, 0.5/)  ! J bounds
real(dp) :: S_bounds(2) = (/0.1, 5.0/)  ! S bounds
```

#### Key Subroutines

##### 2.9.1 `downfold_to_cc(hk_cc, hk_full, norb_full, norb_cc, cc_idx)`

**Purpose**: Static downfolding — extract effective CC Hamiltonian

**Formula**:
$$H_{\text{eff}}^{CC}(\mathbf{k}) = -[G_{CC}(\mathbf{k}, 0)]^{-1}$$

where $G_{\text{full}}(\mathbf{k}, 0) = (0 - H_{\text{seed}}(\mathbf{k}))^{-1}$

---

##### 2.9.2 `determine_cc_indices(cc_idx, norb_full, norb_bare, nbasis_full, nbasis_bare, nsite_full, nsite_bare)`

**Purpose**: Determine CC orbital indices

---

##### 2.9.3 `add_js_coupling(ham_out, norb_bare, J_val, Svec, irpt)`

**Purpose**: Add J·(S·σ)/2 exchange coupling

**Formula**:
$$H_{JS}(\mathbf{R}) = J(\mathbf{R}) \cdot \frac{\mathbf{S} \cdot \boldsymbol{\sigma}}{2}$$

---

##### 2.9.4 `js_objective(params, n_params, val)`

**Purpose**: Bayesian optimization objective function

**Interface**:
```fortran
function js_objective(params, n_params) result(val)
    real(dp), intent(in) :: params(n_params)
    integer, intent(in) :: n_params
    real(dp) :: val
```

**Objective function**:
$$L(J, S) = \frac{1}{N_k} \sum_{\mathbf{k}} \left\| \text{sort}(\lambda(H_{\text{bare}} + H_{JS})) - \text{sort}(\lambda(H_{\text{eff}}^{CC})) \right\|^2$$

---

## 3. Downfolding Architecture

### 3.1 Physical Model

```
H_seed = [ H_CC  H_CF ]
         [ H_FC  H_FF ]    (full system with f-electrons)

H_eff_CC = H_CC - H_CF * H_FF^{-1} * H_FC   (Schur complement)
H_bare   = Wannier90 HR of seedbare (conduction-only)
```

**Key assumption**: FF orbitals are at different atomic sites than CC orbitals, so the CC-FF coupling `H_CF` is hopping-like.

### 3.2 R-space vs K-space Downfolding

**R-space (used here)**:
- Computes `H_eff_CC(R)` directly via Schur complement
- Inverse FT to get `H_eff_CC(k)` for eigenvalue comparison
- Numerically more stable for localized Wannier functions

**K-space (deprecated)**:
- Computes `G_full(k,0) = (0 - H_seed(k))^{-1}`
- Extracts CC block: `H_eff_CC(k) = -[G_CC(k,0)]^{-1}`
- Prone to ghost states when FF energies are near Fermi

### 3.3 ⚠️ Known Limitation: Energy Window Sign Dependency

The Schur complement sign depends on the energy window:

| FF states | Sign | Physical meaning |
|-----------|------|-----------------|
| Below Fermi (filled) | `-H_CF * H_FF^{-1} * H_FC` | Electrons virtual-hop into empty FF states |
| Above Fermi (empty) | `+H_CF * H_FF^{-1} * H_FC` | Virtual hopping from filled FF states |

**Current behavior**: Always uses negative sign regardless of energy window.

**Detection**: Check `H_FF` diagonal at R=0 (on-site energies). If diagonal is positive (above Fermi), flip the sign.

**Test case**: `kagome_f_spinor_test.py` places f-orbitals at `EF + 0.4 eV` → requires positive sign.

---

## 4. Test Documentation

### 4.1 kagome_f_spinor_test.py (Test 6)

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

### 4.2 algo_test.py (Tests 1-5)

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

## 5. Build Process

### 5.1 make.sys.laptop (gfortran + Accelerate)

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

### 5.2 para_serial.f90

**Location**: `modules/para_serial.f90`
**Purpose**: Serial stub for MPI parallel utilities

Provides empty/no-op versions of:
- `init_para`, `finalize_para`
- `distribute_calc` (sets `first_idx=1, last_idx=n`)
- `para_merge_cmplx`, `para_sync_logical`, `para_sync0`

### 5.3 Build Commands

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

## 6. Q&A

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

## 7. Appendix: Fortran Syntax Reference

### 7.1 TYPE Definition

```fortran
TYPE :: type_name
  integer :: field1
  real(dp), allocatable :: array(:,:)
END TYPE type_name
```

### 7.2 Interface Block (callback)

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

### 7.3 Intent Specifiers

| Intent | Meaning |
|--------|---------|
| `intent(in)` | Input — not modified |
| `intent(out)` | Output — will be written to |
| `intent(inout)` | Both — may be modified |

### 7.4 Array Slice

```fortran
real(dp), dimension(10) :: a
a(1:5)        ! First 5 elements
a(2:10:2)    ! Elements 2,4,6,8,10 (step 2)
a(:)         ! All elements
```

### 7.5 Module vs Program

- **MODULE**: Collection of types, constants, procedures; `CONTAINS` marks procedure definitions
- **PROGRAM**: Executable unit; `CALL` procedures, `USE` modules
- **Internal procedures**: Procedures defined after `CONTAINS` inside a module/program; can access module-level variables

### 7.6 ZGEMM (Complex Matrix Multiply)

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

### 7.7 Random Number Generation

```fortran
real(dp) :: u
call random_number(u)  ! u in [0, 1)
u = 2.0_dp * u - 1.0_dp  ! map to [-1, 1)
```

### 7.8 Array Construction

```fortran
real(dp), dimension(3) :: vec
vec = [1.0_dp, 2.0_dp, 3.0_dp]  ! 1D array constructor
! Or with explicit bounds:
real(dp), dimension(0:2) :: vec
```

### 7.9 Deallocate with Check

```fortran
if (allocated(arr)) deallocate(arr)
if (associated(ptr)) nullify(ptr)
```

### 7.10 Merge (Conditional Assignment)

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

---

## Summary

This document provides a comprehensive analysis of all subroutines in the WannChi project, including:

1. **Function description**: The specific role of each subroutine
2. **Interface**: Input/output parameter types and meanings
3. **Implementation**: Algorithm implementation details
4. **Algorithm**: Computational flow and logic
5. **Corresponding formulas**: Physical and mathematical formulas

All subroutines are organized by module and source file for ease of code understanding and maintenance.
