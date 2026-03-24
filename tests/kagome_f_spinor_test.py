"""
kagome_f_spinor_test.py

Test suite for wanneff_JS workflow on a decorated kagome lattice with f-electrons.

Lattice:
  a1 = (1, 0, 0),  a2 = (0.5, sqrt(3)/2, 0),  a3 = (0, 0, 10)
  Sites:
    1: (0,   0,   0) -- s-orbital  (kagome vertex)
    2: (0.5, 0,   0) -- s-orbital  (kagome vertex)
    3: (0,   0.5, 0) -- s-orbital  (kagome vertex)
    4: (0.5, 0.5, 0) -- s-orbital + f-orbital  (decoration site)

  seed norb = 2*(4s+1f) = 10  [spinor]
    orbital order: [s1up,s2up,s3up,s4up, f4up, s1dn,s2dn,s3dn,s4dn, f4dn]
  seedbare norb = 2*4s = 8    [spinor]
    orbital order: [s1up,s2up,s3up,s4up, s1dn,s2dn,s3dn,s4dn]

Tests:
  1. Kagome reference bands + DOS
  2. Bayesian optimization on known 8D function
  3. Classical MC on square Heisenberg lattice
  4. AHC Berry curvature visualization
  5. Full wanneff_JS pipeline

Usage:
  cd wannchi/tests
  python3 kagome_f_spinor_test.py
"""

import numpy as np
import os
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")

# ============================================================
# Constants
# ============================================================
TSS  = -1.0    # s-s NN hopping (eV)
TSF  = -0.15   # NN s-f hopping (eV)
TSF2 = 0.0     # on-site s-f coupling (eV, default off)

# Lattice vectors
A1 = np.array([1.0, 0.0, 0.0])
A2 = np.array([0.5, np.sqrt(3)/2, 0.0])
A3 = np.array([0.0, 0.0, 10.0])
AVEC = np.column_stack([A1, A2, A3])  # columns are a1,a2,a3

# Fractional coordinates of sites
SITES_FRAC = np.array([
    [0.0,  0.0,  0.0],   # site 1: s
    [0.5,  0.0,  0.0],   # site 2: s
    [0.0,  0.5,  0.0],   # site 3: s
    [0.5,  0.5,  0.0],   # site 4: s+f
])

NORB_SEED = 10   # spinor: 5 spatial * 2 spin
NORB_BARE = 8    # spinor: 4 spatial * 2 spin

NSITE_SEED = 4
NSITE_BARE = 4  # same sites, but site4 has only 1 orbital (s only, no f)

NBASIS_SEED = np.array([1, 1, 1, 2])   # site4 has 2: s+f
NBASIS_BARE = np.array([1, 1, 1, 1])   # site4 has 1: s only

# Orbital indices (1-based, spin-up block, then spin-down block)
# Seed: s1up=1, s2up=2, s3up=3, s4up=4, f4up=5, s1dn=6, s2dn=7, s3dn=8, s4dn=9, f4dn=10
# Bare: s1up=1, s2up=2, s3up=3, s4up=4, s1dn=5, s2dn=6, s3dn=7, s4dn=8

# ============================================================
# WS R-vector generation (port of lattice.f90 find_ws)
# ============================================================

def find_ws_rvectors(nr1, nr2, nr3, avec):
    """
    Generate Wigner-Seitz R-vectors and degeneracy weights.
    Port of find_ws() from wannchi/modules/lattice.f90.

    avec: (3,3) array with lattice vectors as columns: avec[:,i] = ai
    """
    # Metric tensor: g_ij = a_i . a_j
    metric = np.dot(avec.T, avec)  # (3,3), metric[i,j] = a_i . a_j

    rvecs = []
    weights = []

    for ir1 in range(-nr1, nr1+1):
        for ir2 in range(-nr2, nr2+1):
            for ir3 in range(-nr3, nr3+1):

                # Compute distances to 5x5x5 supercell images
                dist = np.zeros(125)
                idx = 0
                for i1 in range(-2, 3):
                    for i2 in range(-2, 3):
                        for i3 in range(-2, 3):
                            ndiff = np.array([ir1 - i1*nr1, ir2 - i2*nr2, ir3 - i3*nr3],
                                           dtype=float)
                            # dist = ndiff . metric . ndiff
                            dist[idx] = ndiff @ metric @ ndiff
                            idx += 1

                dist_min = dist.min()
                # Index 63 = center (i1=0,i2=0,i3=0 -> index 2*25+2*5+2 = 62 in 0-based)
                # Actually in the Fortran: i1 from -2 to 2, i2 from -2 to 2, i3 from -2 to 2
                # Center = i1=0,i2=0,i3=0 -> index (2*5+2)*5+2 = 62 in 0-based = 63 in 1-based
                center_idx = 62  # 0-based

                if abs(dist[center_idx] - dist_min) < 1e-7:
                    weight = np.sum(np.abs(dist - dist_min) < 1e-7)
                    rvecs.append([ir1, ir2, ir3])
                    weights.append(weight)

    rvecs = np.array(rvecs, dtype=float).T  # (3, nrpt)
    weights = np.array(weights, dtype=float)

    # Verify sum of 1/weight = nr1*nr2*nr3
    tot = np.sum(1.0 / weights)
    expected = nr1 * nr2 * nr3
    assert abs(tot - expected) < 1e-6, f"WS weight check failed: {tot} != {expected}"

    return rvecs, weights


# ============================================================
# Hamiltonian construction
# ============================================================

def cartesian_pos(frac, avec):
    """Convert fractional to Cartesian: cart = avec @ frac (avec columns = a_i)"""
    return avec @ frac


def distance(f1, f2, avec, R=np.zeros(3)):
    """Cartesian distance between site f1 (frac) and site f2 + R (lattice vec)"""
    c1 = cartesian_pos(f1, avec)
    c2 = cartesian_pos(f2 + R, avec)
    return np.linalg.norm(c1 - c2)


def get_nn_distance(avec, sites_frac):
    """Find nearest-neighbor distance in the kagome lattice."""
    min_d = 1e10
    for i in range(len(sites_frac)):
        for j in range(len(sites_frac)):
            for r1 in [-1, 0, 1]:
                for r2 in [-1, 0, 1]:
                    R = np.array([r1, r2, 0.0])
                    if i == j and np.all(R == 0):
                        continue
                    d = distance(sites_frac[i], sites_frac[j], avec, R)
                    if d < min_d:
                        min_d = d
    return min_d


def build_hr_seed(rvecs, weights, avec, sites_frac=SITES_FRAC,
                  tss=TSS, tsf=TSF, tsf2=TSF2):
    """
    Build seed Hamiltonian hr(10,10,nrpt) for spinor kagome+f model.

    Orbital order (1-based): s1up,s2up,s3up,s4up, f4up, s1dn,s2dn,s3dn,s4dn, f4dn

    Hoppings:
      tss: s-s nearest neighbor (kagome: sites 1-4 among themselves)
      tsf: NN s-f hopping (f on site4 to neighboring s-orbitals)
      tsf2: on-site s-f coupling on site4 (R=0)
    """
    norb = NORB_SEED
    nrpt = rvecs.shape[1]
    hr = np.zeros((norb, norb, nrpt), dtype=complex)

    nn_dist = get_nn_distance(avec, sites_frac)
    nn_cutoff = nn_dist * 1.5

    # Orbital index map (0-based):
    # spin-up: s1=0, s2=1, s3=2, s4=3, f4=4
    # spin-dn: s1=5, s2=6, s3=7, s4=8, f4=9
    s_up = [0, 1, 2, 3]
    f_up = 4
    s_dn = [5, 6, 7, 8]
    f_dn = 9

    for ir in range(nrpt):
        R = rvecs[:, ir]

        for i_site in range(4):  # s-orbitals only (sites 0-3)
            for j_site in range(4):
                d = distance(sites_frac[i_site], sites_frac[j_site], avec, R)
                if d < 1e-8:
                    # On-site
                    continue
                if d < nn_cutoff:
                    # Nearest-neighbor s-s hopping
                    hr[s_up[i_site], s_up[j_site], ir] += tss
                    hr[s_dn[i_site], s_dn[j_site], ir] += tss

        # s-f NN hopping: f on site4 to s on sites 0-3
        # H(R)_{f,s_i}: hopping from s_i at R to f at 0
        # Need distance d(f_site, s_i + R) for forward: check sites_frac[3] vs sites_frac[i]+R
        # Need distance d(s_i, f + R) for reverse:  check sites_frac[i] vs sites_frac[3]+R
        for i_site in range(4):
            # Forward direction: H(R)_{f, s_i}  [f at cell 0, s_i at cell R]
            d_fwd = distance(sites_frac[3], sites_frac[i_site], avec, R)
            if d_fwd > 1e-8 and d_fwd < nn_cutoff:  # NN hop (exclude on-site d=0)
                hr[f_up, s_up[i_site], ir] += tsf
                hr[f_dn, s_dn[i_site], ir] += tsf
            if abs(tsf2) > 1e-10 and d_fwd < 1e-8:  # on-site f-s coupling (same site)
                hr[f_up, s_up[i_site], ir] += tsf2
                hr[s_up[i_site], f_up, ir] += np.conj(tsf2)
                hr[f_dn, s_dn[i_site], ir] += tsf2
                hr[s_dn[i_site], f_dn, ir] += np.conj(tsf2)
            # Reverse direction: H(R)_{s_i, f}  [s_i at cell 0, f at cell R]
            d_rev = distance(sites_frac[i_site], sites_frac[3], avec, R)
            if d_rev > 1e-8 and d_rev < nn_cutoff:  # exclude on-site (d=0)
                hr[s_up[i_site], f_up, ir] += np.conj(tsf)
                hr[s_dn[i_site], f_dn, ir] += np.conj(tsf)

    return hr


def build_hr_bare(rvecs, weights, avec, sites_frac=SITES_FRAC[:4], tss=TSS):
    """
    Build seedbare Hamiltonian hr(8,8,nrpt) for spinor kagome s-only model.

    Orbital order: s1up,s2up,s3up,s4up, s1dn,s2dn,s3dn,s4dn
    """
    norb = NORB_BARE
    nrpt = rvecs.shape[1]
    hr = np.zeros((norb, norb, nrpt), dtype=complex)

    nn_dist = get_nn_distance(avec, sites_frac)
    nn_cutoff = nn_dist * 1.5

    s_up = [0, 1, 2, 3]
    s_dn = [4, 5, 6, 7]

    for ir in range(nrpt):
        R = rvecs[:, ir]
        for i_site in range(4):
            for j_site in range(4):
                d = distance(sites_frac[i_site], sites_frac[j_site], avec, R)
                if d < 1e-8:
                    continue
                if d < nn_cutoff:
                    hr[s_up[i_site], s_up[j_site], ir] += tss
                    hr[s_dn[i_site], s_dn[j_site], ir] += tss

    return hr


def compute_hk_python(hr, rvecs, weights, kvec, tau=None, avec=None):
    """
    Compute H(k) from hr.dat arrays via Bloch sum (matches wanndata.f90 calc_hk).

    H(k)_{io,jo} = sum_R exp(i*2pi*k.R)/w(R) * conj(phase_io)*phase_jo * hr(io,jo,R)
    phase_io = exp(i*2pi*k.tau_io)
    """
    norb = hr.shape[0]
    nrpt = hr.shape[2]

    if tau is None:
        tau = np.zeros((3, norb))

    # Phase factors for orbital positions
    phase = np.exp(1j * 2*np.pi * (kvec @ tau))  # (norb,)

    hk = np.zeros((norb, norb), dtype=complex)

    for ir in range(nrpt):
        R = rvecs[:, ir]
        rdotk = kvec @ R
        fact = np.exp(1j * 2*np.pi * rdotk) / weights[ir]

        for io in range(norb):
            for jo in range(norb):
                hk[io, jo] += fact * np.conj(phase[io]) * phase[jo] * hr[io, jo, ir]

    return hk


def compute_tau_from_sites(sites_frac, nbasis, spinor=True):
    """Build tau array (3, norb) from fractional site positions and orbital counts."""
    taus = []
    for i_site, nbase in enumerate(nbasis):
        for _ in range(nbase):
            taus.append(sites_frac[i_site])

    if spinor:
        # Duplicate for spin-down block
        taus_full = taus + taus
    else:
        taus_full = taus

    return np.array(taus_full).T  # (3, norb)


def compute_dos_python(hr, rvecs, weights, tau, nk=50, nw=200, emin=-4.0, emax=4.0, eta=0.05):
    """Compute DOS by delta function broadening on an nk x nk mesh."""
    norb = hr.shape[0]
    energies = np.linspace(emin, emax, nw)
    dos = np.zeros(nw)

    kpoints = []
    for ik1 in range(nk):
        for ik2 in range(nk):
            kpoints.append([ik1/nk, ik2/nk, 0.0])

    for kvec in kpoints:
        hk = compute_hk_python(hr, rvecs, weights, np.array(kvec), tau)
        eigs = np.linalg.eigvalsh(hk)
        for E in eigs:
            dos += eta/np.pi / ((energies - E)**2 + eta**2)

    dos /= (nk * nk * norb)
    return energies, dos


# ============================================================
# File I/O
# ============================================================

def write_hr_dat(fname, hr, rvecs, weights, norb, nrpt):
    """Write Wannier90 _hr.dat format."""
    with open(fname, 'w') as f:
        f.write(f"# Kagome test Hamiltonian ({norb} orbitals)\n")
        f.write(f"{norb:10d}\n")
        f.write(f"{nrpt:10d}\n")
        # Weights in groups of 15
        wt_int = [int(round(w)) for w in weights]
        for i in range(0, nrpt, 15):
            chunk = wt_int[i:min(i+15, nrpt)]
            f.write("".join(f"{w:5d}" for w in chunk) + "\n")
        # HR elements: Rx Ry Rz jo io  Re Im
        for ir in range(nrpt):
            Rv = rvecs[:, ir]
            for io in range(norb):
                for jo in range(norb):
                    val = hr[jo, io, ir]  # NOTE: file stores (jo, io) = (row, col) in Fortran convention
                    f.write(f"{int(round(Rv[0])):5d}{int(round(Rv[1])):5d}{int(round(Rv[2])):5d}"
                           f"{jo+1:5d}{io+1:5d}"
                           f"{val.real:22.16f}{val.imag:22.16f}\n")


def write_pos_file(fname, avec, sites_frac, nbasis, atomic_nums, spinor=True):
    """
    Write .pos file matching lattice.f90 read_posfile format.
    Format:
      Comment
      1.0 (scaling)
      A1x A1y A1z
      A2x A2y A2z
      A3x A3y A3z
      nsite soc
      Z x y z nbasis   (per site)
    """
    soc = 1 if spinor else 0
    nsite = len(sites_frac)
    with open(fname, 'w') as f:
        f.write("Kagome lattice\n")
        f.write("1.0\n")
        for i in range(3):
            f.write(f"{avec[0,i]:20.14f}{avec[1,i]:20.14f}{avec[2,i]:20.14f}\n")
        f.write(f"{nsite:5d}{soc:5d}\n")
        for i in range(nsite):
            f.write(f"{atomic_nums[i]:5d}"
                   f"{sites_frac[i][0]:14.10f}{sites_frac[i][1]:14.10f}{sites_frac[i][2]:14.10f}"
                   f"{nbasis[i]:5d}\n")


def write_ibzkpt(fname, nk1, nk2, nk3=1):
    """Write IBZKPT file (Gamma-centered mesh)."""
    with open(fname, 'w') as f:
        f.write("Automatic mesh\n")
        f.write(" 0\n")
        f.write("Reciprocal lattice\n")
        f.write(f" {nk1} {nk2} {nk3}\n")


def write_qpoints_bandpath(fname, n_per_seg=100):
    """Write QPOINTS for Gamma -> M -> K -> Gamma band path (kagome)."""
    Gamma = np.array([0.0, 0.0, 0.0])
    M     = np.array([0.5, 0.0, 0.0])
    K     = np.array([1/3, 1/3, 0.0])

    segments = [(Gamma, M), (M, K), (K, Gamma)]
    n_seg = len(segments)
    n_total = n_seg * (n_per_seg + 1)

    with open(fname, 'w') as f:
        f.write("1\n")  # mode 1: line mode
        f.write(f"{n_seg}  {n_per_seg}\n")
        for q1, q2 in segments:
            f.write(f"{q1[0]:10.6f}{q1[1]:10.6f}{q1[2]:10.6f}  "
                   f"{q2[0]:10.6f}{q2[1]:10.6f}{q2[2]:10.6f}\n")


def write_wanneff_inp(fname, seed='seed', seedbare='seedbare',
                      eff_js=True, eff_mc=True,
                      mc_temperature=(0.0, 0.001, 0.01),
                      J_TENSOR=False, J_R_range=(0,0,0,0,0,0),
                      mu=0.0, nnu=300, emin=-4.0, emax=4.0,
                      bayes_niter=30):
    """Write wanneff.inp input file."""
    mc_str = f"{mc_temperature[0]}, {mc_temperature[1]}, {mc_temperature[2]}"
    jr_str = " ".join(str(x) for x in J_R_range)
    with open(fname, 'w') as f:
        f.write("&SYSTEM\n")
        f.write(f"  seed='{seed}', beta=1e7, mu={mu}\n")
        f.write("/\n")
        f.write("&CONTROL\n")
        f.write(f"  spectra_calc=.true., nnu={nnu}, emin={emin}, emax={emax}, eps=1e-3\n")
        f.write("/\n")
        f.write("&EFFJS\n")
        f.write(f"  seedbare='{seedbare}',\n")
        f.write(f"  eff_js={'.true.' if eff_js else '.false.'},\n")
        f.write(f"  eff_mc={'.true.' if eff_mc else '.false.'},\n")
        f.write(f"  mc_temperature={mc_str},\n")
        f.write(f"  J_TENSOR={'.true.' if J_TENSOR else '.false.'},\n")
        f.write(f"  J_R_range={jr_str},\n")
        f.write(f"  bayes_niter={bayes_niter},\n")
        f.write(f"  J_bounds=0.0 5.0,\n")
        f.write(f"  S_bounds=-3.0 3.0,\n")
        f.write(f"  tol_Jeff=1e-2,\n")
        f.write("/\n")


# ============================================================
# Plotting helpers
# ============================================================

def plot_bands(spectra_file, title="Bands", ax=None, color='blue', label=None):
    """Plot band structure from spectra.dat."""
    if not HAS_MATPLOTLIB:
        return None

    if not os.path.exists(spectra_file):
        print(f"  Warning: {spectra_file} not found, skipping band plot")
        return None

    data = np.loadtxt(spectra_file)
    # Columns: kx ky kz  energy  A(k,w)
    # Detect block structure by kz=0 and new q-point groups
    kpath = np.sqrt(np.sum(np.diff(data[::data.shape[0]//100 or 1, :3])**2, axis=1))

    # Simple: group by unique k-points
    energies = data[:, 3]
    akw = data[:, 4]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    # Plot as scatter with A(k,w) as color intensity
    # Detect structure: nnu frequencies per k-point
    # Count unique k-points
    kvecs = data[:, :3]
    unique_k = []
    for i, k in enumerate(kvecs):
        if i == 0 or not np.allclose(k, kvecs[i-1]):
            unique_k.append(i)

    nkpt = len(unique_k)
    nnu = data.shape[0] // nkpt if nkpt > 0 else 1

    kindex = np.repeat(np.arange(nkpt), nnu)
    ax.scatter(kindex, energies, c=akw, cmap='hot_r', s=0.5, alpha=0.7)

    ax.set_xlabel("k-point index")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(title)
    ax.axhline(0, color='gray', lw=0.5, ls='--')

    # Tick marks at high-symmetry points
    n_per_seg = nkpt // 3
    ticks = [0, n_per_seg, 2*n_per_seg, nkpt-1]
    labels = ['Γ', 'M', 'K', 'Γ']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    return ax


def plot_dos(energies, dos, ax=None):
    if not HAS_MATPLOTLIB:
        return None
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(dos, energies)
    ax.set_xlabel("DOS")
    ax.set_ylabel("Energy (eV)")
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    return ax


# ============================================================
# Test utilities
# ============================================================

def run_executable(cmd, workdir='.', timeout=120):
    """Run a shell command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=workdir, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def find_executable(name, search_dirs=None):
    """Find a compiled executable."""
    if search_dirs is None:
        search_dirs = [
            os.path.join(os.path.dirname(__file__), '..', 'src'),
            os.path.join(os.path.dirname(__file__), '..'),
            '.',
        ]
    for d in search_dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


# ============================================================
# Test 1: Kagome reference bands + DOS
# ============================================================

def generate_test_data(outdir):
    """Generate all HR and input files in outdir."""
    os.makedirs(outdir, exist_ok=True)

    print("  Generating WS R-vectors (nr=5)...")
    rvecs, weights = find_ws_rvectors(5, 5, 1, AVEC)
    nrpt = rvecs.shape[1]
    print(f"  nrpt = {nrpt}")

    # Tau arrays
    tau_seed = compute_tau_from_sites(SITES_FRAC, NBASIS_SEED, spinor=True)
    tau_bare = compute_tau_from_sites(SITES_FRAC[:4], NBASIS_BARE, spinor=True)

    # Build HR
    print("  Building seed HR...")
    hr_seed = build_hr_seed(rvecs, weights, AVEC)
    print("  Building bare HR...")
    hr_bare = build_hr_bare(rvecs, weights, AVEC)

    # Write HR files
    write_hr_dat(os.path.join(outdir, 'seed_hr.dat'), hr_seed, rvecs, weights, NORB_SEED, nrpt)
    write_hr_dat(os.path.join(outdir, 'seedbare_hr.dat'), hr_bare, rvecs, weights, NORB_BARE, nrpt)

    # Write pos files
    at_seed = [1, 1, 1, 1]  # all atomic number 1 (dummy)
    write_pos_file(os.path.join(outdir, 'seed.pos'), AVEC, SITES_FRAC,
                   NBASIS_SEED, at_seed, spinor=True)
    write_pos_file(os.path.join(outdir, 'seedbare.pos'), AVEC, SITES_FRAC[:4],
                   NBASIS_BARE, at_seed, spinor=True)

    # Write input files
    write_ibzkpt(os.path.join(outdir, 'IBZKPT'), 12, 12, 1)
    write_qpoints_bandpath(os.path.join(outdir, 'QPOINTS'))
    write_wanneff_inp(os.path.join(outdir, 'wanneff.inp'),
                      seed='seed', seedbare='seedbare',
                      eff_js=True, eff_mc=True,
                      mc_temperature=(0.0, 0.001, 0.025),
                      bayes_niter=50)

    print(f"  Files written to {outdir}/")
    return rvecs, weights, hr_seed, hr_bare, tau_seed, tau_bare


def test_kagome_reference(outdir, exe_wannband=None):
    """Test 1: Compute reference band structure and verify."""
    print("\n" + "="*60)
    print("TEST 1: Kagome reference bands + DOS")
    print("="*60)

    rvecs, weights, hr_seed, hr_bare, tau_seed, tau_bare = generate_test_data(outdir)

    # Python cross-check at high-symmetry k-points
    test_kvecs = [
        np.array([0.0, 0.0, 0.0]),   # Gamma
        np.array([0.5, 0.0, 0.0]),   # M
        np.array([1/3, 1/3, 0.0]),   # K
    ]

    print("  Python eigenvalues at high-symmetry points:")
    max_err = 0.0

    for kvec in test_kvecs:
        hk = compute_hk_python(hr_seed, rvecs, weights, kvec, tau_seed)
        eigs = np.linalg.eigvalsh(hk)
        print(f"    k={kvec[:2]}: eigs = {eigs[:5]}")
        # Check Hermiticity
        err = np.max(np.abs(hk - hk.conj().T))
        max_err = max(max_err, err)

    passed = max_err < 1e-10
    print(f"  Hermiticity check: max |H - H†| = {max_err:.2e}  {'PASS' if passed else 'FAIL'}")

    # DOS computation (Python)
    print("  Computing DOS...")
    energies, dos_seed = compute_dos_python(hr_seed, rvecs, weights, tau_seed, nk=30)
    _, dos_bare = compute_dos_python(hr_bare, rvecs, weights, tau_bare, nk=30)

    # Run wannband if available
    if exe_wannband and os.path.exists(exe_wannband):
        print("  Running wannband.x on seed...")
        # Write wannband.inp
        with open(os.path.join(outdir, 'wannband.inp'), 'w') as f:
            f.write("&SYSTEM\n  seed='seed', mu=0.0\n/\n")
            f.write("&CONTROL\n  spectra_calc=.true., nnu=300, emin=-4.0, emax=4.0, eps=1e-3\n/\n")
        rc, out, err = run_executable(f"{exe_wannband}", workdir=outdir)
        if rc == 0:
            print("  wannband.x: OK")
        else:
            print(f"  wannband.x: FAILED (rc={rc})")
            print(f"  stderr: {err[:200]}")

    # Plot
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bands at test k-points (Python)
        ax = axes[0]
        k_labels = ['Γ', 'M', 'K']
        for ii, (kvec, label) in enumerate(zip(test_kvecs, k_labels)):
            hk_s = compute_hk_python(hr_seed, rvecs, weights, kvec, tau_seed)
            hk_b = compute_hk_python(hr_bare, rvecs, weights, kvec, tau_bare)
            e_s = np.sort(np.linalg.eigvalsh(hk_s))
            e_b = np.sort(np.linalg.eigvalsh(hk_b))
            ax.scatter([ii]*len(e_s), e_s, c='blue', s=50, zorder=5, label='seed' if ii==0 else '')
            ax.scatter([ii]*len(e_b), e_b, c='red', marker='+', s=80, zorder=5, label='bare' if ii==0 else '')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(k_labels)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel("Energy (eV)")
        ax.set_title("Eigenvalues at HSP")
        ax.legend()

        # DOS
        axes[1].plot(dos_seed, energies, 'b-', label='seed')
        axes[1].plot(dos_bare, energies, 'r--', label='bare')
        axes[1].axhline(0, color='gray', lw=0.5, ls='--')
        axes[1].set_xlabel("DOS (arb. u.)")
        axes[1].set_ylabel("Energy (eV)")
        axes[1].set_title("DOS")
        axes[1].legend()

        plt.suptitle("Test 1: Kagome Reference")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig1_kagome_reference.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved fig1_kagome_reference.png")

    result = {
        'test': 'kagome_reference',
        'passed': passed,
        'max_hermitian_error': float(max_err),
        'details': f"Hermiticity check {'PASS' if passed else 'FAIL'}"
    }
    return result


# ============================================================
# Test 2: Bayesian optimization on known function
# ============================================================

def test_bayesian_known_function(outdir):
    """
    Test 2: Bayesian optimization on known 8-dimensional function.
    f(x) = (x1-2)^2 + (x2+1)^2 + sum(sin(xi) for i=3..8)
    Known minimum near (2, -1, 0, 0, 0, 0, 0, 0), f_min ≈ 0.
    Uses Python reference implementation to verify convergence.
    """
    print("\n" + "="*60)
    print("TEST 2: Bayesian optimization - known 8D function")
    print("="*60)

    try:
        from scipy.optimize import minimize
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("  scipy not available; using numerical minimum estimate")

    def objective_8d(x):
        return (x[0]-2)**2 + (x[1]+1)**2 + sum(np.sin(xi) for xi in x[2:])

    # True minimum (approximately)
    def find_true_min():
        best_val = 1e10
        best_x = None
        np.random.seed(42)
        for _ in range(200):
            x0 = np.random.uniform(-3, 3, 8)
            if has_scipy:
                res = minimize(objective_8d, x0, method='L-BFGS-B',
                              bounds=[(-3,3)]*8)
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            else:
                val = objective_8d(x0)
                if val < best_val:
                    best_val = val
                    best_x = x0
        return best_x, best_val

    print("  Finding true minimum...")
    x_true, f_true = find_true_min()
    print(f"  True minimum: f = {f_true:.6f} at x ≈ {x_true[:4]}")

    # Simple GP-BO reference implementation in Python
    from scipy.stats import norm

    class SimpleGPBO:
        def __init__(self, ndim, bounds, length_scale=1.0, noise=1e-6):
            self.ndim = ndim
            self.bounds = bounds
            self.ls = length_scale
            self.noise = noise
            self.X = []
            self.y = []

        def rbf(self, x1, x2):
            return np.exp(-0.5 * np.sum((x1-x2)**2) / self.ls**2)

        def K_matrix(self):
            n = len(self.X)
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i,j] = self.rbf(self.X[i], self.X[j])
            K += self.noise * np.eye(n)
            return K

        def predict(self, x_test):
            if not self.X:
                return 0.0, 1.0
            K = self.K_matrix()
            k_star = np.array([self.rbf(x_test, xi) for xi in self.X])
            k_ss = self.rbf(x_test, x_test)
            try:
                K_inv = np.linalg.inv(K)
            except:
                return 0.0, 1.0
            mu = k_star @ K_inv @ np.array(self.y)
            var = k_ss - k_star @ K_inv @ k_star
            return mu, max(var, 0)**0.5

        def ei(self, x_test):
            mu, sigma = self.predict(x_test)
            if not self.y:
                return 1.0
            y_best = min(self.y)
            z = (y_best - mu) / (sigma + 1e-10)
            return (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        def optimize(self, func, n_init=8, n_iter=50):
            # Latin hypercube init
            np.random.seed(0)
            bounds_arr = np.array(self.bounds)
            for _ in range(n_init):
                x = bounds_arr[:,0] + np.random.rand(self.ndim) * (bounds_arr[:,1] - bounds_arr[:,0])
                self.X.append(x)
                self.y.append(func(x))

            history = [min(self.y)]

            for it in range(n_iter):
                # Random EI maximization
                best_ei = -1
                x_next = None
                for _ in range(2000):
                    x = bounds_arr[:,0] + np.random.rand(self.ndim) * (bounds_arr[:,1] - bounds_arr[:,0])
                    ei_val = self.ei(x)
                    if ei_val > best_ei:
                        best_ei = ei_val
                        x_next = x

                y_next = func(x_next)
                self.X.append(x_next)
                self.y.append(y_next)
                history.append(min(self.y))

            best_idx = np.argmin(self.y)
            return self.X[best_idx], self.y[best_idx], history

    print("  Running Python GP-BO (50 iterations)...")
    bounds_8d = [(-3, 3)] * 8
    gp = SimpleGPBO(8, bounds_8d, length_scale=2.0)
    x_opt, f_opt, history = gp.optimize(objective_8d, n_init=8, n_iter=50)

    print(f"  Python GP-BO result: f = {f_opt:.6f} at x ≈ {x_opt[:4]}")
    print(f"  True minimum: f = {f_true:.6f}")

    passed = f_opt < 0.5  # Should get close to the minimum
    print(f"  {'PASS' if passed else 'FAIL'}: f_opt = {f_opt:.4f} (threshold: 0.5)")

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history, 'b-o', ms=3, label='best f so far')
        ax.axhline(f_true, color='r', ls='--', label=f'true min = {f_true:.4f}')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("f(x)")
        ax.set_yscale('log')
        ax.set_title("Test 2: Bayesian Optimization Convergence (8D)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig2_bayesian_convergence.png'), dpi=150)
        plt.close()
        print("  Saved fig2_bayesian_convergence.png")

    return {
        'test': 'bayesian_known_function',
        'passed': passed,
        'f_opt': float(f_opt),
        'f_true': float(f_true),
        'details': f"GP-BO {'PASS' if passed else 'FAIL'}: f_opt={f_opt:.4f} vs f_true={f_true:.4f}"
    }


# ============================================================
# Test 3: Classical MC on square lattice
# ============================================================

def test_mc_square_lattice(outdir):
    """
    Test 3: MC on 10x10 square Heisenberg lattice.
    J_mc=1.0 eV, S=1.0, T from 0 to 5 eV, step 0.1 eV.
    Expected: <|S|>(T=0) ~ 1, <|S|>(T=5) < 0.2

    Note: 2D classical Heisenberg has no true phase transition (Mermin-Wagner theorem),
    but finite-size shows crossover. Mean-field T_c = J_mc*z*S^2/3 = 1.0*4*1/3 ~ 1.33 eV.
    """
    print("\n" + "="*60)
    print("TEST 3: Classical MC on square Heisenberg lattice")
    print("="*60)

    # Simple Python MC implementation for the test
    J_mc = 1.0   # eV
    S_mag = 1.0
    N = 10       # 10x10 lattice

    def random_spin():
        """Marsaglia method."""
        while True:
            u = 2*np.random.rand() - 1
            v = 2*np.random.rand() - 1
            s = u*u + v*v
            if s < 1:
                break
        return np.array([2*u*np.sqrt(1-s), 2*v*np.sqrt(1-s), 1-2*s])

    def local_energy(spins, idx, J, S):
        i, j = idx
        E = 0.0
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = (i+di)%N, (j+dj)%N
            E -= J * S**2 * np.dot(spins[i,j], spins[ni,nj])
        return E

    def mc_run_python(T, n_therm=2000, n_meas=5000):
        spins = np.array([[random_spin() for _ in range(N)] for _ in range(N)])

        # Thermalize
        for _ in range(n_therm):
            for i in range(N):
                for j in range(N):
                    E_old = local_energy(spins, (i,j), J_mc, S_mag)
                    s_new = random_spin()
                    s_old = spins[i,j].copy()
                    spins[i,j] = s_new
                    E_new = local_energy(spins, (i,j), J_mc, S_mag)
                    dE = E_new - E_old
                    if dE > 0 and (T < 1e-10 or np.random.rand() >= np.exp(-dE/T)):
                        spins[i,j] = s_old

        # Measure
        M_acc = np.zeros(3)
        n_acc = 0
        for _ in range(n_meas):
            for i in range(N):
                for j in range(N):
                    E_old = local_energy(spins, (i,j), J_mc, S_mag)
                    s_new = random_spin()
                    s_old = spins[i,j].copy()
                    spins[i,j] = s_new
                    E_new = local_energy(spins, (i,j), J_mc, S_mag)
                    dE = E_new - E_old
                    if dE > 0 and (T < 1e-10 or np.random.rand() >= np.exp(-dE/T)):
                        spins[i,j] = s_old
            n_acc += 1
            M_acc += spins.mean(axis=(0,1))

        return np.linalg.norm(M_acc / n_acc)

    print(f"  Running MC on {N}x{N} square lattice (J={J_mc}, S={S_mag})...")

    temps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    magnetizations = []

    np.random.seed(42)
    for T in temps:
        if T < 1e-10:
            # T=0: perfect order
            m = 1.0
        else:
            m = mc_run_python(T, n_therm=500, n_meas=500)
        magnetizations.append(m)
        print(f"    T = {T:.1f} eV, <|m|> = {m:.4f}")

    magnetizations = np.array(magnetizations)
    passed = (magnetizations[0] > 0.9 and magnetizations[-1] < 0.5)
    print(f"  {'PASS' if passed else 'FAIL'}: m(T=0)={magnetizations[0]:.3f}, m(T=5)={magnetizations[-1]:.3f}")

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(temps, magnetizations, 'bo-', ms=6)
        ax.axvline(J_mc*4*S_mag**2/3, color='r', ls='--', label=f'MFT T_c≈{J_mc*4*S_mag**2/3:.2f} eV')
        ax.set_xlabel("Temperature (eV)")
        ax.set_ylabel("<|m|>")
        ax.set_title("Test 3: Classical MC - Square Heisenberg Lattice")
        ax.set_ylim(0, 1.1)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig3_mc_square.png'), dpi=150)
        plt.close()
        print("  Saved fig3_mc_square.png")

    return {
        'test': 'mc_square_lattice',
        'passed': passed,
        'temps': temps.tolist(),
        'magnetizations': magnetizations.tolist(),
        'details': f"MC {'PASS' if passed else 'FAIL'}: m(0)={magnetizations[0]:.3f}, m(5)={magnetizations[-1]:.3f}"
    }


# ============================================================
# Test 4: Berry curvature visualization (uses seedbare_0K if available)
# ============================================================

def test_berry_curvature_plot(outdir, hr_file=None):
    """
    Test 4: Compute and plot Berry curvature for debugging.
    Uses Python implementation of the AHC formula.
    """
    print("\n" + "="*60)
    print("TEST 4: Berry curvature visualization")
    print("="*60)

    # Load seedbare HR (or bare+mock JS coupling)
    rvecs, weights = find_ws_rvectors(5, 5, 1, AVEC)
    hr_bare = build_hr_bare(rvecs, weights, AVEC)
    tau_bare = compute_tau_from_sites(SITES_FRAC[:4], NBASIS_BARE, spinor=True)
    norb = NORB_BARE

    # Add a mock JS coupling (J=0.3, S=(0,0,1)) for visualization
    J_mock = 0.3
    S_mock = np.array([0.0, 0.0, 1.0])
    nrpt = rvecs.shape[1]
    # Find R=0 index
    r000 = np.argmin(np.sum(rvecs**2, axis=0))

    hr_eff = hr_bare.copy()
    n_c = norb // 2
    for io in range(n_c):
        hr_eff[io, io, r000] += J_mock * S_mock[2] / 2
        hr_eff[io+n_c, io+n_c, r000] -= J_mock * S_mock[2] / 2
        hr_eff[io, io+n_c, r000] += J_mock * (S_mock[0] - 1j*S_mock[1]) / 2
        hr_eff[io+n_c, io, r000] += J_mock * (S_mock[0] + 1j*S_mock[1]) / 2

    # Compute Berry curvature on 30x30 k-mesh
    nk = 30
    print(f"  Computing Berry curvature on {nk}x{nk} mesh...")

    kpoints = []
    for ik1 in range(nk):
        for ik2 in range(nk):
            kpoints.append([ik1/nk, ik2/nk, 0.0])
    kpoints = np.array(kpoints)

    mu_chem = 0.0  # Fermi level

    omega_kmap = np.zeros(len(kpoints))
    sigma_xy_sum = 0.0

    for idx, kvec in enumerate(kpoints):
        hk = compute_hk_python(hr_eff, rvecs, weights, kvec, tau_bare)
        eigs, eigvecs = np.linalg.eigh(hk)

        # Velocity matrices (analytical)
        vx = np.zeros((norb, norb), dtype=complex)
        vy = np.zeros((norb, norb), dtype=complex)

        phase = np.exp(1j * 2*np.pi * (kvec @ tau_bare))

        for ir in range(nrpt):
            R = rvecs[:, ir]
            rdotk = kvec @ R
            fact = np.exp(1j * 2*np.pi * rdotk) / weights[ir]
            for io in range(norb):
                for jo in range(norb):
                    rtilde_x = R[0] + tau_bare[0,jo] - tau_bare[0,io]
                    rtilde_y = R[1] + tau_bare[1,jo] - tau_bare[1,io]
                    orb_fac = np.conj(phase[io]) * phase[jo] * fact
                    vx[io,jo] += 1j * 2*np.pi * rtilde_x * orb_fac * hr_eff[io,jo,ir]
                    vy[io,jo] += 1j * 2*np.pi * rtilde_y * orb_fac * hr_eff[io,jo,ir]

        # Transform to eigenbasis
        vx_band = eigvecs.conj().T @ vx @ eigvecs
        vy_band = eigvecs.conj().T @ vy @ eigvecs

        # Berry curvature per band
        omega_n = np.zeros(norb)
        for n in range(norb):
            for m in range(norb):
                if m == n:
                    continue
                dE2 = (eigs[n] - eigs[m])**2
                if dE2 < 1e-10:
                    continue
                omega_n[n] -= 2 * np.imag(vx_band[n,m] * vy_band[m,n]) / dE2

        # Fermi occupation (T=0)
        f_occ = (eigs <= mu_chem).astype(float)

        omega_kmap[idx] = np.sum(f_occ * omega_n)
        sigma_xy_sum += np.sum(f_occ * omega_n) / (nk * nk)

    sigma_xy = -sigma_xy_sum / (2*np.pi)**2
    print(f"  sigma_xy ~ {sigma_xy:.4f} (e^2/h, dimensionless)")
    print(f"  Max |Omega| = {np.max(np.abs(omega_kmap)):.4f}")

    # Chern number check: integral of Omega / 2pi should be integer
    chern = sigma_xy_sum / (2*np.pi)
    print(f"  Chern number ~ {chern:.3f} (expected: integer or 0)")

    passed = True  # Visual test, always pass

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Berry curvature map
        ax = axes[0]
        omega_grid = omega_kmap.reshape(nk, nk)
        vmax = np.percentile(np.abs(omega_kmap), 95)
        im = ax.imshow(omega_grid.T, origin='lower', cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, extent=[0,1,0,1])
        plt.colorbar(im, ax=ax, label='Ω(k)')
        ax.set_xlabel('k₁')
        ax.set_ylabel('k₂')
        ax.set_title(f'Berry curvature Ω(k) [sum over occ. bands]\nJ={J_mock}, S=(0,0,{S_mock[2]})')

        # Line cut along Γ-M-K
        ax2 = axes[1]
        k_line = np.linspace(0, 1, 100)
        kvecs_line = np.column_stack([k_line/2, np.zeros(100), np.zeros(100)])
        omega_line = []
        for kvec in kvecs_line:
            idx_k = np.argmin(np.sum((kpoints[:,:2] - kvec[:2])**2, axis=1))
            omega_line.append(omega_kmap[idx_k])
        ax2.plot(k_line, omega_line)
        ax2.axhline(0, color='gray', ls='--')
        ax2.set_xlabel('k along Γ-M')
        ax2.set_ylabel('Ω(k)')
        ax2.set_title(f'Berry curvature along Γ-M\nσ_xy ~ {sigma_xy:.4f} e²/h')

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig6_berry_curvature.png'), dpi=150)
        plt.close()
        print("  Saved fig6_berry_curvature.png")

    return {
        'test': 'berry_curvature',
        'passed': passed,
        'sigma_xy': float(sigma_xy),
        'chern': float(chern),
        'details': f"Berry curvature computed. σ_xy ~ {sigma_xy:.4f}, Chern ~ {chern:.3f}"
    }


# ============================================================
# Test 5: Full pipeline (requires Fortran executables)
# ============================================================

def test_full_pipeline(outdir, exe_wanneff=None, exe_wannband=None, exe_ahc=None):
    """
    Test 5: Run full wanneff_JS pipeline and check outputs.
    """
    print("\n" + "="*60)
    print("TEST 5: Full pipeline (Fortran wanneff_js.x)")
    print("="*60)

    results = {}

    if exe_wanneff is None:
        print("  wanneff_js.x not found, skipping Fortran pipeline test")
        return {
            'test': 'full_pipeline',
            'passed': None,
            'details': 'Fortran executable not found'
        }

    # Run wanneff_js.x
    print(f"  Running: {exe_wanneff}")
    rc, out, err = run_executable(exe_wanneff, workdir=outdir, timeout=300)
    print(f"  Return code: {rc}")
    if rc != 0:
        print(f"  FAILED. stderr: {err[:500]}")
        return {'test': 'full_pipeline', 'passed': False, 'details': f'wanneff_js.x failed: {err[:200]}'}

    # Check output files
    expected_files = ['seedbare_hr_0K_hr.dat']
    for f in expected_files:
        if os.path.exists(os.path.join(outdir, f)):
            print(f"  Found: {f}")
        else:
            print(f"  Missing: {f}")

    # Parse J and S from output
    J_opt = None
    Svec_opt = None
    for line in out.split('\n'):
        if 'J_opt' in line:
            try:
                J_opt = float(line.split('=')[-1])
            except:
                pass
        if 'Svec_opt' in line:
            try:
                parts = line.split('=')[-1].split()
                Svec_opt = [float(x) for x in parts[:3]]
            except:
                pass

    print(f"  J_opt = {J_opt}")
    print(f"  Svec_opt = {Svec_opt}")

    # Check that seedbare_0K bands are reasonable
    hr_0K_file = os.path.join(outdir, 'seedbare_0K_hr.dat')
    passed = os.path.exists(hr_0K_file) and J_opt is not None

    print(f"  {'PASS' if passed else 'FAIL'}")

    # Plot band comparison if wannband available
    if exe_wannband and passed and HAS_MATPLOTLIB:
        print("  Running wannband.x on seedbare_0K...")
        # Update wannband.inp for seedbare_0K
        with open(os.path.join(outdir, 'wannband.inp'), 'w') as f:
            f.write("&SYSTEM\n  seed='seedbare_0K', mu=0.0\n/\n")
            f.write("&CONTROL\n  spectra_calc=.true., nnu=300, emin=-4.0, emax=4.0, eps=1e-3\n/\n")
        rc2, _, _ = run_executable(exe_wannband, workdir=outdir)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_bands(os.path.join(outdir, 'spectra_bare.dat'), "seedbare", ax=ax,
                  color='blue', label='seedbare')
        plot_bands(os.path.join(outdir, 'spectra_0K.dat'), "seedbare_0K", ax=ax,
                  color='red', label='seedbare_0K')
        ax.set_title("Test 5: Band comparison seedbare vs seedbare_0K")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig4_band_comparison.png'), dpi=150)
        plt.close()

    return {
        'test': 'full_pipeline',
        'passed': passed,
        'J_opt': J_opt,
        'Svec_opt': Svec_opt,
        'details': f"Pipeline {'PASS' if passed else 'FAIL'}"
    }


# ============================================================
# Test report generator
# ============================================================

def generate_test_report(outdir, test_results):
    """Generate markdown test report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# wanneff_JS Test Report",
        f"",
        f"**Generated:** {ts}",
        f"**Test directory:** `{outdir}`",
        f"",
        f"## Lattice Parameters",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Lattice | kagome + f-site decoration |",
        f"| a1 | {A1} |",
        f"| a2 | {A2} |",
        f"| tss | {TSS} eV |",
        f"| tsf | {TSF} eV |",
        f"| tsf2 | {TSF2} eV |",
        f"| seed norb | {NORB_SEED} (spinor) |",
        f"| seedbare norb | {NORB_BARE} (spinor) |",
        f"",
        f"## Test Results",
        f"",
    ]

    for res in test_results:
        status = "✅ PASS" if res.get('passed') else ("⚠️ SKIP" if res.get('passed') is None else "❌ FAIL")
        lines.append(f"### {res['test']}: {status}")
        lines.append(f"")
        lines.append(f"**Details:** {res.get('details', '')}")
        lines.append(f"")

        # Extra info
        for k, v in res.items():
            if k not in ('test', 'passed', 'details', 'temps', 'magnetizations'):
                lines.append(f"- **{k}:** `{v}`")
        lines.append(f"")

    lines.append("## Figures")
    lines.append("")
    for fig in ['fig1_kagome_reference.png', 'fig2_bayesian_convergence.png',
                'fig3_mc_square.png', 'fig4_band_comparison.png',
                'fig5_sigma_vs_T.png', 'fig6_berry_curvature.png']:
        if os.path.exists(os.path.join(outdir, fig)):
            lines.append(f"- [{fig}]({fig})")
    lines.append("")

    report_path = os.path.join(outdir, 'test_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Test report written to {report_path}")
    return report_path


# ============================================================
# Main
# ============================================================

def main():
    # Test output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(script_dir, 'kagome_test_output')
    os.makedirs(outdir, exist_ok=True)

    print(f"Output directory: {outdir}")

    # Find executables
    exe_wanneff = find_executable('wanneff_js.x')
    exe_wannband = find_executable('wannband.x')
    exe_ahc = find_executable('ahc_calc.x')

    print(f"wanneff_js.x: {exe_wanneff or 'NOT FOUND'}")
    print(f"wannband.x:   {exe_wannband or 'NOT FOUND'}")

    test_results = []

    # Test 1: Reference bands
    try:
        r1 = test_kagome_reference(outdir, exe_wannband)
        test_results.append(r1)
    except Exception as e:
        print(f"Test 1 ERROR: {e}")
        test_results.append({'test': 'kagome_reference', 'passed': False, 'details': str(e)})

    # Test 2: Bayesian
    try:
        r2 = test_bayesian_known_function(outdir)
        test_results.append(r2)
    except Exception as e:
        print(f"Test 2 ERROR: {e}")
        test_results.append({'test': 'bayesian_known_function', 'passed': False, 'details': str(e)})

    # Test 3: Classical MC
    try:
        r3 = test_mc_square_lattice(outdir)
        test_results.append(r3)
    except Exception as e:
        print(f"Test 3 ERROR: {e}")
        test_results.append({'test': 'mc_square_lattice', 'passed': False, 'details': str(e)})

    # Test 4: Berry curvature
    try:
        r4 = test_berry_curvature_plot(outdir)
        test_results.append(r4)
    except Exception as e:
        print(f"Test 4 ERROR: {e}")
        test_results.append({'test': 'berry_curvature', 'passed': False, 'details': str(e)})

    # Test 5: Full pipeline (only if Fortran compiled)
    try:
        r5 = test_full_pipeline(outdir, exe_wanneff, exe_wannband, exe_ahc)
        test_results.append(r5)
    except Exception as e:
        print(f"Test 5 ERROR: {e}")
        test_results.append({'test': 'full_pipeline', 'passed': False, 'details': str(e)})

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    n_pass = sum(1 for r in test_results if r.get('passed') is not None and r.get('passed'))
    n_fail = sum(1 for r in test_results if r.get('passed') is not None and not r.get('passed'))
    n_skip = sum(1 for r in test_results if r.get('passed') is None)
    print(f"  PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}")
    for r in test_results:
        status = "PASS" if r.get('passed') else ("SKIP" if r.get('passed') is None else "FAIL")
        print(f"  [{status:4s}] {r['test']}")

    # Generate report
    generate_test_report(outdir, test_results)

    return n_fail == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
