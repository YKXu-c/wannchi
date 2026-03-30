"""
kagome_f_spinor_test.py

End-to-end pipeline test (Test 6) for wanneff_JS using the kagome+f model.

This file:
  - Generates test input files (HR, pos, IBZKPT, QPOINTS, wanneff.inp)
  - Runs compiled Fortran executables (wanneff_js.x, wannband.x)
  - Parses output files produced by Fortran
  - Plots results (AHC vs T, band structures)

Python does NO algorithmic computation here. All physics is in Fortran.
For Python algorithm validation tests (Tests 1-5), see algo_test.py.

Lattice:
  a1=(1,0,0), a2=(0.5,sqrt(3)/2,0), a3=(0,0,10)
  seed:     4 sites, site4 has nbasis=2 (s+f), total 10 spinor orbitals
  seedbare: 4 sites, site4 has nbasis=1 (s only), total 8 spinor orbitals

Usage:
  cd wannchi/tests
  source /Users/ykxu/Projects/hrJS/hrJS/bin/activate
  python3 kagome_f_spinor_test.py
"""

import numpy as np
import os
import subprocess
import sys
import glob
import shutil
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")

# ============================================================
# Constants
# ============================================================
TSS  = -1.0
TSF  = -0.5    # increased from 0.2: J_SW = tsf^2/ef = 0.25/0.4 ~ 0.63 eV
TSF2 = 0.0
EF_KAGOME = 0.4  # f within ±0.5 eV of E_F, still above for stable downfolding

A1 = np.array([1.0, 0.0, 0.0])
A2 = np.array([0.5, np.sqrt(3)/2, 0.0])
A3 = np.array([0.0, 0.0, 10.0])
AVEC = np.column_stack([A1, A2, A3])

SITES_FRAC = np.array([
    [0.0,  0.0,  0.0],
    [0.5,  0.0,  0.0],
    [0.0,  0.5,  0.0],
    [0.5,  0.5,  0.0],
])

NORB_SEED = 10
NORB_BARE = 8
NSITE_SEED = 4
NSITE_BARE = 4
NBASIS_SEED = np.array([1, 1, 1, 2])
NBASIS_BARE = np.array([1, 1, 1, 1])

# 3D Simple Cubic constants
AVEC_CUBIC = np.column_stack([[3.0,0,0], [0,3.0,0], [0,0,3.0]])
SITES_CUBIC = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
NORB_SEED_CUBIC = 6    # 3 spatial x 2 spin
NORB_BARE_CUBIC = 4    # 2 spatial x 2 spin
NBASIS_SEED_CUBIC = np.array([1, 2])  # site1: 1 orb(s), site2: 2 orbs(s+f)
NBASIS_BARE_CUBIC = np.array([1, 1])  # site1: 1 orb(s), site2: 1 orb(s)
TSS_CUBIC = -1.0   # s-s NN hopping (eV)
TSF_CUBIC = -0.5   # increased from 0.2: J_SW = tsf^2/ef
EF_CUBIC  = 0.4    # f within ±0.5 eV of E_F

# ============================================================
# WS R-vector generation (needed for HR building)
# ============================================================

def find_ws_rvectors(nr1, nr2, nr3, avec):
    metric = np.dot(avec.T, avec)
    rvecs = []; weights = []
    for ir1 in range(-nr1, nr1+1):
        for ir2 in range(-nr2, nr2+1):
            for ir3 in range(-nr3, nr3+1):
                dist = np.zeros(125); idx = 0
                for i1 in range(-2, 3):
                    for i2 in range(-2, 3):
                        for i3 in range(-2, 3):
                            ndiff = np.array([ir1-i1*nr1, ir2-i2*nr2, ir3-i3*nr3], dtype=float)
                            dist[idx] = ndiff @ metric @ ndiff; idx += 1
                dist_min = dist.min(); center_idx = 62
                if abs(dist[center_idx] - dist_min) < 1e-7:
                    weight = np.sum(np.abs(dist - dist_min) < 1e-7)
                    rvecs.append([ir1, ir2, ir3]); weights.append(weight)
    rvecs = np.array(rvecs, dtype=float).T
    weights = np.array(weights, dtype=float)
    assert abs(np.sum(1.0/weights) - nr1*nr2*nr3) < 1e-6
    return rvecs, weights


def compute_tau_from_sites(sites_frac, nbasis, spinor=True):
    taus = []
    for i_site, nbase in enumerate(nbasis):
        for _ in range(nbase):
            taus.append(sites_frac[i_site])
    if spinor:
        taus_full = taus + taus
    else:
        taus_full = taus
    return np.array(taus_full).T


def distance(f1, f2, avec, R=np.zeros(3)):
    return np.linalg.norm(avec @ f1 - avec @ (f2 + R))


def get_nn_distance(avec, sites_frac):
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


def build_hr_seed(rvecs, weights, avec):
    norb = NORB_SEED; nrpt = rvecs.shape[1]
    hr = np.zeros((norb, norb, nrpt), dtype=complex)
    nn_dist = get_nn_distance(avec, SITES_FRAC)
    nn_cutoff = nn_dist * 1.5
    s_up = [0,1,2,3]; f_up = 4; s_dn = [5,6,7,8]; f_dn = 9
    r000 = np.argmin(np.sum(rvecs**2, axis=0))
    for ir in range(nrpt):
        R = rvecs[:, ir]
        for i in range(4):
            for j in range(4):
                d = distance(SITES_FRAC[i], SITES_FRAC[j], avec, R)
                if d > 1e-8 and d < nn_cutoff:
                    hr[s_up[i], s_up[j], ir] += TSS
                    hr[s_dn[i], s_dn[j], ir] += TSS
        for i in range(4):
            d_fwd = distance(SITES_FRAC[3], SITES_FRAC[i], avec, R)
            if d_fwd > 1e-8 and d_fwd < nn_cutoff:
                hr[f_up, s_up[i], ir] += TSF
                hr[f_dn, s_dn[i], ir] += TSF
            d_rev = distance(SITES_FRAC[i], SITES_FRAC[3], avec, R)
            if d_rev > 1e-8 and d_rev < nn_cutoff:
                hr[s_up[i], f_up, ir] += np.conj(TSF)
                hr[s_dn[i], f_dn, ir] += np.conj(TSF)
    # On-site f-orbital energy: place well above E_F for stable downfolding
    hr[f_up, f_up, r000] += EF_KAGOME
    hr[f_dn, f_dn, r000] += EF_KAGOME
    return hr


def build_hr_bare(rvecs, weights, avec):
    norb = NORB_BARE; nrpt = rvecs.shape[1]
    hr = np.zeros((norb, norb, nrpt), dtype=complex)
    nn_dist = get_nn_distance(avec, SITES_FRAC[:4])
    nn_cutoff = nn_dist * 1.5
    s_up = [0,1,2,3]; s_dn = [4,5,6,7]
    for ir in range(nrpt):
        R = rvecs[:, ir]
        for i in range(4):
            for j in range(4):
                d = distance(SITES_FRAC[i], SITES_FRAC[j], avec, R)
                if d > 1e-8 and d < nn_cutoff:
                    hr[s_up[i], s_up[j], ir] += TSS
                    hr[s_dn[i], s_dn[j], ir] += TSS
    return hr


# ============================================================
# File I/O helpers
# ============================================================

def write_hr_dat(fname, hr, rvecs, weights, norb, nrpt):
    with open(fname, 'w') as f:
        f.write(f"# Kagome test Hamiltonian ({norb} orbitals)\n")
        f.write(f"{norb:10d}\n"); f.write(f"{nrpt:10d}\n")
        wt_int = [int(round(w)) for w in weights]
        for i in range(0, nrpt, 15):
            chunk = wt_int[i:min(i+15, nrpt)]
            f.write("".join(f"{w:5d}" for w in chunk) + "\n")
        for ir in range(nrpt):
            Rv = rvecs[:, ir]
            for io in range(norb):
                for jo in range(norb):
                    val = hr[jo, io, ir]
                    f.write(f"{int(round(Rv[0])):5d}{int(round(Rv[1])):5d}{int(round(Rv[2])):5d}"
                           f"{jo+1:5d}{io+1:5d}{val.real:22.16f}{val.imag:22.16f}\n")


def read_hr_dat(fname):
    """Read Wannier90 _hr.dat file. Returns (hr, rvecs, weights, norb, nrpt)."""
    with open(fname, 'r') as f:
        lines = f.readlines()
    i = 0
    # Skip comment line
    while i < len(lines) and lines[i].strip().startswith('#'):
        i += 1
    norb = int(lines[i].strip()); i += 1
    nrpt = int(lines[i].strip()); i += 1
    # Read weights (15 per line)
    weights = []
    while len(weights) < nrpt:
        weights.extend([int(x) for x in lines[i].split()])
        i += 1
    weights = np.array(weights[:nrpt], dtype=float)
    # Read HR elements
    hr = np.zeros((norb, norb, nrpt), dtype=complex)
    rvecs_set = {}
    rvec_list = []
    for _ in range(nrpt * norb * norb):
        parts = lines[i].split(); i += 1
        rx, ry, rz = int(parts[0]), int(parts[1]), int(parts[2])
        jo, io = int(parts[3])-1, int(parts[4])-1
        re, im = float(parts[5]), float(parts[6])
        key = (rx, ry, rz)
        if key not in rvecs_set:
            rvecs_set[key] = len(rvec_list)
            rvec_list.append([rx, ry, rz])
        ir = rvecs_set[key]
        hr[jo, io, ir] = complex(re, im)
    rvecs = np.array(rvec_list, dtype=float).T
    return hr, rvecs, weights, norb, nrpt


def write_pos_file(fname, avec, sites_frac, nbasis, atomic_nums, spinor=True):
    soc = 1 if spinor else 0
    nsite = len(sites_frac)
    with open(fname, 'w') as f:
        f.write("Kagome lattice\n"); f.write("1.0\n")
        for i in range(3):
            f.write(f"{avec[0,i]:20.14f}{avec[1,i]:20.14f}{avec[2,i]:20.14f}\n")
        f.write(f"{nsite:5d}{soc:5d}\n")
        for i in range(nsite):
            f.write(f"{atomic_nums[i]:5d}"
                   f"{sites_frac[i][0]:14.10f}{sites_frac[i][1]:14.10f}{sites_frac[i][2]:14.10f}"
                   f"{nbasis[i]:5d}\n")


def write_ibzkpt(fname, nk1, nk2, nk3=1):
    with open(fname, 'w') as f:
        f.write("Automatic mesh\n"); f.write(" 0\n")
        f.write("Reciprocal lattice\n"); f.write(f" {nk1} {nk2} {nk3}\n")


def write_qpoints_bandpath(fname, n_per_seg=100):
    Gamma = np.array([0.0, 0.0, 0.0])
    M     = np.array([0.5, 0.0, 0.0])
    K     = np.array([1/3, 1/3, 0.0])
    segments = [(Gamma, M), (M, K), (K, Gamma)]
    with open(fname, 'w') as f:
        f.write("1\n"); f.write(f"{len(segments)}  {n_per_seg}\n")
        for q1, q2 in segments:
            f.write(f"{q1[0]:10.6f}{q1[1]:10.6f}{q1[2]:10.6f}  "
                   f"{q2[0]:10.6f}{q2[1]:10.6f}{q2[2]:10.6f}\n")


def write_wanneff_inp(fname, seed='seed', seedbare='seedbare',
                      eff_js=True, eff_mc=True,
                      mc_temperature=(0.0, 0.000172, 0.01724),
                      eff_mode=1, J_R_range=(0,0,0,0,0,0),
                      mu=0.0, nnu=300, emin=-4.0, emax=4.0,
                      bayes_niter=100, berry_curvature_output=False,
                      ff_orbital_indices=None):
    mc_str = f"{mc_temperature[0]}, {mc_temperature[1]}, {mc_temperature[2]}"
    jr_str = " ".join(str(x) for x in J_R_range)
    with open(fname, 'w') as f:
        f.write("&SYSTEM\n")
        f.write(f"  seed='{seed}', beta=1e7, mu={mu}, spectra_calc=.true.\n/\n")
        f.write("&CONTROL\n")
        f.write(f"  nnu={nnu}, emin={emin}, emax={emax}, eps=1e-3\n/\n")
        f.write("&EFFJS\n")
        f.write(f"  seedbare='{seedbare}',\n")
        f.write(f"  eff_js={'.true.' if eff_js else '.false.'},\n")
        f.write(f"  eff_mc={'.true.' if eff_mc else '.false.'},\n")
        f.write(f"  mc_temperature={mc_str},\n")
        f.write(f"  eff_mode={eff_mode},\n")
        f.write(f"  J_R_range={jr_str},\n")
        f.write(f"  bayes_niter={bayes_niter},\n")
        f.write(f"  J_bounds=0.0 5.0,\n")
        f.write(f"  S_bounds=-3.0 3.0,\n")
        f.write(f"  tol_Jeff=1e-2,\n")
        if berry_curvature_output:
            f.write(f"  berry_curvature_output=.true.,\n")
        if ff_orbital_indices is not None:
            ff_str = ", ".join(str(x) for x in ff_orbital_indices)
            f.write(f"  n_ff_orbital_indices={len(ff_orbital_indices)},\n")
            f.write(f"  ff_orbital_indices={ff_str},\n")
        f.write("/\n")


def parse_js_output(fname):
    """Parse seed_JS.output file into a dict."""
    result = {}
    if not os.path.exists(fname):
        return result
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]; val = parts[1]
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


def parse_transport_vs_T(fname):
    """Parse seed_transport_vs_T.dat. Returns (temps_K, sigma_xy, sigma_xx) arrays."""
    if not os.path.exists(fname):
        return np.array([]), np.array([]), np.array([])
    data = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    pass
            elif len(parts) >= 2:
                try:
                    data.append([float(parts[0]), float(parts[1]), 0.0])
                except ValueError:
                    pass
    if not data:
        return np.array([]), np.array([]), np.array([])
    arr = np.array(data)
    return arr[:, 0], arr[:, 1], arr[:, 2]


# ============================================================
# Test utilities
# ============================================================

def run_executable(cmd, workdir='.', timeout=120):
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=workdir, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def find_executable(name, search_dirs=None):
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
# Plotting helpers
# ============================================================

def plot_bands(spectra_file, title="Bands", ax=None, color='blue', label=None):
    if not HAS_MATPLOTLIB:
        return None
    if not os.path.exists(spectra_file):
        print(f"  Warning: {spectra_file} not found, skipping band plot")
        return None
    data = np.loadtxt(spectra_file)
    energies = data[:, 3]; akw = data[:, 4]
    kvecs = data[:, :3]
    # Detect unique k-points and nnu from actual data structure
    unique_k = [i for i, k in enumerate(kvecs) if i == 0 or not np.allclose(k, kvecs[i-1])]
    nkpt = len(unique_k)
    # nnu = number of frequency points for first k-point (gap between first two unique k indices)
    nnu = unique_k[1] if nkpt > 1 else data.shape[0]
    n_use = nkpt * nnu
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    # Reshape to 2D grid: A(k_index, energy_index)
    A_grid = akw[:n_use].reshape(nkpt, nnu)
    E_grid = energies[:n_use].reshape(nkpt, nnu)
    e_axis = E_grid[0, :]  # energy values (same for all k-points)
    # Use imshow for proper heatmap (energy on y-axis, k on x-axis)
    vmax = max(np.percentile(A_grid, 99.5), 1e-12)
    ax.imshow(A_grid.T, aspect='auto', origin='lower', cmap='hot_r',
              extent=[0, nkpt-1, e_axis[0], e_axis[-1]],
              vmin=0, vmax=vmax, interpolation='bilinear')
    ax.set_xlabel("k-point index"); ax.set_ylabel("Energy (eV)")
    ax.set_title(title); ax.axhline(0, color='cyan', lw=0.5, ls='--')
    n_per_seg = nkpt // 3
    ax.set_xticks([0, n_per_seg, 2*n_per_seg, nkpt-1])
    ax.set_xticklabels(['Γ', 'M', 'K', 'Γ'])
    return ax


# ============================================================
# Generate test data
# ============================================================

def generate_test_data(outdir):
    """Generate all HR and input files in outdir."""
    os.makedirs(outdir, exist_ok=True)
    print("  Generating WS R-vectors (nr=5)...")
    rvecs, weights = find_ws_rvectors(5, 5, 1, AVEC)
    nrpt = rvecs.shape[1]
    print(f"  nrpt = {nrpt}")
    print("  Building seed HR...")
    hr_seed = build_hr_seed(rvecs, weights, AVEC)
    print("  Building bare HR...")
    hr_bare = build_hr_bare(rvecs, weights, AVEC)
    write_hr_dat(os.path.join(outdir, 'seed_hr.dat'), hr_seed, rvecs, weights, NORB_SEED, nrpt)
    write_hr_dat(os.path.join(outdir, 'seedbare_hr.dat'), hr_bare, rvecs, weights, NORB_BARE, nrpt)
    at_nums = [1, 1, 1, 1]
    write_pos_file(os.path.join(outdir, 'seed.pos'), AVEC, SITES_FRAC,
                   NBASIS_SEED, at_nums, spinor=True)
    write_pos_file(os.path.join(outdir, 'seedbare.pos'), AVEC, SITES_FRAC[:4],
                   NBASIS_BARE, at_nums, spinor=True)
    write_ibzkpt(os.path.join(outdir, 'IBZKPT'), 12, 12, 1)
    write_qpoints_bandpath(os.path.join(outdir, 'QPOINTS'))
    print(f"  Files written to {outdir}/")
    return rvecs, weights, hr_seed, hr_bare


# ============================================================
# Test 6: Full end-to-end pipeline
# ============================================================

def test_full_pipeline_e2e(outdir, exe_wanneff=None, exe_wannband=None):
    """
    Test 6: Full Fortran pipeline.
    Python controls I/O only; all computation is in Fortran.

    wanneff_js.x: Bayesian fit -> MC sweep -> AHC at each T -> HR output
    wannband.x:   Band structure at selected temperatures
    """
    print("\n" + "="*60)
    print("TEST 6: Full end-to-end pipeline (Fortran)")
    print("="*60)

    if exe_wanneff is None:
        print("  wanneff_js.x not found, skipping")
        return {'test': 'full_pipeline_e2e', 'passed': None, 'details': 'wanneff_js.x not found'}

    # Step 1: Generate test input data
    print("\n  [1] Generating test data...")
    generate_test_data(outdir)

    # Step 2: Write wanneff.inp with temperature sweep 0K-200K (step ~2K)
    print("  [2] Writing wanneff.inp (mc_temperature covers 0-200K)...")
    write_wanneff_inp(
        os.path.join(outdir, 'wanneff.inp'),
        seed='seed', seedbare='seedbare',
        eff_js=True, eff_mc=True,
        mc_temperature=(0.0, 0.000172, 0.01724),  # eV, 0 to ~200K
        eff_mode=1,  # scalar J mode
        bayes_niter=100,
        ff_orbital_indices=[5, 10]  # f_up, f_dn in kagome seed
    )

    # Step 3: Run wanneff_js.x
    print(f"  [3] Running wanneff_js.x...")
    print(f"      exe: {exe_wanneff}")
    try:
        rc, out, err = run_executable(exe_wanneff, workdir=outdir, timeout=600)
    except subprocess.TimeoutExpired:
        return {'test': 'full_pipeline_e2e', 'passed': False, 'details': 'wanneff_js.x timeout (600s)'}

    print(f"      Return code: {rc}")
    if rc != 0:
        print(f"      FAILED. stderr:\n{err[:1000]}")
        return {'test': 'full_pipeline_e2e', 'passed': False,
                'details': f'wanneff_js.x failed rc={rc}: {err[:300]}'}

    # Step 4: Parse seed_JS.output
    print("  [4] Parsing seed_JS.output...")
    js_out = parse_js_output(os.path.join(outdir, 'seed_JS.output'))
    J_opt = js_out.get('J_opt')
    S_mag = js_out.get('S_mag')
    print(f"      J_opt  = {J_opt}")
    print(f"      S_mag  = {S_mag}")
    print(f"      S_x    = {js_out.get('S_x')}")
    print(f"      S_y    = {js_out.get('S_y')}")
    print(f"      S_z    = {js_out.get('S_z')}")

    # Step 5: Parse seed_transport_vs_T.dat
    print("  [5] Parsing seed_transport_vs_T.dat...")
    transport_fname = os.path.join(outdir, 'seed_transport_vs_T.dat')
    temps_K, sigma_xy, sigma_xx = parse_transport_vs_T(transport_fname)
    print(f"      Transport data points: {len(temps_K)}")
    if len(temps_K) > 0:
        print(f"      T range: {temps_K[0]:.1f} - {temps_K[-1]:.1f} K")
        print(f"      sigma_xy(0K) = {sigma_xy[0]:.6f} e^2/h")
        print(f"      sigma_xx(0K) = {sigma_xx[0]:.6f} e^2/h")

    # Step 6: Discover HR files
    hr_files = sorted(glob.glob(os.path.join(outdir, 'seedbare_hr_*K_hr.dat')))
    print(f"  [6] Found {len(hr_files)} temperature HR files")

    # Step 6b: Band comparison — 4 panels: seed, seed_downfold, seedbare, seedbare+JS
    print("  [6b] Band comparison: seed / seed_downfold / seedbare / seedbare+JS...")
    if exe_wannband:
        for src_seed, spec_label, pos_src in [
            ('seed',           'full',      'seed.pos'),
            ('seed_downfold',  'downfold',  'seedbare.pos'),   # CC block from seed
            ('seedbare',       'bare',      'seedbare.pos'),
            ('seedbare_hr_0K', 'effective', 'seedbare.pos'),
        ]:
            pos_dst = os.path.join(outdir, f'{src_seed}.pos')
            if not os.path.exists(pos_dst):
                shutil.copy(os.path.join(outdir, pos_src), pos_dst)
            with open(os.path.join(outdir, 'wannband.inp'), 'w') as f:
                f.write(f"&SYSTEM\n  seed='{src_seed}', mu=0.0, spectra_calc=.true.\n/\n")
                f.write("&CONTROL\n  nnu=200, emin=-6.0, emax=6.0, eps=1e-3\n/\n")
            rc_b, _, _ = run_executable(exe_wannband, workdir=outdir, timeout=180)
            spec_out = os.path.join(outdir, f'spectra_{spec_label}.dat')
            if rc_b == 0 and os.path.exists(os.path.join(outdir, 'spectra.dat')):
                shutil.move(os.path.join(outdir, 'spectra.dat'), spec_out)
                print(f"      {src_seed}: bands OK -> spectra_{spec_label}.dat")
            else:
                print(f"      {src_seed}: wannband.x failed (rc={rc_b})")
        if HAS_MATPLOTLIB:
            fig_cmp, axes_cmp = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
            plot_bands(os.path.join(outdir, 'spectra_full.dat'),
                       title="seed (full, with f)", ax=axes_cmp[0])
            plot_bands(os.path.join(outdir, 'spectra_downfold.dat'),
                       title="seed_downfold (H_CC from downfold)", ax=axes_cmp[1])
            plot_bands(os.path.join(outdir, 'spectra_bare.dat'),
                       title="seedbare (bare, no f)", ax=axes_cmp[2])
            eff_title = f"seedbare+J·S (J={J_opt:.3f}, S=({js_out.get('S_x',0):.2f},{js_out.get('S_y',0):.2f},{js_out.get('S_z',0):.2f}))"
            plot_bands(os.path.join(outdir, 'spectra_effective.dat'),
                       title=eff_title, ax=axes_cmp[3])
            plt.suptitle("Band comparison: seed vs seed_downfold vs seedbare vs bare+J·S (kagome+f)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'fig_band_comparison.png'), dpi=150)
            plt.close()
            print("      Saved fig_band_comparison.png")

    # Step 7: Plot transport vs T
    print("  [7] Plotting transport vs T...")
    if HAS_MATPLOTLIB and len(temps_K) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(temps_K, sigma_xy, 'b-o', ms=2, lw=1.0)
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel(r"$\sigma_{xy}$ ($e^2/h$)")
        ax1.set_title("Hall conductivity (AHC)")
        ax1.axhline(0, color='gray', lw=0.5, ls='--')
        ax2.plot(temps_K, sigma_xx, 'r-o', ms=2, lw=1.0)
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel(r"$\sigma_{xx}$ ($e^2/h$)")
        ax2.set_title("Longitudinal conductivity (Kubo-Greenwood)")
        ax2.axhline(0, color='gray', lw=0.5, ls='--')
        plt.suptitle("Transport vs Temperature (kagome+f, Fortran pipeline)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig7_transport_vs_T.png'), dpi=150)
        plt.close()
        print("      Saved fig7_transport_vs_T.png")

    # Step 8: Run wannband.x at selected temperatures
    target_T_K = [0, 10, 30, 90, 200]
    bands_files = {}
    if exe_wannband:
        print(f"  [8] Running wannband.x at T = {target_T_K} K...")
        for T_K in target_T_K:
            # Find closest available HR file
            avail_T = []
            for f in hr_files:
                basename = os.path.basename(f)
                try:
                    t_str = basename.replace('seedbare_hr_', '').replace('K_hr.dat', '')
                    avail_T.append((abs(int(t_str) - T_K), int(t_str), f))
                except:
                    pass
            if not avail_T:
                print(f"      T={T_K}K: no HR file found, skipping")
                continue
            avail_T.sort()
            _, T_actual, hr_file = avail_T[0]
            seed_name = f'seedbare_hr_{T_actual}K'
            print(f"      T={T_K}K -> using {seed_name}_hr.dat (actual {T_actual}K)")

            # Copy seedbare.pos -> seedbare_hr_{T}K.pos (wannband needs {seed}.pos)
            pos_src = os.path.join(outdir, 'seedbare.pos')
            pos_dst = os.path.join(outdir, f'{seed_name}.pos')
            shutil.copy(pos_src, pos_dst)

            # Write wannband.inp
            with open(os.path.join(outdir, 'wannband.inp'), 'w') as f:
                f.write(f"&SYSTEM\n  seed='{seed_name}', mu=0.0, spectra_calc=.true.\n/\n")
                f.write("&CONTROL\n  nnu=200, emin=-4.0, emax=4.0, eps=1e-3\n/\n")

            # Run wannband.x
            rc2, _, err2 = run_executable(exe_wannband, workdir=outdir, timeout=120)
            if rc2 == 0:
                spectra_src = os.path.join(outdir, 'spectra.dat')
                spectra_dst = os.path.join(outdir, f'spectra_{T_K}K.dat')
                if os.path.exists(spectra_src):
                    shutil.move(spectra_src, spectra_dst)
                    bands_files[T_K] = spectra_dst
                    print(f"      T={T_K}K: bands OK -> spectra_{T_K}K.dat")
            else:
                print(f"      T={T_K}K: wannband.x failed (rc={rc2})")

    # Step 9: Plot 5-panel band structures
    print("  [9] Plotting band structures...")
    if HAS_MATPLOTLIB and bands_files:
        n_panels = len(target_T_K)
        fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, 5), sharey=True)
        if n_panels == 1:
            axes = [axes]
        for ax, T_K in zip(axes, target_T_K):
            if T_K in bands_files:
                plot_bands(bands_files[T_K], title=f"T = {T_K} K", ax=ax)
            else:
                ax.text(0.5, 0.5, f"T={T_K}K\n(no data)", ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f"T = {T_K} K")
        plt.suptitle("Band structure vs Temperature (kagome+f, Fortran)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig8_bands_vs_T.png'), dpi=150)
        plt.close()
        print("      Saved fig8_bands_vs_T.png")

    # Pass/fail criteria
    # Note: AHC may be zero for simple s-orbital kagome (no topological gap);
    # the pipeline completeness is what we validate here.
    passed = (rc == 0 and
              J_opt is not None and
              len(temps_K) > 5 and
              len(hr_files) >= 5 and
              len(bands_files) >= 3)

    details_parts = [
        f"wanneff_js.x rc={rc}",
        f"J_opt={J_opt}",
        f"S_mag={S_mag}",
        f"transport points={len(temps_K)}",
        f"HR files={len(hr_files)}",
        f"band plots={len(bands_files)}",
    ]
    details = ", ".join(details_parts)
    print(f"\n  {'PASS' if passed else 'FAIL'}: {details}")

    return {
        'test': 'full_pipeline_e2e',
        'passed': passed,
        'J_opt': J_opt,
        'S_mag': S_mag,
        'n_ahc_points': len(temps_K),
        'n_hr_files': len(hr_files),
        'details': details
    }


# ============================================================
# Test 7: 3D Simple Cubic lattice helpers
# ============================================================

def build_hr_seed_cubic(rvecs, weights, avec):
    """Build seed HR (6x6 spinor) for simple cubic with s+f decoration.
    Sites: (0,0,0) has 1 s-orb, (1/2,1/2,1/2) has s+f.
    Spinor order: [s1_up, s2_up, f2_up, s1_dn, s2_dn, f2_dn]
    """
    norb = NORB_SEED_CUBIC; nrpt = rvecs.shape[1]
    hr = np.zeros((norb, norb, nrpt), dtype=complex)
    nn_dist = get_nn_distance(avec, SITES_CUBIC)
    nn_cutoff = nn_dist * 1.5
    # Orbital indices (0-based): s1_up=0, s2_up=1, f2_up=2, s1_dn=3, s2_dn=4, f2_dn=5
    s_up = [0, 1]; f_up = 2; s_dn = [3, 4]; f_dn = 5
    r000 = np.argmin(np.sum(rvecs**2, axis=0))
    for ir in range(nrpt):
        R = rvecs[:, ir]
        # s-s NN hoppings (site1 <-> site2)
        for i_s, i_site in enumerate(range(2)):
            for j_s, j_site in enumerate(range(2)):
                d = distance(SITES_CUBIC[i_site], SITES_CUBIC[j_site], avec, R)
                if d > 1e-8 and d < nn_cutoff:
                    hr[s_up[i_s], s_up[j_s], ir] += TSS_CUBIC
                    hr[s_dn[i_s], s_dn[j_s], ir] += TSS_CUBIC
        # s-f hoppings: f on site2 to s on site1 and site2
        for i_s, i_site in enumerate(range(2)):
            # f at site2(0.5,0.5,0.5) to s_i at site i_site
            d_fwd = distance(SITES_CUBIC[1], SITES_CUBIC[i_site], avec, R)
            if d_fwd > 1e-8 and d_fwd < nn_cutoff:
                hr[f_up, s_up[i_s], ir] += TSF_CUBIC
                hr[f_dn, s_dn[i_s], ir] += TSF_CUBIC
            d_rev = distance(SITES_CUBIC[i_site], SITES_CUBIC[1], avec, R)
            if d_rev > 1e-8 and d_rev < nn_cutoff:
                hr[s_up[i_s], f_up, ir] += np.conj(TSF_CUBIC)
                hr[s_dn[i_s], f_dn, ir] += np.conj(TSF_CUBIC)
    # On-site f energy
    hr[f_up, f_up, r000] += EF_CUBIC
    hr[f_dn, f_dn, r000] += EF_CUBIC
    return hr


def build_hr_bare_cubic(rvecs, weights, avec):
    """Build seedbare HR (4x4 spinor) for simple cubic s-only.
    Sites: (0,0,0) has 1 s-orb, (1/2,1/2,1/2) has 1 s-orb.
    Spinor order: [s1_up, s2_up, s1_dn, s2_dn]
    """
    norb = NORB_BARE_CUBIC; nrpt = rvecs.shape[1]
    hr = np.zeros((norb, norb, nrpt), dtype=complex)
    nn_dist = get_nn_distance(avec, SITES_CUBIC)
    nn_cutoff = nn_dist * 1.5
    s_up = [0, 1]; s_dn = [2, 3]
    for ir in range(nrpt):
        R = rvecs[:, ir]
        for i_s in range(2):
            for j_s in range(2):
                d = distance(SITES_CUBIC[i_s], SITES_CUBIC[j_s], avec, R)
                if d > 1e-8 and d < nn_cutoff:
                    hr[s_up[i_s], s_up[j_s], ir] += TSS_CUBIC
                    hr[s_dn[i_s], s_dn[j_s], ir] += TSS_CUBIC
    return hr


def write_qpoints_cubic_bandpath(fname, n_per_seg=80):
    """QPOINTS for simple cubic BZ: Gamma -> X -> M -> Gamma -> R."""
    Gamma = np.array([0.0, 0.0, 0.0])
    X     = np.array([0.5, 0.0, 0.0])
    M     = np.array([0.5, 0.5, 0.0])
    R     = np.array([0.5, 0.5, 0.5])
    segments = [(Gamma, X), (X, M), (M, Gamma), (Gamma, R)]
    with open(fname, 'w') as f:
        f.write("1\n")
        f.write(f"{len(segments)}  {n_per_seg}\n")
        for q1, q2 in segments:
            f.write(f"{q1[0]:10.6f}{q1[1]:10.6f}{q1[2]:10.6f}  "
                   f"{q2[0]:10.6f}{q2[1]:10.6f}{q2[2]:10.6f}\n")


def generate_cubic_test_data(outdir):
    """Generate all HR and input files for 3D simple cubic test."""
    os.makedirs(outdir, exist_ok=True)
    print("  Generating WS R-vectors (nr=3, 3D cubic)...")
    rvecs, weights = find_ws_rvectors(3, 3, 3, AVEC_CUBIC)
    nrpt = rvecs.shape[1]
    print(f"  nrpt = {nrpt}")
    print("  Building cubic seed HR...")
    hr_seed = build_hr_seed_cubic(rvecs, weights, AVEC_CUBIC)
    print("  Building cubic bare HR...")
    hr_bare = build_hr_bare_cubic(rvecs, weights, AVEC_CUBIC)
    write_hr_dat(os.path.join(outdir, 'seed_hr.dat'), hr_seed, rvecs, weights, NORB_SEED_CUBIC, nrpt)
    write_hr_dat(os.path.join(outdir, 'seedbare_hr.dat'), hr_bare, rvecs, weights, NORB_BARE_CUBIC, nrpt)
    at_nums = [1, 1]
    write_pos_file(os.path.join(outdir, 'seed.pos'), AVEC_CUBIC, SITES_CUBIC,
                   NBASIS_SEED_CUBIC, at_nums, spinor=True)
    write_pos_file(os.path.join(outdir, 'seedbare.pos'), AVEC_CUBIC, SITES_CUBIC,
                   NBASIS_BARE_CUBIC, at_nums, spinor=True)
    write_ibzkpt(os.path.join(outdir, 'IBZKPT'), 8, 8, 8)
    write_qpoints_cubic_bandpath(os.path.join(outdir, 'QPOINTS'))
    write_wanneff_inp(
        os.path.join(outdir, 'wanneff.inp'),
        seed='seed', seedbare='seedbare',
        eff_js=True, eff_mc=True,
        mc_temperature=(0.0, 0.00026, 0.026),  # ~0 to 300K, step ~3K
        eff_mode=1, bayes_niter=100,
        ff_orbital_indices=[3, 6]  # f2_up, f2_dn in cubic seed
    )
    print(f"  Files written to {outdir}/")
    return rvecs, weights, hr_seed, hr_bare


def test_cubic_pipeline(outdir, exe_wanneff=None, exe_wannband=None):
    """
    Test 7: Full Fortran pipeline on 3D simple cubic lattice.
    """
    print("\n" + "="*60)
    print("TEST 7: 3D simple cubic pipeline (Fortran)")
    print("="*60)

    if exe_wanneff is None:
        print("  wanneff_js.x not found, skipping")
        return {'test': 'cubic_pipeline', 'passed': None, 'details': 'wanneff_js.x not found'}

    # Step 1-2: Generate data
    print("\n  [1] Generating cubic test data...")
    generate_cubic_test_data(outdir)

    # Step 3: Run wanneff_js.x
    print(f"  [3] Running wanneff_js.x (3D, may be slower)...")
    try:
        rc, out, err = run_executable(exe_wanneff, workdir=outdir, timeout=600)
    except subprocess.TimeoutExpired:
        return {'test': 'cubic_pipeline', 'passed': False, 'details': 'wanneff_js.x timeout (600s)'}
    print(f"      Return code: {rc}")
    if rc != 0:
        print(f"      FAILED. stderr:\n{err[:1000]}")
        return {'test': 'cubic_pipeline', 'passed': False,
                'details': f'wanneff_js.x failed rc={rc}: {err[:300]}'}

    # Step 4: Parse seed_JS.output
    print("  [4] Parsing seed_JS.output...")
    js_out = parse_js_output(os.path.join(outdir, 'seed_JS.output'))
    J_opt = js_out.get('J_opt'); S_mag = js_out.get('S_mag')
    print(f"      J_opt={J_opt}, S_mag={S_mag}")

    # Step 5: Parse transport
    print("  [5] Parsing seed_transport_vs_T.dat...")
    temps_K, sigma_xy, sigma_xx = parse_transport_vs_T(os.path.join(outdir, 'seed_transport_vs_T.dat'))
    print(f"      Transport data points: {len(temps_K)}")
    if len(temps_K) > 0:
        print(f"      T range: {temps_K[0]:.1f} - {temps_K[-1]:.1f} K")

    # Step 6: Discover HR files
    hr_files = sorted(glob.glob(os.path.join(outdir, 'seedbare_hr_*K_hr.dat')))
    print(f"  [6] Found {len(hr_files)} temperature HR files")

    # Step 6b: Band comparison — 4 panels: seed, seed_downfold, seedbare, seedbare+JS
    print("  [6b] Band comparison: seed / seed_downfold / seedbare / seedbare+JS (cubic)...")
    if exe_wannband:
        for src_seed, spec_label, pos_src in [
            ('seed',           'cubic_full',      'seed.pos'),
            ('seed_downfold',  'cubic_downfold',  'seedbare.pos'),
            ('seedbare',       'cubic_bare',      'seedbare.pos'),
            ('seedbare_hr_0K', 'cubic_effective', 'seedbare.pos'),
        ]:
            pos_dst = os.path.join(outdir, f'{src_seed}.pos')
            if not os.path.exists(pos_dst):
                shutil.copy(os.path.join(outdir, pos_src), pos_dst)
            with open(os.path.join(outdir, 'wannband.inp'), 'w') as f:
                f.write(f"&SYSTEM\n  seed='{src_seed}', mu=0.0, spectra_calc=.true.\n/\n")
                f.write("&CONTROL\n  nnu=200, emin=-6.0, emax=6.0, eps=1e-3\n/\n")
            rc_b, _, _ = run_executable(exe_wannband, workdir=outdir, timeout=180)
            spec_out = os.path.join(outdir, f'spectra_{spec_label}.dat')
            if rc_b == 0 and os.path.exists(os.path.join(outdir, 'spectra.dat')):
                shutil.move(os.path.join(outdir, 'spectra.dat'), spec_out)
                print(f"      {src_seed}: bands OK -> spectra_{spec_label}.dat")
            else:
                print(f"      {src_seed}: wannband.x failed (rc={rc_b})")
        if HAS_MATPLOTLIB:
            fig_cmp, axes_cmp = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
            plot_bands(os.path.join(outdir, 'spectra_cubic_full.dat'),
                       title="seed (full, with f)", ax=axes_cmp[0])
            plot_bands(os.path.join(outdir, 'spectra_cubic_downfold.dat'),
                       title="seed_downfold (H_CC from downfold)", ax=axes_cmp[1])
            plot_bands(os.path.join(outdir, 'spectra_cubic_bare.dat'),
                       title="seedbare (bare, no f)", ax=axes_cmp[2])
            eff_title = f"seedbare+J·S (J={J_opt:.3f}, S=({js_out.get('S_x',0):.2f},{js_out.get('S_y',0):.2f},{js_out.get('S_z',0):.2f}))"
            plot_bands(os.path.join(outdir, 'spectra_cubic_effective.dat'),
                       title=eff_title, ax=axes_cmp[3])
            plt.suptitle("Band comparison: seed vs seed_downfold vs seedbare vs bare+J·S (3D cubic)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'fig_cubic_band_comparison.png'), dpi=150)
            plt.close()
            print("      Saved fig_cubic_band_comparison.png")

    # Step 7: Plot transport vs T
    print("  [7] Plotting transport vs T...")
    if HAS_MATPLOTLIB and len(temps_K) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(temps_K, sigma_xy, 'b-o', ms=2, lw=1.0)
        ax1.set_xlabel("Temperature (K)"); ax1.set_ylabel(r"$\sigma_{xy}$ ($e^2/h$)")
        ax1.set_title("Hall conductivity (AHC)"); ax1.axhline(0, color='gray', lw=0.5, ls='--')
        ax2.plot(temps_K, sigma_xx, 'r-o', ms=2, lw=1.0)
        ax2.set_xlabel("Temperature (K)"); ax2.set_ylabel(r"$\sigma_{xx}$ ($e^2/h$)")
        ax2.set_title("Longitudinal conductivity"); ax2.axhline(0, color='gray', lw=0.5, ls='--')
        plt.suptitle("Transport vs T (3D simple cubic, Fortran)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig9_cubic_transport_vs_T.png'), dpi=150)
        plt.close()
        print("      Saved fig9_cubic_transport_vs_T.png")

    # Step 8: Run wannband.x at selected temperatures
    target_T_K = [0, 30, 100, 200, 300]
    bands_files = {}
    if exe_wannband:
        print(f"  [8] Running wannband.x at T = {target_T_K} K...")
        for T_K in target_T_K:
            avail_T = []
            for f in hr_files:
                basename = os.path.basename(f)
                try:
                    t_str = basename.replace('seedbare_hr_', '').replace('K_hr.dat', '')
                    avail_T.append((abs(int(t_str) - T_K), int(t_str), f))
                except:
                    pass
            if not avail_T:
                print(f"      T={T_K}K: no HR file found"); continue
            avail_T.sort()
            _, T_actual, hr_file = avail_T[0]
            seed_name = f'seedbare_hr_{T_actual}K'
            print(f"      T={T_K}K -> {seed_name}")
            shutil.copy(os.path.join(outdir, 'seedbare.pos'), os.path.join(outdir, f'{seed_name}.pos'))
            with open(os.path.join(outdir, 'wannband.inp'), 'w') as f:
                f.write(f"&SYSTEM\n  seed='{seed_name}', mu=0.0, spectra_calc=.true.\n/\n")
                f.write("&CONTROL\n  nnu=200, emin=-6.0, emax=6.0, eps=1e-3\n/\n")
            rc2, _, _ = run_executable(exe_wannband, workdir=outdir, timeout=120)
            if rc2 == 0:
                spectra_src = os.path.join(outdir, 'spectra.dat')
                spectra_dst = os.path.join(outdir, f'spectra_cubic_{T_K}K.dat')
                if os.path.exists(spectra_src):
                    shutil.move(spectra_src, spectra_dst)
                    bands_files[T_K] = spectra_dst
                    print(f"      T={T_K}K: bands OK")
            else:
                print(f"      T={T_K}K: wannband.x failed (rc={rc2})")

    # Step 9: Plot bands
    print("  [9] Plotting band structures...")
    if HAS_MATPLOTLIB and bands_files:
        n_panels = len(target_T_K)
        fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, 5), sharey=True)
        if n_panels == 1: axes = [axes]
        for ax, T_K in zip(axes, target_T_K):
            if T_K in bands_files:
                plot_bands(bands_files[T_K], title=f"T = {T_K} K", ax=ax)
            else:
                ax.text(0.5, 0.5, f"T={T_K}K\n(no data)", ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f"T = {T_K} K")
        # Relabel x-ticks for cubic BZ path
        for ax in axes:
            n_per_seg = 80; nkpt_est = 4 * n_per_seg + 1
            ticks = [0, n_per_seg, 2*n_per_seg, 3*n_per_seg, nkpt_est-1]
            labels_bz = ['Γ', 'X', 'M', 'Γ', 'R']
            ax.set_xticks(ticks); ax.set_xticklabels(labels_bz)
        plt.suptitle("Band structure vs T (3D simple cubic, Fortran)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig10_cubic_bands_vs_T.png'), dpi=150)
        plt.close()
        print("      Saved fig10_cubic_bands_vs_T.png")

    # Pass/fail
    passed = (rc == 0 and J_opt is not None and
              len(temps_K) > 5 and len(hr_files) >= 5 and len(bands_files) >= 3)
    details = (f"rc={rc}, J_opt={J_opt}, S_mag={S_mag}, "
               f"transport={len(temps_K)}, HR={len(hr_files)}, bands={len(bands_files)}")
    print(f"\n  {'PASS' if passed else 'FAIL'}: {details}")
    return {'test': 'cubic_pipeline', 'passed': passed,
            'J_opt': J_opt, 'S_mag': S_mag,
            'n_transport_points': len(temps_K), 'n_hr_files': len(hr_files),
            'details': details}


# ============================================================
# Test 8: J_TENSOR pipeline (generic for kagome and cubic)
# ============================================================

def test_tensor_pipeline(outdir, exe_wanneff=None, exe_wannband=None,
                         label='tensor', generate_fn=None,
                         temps=None, emin=-6.0, emax=6.0):
    """
    Test 8 (a or b): Full pipeline with J_TENSOR=True.
    generate_fn: either generate_test_data or generate_cubic_test_data
    label: short label for figure naming
    temps: list of target temperatures in K for band plots
    """
    if temps is None:
        temps = [0, 10, 30, 90, 200]
    print("\n" + "="*60)
    print(f"TEST 8 ({label}): J_TENSOR pipeline (Fortran)")
    print("="*60)

    if exe_wanneff is None:
        print("  wanneff_js.x not found, skipping")
        return {'test': f'tensor_pipeline_{label}', 'passed': None,
                'details': 'wanneff_js.x not found'}

    # Step 1: Generate test data
    print(f"\n  [1] Generating {label} test data...")
    generate_fn(outdir)

    # Determine mc_temperature based on label
    if 'cubic' in label:
        mc_temp = (0.0, 0.00026, 0.026)  # 0-300K
    else:
        mc_temp = (0.0, 0.000172, 0.01724)  # 0-200K

    # Step 2: Write wanneff.inp with J_TENSOR=True
    print(f"  [2] Writing wanneff.inp (J_TENSOR=True, J_R_range=±1)...")
    # Determine FF indices based on label
    if 'cubic' in label:
        ff_indices = [3, 6]  # cubic: f2_up, f2_dn
    else:
        ff_indices = [5, 10]  # kagome: f_up, f_dn
    write_wanneff_inp(
        os.path.join(outdir, 'wanneff.inp'),
        seed='seed', seedbare='seedbare',
        eff_js=True, eff_mc=True,
        mc_temperature=mc_temp,
        eff_mode=2,  # J_TENSOR mode
        J_R_range=(-1, 1, -1, 1, -1, 1),
        bayes_niter=100,
        emin=emin, emax=emax,
        ff_orbital_indices=ff_indices
    )

    # Step 3: Run wanneff_js.x
    print(f"  [3] Running wanneff_js.x (J_TENSOR mode, may be slower)...")
    try:
        rc, out, err = run_executable(exe_wanneff, workdir=outdir, timeout=900)
    except subprocess.TimeoutExpired:
        return {'test': f'tensor_pipeline_{label}', 'passed': False,
                'details': 'wanneff_js.x timeout (900s)'}
    print(f"      Return code: {rc}")
    if rc != 0:
        print(f"      FAILED. stderr:\n{err[:1000]}")
        return {'test': f'tensor_pipeline_{label}', 'passed': False,
                'details': f'rc={rc}: {err[:300]}'}

    # Step 4: Parse outputs
    print("  [4] Parsing seed_JS.output...")
    js_out = parse_js_output(os.path.join(outdir, 'seed_JS.output'))
    J_opt = js_out.get('J_opt')  # scalar summary (may be 1.0 for tensor mode)
    S_mag = js_out.get('S_mag')
    n_jrpt = js_out.get('n_jrpt')
    print(f"      S_mag={S_mag}, n_jrpt={n_jrpt}")

    # Step 5: Parse transport
    print("  [5] Parsing seed_transport_vs_T.dat...")
    temps_K, sigma_xy, sigma_xx = parse_transport_vs_T(
        os.path.join(outdir, 'seed_transport_vs_T.dat'))
    print(f"      Transport data points: {len(temps_K)}")

    # Step 6: Discover HR files
    hr_files = sorted(glob.glob(os.path.join(outdir, 'seedbare_hr_*K_hr.dat')))
    print(f"  [6] Found {len(hr_files)} temperature HR files")

    # Step 6b: Band comparison — 4 panels: seed, seed_downfold, seedbare, seedbare+JS
    print(f"  [6b] Band comparison: seed / seed_downfold / seedbare / seedbare+JS ({label})...")
    if exe_wannband:
        for src_seed, spec_label, pos_src in [
            ('seed',           f'{label}_full',      'seed.pos'),
            ('seed_downfold',  f'{label}_downfold', 'seedbare.pos'),
            ('seedbare',       f'{label}_bare',     'seedbare.pos'),
            ('seedbare_hr_0K', f'{label}_effective','seedbare.pos'),
        ]:
            pos_dst = os.path.join(outdir, f'{src_seed}.pos')
            if not os.path.exists(pos_dst):
                shutil.copy(os.path.join(outdir, pos_src), pos_dst)
            with open(os.path.join(outdir, 'wannband.inp'), 'w') as f:
                f.write(f"&SYSTEM\n  seed='{src_seed}', mu=0.0, spectra_calc=.true.\n/\n")
                f.write(f"&CONTROL\n  nnu=200, emin={emin}, emax={emax}, eps=1e-3\n/\n")
            rc_b, _, _ = run_executable(exe_wannband, workdir=outdir, timeout=180)
            spec_out = os.path.join(outdir, f'spectra_{spec_label}.dat')
            if rc_b == 0 and os.path.exists(os.path.join(outdir, 'spectra.dat')):
                shutil.move(os.path.join(outdir, 'spectra.dat'), spec_out)
                print(f"      {src_seed}: bands OK -> spectra_{spec_label}.dat")
            else:
                print(f"      {src_seed}: wannband.x failed (rc={rc_b})")
        if HAS_MATPLOTLIB:
            fig_cmp, axes_cmp = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
            plot_bands(os.path.join(outdir, f'spectra_{label}_full.dat'),
                       title="seed (full, with f)", ax=axes_cmp[0])
            plot_bands(os.path.join(outdir, f'spectra_{label}_downfold.dat'),
                       title="seed_downfold (H_CC)", ax=axes_cmp[1])
            plot_bands(os.path.join(outdir, f'spectra_{label}_bare.dat'),
                       title="seedbare (bare, no f)", ax=axes_cmp[2])
            eff_title = f"bare+J·S (J_TENSOR, S={S_mag:.2f})"
            plot_bands(os.path.join(outdir, f'spectra_{label}_effective.dat'),
                       title=eff_title, ax=axes_cmp[3])
            plt.suptitle(f"Band comparison: J_TENSOR ({label})")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'fig_{label}_band_comparison.png'), dpi=150)
            plt.close()
            print(f"      Saved fig_{label}_band_comparison.png")

    # Step 7: Plot transport
    print("  [7] Plotting transport vs T...")
    if HAS_MATPLOTLIB and len(temps_K) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(temps_K, sigma_xy, 'b-o', ms=2, lw=1.0)
        ax1.set_xlabel("T (K)"); ax1.set_ylabel(r"$\sigma_{xy}$ ($e^2/h$)")
        ax1.set_title("Hall (AHC)"); ax1.axhline(0, color='gray', lw=0.5, ls='--')
        ax2.plot(temps_K, sigma_xx, 'r-o', ms=2, lw=1.0)
        ax2.set_xlabel("T (K)"); ax2.set_ylabel(r"$\sigma_{xx}$ ($e^2/h$)")
        ax2.set_title("Longitudinal (Drude+Kubo)"); ax2.axhline(0, color='gray', lw=0.5, ls='--')
        plt.suptitle(f"Transport vs T: J_TENSOR ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'fig_{label}_transport_vs_T.png'), dpi=150)
        plt.close()
        print(f"      Saved fig_{label}_transport_vs_T.png")

    # Step 8: Band plots at selected T
    bands_files = {}
    if exe_wannband:
        print(f"  [8] Running wannband.x at T = {temps} K...")
        for T_K in temps:
            avail_T = []
            for f in hr_files:
                try:
                    t_str = os.path.basename(f).replace('seedbare_hr_','').replace('K_hr.dat','')
                    avail_T.append((abs(int(t_str)-T_K), int(t_str), f))
                except: pass
            if not avail_T: continue
            avail_T.sort()
            _, T_actual, _ = avail_T[0]
            seed_name = f'seedbare_hr_{T_actual}K'
            shutil.copy(os.path.join(outdir,'seedbare.pos'), os.path.join(outdir,f'{seed_name}.pos'))
            with open(os.path.join(outdir,'wannband.inp'),'w') as f:
                f.write(f"&SYSTEM\n  seed='{seed_name}', mu=0.0, spectra_calc=.true.\n/\n")
                f.write(f"&CONTROL\n  nnu=200, emin={emin}, emax={emax}, eps=1e-3\n/\n")
            rc2, _, _ = run_executable(exe_wannband, workdir=outdir, timeout=120)
            if rc2 == 0 and os.path.exists(os.path.join(outdir,'spectra.dat')):
                dst = os.path.join(outdir,f'spectra_{label}_{T_K}K.dat')
                shutil.move(os.path.join(outdir,'spectra.dat'), dst)
                bands_files[T_K] = dst
                print(f"      T={T_K}K: OK")

    if HAS_MATPLOTLIB and bands_files:
        n_p = len(temps)
        fig, axes = plt.subplots(1, n_p, figsize=(4*n_p, 5), sharey=True)
        if n_p == 1: axes = [axes]
        for ax, T_K in zip(axes, temps):
            if T_K in bands_files:
                plot_bands(bands_files[T_K], title=f"T={T_K}K", ax=ax)
            else:
                ax.text(0.5, 0.5, f"no data", ha='center', va='center', transform=ax.transAxes)
        plt.suptitle(f"Bands vs T: J_TENSOR ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'fig_{label}_bands_vs_T.png'), dpi=150)
        plt.close()
        print(f"      Saved fig_{label}_bands_vs_T.png")

    passed = (rc == 0 and S_mag is not None and len(temps_K) > 5 and len(hr_files) >= 5)
    details = f"rc={rc}, S_mag={S_mag}, n_jrpt={n_jrpt}, T_points={len(temps_K)}, HR={len(hr_files)}"
    print(f"\n  {'PASS' if passed else 'FAIL'}: {details}")
    return {'test': f'tensor_pipeline_{label}', 'passed': passed,
            'S_mag': S_mag, 'n_jrpt': n_jrpt, 'details': details}


# ============================================================
# Test 10: J_S_TENSOR pipeline (generic for kagome and cubic)
# ============================================================

def test_js_tensor_pipeline(outdir, exe_wanneff=None, exe_wannband=None,
                            label='js_tensor', generate_fn=None,
                            temps=None, emin=-6.0, emax=6.0):
    """
    Test 10 (a,b,c): Full pipeline with eff_mode=3 (J_S_TENSOR).
    J and S both vary with R: params = (J_R, S_x(R), S_y(R), S_z(R))
    generate_fn: either generate_test_data or generate_cubic_test_data
    label: short label for figure naming
    temps: list of target temperatures in K for band plots
    """
    if temps is None:
        temps = [0, 10, 30, 90, 200]
    print("\n" + "="*60)
    print(f"TEST 10 ({label}): J_S_TENSOR pipeline (Fortran)")
    print("="*60)

    if exe_wanneff is None:
        print("  wanneff_js.x not found, skipping")
        return {'test': f'js_tensor_pipeline_{label}', 'passed': None,
                'details': 'wanneff_js.x not found'}

    # Step 1: Generate test data
    print(f"\n  [1] Generating {label} test data...")
    generate_fn(outdir)

    # Determine mc_temperature based on label
    if 'cubic' in label:
        mc_temp = (0.0, 0.00026, 0.026)  # 0-300K
    else:
        mc_temp = (0.0, 0.000172, 0.01724)  # 0-200K

    # Step 2: Write wanneff.inp with eff_mode=3 (J_S_TENSOR)
    print(f"  [2] Writing wanneff.inp (eff_mode=3, J_S_TENSOR, J_R_range=±1)...")
    # Determine FF indices based on label
    if 'cubic' in label:
        ff_indices = [3, 6]  # cubic: f2_up, f2_dn
    elif 'k3' in label or 'kagome3' in label:
        ff_indices = [4, 8]  # kagome3: f2_up, f2_dn
    else:
        ff_indices = [5, 10]  # kagome: f_up, f_dn
    write_wanneff_inp(
        os.path.join(outdir, 'wanneff.inp'),
        seed='seed', seedbare='seedbare',
        eff_js=True, eff_mc=True,
        mc_temperature=mc_temp,
        eff_mode=3,  # J_S_TENSOR mode
        J_R_range=(0, 0, 0, 0, 0, 0),  # use full R-grid from ham_bare
        bayes_niter=100,
        emin=emin, emax=emax,
        ff_orbital_indices=ff_indices
    )

    # Step 3: Run wanneff_js.x
    print(f"  [3] Running wanneff_js.x (J_S_TENSOR mode, may be slower)...")
    try:
        rc, out, err = run_executable(exe_wanneff, workdir=outdir, timeout=900)
    except subprocess.TimeoutExpired:
        return {'test': f'js_tensor_pipeline_{label}', 'passed': False,
                'details': 'wanneff_js.x timeout (900s)'}
    print(f"      Return code: {rc}")
    if rc != 0:
        print(f"      FAILED. stderr:\n{err[:1000]}")
        return {'test': f'js_tensor_pipeline_{label}', 'passed': False,
                'details': f'rc={rc}: {err[:300]}'}

    # Step 4: Parse outputs
    print("  [4] Parsing seed_JS.output...")
    js_out = parse_js_output(os.path.join(outdir, 'seed_JS.output'))
    J_opt = js_out.get('J_opt')  # scalar summary (may be 1.0 for tensor mode)
    S_mag = js_out.get('S_mag')
    n_jrpt = js_out.get('n_jrpt')
    print(f"      S_mag={S_mag}, n_jrpt={n_jrpt}")
    # Print J and S vectors per R if available
    if 'S_R' in js_out or 'J_R' in js_out:
        print(f"      J_R: {js_out.get('J_R', 'N/A')}")
        print(f"      S_R: {js_out.get('S_R', 'N/A')}")

    # Step 5: Parse transport
    print("  [5] Parsing seed_transport_vs_T.dat...")
    temps_K, sigma_xy, sigma_xx = parse_transport_vs_T(
        os.path.join(outdir, 'seed_transport_vs_T.dat'))
    print(f"      Transport data points: {len(temps_K)}")

    # Step 6: Discover HR files
    hr_files = sorted(glob.glob(os.path.join(outdir, 'seedbare_hr_*K_hr.dat')))
    print(f"  [6] Found {len(hr_files)} temperature HR files")

    # Step 6b: Band comparison — 4 panels: seed, seed_downfold, seedbare, seedbare+JS
    print(f"  [6b] Band comparison: seed / seed_downfold / seedbare / seedbare+JS ({label})...")
    if exe_wannband:
        for src_seed, spec_label, pos_src in [
            ('seed',           f'{label}_full',      'seed.pos'),
            ('seed_downfold',  f'{label}_downfold', 'seedbare.pos'),
            ('seedbare',       f'{label}_bare',     'seedbare.pos'),
            ('seedbare_hr_0K', f'{label}_effective','seedbare.pos'),
        ]:
            pos_dst = os.path.join(outdir, f'{src_seed}.pos')
            if not os.path.exists(pos_dst):
                shutil.copy(os.path.join(outdir, pos_src), pos_dst)
            with open(os.path.join(outdir, 'wannband.inp'), 'w') as f:
                f.write(f"&SYSTEM\n  seed='{src_seed}', mu=0.0, spectra_calc=.true.\n/\n")
                f.write(f"&CONTROL\n  nnu=200, emin={emin}, emax={emax}, eps=1e-3\n/\n")
            rc_b, _, _ = run_executable(exe_wannband, workdir=outdir, timeout=180)
            spec_out = os.path.join(outdir, f'spectra_{spec_label}.dat')
            if rc_b == 0 and os.path.exists(os.path.join(outdir, 'spectra.dat')):
                shutil.move(os.path.join(outdir, 'spectra.dat'), spec_out)
                print(f"      {src_seed}: bands OK -> spectra_{spec_label}.dat")
            else:
                print(f"      {src_seed}: wannband.x failed (rc={rc_b})")
        if HAS_MATPLOTLIB:
            fig_cmp, axes_cmp = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
            plot_bands(os.path.join(outdir, f'spectra_{label}_full.dat'),
                       title="seed (full, with f)", ax=axes_cmp[0])
            plot_bands(os.path.join(outdir, f'spectra_{label}_downfold.dat'),
                       title="seed_downfold (H_CC)", ax=axes_cmp[1])
            plot_bands(os.path.join(outdir, f'spectra_{label}_bare.dat'),
                       title="seedbare (bare, no f)", ax=axes_cmp[2])
            eff_title = f"bare+J·S (J_S_TENSOR, S={S_mag:.2f})"
            plot_bands(os.path.join(outdir, f'spectra_{label}_effective.dat'),
                       title=eff_title, ax=axes_cmp[3])
            plt.suptitle(f"Band comparison: J_S_TENSOR ({label})")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'fig_{label}_band_comparison.png'), dpi=150)
            plt.close()
            print(f"      Saved fig_{label}_band_comparison.png")

    # Step 7: Plot transport
    print("  [7] Plotting transport vs T...")
    if HAS_MATPLOTLIB and len(temps_K) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(temps_K, sigma_xy, 'b-o', ms=2, lw=1.0)
        ax1.set_xlabel("T (K)"); ax1.set_ylabel(r"$\sigma_{xy}$ ($e^2/h$)")
        ax1.set_title("Hall (AHC)"); ax1.axhline(0, color='gray', lw=0.5, ls='--')
        ax2.plot(temps_K, sigma_xx, 'r-o', ms=2, lw=1.0)
        ax2.set_xlabel("T (K)"); ax2.set_ylabel(r"$\sigma_{xx}$ ($e^2/h$)")
        ax2.set_title("Longitudinal (Drude+Kubo)"); ax2.axhline(0, color='gray', lw=0.5, ls='--')
        plt.suptitle(f"Transport vs T: J_S_TENSOR ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'fig_{label}_transport_vs_T.png'), dpi=150)
        plt.close()
        print(f"      Saved fig_{label}_transport_vs_T.png")

    # Step 8: Band plots at selected T
    bands_files = {}
    if exe_wannband:
        print(f"  [8] Running wannband.x at T = {temps} K...")
        for T_K in temps:
            avail_T = []
            for f in hr_files:
                try:
                    t_str = os.path.basename(f).replace('seedbare_hr_','').replace('K_hr.dat','')
                    avail_T.append((abs(int(t_str)-T_K), int(t_str), f))
                except: pass
            if not avail_T: continue
            avail_T.sort()
            _, T_actual, _ = avail_T[0]
            seed_name = f'seedbare_hr_{T_actual}K'
            shutil.copy(os.path.join(outdir,'seedbare.pos'), os.path.join(outdir,f'{seed_name}.pos'))
            with open(os.path.join(outdir,'wannband.inp'),'w') as f:
                f.write(f"&SYSTEM\n  seed='{seed_name}', mu=0.0, spectra_calc=.true.\n/\n")
                f.write(f"&CONTROL\n  nnu=200, emin={emin}, emax={emax}, eps=1e-3\n/\n")
            rc2, _, _ = run_executable(exe_wannband, workdir=outdir, timeout=120)
            if rc2 == 0 and os.path.exists(os.path.join(outdir,'spectra.dat')):
                dst = os.path.join(outdir,f'spectra_{label}_{T_K}K.dat')
                shutil.move(os.path.join(outdir,'spectra.dat'), dst)
                bands_files[T_K] = dst
                print(f"      T={T_K}K: OK")

    if HAS_MATPLOTLIB and bands_files:
        n_p = len(temps)
        fig, axes = plt.subplots(1, n_p, figsize=(4*n_p, 5), sharey=True)
        if n_p == 1: axes = [axes]
        for ax, T_K in zip(axes, temps):
            if T_K in bands_files:
                plot_bands(bands_files[T_K], title=f"T={T_K}K", ax=ax)
            else:
                ax.text(0.5, 0.5, f"no data", ha='center', va='center', transform=ax.transAxes)
        plt.suptitle(f"Bands vs T: J_S_TENSOR ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'fig_{label}_bands_vs_T.png'), dpi=150)
        plt.close()
        print(f"      Saved fig_{label}_bands_vs_T.png")

    passed = (rc == 0 and S_mag is not None and len(temps_K) > 5 and len(hr_files) >= 5)
    details = f"rc={rc}, S_mag={S_mag}, n_jrpt={n_jrpt}, T_points={len(temps_K)}, HR={len(hr_files)}"
    print(f"\n  {'PASS' if passed else 'FAIL'}: {details}")
    return {'test': f'js_tensor_pipeline_{label}', 'passed': passed,
            'S_mag': S_mag, 'n_jrpt': n_jrpt, 'details': details}


# ============================================================
# Test 9: 3-site kagome lattice (proper flat band + f-decoration)
# ============================================================

# 3-site kagome constants: 3 vertices only, site 2 has s+f
AVEC_K3 = AVEC  # same hexagonal lattice vectors
SITES_K3 = np.array([
    [0.5, 0.0, 0.0],   # kagome vertex 1 (s)
    [0.0, 0.5, 0.0],   # kagome vertex 2 (s)
    [0.5, 0.5, 0.0],   # kagome vertex 3 (s+f)
])
NBASIS_SEED_K3 = np.array([1, 1, 2])   # site2 has s+f
NBASIS_BARE_K3 = np.array([1, 1, 1])   # s only → flat band
NORB_SEED_K3 = 8   # (1+1+2)*2 spin
NORB_BARE_K3 = 6   # 3*2 spin
TSF_K3 = -0.5      # increased from 0.2: J_SW = tsf^2/ef


def build_hr_seed_kagome3(rvecs, weights, avec):
    """Build 8x8 spinor HR for 3-site kagome with f-decoration at site 2.
    Spinor order: [s0_up, s1_up, s2_up, f2_up, s0_dn, s1_dn, s2_dn, f2_dn]
    """
    norb = NORB_SEED_K3; nrpt = rvecs.shape[1]
    hr = np.zeros((norb, norb, nrpt), dtype=complex)
    nn_dist = get_nn_distance(avec, SITES_K3)
    nn_cutoff = nn_dist * 1.5
    s_up = [0, 1, 2]; f_up = 3; s_dn = [4, 5, 6]; f_dn = 7
    r000 = np.argmin(np.sum(rvecs**2, axis=0))
    for ir in range(nrpt):
        R = rvecs[:, ir]
        # s-s hoppings between all 3 sites
        for i in range(3):
            for j in range(3):
                d = distance(SITES_K3[i], SITES_K3[j], avec, R)
                if d > 1e-8 and d < nn_cutoff:
                    hr[s_up[i], s_up[j], ir] += TSS
                    hr[s_dn[i], s_dn[j], ir] += TSS
        # s-f hoppings: f on site2 to s on sites 0,1,2
        for i in range(3):
            d_fwd = distance(SITES_K3[2], SITES_K3[i], avec, R)
            if d_fwd > 1e-8 and d_fwd < nn_cutoff:
                hr[f_up, s_up[i], ir] += TSF_K3
                hr[f_dn, s_dn[i], ir] += TSF_K3
            d_rev = distance(SITES_K3[i], SITES_K3[2], avec, R)
            if d_rev > 1e-8 and d_rev < nn_cutoff:
                hr[s_up[i], f_up, ir] += np.conj(TSF_K3)
                hr[s_dn[i], f_dn, ir] += np.conj(TSF_K3)
    # f on-site energy (EF_KAGOME) at R=0
    hr[f_up, f_up, r000] += EF_KAGOME
    hr[f_dn, f_dn, r000] += EF_KAGOME
    return hr


def build_hr_bare_kagome3(rvecs, weights, avec):
    """Build 6x6 spinor HR for 3-site kagome (s-only → flat band).
    Spinor order: [s0_up, s1_up, s2_up, s0_dn, s1_dn, s2_dn]
    """
    norb = NORB_BARE_K3; nrpt = rvecs.shape[1]
    hr = np.zeros((norb, norb, nrpt), dtype=complex)
    nn_dist = get_nn_distance(avec, SITES_K3)
    nn_cutoff = nn_dist * 1.5
    s_up = [0, 1, 2]; s_dn = [3, 4, 5]
    for ir in range(nrpt):
        R = rvecs[:, ir]
        for i in range(3):
            for j in range(3):
                d = distance(SITES_K3[i], SITES_K3[j], avec, R)
                if d > 1e-8 and d < nn_cutoff:
                    hr[s_up[i], s_up[j], ir] += TSS
                    hr[s_dn[i], s_dn[j], ir] += TSS
    return hr


def generate_kagome3_test_data(outdir):
    """Generate HR and input files for 3-site kagome test."""
    os.makedirs(outdir, exist_ok=True)
    print("  Generating WS R-vectors (3-site kagome, nr=5)...")
    rvecs, weights = find_ws_rvectors(5, 5, 1, AVEC_K3)
    nrpt = rvecs.shape[1]
    print(f"  nrpt = {nrpt}")
    print("  Building 3-site kagome seed HR (flat band + f)...")
    hr_seed = build_hr_seed_kagome3(rvecs, weights, AVEC_K3)
    print("  Building 3-site kagome bare HR (flat band)...")
    hr_bare = build_hr_bare_kagome3(rvecs, weights, AVEC_K3)
    write_hr_dat(os.path.join(outdir, 'seed_hr.dat'),
                 hr_seed, rvecs, weights, NORB_SEED_K3, nrpt)
    write_hr_dat(os.path.join(outdir, 'seedbare_hr.dat'),
                 hr_bare, rvecs, weights, NORB_BARE_K3, nrpt)
    at_nums = [1, 1, 1]
    write_pos_file(os.path.join(outdir, 'seed.pos'), AVEC_K3, SITES_K3,
                   NBASIS_SEED_K3, at_nums, spinor=True)
    write_pos_file(os.path.join(outdir, 'seedbare.pos'), AVEC_K3, SITES_K3,
                   NBASIS_BARE_K3, at_nums, spinor=True)
    write_ibzkpt(os.path.join(outdir, 'IBZKPT'), 12, 12, 1)
    write_qpoints_bandpath(os.path.join(outdir, 'QPOINTS'))  # Γ-M-K-Γ
    write_wanneff_inp(
        os.path.join(outdir, 'wanneff.inp'),
        seed='seed', seedbare='seedbare',
        eff_js=True, eff_mc=True,
        mc_temperature=(0.0, 0.000172, 0.01724),
        eff_mode=1, bayes_niter=100,
        ff_orbital_indices=[4, 8]  # f2_up, f2_dn in kagome3 seed
    )
    print(f"  Files written to {outdir}/")
    return rvecs, weights, hr_seed, hr_bare


def test_kagome3_pipeline(outdir, exe_wanneff=None, exe_wannband=None):
    """
    Test 9: 3-site kagome with proper flat band + f-decoration.
    seedbare has 6 spinor orbs → classic kagome flat band.
    seed has 8 spinor orbs → hybridized bands clearly distinct from seedbare.
    """
    print("\n" + "="*60)
    print("TEST 9: 3-site kagome flat band + f-decoration (Fortran)")
    print("="*60)

    if exe_wanneff is None:
        print("  wanneff_js.x not found, skipping")
        return {'test': 'kagome3_pipeline', 'passed': None, 'details': 'wanneff_js.x not found'}

    # Step 1: Generate test data
    print("\n  [1] Generating 3-site kagome test data...")
    generate_kagome3_test_data(outdir)

    # Step 3: Run wanneff_js.x
    print(f"  [3] Running wanneff_js.x...")
    try:
        rc, out, err = run_executable(exe_wanneff, workdir=outdir, timeout=600)
    except subprocess.TimeoutExpired:
        return {'test': 'kagome3_pipeline', 'passed': False, 'details': 'timeout (600s)'}
    print(f"      Return code: {rc}")
    if rc != 0:
        print(f"      FAILED.\n{err[:500]}")
        return {'test': 'kagome3_pipeline', 'passed': False, 'details': f'rc={rc}: {err[:200]}'}

    # Step 4: Parse outputs
    print("  [4] Parsing seed_JS.output...")
    js_out = parse_js_output(os.path.join(outdir, 'seed_JS.output'))
    J_opt = js_out.get('J_opt'); S_mag = js_out.get('S_mag'); L2 = js_out.get('L2_best')
    print(f"      J_opt={J_opt}, S_mag={S_mag}, L2_best={L2}")

    # Step 5: Parse transport
    temps_K, sigma_xy, sigma_xx = parse_transport_vs_T(
        os.path.join(outdir, 'seed_transport_vs_T.dat'))
    print(f"  [5] Transport data points: {len(temps_K)}")

    # Step 6: HR files
    hr_files = sorted(glob.glob(os.path.join(outdir, 'seedbare_hr_*K_hr.dat')))
    print(f"  [6] Found {len(hr_files)} temperature HR files")

    # Step 6b: Band comparison — 4 panels: seed, seed_downfold, seedbare, seedbare+JS
    print("  [6b] Band comparison: seed / seed_downfold / seedbare / seedbare+JS (kagome3)...")
    if exe_wannband:
        for src_seed, spec_label, pos_src in [
            ('seed',           'k3_full',      'seed.pos'),
            ('seed_downfold',  'k3_downfold',  'seedbare.pos'),
            ('seedbare',       'k3_bare',      'seedbare.pos'),
            ('seedbare_hr_0K', 'k3_effective', 'seedbare.pos'),
        ]:
            pos_dst = os.path.join(outdir, f'{src_seed}.pos')
            if not os.path.exists(pos_dst):
                shutil.copy(os.path.join(outdir, pos_src), pos_dst)
            with open(os.path.join(outdir, 'wannband.inp'), 'w') as f:
                f.write(f"&SYSTEM\n  seed='{src_seed}', mu=0.0, spectra_calc=.true.\n/\n")
                f.write("&CONTROL\n  nnu=200, emin=-6.0, emax=6.0, eps=1e-3\n/\n")
            rc_b, _, _ = run_executable(exe_wannband, workdir=outdir, timeout=180)
            spec_out = os.path.join(outdir, f'spectra_{spec_label}.dat')
            if rc_b == 0 and os.path.exists(os.path.join(outdir, 'spectra.dat')):
                shutil.move(os.path.join(outdir, 'spectra.dat'), spec_out)
                print(f"      {src_seed}: bands OK -> spectra_{spec_label}.dat")
            else:
                print(f"      {src_seed}: wannband.x failed (rc={rc_b})")
        if HAS_MATPLOTLIB:
            fig_cmp, axes_cmp = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
            plot_bands(os.path.join(outdir, 'spectra_k3_full.dat'),
                       title="seed (full, with f)", ax=axes_cmp[0])
            plot_bands(os.path.join(outdir, 'spectra_k3_downfold.dat'),
                       title="seed_downfold (H_CC from downfold)", ax=axes_cmp[1])
            plot_bands(os.path.join(outdir, 'spectra_k3_bare.dat'),
                       title="seedbare (bare, no f)", ax=axes_cmp[2])
            eff_title = f"bare+J·S (J={J_opt:.3f}, S=({js_out.get('S_x',0):.2f},{js_out.get('S_y',0):.2f},{js_out.get('S_z',0):.2f}))"
            plot_bands(os.path.join(outdir, 'spectra_k3_effective.dat'),
                       title=eff_title, ax=axes_cmp[3])
            plt.suptitle("Band comparison: 3-site kagome — seed vs seed_downfold vs seedbare vs bare+J·S")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'fig_k3_band_comparison.png'), dpi=150)
            plt.close()
            print("      Saved fig_k3_band_comparison.png")

    # Step 7: Transport plot
    if HAS_MATPLOTLIB and len(temps_K) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(temps_K, sigma_xy, 'b-o', ms=2, lw=1.0)
        ax1.set_xlabel("T (K)"); ax1.set_ylabel(r"$\sigma_{xy}$ ($e^2/h$)")
        ax1.set_title("Hall (AHC)"); ax1.axhline(0, color='gray', lw=0.5, ls='--')
        ax2.plot(temps_K, sigma_xx, 'r-o', ms=2, lw=1.0)
        ax2.set_xlabel("T (K)"); ax2.set_ylabel(r"$\sigma_{xx}$ ($e^2/h$)")
        ax2.set_title("Longitudinal (Kubo bubble)"); ax2.axhline(0, color='gray', lw=0.5, ls='--')
        plt.suptitle("Transport vs T: 3-site kagome + f")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig_k3_transport_vs_T.png'), dpi=150)
        plt.close()
        print("      Saved fig_k3_transport_vs_T.png")

    # Step 8: Band plots at selected T
    target_T = [0, 10, 30, 90, 200]
    bands_files = {}
    if exe_wannband:
        for T_K in target_T:
            avail_T = []
            for f in hr_files:
                try:
                    t_str = os.path.basename(f).replace('seedbare_hr_','').replace('K_hr.dat','')
                    avail_T.append((abs(int(t_str)-T_K), int(t_str), f))
                except: pass
            if not avail_T: continue
            avail_T.sort()
            _, T_act, _ = avail_T[0]
            sn = f'seedbare_hr_{T_act}K'
            shutil.copy(os.path.join(outdir,'seedbare.pos'), os.path.join(outdir,f'{sn}.pos'))
            with open(os.path.join(outdir,'wannband.inp'),'w') as f:
                f.write(f"&SYSTEM\n  seed='{sn}', mu=0.0, spectra_calc=.true.\n/\n")
                f.write("&CONTROL\n  nnu=200, emin=-6.0, emax=6.0, eps=1e-3\n/\n")
            rc2, _, _ = run_executable(exe_wannband, workdir=outdir, timeout=120)
            if rc2 == 0 and os.path.exists(os.path.join(outdir,'spectra.dat')):
                dst = os.path.join(outdir,f'spectra_k3_{T_K}K.dat')
                shutil.move(os.path.join(outdir,'spectra.dat'), dst)
                bands_files[T_K] = dst

    if HAS_MATPLOTLIB and bands_files:
        n_p = len(target_T); fig, axes = plt.subplots(1, n_p, figsize=(4*n_p, 5), sharey=True)
        if n_p == 1: axes = [axes]
        for ax, T_K in zip(axes, target_T):
            if T_K in bands_files:
                plot_bands(bands_files[T_K], title=f"T={T_K}K", ax=ax)
        plt.suptitle("Bands vs T: 3-site kagome + f")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig_k3_bands_vs_T.png'), dpi=150)
        plt.close()
        print("      Saved fig_k3_bands_vs_T.png")

    passed = (rc == 0 and J_opt is not None and len(temps_K) > 5 and len(hr_files) >= 5)
    details = f"rc={rc}, J_opt={J_opt}, S_mag={S_mag}, L2={L2}, T_pts={len(temps_K)}, HR={len(hr_files)}"
    print(f"\n  {'PASS' if passed else 'FAIL'}: {details}")
    return {'test': 'kagome3_pipeline', 'passed': passed,
            'J_opt': J_opt, 'S_mag': S_mag, 'L2_best': L2, 'details': details}


# ============================================================
# Test report
# ============================================================

def generate_test_report(outdir, test_results):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# wanneff_JS End-to-End Test Report",
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
        for k, v in res.items():
            if k not in ('test', 'passed', 'details'):
                lines.append(f"- **{k}:** `{v}`")
        lines.append(f"")
    lines.append("## Figures")
    lines.append("")
    for fig in ['fig7_transport_vs_T.png', 'fig8_bands_vs_T.png']:
        if os.path.exists(os.path.join(outdir, fig)):
            lines.append(f"- [{fig}]({fig})")
    lines.append("")
    report_path = os.path.join(outdir, 'test6_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Test report written to {report_path}")
    return report_path


# ============================================================
# Main
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(script_dir, 'test_tsf0.5_output')
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")

    exe_wanneff = find_executable('wanneff_js.x')
    exe_wannband = find_executable('wannband.x')
    print(f"wanneff_js.x: {exe_wanneff or 'NOT FOUND'}")
    print(f"wannband.x:   {exe_wannband or 'NOT FOUND'}")

    test_results = []

    try:
        r6 = test_full_pipeline_e2e(outdir, exe_wanneff, exe_wannband)
        test_results.append(r6)
    except Exception as e:
        import traceback
        print(f"Test 6 ERROR: {e}")
        traceback.print_exc()
        test_results.append({'test': 'full_pipeline_e2e', 'passed': False, 'details': str(e)})

    # Test 7: 3D simple cubic
    outdir_cubic = os.path.join(script_dir, 'test_tsf0.5_output', 'cubic')
    os.makedirs(outdir_cubic, exist_ok=True)
    try:
        r7 = test_cubic_pipeline(outdir_cubic, exe_wanneff, exe_wannband)
        test_results.append(r7)
    except Exception as e:
        import traceback
        print(f"Test 7 ERROR: {e}")
        traceback.print_exc()
        test_results.append({'test': 'cubic_pipeline', 'passed': False, 'details': str(e)})

    # Test 8a: Kagome J_TENSOR
    outdir_tensor_k = os.path.join(script_dir, 'test_tsf0.5_output', 'kagome_tensor')
    os.makedirs(outdir_tensor_k, exist_ok=True)
    try:
        r8a = test_tensor_pipeline(outdir_tensor_k, exe_wanneff, exe_wannband,
                                    label='kagome_tensor',
                                    generate_fn=generate_test_data,
                                    temps=[0, 10, 30, 90, 200])
        test_results.append(r8a)
    except Exception as e:
        import traceback; traceback.print_exc()
        test_results.append({'test': 'tensor_pipeline_kagome_tensor', 'passed': False, 'details': str(e)})

    # Test 8b: Cubic J_TENSOR
    outdir_tensor_c = os.path.join(script_dir, 'test_tsf0.5_output', 'cubic_tensor')
    os.makedirs(outdir_tensor_c, exist_ok=True)
    try:
        r8b = test_tensor_pipeline(outdir_tensor_c, exe_wanneff, exe_wannband,
                                    label='cubic_tensor',
                                    generate_fn=generate_cubic_test_data,
                                    temps=[0, 30, 100, 200, 300])
        test_results.append(r8b)
    except Exception as e:
        import traceback; traceback.print_exc()
        test_results.append({'test': 'tensor_pipeline_cubic_tensor', 'passed': False, 'details': str(e)})

    # Test 9: 3-site kagome flat band + f-decoration
    outdir_k3 = os.path.join(script_dir, 'test_tsf0.5_output', 'kagome3')
    os.makedirs(outdir_k3, exist_ok=True)
    try:
        r9 = test_kagome3_pipeline(outdir_k3, exe_wanneff, exe_wannband)
        test_results.append(r9)
    except Exception as e:
        import traceback; traceback.print_exc()
        test_results.append({'test': 'kagome3_pipeline', 'passed': False, 'details': str(e)})

    # Test 10a: Kagome J_S_TENSOR
    outdir_js_k = os.path.join(script_dir, 'test_tsf0.5_output', 'kagome_JStensor')
    os.makedirs(outdir_js_k, exist_ok=True)
    try:
        r10a = test_js_tensor_pipeline(outdir_js_k, exe_wanneff, exe_wannband,
                                       label='kagome_JStensor',
                                       generate_fn=generate_test_data,
                                       temps=[0, 10, 30, 90, 200])
        test_results.append(r10a)
    except Exception as e:
        import traceback; traceback.print_exc()
        test_results.append({'test': 'js_tensor_pipeline_kagome_JStensor', 'passed': False, 'details': str(e)})

    # Test 10b: Cubic J_S_TENSOR
    outdir_js_c = os.path.join(script_dir, 'test_tsf0.5_output', 'cubic_JStensor')
    os.makedirs(outdir_js_c, exist_ok=True)
    try:
        r10b = test_js_tensor_pipeline(outdir_js_c, exe_wanneff, exe_wannband,
                                       label='cubic_JStensor',
                                       generate_fn=generate_cubic_test_data,
                                       temps=[0, 30, 100, 200, 300])
        test_results.append(r10b)
    except Exception as e:
        import traceback; traceback.print_exc()
        test_results.append({'test': 'js_tensor_pipeline_cubic_JStensor', 'passed': False, 'details': str(e)})

    # Test 10c: Kagome3 J_S_TENSOR
    outdir_js_k3 = os.path.join(script_dir, 'test_tsf0.5_output', 'kagome3_JStensor')
    os.makedirs(outdir_js_k3, exist_ok=True)
    try:
        r10c = test_js_tensor_pipeline(outdir_js_k3, exe_wanneff, exe_wannband,
                                       label='kagome3_JStensor',
                                       generate_fn=generate_kagome3_test_data,
                                       temps=[0, 10, 30, 90, 200])
        test_results.append(r10c)
    except Exception as e:
        import traceback; traceback.print_exc()
        test_results.append({'test': 'js_tensor_pipeline_kagome3_JStensor', 'passed': False, 'details': str(e)})

    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    n_pass = sum(1 for r in test_results if r.get('passed') is not None and r.get('passed'))
    n_fail = sum(1 for r in test_results if r.get('passed') is not None and not r.get('passed'))
    n_skip = sum(1 for r in test_results if r.get('passed') is None)
    print(f"  PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}")
    for r in test_results:
        status = "PASS" if r.get('passed') else ("SKIP" if r.get('passed') is None else "FAIL")
        print(f"  [{status:4s}] {r['test']}")

    generate_test_report(outdir, test_results)
    return n_fail == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
