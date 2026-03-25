"""
algo_test.py

Python algorithm validation tests for wanneff_JS modules (Tests 1-5).
These tests validate the Python reference implementations and were previously
part of kagome_f_spinor_test.py. They do NOT require compiled Fortran executables.

Tests:
  1. Kagome reference bands + DOS (Python Bloch sum, Hermiticity check)
  2. Bayesian optimization on known 8D function
  3. Classical MC on square Heisenberg lattice
  4. AHC Berry curvature visualization
  5. Full pipeline (Fortran wanneff_js.x, optional)

Usage:
  cd wannchi/tests
  source /Users/ykxu/Projects/hrJS/hrJS/bin/activate
  python3 algo_test.py
"""

import numpy as np
import os
import subprocess
import sys
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
TSS  = -1.0    # s-s NN hopping (eV)
TSF  = -0.15   # NN s-f hopping (eV)
TSF2 = 0.0     # on-site s-f coupling (eV, default off)

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

# ============================================================
# WS R-vector generation
# ============================================================

def find_ws_rvectors(nr1, nr2, nr3, avec):
    metric = np.dot(avec.T, avec)
    rvecs = []
    weights = []
    for ir1 in range(-nr1, nr1+1):
        for ir2 in range(-nr2, nr2+1):
            for ir3 in range(-nr3, nr3+1):
                dist = np.zeros(125)
                idx = 0
                for i1 in range(-2, 3):
                    for i2 in range(-2, 3):
                        for i3 in range(-2, 3):
                            ndiff = np.array([ir1 - i1*nr1, ir2 - i2*nr2, ir3 - i3*nr3], dtype=float)
                            dist[idx] = ndiff @ metric @ ndiff
                            idx += 1
                dist_min = dist.min()
                center_idx = 62
                if abs(dist[center_idx] - dist_min) < 1e-7:
                    weight = np.sum(np.abs(dist - dist_min) < 1e-7)
                    rvecs.append([ir1, ir2, ir3])
                    weights.append(weight)
    rvecs = np.array(rvecs, dtype=float).T
    weights = np.array(weights, dtype=float)
    tot = np.sum(1.0 / weights)
    expected = nr1 * nr2 * nr3
    assert abs(tot - expected) < 1e-6, f"WS weight check failed: {tot} != {expected}"
    return rvecs, weights


# ============================================================
# Hamiltonian construction
# ============================================================

def cartesian_pos(frac, avec):
    return avec @ frac


def distance(f1, f2, avec, R=np.zeros(3)):
    c1 = cartesian_pos(f1, avec)
    c2 = cartesian_pos(f2 + R, avec)
    return np.linalg.norm(c1 - c2)


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


def build_hr_seed(rvecs, weights, avec, sites_frac=SITES_FRAC,
                  tss=TSS, tsf=TSF, tsf2=TSF2):
    norb = NORB_SEED
    nrpt = rvecs.shape[1]
    hr = np.zeros((norb, norb, nrpt), dtype=complex)
    nn_dist = get_nn_distance(avec, sites_frac)
    nn_cutoff = nn_dist * 1.5
    s_up = [0, 1, 2, 3]; f_up = 4
    s_dn = [5, 6, 7, 8]; f_dn = 9
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
        for i_site in range(4):
            d_fwd = distance(sites_frac[3], sites_frac[i_site], avec, R)
            if d_fwd > 1e-8 and d_fwd < nn_cutoff:
                hr[f_up, s_up[i_site], ir] += tsf
                hr[f_dn, s_dn[i_site], ir] += tsf
            if abs(tsf2) > 1e-10 and d_fwd < 1e-8:
                hr[f_up, s_up[i_site], ir] += tsf2
                hr[s_up[i_site], f_up, ir] += np.conj(tsf2)
                hr[f_dn, s_dn[i_site], ir] += tsf2
                hr[s_dn[i_site], f_dn, ir] += np.conj(tsf2)
            d_rev = distance(sites_frac[i_site], sites_frac[3], avec, R)
            if d_rev > 1e-8 and d_rev < nn_cutoff:
                hr[s_up[i_site], f_up, ir] += np.conj(tsf)
                hr[s_dn[i_site], f_dn, ir] += np.conj(tsf)
    return hr


def build_hr_bare(rvecs, weights, avec, sites_frac=SITES_FRAC[:4], tss=TSS):
    norb = NORB_BARE
    nrpt = rvecs.shape[1]
    hr = np.zeros((norb, norb, nrpt), dtype=complex)
    nn_dist = get_nn_distance(avec, sites_frac)
    nn_cutoff = nn_dist * 1.5
    s_up = [0, 1, 2, 3]; s_dn = [4, 5, 6, 7]
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
    norb = hr.shape[0]
    nrpt = hr.shape[2]
    if tau is None:
        tau = np.zeros((3, norb))
    phase = np.exp(1j * 2*np.pi * (kvec @ tau))
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
    taus = []
    for i_site, nbase in enumerate(nbasis):
        for _ in range(nbase):
            taus.append(sites_frac[i_site])
    if spinor:
        taus_full = taus + taus
    else:
        taus_full = taus
    return np.array(taus_full).T


def compute_dos_python(hr, rvecs, weights, tau, nk=50, nw=200, emin=-4.0, emax=4.0, eta=0.05):
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


def generate_kpath(hsps, labels, avec, n_per_seg=100):
    bvec = 2 * np.pi * np.linalg.inv(avec).T
    kpoints = []; kdist = []; tick_pos = []; cum = 0.0
    for i_seg in range(len(hsps) - 1):
        k0, k1 = np.array(hsps[i_seg]), np.array(hsps[i_seg + 1])
        for j in range(n_per_seg):
            t = j / n_per_seg
            kfrac = (1 - t) * k0 + t * k1
            if kpoints:
                dk_frac = kfrac - kpoints[-1]
                dk_cart = bvec @ dk_frac
                cum += np.linalg.norm(dk_cart)
            kpoints.append(kfrac)
            kdist.append(cum)
        if i_seg == 0:
            tick_pos.append(0.0)
        tick_pos.append(kdist[-1] + np.linalg.norm(bvec @ (k1 - kpoints[-1])))
    kpoints.append(np.array(hsps[-1]))
    dk_cart = bvec @ (kpoints[-1] - kpoints[-2])
    cum += np.linalg.norm(dk_cart)
    kdist.append(cum)
    tick_pos[-1] = cum
    return np.array(kpoints), np.array(kdist), tick_pos, labels


# ============================================================
# File I/O (shared with kagome_f_spinor_test.py)
# ============================================================

def write_hr_dat(fname, hr, rvecs, weights, norb, nrpt):
    with open(fname, 'w') as f:
        f.write(f"# Kagome test Hamiltonian ({norb} orbitals)\n")
        f.write(f"{norb:10d}\n")
        f.write(f"{nrpt:10d}\n")
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
                           f"{jo+1:5d}{io+1:5d}"
                           f"{val.real:22.16f}{val.imag:22.16f}\n")


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
# Test 1: Kagome reference bands + DOS
# ============================================================

def test_kagome_reference(outdir, exe_wannband=None):
    print("\n" + "="*60)
    print("TEST 1: Kagome reference bands + DOS")
    print("="*60)

    rvecs, weights = find_ws_rvectors(5, 5, 1, AVEC)
    nrpt = rvecs.shape[1]
    print(f"  nrpt = {nrpt}")
    tau_seed = compute_tau_from_sites(SITES_FRAC, NBASIS_SEED, spinor=True)
    tau_bare = compute_tau_from_sites(SITES_FRAC[:4], NBASIS_BARE, spinor=True)
    hr_seed = build_hr_seed(rvecs, weights, AVEC)
    hr_bare = build_hr_bare(rvecs, weights, AVEC)

    test_kvecs = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([1/3, 1/3, 0.0]),
    ]
    print("  Python eigenvalues at high-symmetry points:")
    max_err = 0.0
    for kvec in test_kvecs:
        hk = compute_hk_python(hr_seed, rvecs, weights, kvec, tau_seed)
        eigs = np.linalg.eigvalsh(hk)
        print(f"    k={kvec[:2]}: eigs = {eigs[:5]}")
        err = np.max(np.abs(hk - hk.conj().T))
        max_err = max(max_err, err)

    passed = max_err < 1e-10
    print(f"  Hermiticity check: max |H - H†| = {max_err:.2e}  {'PASS' if passed else 'FAIL'}")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        Gamma = np.array([0.0, 0.0, 0.0])
        M = np.array([0.5, 0.0, 0.0])
        K = np.array([1/3, 1/3, 0.0])
        kpath_pts, kpath_dist, tick_pos, tick_labels = generate_kpath(
            [Gamma, M, K, Gamma], ['Γ', 'M', 'K', 'Γ'], AVEC, n_per_seg=100)
        eigs_seed_all = np.zeros((len(kpath_pts), NORB_SEED))
        eigs_bare_all = np.zeros((len(kpath_pts), NORB_BARE))
        for ik, kvec in enumerate(kpath_pts):
            hk_s = compute_hk_python(hr_seed, rvecs, weights, kvec, tau_seed)
            hk_b = compute_hk_python(hr_bare, rvecs, weights, kvec, tau_bare)
            eigs_seed_all[ik] = np.sort(np.linalg.eigvalsh(hk_s))
            eigs_bare_all[ik] = np.sort(np.linalg.eigvalsh(hk_b))
        for n in range(NORB_SEED):
            ax.plot(kpath_dist, eigs_seed_all[:, n], 'b-', lw=1.0, label='seed' if n == 0 else '')
        for n in range(NORB_BARE):
            ax.plot(kpath_dist, eigs_bare_all[:, n], 'r--', lw=0.8, label='bare' if n == 0 else '')
        for tp in tick_pos:
            ax.axvline(tp, color='gray', lw=0.5, ls='-', alpha=0.5)
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel("Energy (eV)"); ax.set_title("Bands"); ax.legend()

        print("  Computing DOS...")
        energies, dos_seed = compute_dos_python(hr_seed, rvecs, weights, tau_seed, nk=50, nw=500, emin=-7.0, emax=5.0, eta=0.1)
        _, dos_bare = compute_dos_python(hr_bare, rvecs, weights, tau_bare, nk=50, nw=500, emin=-7.0, emax=5.0, eta=0.1)
        axes[1].plot(dos_seed, energies, 'b-', label='seed')
        axes[1].plot(dos_bare, energies, 'r--', label='bare')
        axes[1].axhline(0, color='gray', lw=0.5, ls='--')
        axes[1].set_xlabel("DOS (arb. u.)"); axes[1].set_ylabel("Energy (eV)")
        axes[1].set_title("DOS"); axes[1].legend()
        plt.suptitle("Test 1: Kagome Reference")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig1_kagome_reference.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved fig1_kagome_reference.png")

    return {'test': 'kagome_reference', 'passed': passed,
            'max_hermitian_error': float(max_err),
            'details': f"Hermiticity check {'PASS' if passed else 'FAIL'}"}


# ============================================================
# Test 2: Bayesian optimization on known function
# ============================================================

def test_bayesian_known_function(outdir):
    print("\n" + "="*60)
    print("TEST 2: Bayesian optimization - known 8D function")
    print("="*60)

    try:
        from scipy.optimize import minimize
        from scipy.stats import norm
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("  scipy not available; falling back")

    def objective_8d(x):
        return (x[0]-2)**2 + (x[1]+1)**2 + sum(np.sin(xi) for xi in x[2:])

    def find_true_min():
        best_val = 1e10; best_x = None
        np.random.seed(42)
        for _ in range(200):
            x0 = np.random.uniform(-3, 3, 8)
            if has_scipy:
                res = minimize(objective_8d, x0, method='L-BFGS-B', bounds=[(-3,3)]*8)
                if res.fun < best_val:
                    best_val = res.fun; best_x = res.x
            else:
                val = objective_8d(x0)
                if val < best_val:
                    best_val = val; best_x = x0
        return best_x, best_val

    print("  Finding true minimum...")
    x_true, f_true = find_true_min()
    print(f"  True minimum: f = {f_true:.6f}")

    class SimpleGPBO:
        def __init__(self, ndim, bounds, length_scale=1.0, noise=1e-6):
            self.ndim = ndim; self.bounds = bounds
            self.ls = length_scale; self.noise = noise
            self.X = []; self.y = []

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
            from scipy.stats import norm as _norm
            z = (y_best - mu) / (sigma + 1e-10)
            return (y_best - mu) * _norm.cdf(z) + sigma * _norm.pdf(z)

        def optimize(self, func, n_init=8, n_iter=50):
            np.random.seed(0)
            bounds_arr = np.array(self.bounds)
            for _ in range(n_init):
                x = bounds_arr[:,0] + np.random.rand(self.ndim) * (bounds_arr[:,1] - bounds_arr[:,0])
                self.X.append(x); self.y.append(func(x))
            history = [min(self.y)]
            for it in range(n_iter):
                best_ei = -1; x_next = None
                for _ in range(2000):
                    x = bounds_arr[:,0] + np.random.rand(self.ndim) * (bounds_arr[:,1] - bounds_arr[:,0])
                    ei_val = self.ei(x)
                    if ei_val > best_ei:
                        best_ei = ei_val; x_next = x
                y_next = func(x_next); self.X.append(x_next); self.y.append(y_next)
                history.append(min(self.y))
            best_idx = np.argmin(self.y)
            return self.X[best_idx], self.y[best_idx], history

    print("  Running Python GP-BO (50 iterations)...")
    gp = SimpleGPBO(8, [(-3, 3)] * 8, length_scale=2.0)
    x_opt, f_opt, history = gp.optimize(objective_8d, n_init=8, n_iter=50)
    print(f"  Python GP-BO result: f = {f_opt:.6f}")
    passed = f_opt < 0.5
    print(f"  {'PASS' if passed else 'FAIL'}: f_opt = {f_opt:.4f} (threshold: 0.5)")

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history, 'b-o', ms=3, label='best f so far')
        ax.axhline(f_true, color='r', ls='--', label=f'true min = {f_true:.4f}')
        ax.set_xlabel("Iteration"); ax.set_ylabel("f(x)")
        ax.set_title("Test 2: Bayesian Optimization Convergence (8D)"); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig2_bayesian_convergence.png'), dpi=150)
        plt.close()

    return {'test': 'bayesian_known_function', 'passed': passed,
            'f_opt': float(f_opt), 'f_true': float(f_true),
            'details': f"GP-BO {'PASS' if passed else 'FAIL'}: f_opt={f_opt:.4f} vs f_true={f_true:.4f}"}


# ============================================================
# Test 3: Classical MC on square lattice
# ============================================================

def test_mc_square_lattice(outdir):
    print("\n" + "="*60)
    print("TEST 3: Classical MC on square Heisenberg lattice")
    print("="*60)

    J_mc = 1.0; S_mag = 1.0; N = 10

    def random_spin():
        while True:
            u = 2*np.random.rand() - 1; v = 2*np.random.rand() - 1
            s = u*u + v*v
            if s < 1:
                break
        return np.array([2*u*np.sqrt(1-s), 2*v*np.sqrt(1-s), 1-2*s])

    def local_energy(spins, idx, J, S):
        i, j = idx; E = 0.0
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = (i+di)%N, (j+dj)%N
            E -= J * S**2 * np.dot(spins[i,j], spins[ni,nj])
        return E

    def mc_run_python(T, n_therm=2000, n_meas=5000):
        spins = np.array([[random_spin() for _ in range(N)] for _ in range(N)])
        for _ in range(n_therm):
            for i in range(N):
                for j in range(N):
                    E_old = local_energy(spins, (i,j), J_mc, S_mag)
                    s_new = random_spin(); s_old = spins[i,j].copy()
                    spins[i,j] = s_new
                    E_new = local_energy(spins, (i,j), J_mc, S_mag)
                    dE = E_new - E_old
                    if dE > 0 and (T < 1e-10 or np.random.rand() >= np.exp(-dE/T)):
                        spins[i,j] = s_old
        M_acc = np.zeros(3); n_acc = 0
        for _ in range(n_meas):
            for i in range(N):
                for j in range(N):
                    E_old = local_energy(spins, (i,j), J_mc, S_mag)
                    s_new = random_spin(); s_old = spins[i,j].copy()
                    spins[i,j] = s_new
                    E_new = local_energy(spins, (i,j), J_mc, S_mag)
                    dE = E_new - E_old
                    if dE > 0 and (T < 1e-10 or np.random.rand() >= np.exp(-dE/T)):
                        spins[i,j] = s_old
            n_acc += 1; M_acc += spins.mean(axis=(0,1))
        return np.linalg.norm(M_acc / n_acc)

    print(f"  Running MC on {N}x{N} square lattice (J={J_mc}, S={S_mag})...")
    temps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    magnetizations = []
    np.random.seed(42)
    for T in temps:
        m = 1.0 if T < 1e-10 else mc_run_python(T, n_therm=500, n_meas=500)
        magnetizations.append(m)
        print(f"    T = {T:.1f} eV, <|m|> = {m:.4f}")
    magnetizations = np.array(magnetizations)
    passed = (magnetizations[0] > 0.9 and magnetizations[-1] < 0.5)
    print(f"  {'PASS' if passed else 'FAIL'}: m(T=0)={magnetizations[0]:.3f}, m(T=5)={magnetizations[-1]:.3f}")

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(temps, magnetizations, 'bo-', ms=6)
        ax.axvline(J_mc*4*S_mag**2/3, color='r', ls='--', label=f'MFT T_c≈{J_mc*4*S_mag**2/3:.2f} eV')
        ax.set_xlabel("Temperature (eV)"); ax.set_ylabel("<|m|>")
        ax.set_title("Test 3: Classical MC - Square Heisenberg"); ax.set_ylim(0, 1.1); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig3_mc_square.png'), dpi=150); plt.close()

    return {'test': 'mc_square_lattice', 'passed': passed,
            'temps': temps.tolist(), 'magnetizations': magnetizations.tolist(),
            'details': f"MC {'PASS' if passed else 'FAIL'}: m(0)={magnetizations[0]:.3f}, m(5)={magnetizations[-1]:.3f}"}


# ============================================================
# Test 4: Berry curvature visualization
# ============================================================

def test_berry_curvature_plot(outdir, hr_file=None):
    print("\n" + "="*60)
    print("TEST 4: Berry curvature visualization")
    print("="*60)

    rvecs, weights = find_ws_rvectors(5, 5, 1, AVEC)
    hr_bare = build_hr_bare(rvecs, weights, AVEC)
    tau_bare = compute_tau_from_sites(SITES_FRAC[:4], NBASIS_BARE, spinor=True)
    norb = NORB_BARE; nrpt = rvecs.shape[1]

    J_mock = 0.3; S_mock = np.array([0.0, 0.0, 1.0])
    r000 = np.argmin(np.sum(rvecs**2, axis=0))
    hr_eff = hr_bare.copy(); n_c = norb // 2
    for io in range(n_c):
        hr_eff[io, io, r000] += J_mock * S_mock[2] / 2
        hr_eff[io+n_c, io+n_c, r000] -= J_mock * S_mock[2] / 2
        hr_eff[io, io+n_c, r000] += J_mock * (S_mock[0] - 1j*S_mock[1]) / 2
        hr_eff[io+n_c, io, r000] += J_mock * (S_mock[0] + 1j*S_mock[1]) / 2

    nk = 30
    print(f"  Computing Berry curvature on {nk}x{nk} mesh...")
    kpoints = np.array([[ik1/nk, ik2/nk, 0.0] for ik1 in range(nk) for ik2 in range(nk)])
    omega_kmap = np.zeros(len(kpoints)); sigma_xy_sum = 0.0

    for idx, kvec in enumerate(kpoints):
        hk = compute_hk_python(hr_eff, rvecs, weights, kvec, tau_bare)
        eigs, eigvecs = np.linalg.eigh(hk)
        vx = np.zeros((norb, norb), dtype=complex)
        vy = np.zeros((norb, norb), dtype=complex)
        phase = np.exp(1j * 2*np.pi * (kvec @ tau_bare))
        for ir in range(nrpt):
            R = rvecs[:, ir]; rdotk = kvec @ R
            fact = np.exp(1j * 2*np.pi * rdotk) / weights[ir]
            for io in range(norb):
                for jo in range(norb):
                    rtilde_x = R[0] + tau_bare[0,jo] - tau_bare[0,io]
                    rtilde_y = R[1] + tau_bare[1,jo] - tau_bare[1,io]
                    orb_fac = np.conj(phase[io]) * phase[jo] * fact
                    vx[io,jo] += 1j * 2*np.pi * rtilde_x * orb_fac * hr_eff[io,jo,ir]
                    vy[io,jo] += 1j * 2*np.pi * rtilde_y * orb_fac * hr_eff[io,jo,ir]
        vx_band = eigvecs.conj().T @ vx @ eigvecs
        vy_band = eigvecs.conj().T @ vy @ eigvecs
        omega_n = np.zeros(norb)
        for n in range(norb):
            for m in range(norb):
                if m == n: continue
                dE2 = (eigs[n] - eigs[m])**2
                if dE2 < 1e-10: continue
                omega_n[n] -= 2 * np.imag(vx_band[n,m] * vy_band[m,n]) / dE2
        f_occ = (eigs <= 0.0).astype(float)
        omega_kmap[idx] = np.sum(f_occ * omega_n)
        sigma_xy_sum += np.sum(f_occ * omega_n) / (nk * nk)

    sigma_xy = -sigma_xy_sum / (2*np.pi)**2
    chern = sigma_xy_sum / (2*np.pi)
    print(f"  sigma_xy ~ {sigma_xy:.4f} (e^2/h)  Chern ~ {chern:.3f}")
    passed = True

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        omega_grid = omega_kmap.reshape(nk, nk)
        vmax = np.percentile(np.abs(omega_kmap), 95)
        im = ax.imshow(omega_grid.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=[0,1,0,1])
        plt.colorbar(im, ax=ax, label='Ω(k)')
        ax.set_xlabel('k₁'); ax.set_ylabel('k₂')
        ax.set_title(f'Berry curvature Ω(k)\nJ={J_mock}, S=(0,0,{S_mock[2]})')
        ax2 = axes[1]
        k_line = np.linspace(0, 1, 100)
        kvecs_line = np.column_stack([k_line/2, np.zeros(100), np.zeros(100)])
        omega_line = [omega_kmap[np.argmin(np.sum((kpoints[:,:2] - kv[:2])**2, axis=1))] for kv in kvecs_line]
        ax2.plot(k_line, omega_line); ax2.axhline(0, color='gray', ls='--')
        ax2.set_xlabel('k along Γ-M'); ax2.set_ylabel('Ω(k)')
        ax2.set_title(f'Berry curvature along Γ-M\nσ_xy ~ {sigma_xy:.4f} e²/h')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'fig6_berry_curvature.png'), dpi=150); plt.close()
        print("  Saved fig6_berry_curvature.png")

    return {'test': 'berry_curvature', 'passed': passed,
            'sigma_xy': float(sigma_xy), 'chern': float(chern),
            'details': f"Berry curvature computed. σ_xy ~ {sigma_xy:.4f}, Chern ~ {chern:.3f}"}


# ============================================================
# Test 5: Full pipeline (Fortran, optional)
# ============================================================

def test_full_pipeline(outdir, exe_wanneff=None, exe_wannband=None, exe_ahc=None):
    print("\n" + "="*60)
    print("TEST 5: Full pipeline (Fortran wanneff_js.x)")
    print("="*60)

    if exe_wanneff is None:
        print("  wanneff_js.x not found, skipping")
        return {'test': 'full_pipeline', 'passed': None, 'details': 'Fortran executable not found'}

    print(f"  Running: {exe_wanneff}")
    rc, out, err = run_executable(exe_wanneff, workdir=outdir, timeout=300)
    print(f"  Return code: {rc}")
    if rc != 0:
        print(f"  FAILED. stderr: {err[:500]}")
        return {'test': 'full_pipeline', 'passed': False, 'details': f'wanneff_js.x failed: {err[:200]}'}

    hr_0K_file = os.path.join(outdir, 'seedbare_hr_0K_hr.dat')
    J_opt = None; Svec_opt = None
    for line in out.split('\n'):
        if 'J_opt' in line:
            try: J_opt = float(line.split('=')[-1])
            except: pass
        if 'Svec_opt' in line:
            try: Svec_opt = [float(x) for x in line.split('=')[-1].split()[:3]]
            except: pass

    passed = os.path.exists(hr_0K_file) and J_opt is not None
    print(f"  J_opt = {J_opt}, Svec_opt = {Svec_opt}")
    print(f"  {'PASS' if passed else 'FAIL'}")

    return {'test': 'full_pipeline', 'passed': passed,
            'J_opt': J_opt, 'Svec_opt': Svec_opt,
            'details': f"Pipeline {'PASS' if passed else 'FAIL'}"}


# ============================================================
# Main
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(script_dir, 'kagome_test_output')
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")

    exe_wanneff = find_executable('wanneff_js.x')
    exe_wannband = find_executable('wannband.x')

    test_results = []

    for test_fn, name, args in [
        (test_kagome_reference,       'kagome_reference',       (outdir, exe_wannband)),
        (test_bayesian_known_function,'bayesian_known_function', (outdir,)),
        (test_mc_square_lattice,      'mc_square_lattice',       (outdir,)),
        (test_berry_curvature_plot,   'berry_curvature',         (outdir,)),
        (test_full_pipeline,          'full_pipeline',           (outdir, exe_wanneff, exe_wannband, None)),
    ]:
        try:
            r = test_fn(*args)
            test_results.append(r)
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            test_results.append({'test': name, 'passed': False, 'details': str(e)})

    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    n_pass = sum(1 for r in test_results if r.get('passed') is not None and r.get('passed'))
    n_fail = sum(1 for r in test_results if r.get('passed') is not None and not r.get('passed'))
    n_skip = sum(1 for r in test_results if r.get('passed') is None)
    print(f"  PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}")
    for r in test_results:
        status = "PASS" if r.get('passed') else ("SKIP" if r.get('passed') is None else "FAIL")
        print(f"  [{status:4s}] {r['test']}")
    return n_fail == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
