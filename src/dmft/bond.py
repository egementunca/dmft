"""Bond-scheme Ghost-DMFT solver and temperature sweep.

Implements the professor's self-consistency loop for the bond scheme on
the square lattice:

1. ``solve_singlesite`` — sequential h/g-sector correlator matching
   (2M equations each) using ``least_squares``.

2. ``solve_bond`` — simultaneous (8M+1)-variable solver matching
   lattice ↔ bond gateway (h-sector) and BPK[gateways] = BPK[imp+H2]
   (g-sector) plus half-filling.

3. ``run_temperature_sweep`` — warm-started sweep from high to low T.

4. ``save_results`` — ``.dat`` output compatible with the professor's
   standalone script format.

All unknowns (eta, W, etab, Bh, eps, V, epsb, Bg, dmu) are stored as
plain arrays and dicts — no PoleParams dependency.
"""

import sys
import time
from math import comb

import numpy as np
from scipy.optimize import least_squares

from .lattice import (
    make_square_lattice,
    lattice_statics,
    bond_lattice_statics,
)
from .gateway import (
    gateway_statics,
    bond_gateway_statics,
)
from .bond_ed import (
    impurity_statics,
    build_H2,
)


# ═══════════════════════════════════════════════════════════
# Single-site solver (general M)
# ═══════════════════════════════════════════════════════════

def solve_singlesite(beta, eta0, W0, eps0, V0, M, U, t, mu, shift,
                     EPS, EPS_W, mix=0.5, tol=1e-9, maxiter=200,
                     verbose=False):
    """Sequential 2M-equation TRF fits (h-sector then g-sector).

    Parameters
    ----------
    beta : float
        Inverse temperature.
    eta0, W0, eps0, V0 : arrays, shape (M,)
        Initial guesses.
    M : int
        Number of ghost poles.
    U, t, mu, shift : float
        Physical parameters.
    EPS : array
        Lattice dispersion.
    EPS_W : array
        BZ weights.
    mix : float
        Linear mixing parameter.
    tol : float
        Convergence tolerance on parameter change.
    maxiter : int
        Max iterations.
    verbose : bool
        Print per-iteration diagnostics.

    Returns
    -------
    dict with keys: eta, W, eps, V, docc, iters.
    """
    eta = np.array(eta0, dtype=float)
    W = np.array(W0, dtype=float)
    eps = np.array(eps0, dtype=float)
    V = np.array(V0, dtype=float)

    for it in range(1, maxiter + 1):
        nh_lat, dh_lat = lattice_statics(beta, eta, W, M, EPS, EPS_W, shift)

        def r_h(p):
            nh_gw, dh_gw, _, _ = gateway_statics(
                beta, eta, W, p[:M], p[M:], M, shift)
            return np.concatenate([nh_gw - nh_lat, dh_gw - dh_lat])

        sol_h = least_squares(r_h, np.concatenate([eps, V]), method='trf',
                              ftol=1e-12, xtol=1e-12, gtol=1e-12,
                              max_nfev=5000)
        eps_new = sol_h.x[:M]
        V_new = sol_h.x[M:]

        ng_imp, dg_imp, _ = impurity_statics(beta, eps_new, V_new, M, U, mu)

        def r_g(p):
            _, _, ng_gw, dg_gw = gateway_statics(
                beta, p[:M], p[M:], eps_new, V_new, M, shift)
            return np.concatenate([ng_gw - ng_imp, dg_gw - dg_imp])

        sol_g = least_squares(r_g, np.concatenate([eta, W]), method='trf',
                              ftol=1e-12, xtol=1e-12, gtol=1e-12,
                              max_nfev=5000)
        eta_new = sol_g.x[:M]
        W_new = sol_g.x[M:]

        dp = float(np.linalg.norm(
            np.concatenate([eta_new - eta, W_new - W,
                            eps_new - eps, V_new - V])))
        eta = mix * eta_new + (1 - mix) * eta
        W = mix * W_new + (1 - mix) * W
        eps = mix * eps_new + (1 - mix) * eps
        V = mix * V_new + (1 - mix) * V

        if verbose:
            _, _, docc = impurity_statics(beta, eps, V, M, U, mu)
            print(f'  SS it={it:3d}  dp={dp:.2e}  docc={docc:.6f}')
            sys.stdout.flush()
        if dp < tol:
            break

    _, _, docc = impurity_statics(beta, eps, V, M, U, mu)
    return dict(eta=eta, W=W, eps=eps, V=V, docc=docc, iters=it)


# ═══════════════════════════════════════════════════════════
# Bond solver — NO FROZEN TARGETS (general M)
# ═══════════════════════════════════════════════════════════

def solve_bond(beta, ss, M, U, t, mu, shift, EPS, GAM, EPS_W, z,
               mix=0.3, tol=1e-9, maxiter=200, verbose=False):
    """Fully simultaneous (8M+1)-variable solve.

    NO frozen targets — all correlators computed fresh at every evaluation.

    Variables: eta[M], W[M], etab[M], Bh[M], eps[M], V[M], epsb[M], Bg[M], dmu
    Total: 8M+1

    Matching conditions (8M+1):
      h-sector (4M): lattice ↔ bond gateway
      g-sector (4M): BPK[gateways] = BPK[imp+H2]
      half-filling (1): n_site = 0.5

    Parameters
    ----------
    beta : float
        Inverse temperature.
    ss : dict
        Single-site solution (keys: eta, W, eps, V, and optionally
        etab, Bh, epsb, Bg, dmu for warm-starting).
    M : int
        Number of ghost poles.
    U, t, mu, shift : float
        Physical parameters.
    EPS, GAM, EPS_W : arrays
        Lattice dispersion, bond form factor, BZ weights.
    z : int
        Coordination number.
    mix : float
        Linear mixing parameter.
    tol : float
        Convergence tolerance.
    maxiter : int
        Max iterations.
    verbose : bool
        Print per-iteration diagnostics.

    Returns
    -------
    dict with keys: eta, W, etab, Bh, eps, V, epsb, Bg, dmu,
                    docc_1, docc_2, docc_bpk, hop, res, iters.
    """
    eta = ss['eta'].copy()
    W = ss['W'].copy()
    eps = ss['eps'].copy()
    V = ss['V'].copy()
    etab = ss.get('etab', np.zeros(M)).copy()
    Bh = ss.get('Bh', np.zeros(M)).copy()
    epsb = ss.get('epsb', np.zeros(M)).copy()
    Bg = ss.get('Bg', np.zeros(M)).copy()
    dmu = float(ss.get('dmu', 0.0))

    def residuals(p):
        eta_ = p[0 * M:1 * M]
        W_ = p[1 * M:2 * M]
        etab_ = p[2 * M:3 * M]
        Bh_ = p[3 * M:4 * M]
        eps_ = p[4 * M:5 * M]
        V_ = p[5 * M:6 * M]
        epsb_ = p[6 * M:7 * M]
        Bg_ = p[7 * M:8 * M]
        dmu_ = float(p[8 * M])

        try:
            # Lattice
            nh_l, dh_l = lattice_statics(
                beta, eta_, W_, M, EPS, EPS_W, shift)
            _, _, nhb_l, dhb_l = bond_lattice_statics(
                beta, eta_, W_, etab_, Bh_, M, EPS, GAM, EPS_W, shift)

            # Single-site gateway
            nh1, dh1, ng1_, dg1_ = gateway_statics(
                beta, eta_, W_, eps_, V_, M, shift)

            # Two-site gateway
            (nh2, dh2, nhb2, dhb2,
             ng2_, dg2_, ngb2_, dgb2_, ns) = bond_gateway_statics(
                beta, eta_, W_, eps_, V_, etab_, Bh_, epsb_, Bg_,
                M, t, shift)

            # Impurity and H2 — fresh, no freezing
            ng_imp, dg_imp, _ = impurity_statics(
                beta, eps_, V_, M, U, mu)
            ng_H2, dg_H2, ngb_H2, dgb_H2, _, _, _ = build_H2(
                beta, eps_, V_, epsb_, Bg_, dmu_, M, U, mu, t)

        except np.linalg.LinAlgError:
            return np.ones(8 * M + 1) * 1e6

        r = []
        for l in range(M):
            # h-sector: lattice ↔ bond gateway
            r.append((dh2[l] - dh1[l]) - (dh_l[l] - dh1[l]) / z)
            r.append((nh2[l] - nh1[l]) - (nh_l[l] - nh1[l]) / z)
            r.append(nhb2[l] - nhb_l[l])
            r.append(dhb2[l] - dhb_l[l])
            # g-sector: BPK[gateways] = BPK[imp+H2]
            r.append((1 - z) * dg1_[l] + z * dg2_[l]
                     - ((1 - z) * dg_imp[l] + z * dg_H2[l]))
            r.append((1 - z) * ng1_[l] + z * ng2_[l]
                     - ((1 - z) * ng_imp[l] + z * ng_H2[l]))
            r.append(ngb2_[l] - ngb_H2[l])
            r.append(dgb2_[l] - dgb_H2[l])
        # half-filling
        r.append(ns - 0.5)
        return np.array(r)

    x0 = np.concatenate([eta, W, etab, Bh, eps, V, epsb, Bg, [dmu]])
    res_norm = np.inf

    for it in range(1, maxiter + 1):
        xc = np.clip(x0, -7.9, 7.9)
        xc[-1] = np.clip(xc[-1], -2.9, 2.9)

        sol = least_squares(
            residuals, xc, method='trf',
            ftol=1e-12, xtol=1e-12, gtol=1e-12,
            max_nfev=3000,
            bounds=([-8] * (8 * M) + [-3], [8] * (8 * M) + [3]))

        dp = float(np.linalg.norm(sol.x - xc))
        res_norm = float(np.linalg.norm(sol.fun))

        x0 = mix * sol.x + (1 - mix) * xc

        if verbose:
            _, _, _, _, docc_2, hop, _ = build_H2(
                beta, x0[4 * M:5 * M], x0[5 * M:6 * M],
                x0[6 * M:7 * M], x0[7 * M:8 * M],
                float(x0[8 * M]), M, U, mu, t)
            print(f'  it={it:3d}  |res|={res_norm:.2e}  dp={dp:.2e}'
                  f'  docc_2={docc_2:.5f}  nfev={sol.nfev}')
            sys.stdout.flush()

        if dp < tol and res_norm < tol:
            break

    eta = x0[0 * M:1 * M]
    W = x0[1 * M:2 * M]
    etab = x0[2 * M:3 * M]
    Bh = x0[3 * M:4 * M]
    eps = x0[4 * M:5 * M]
    V = x0[5 * M:6 * M]
    epsb = x0[6 * M:7 * M]
    Bg = x0[7 * M:8 * M]
    dmu = float(x0[8 * M])

    # Final BPK double occupancy
    _, _, docc_1_f = impurity_statics(beta, eps, V, M, U, mu)
    _, _, _, _, docc_2_f, hop_f, _ = build_H2(
        beta, eps, V, epsb, Bg, dmu, M, U, mu, t)
    docc_bpk = (1 - z) * docc_1_f + z * docc_2_f

    return dict(
        eta=eta, W=W, etab=etab, Bh=Bh,
        eps=eps, V=V, epsb=epsb, Bg=Bg,
        dmu=dmu, docc_1=docc_1_f, docc_2=docc_2_f,
        docc_bpk=docc_bpk, hop=hop_f, res=res_norm, iters=it)


# ═══════════════════════════════════════════════════════════
# Temperature sweep
# ═══════════════════════════════════════════════════════════

def run_temperature_sweep(U=1.3, t=0.5, M=1, mode='both', n_k=30,
                           T_vals=None,
                           mix_ss=0.5, mix_bond=0.3,
                           tol_ss=1e-9, tol_bond=1e-9,
                           maxiter_ss=200, maxiter_bond=200,
                           verbose=False):
    """Run bond-scheme DMFT over a range of temperatures.

    Parameters
    ----------
    U : float
        Hubbard U.
    t : float
        Hopping parameter.
    M : int
        Number of ghost poles (1 or 2).
    mode : str
        'ss', 'bond', or 'both'.
    n_k : int
        k-grid size per direction.
    T_vals : array, optional
        Temperature values. Default: 19 points from 1.0 to 0.05.
    mix_ss, mix_bond : float
        Mixing parameters for single-site and bond solvers.
    tol_ss, tol_bond : float
        Convergence tolerances.
    maxiter_ss, maxiter_bond : int
        Max iterations.
    verbose : bool
        Print per-iteration diagnostics.

    Returns
    -------
    results : list of dicts
        Per-temperature results.
    D : float
        Half-bandwidth.
    """
    mu = U / 2.0
    shift = 0.0
    EPS, GAM, EPS_W, D, z = make_square_lattice(t, n_k=n_k)

    if T_vals is None:
        T_vals = np.array([
            1.0, 0.8, 0.667, 0.5, 0.4, 0.333, 0.25, 0.2,
            0.167, 0.143, 0.125, 0.111, 0.1, 0.091, 0.083,
            0.071, 0.063, 0.056, 0.05])

    # Initial guesses
    if M == 1:
        eta0 = np.array([-0.3])
        W0 = np.array([0.2])
        eps0 = np.array([-0.1])
        V0 = np.array([0.4])
    else:
        eta0 = np.array([-0.3, -0.1])
        W0 = np.array([0.2, 0.1])
        eps0 = np.array([-0.1, -0.05])
        V0 = np.array([0.4, 0.3])

    results = []
    # Bond warm-start state — persists across temperatures (P1 fix)
    bond_state = {}

    print(f'\nGhost-DMFT  M={M}  U={U}  t={t}  D={D:.1f}  mode={mode}')
    print(f'  {8 * M + 1} equations, {8 * M + 1} unknowns')
    print(f'  H2: nps={2 + 3 * M}, states={comb(2 * (2 + 3 * M), 2 + 3 * M)}')

    hdr = f'{"T":>8}  {"T/D":>7}  {"docc_ss":>10}'
    if mode in ('bond', 'both'):
        hdr += (f'  {"docc_BPK":>10}  {"docc_2":>8}'
                f'  {"hop":>8}  {"res":>8}  {"its":>4}')
    print(hdr)
    print('-' * (len(hdr) + 8))
    sys.stdout.flush()

    for Tv in T_vals:
        beta = 1.0 / Tv
        t0 = time.time()

        ss = solve_singlesite(
            beta, eta0, W0, eps0, V0, M, U, t, mu, shift,
            EPS, EPS_W, mix=mix_ss, tol=tol_ss, maxiter=maxiter_ss)

        row = dict(T=Tv, ToverD=Tv / D, beta=beta, docc_ss=ss['docc'])

        rb = None
        if mode in ('bond', 'both'):
            # Inject persisted bond warm-start into ss before calling solver
            ss.update(bond_state)
            rb = solve_bond(
                beta, ss, M, U, t, mu, shift, EPS, GAM, EPS_W, z,
                mix=mix_bond, tol=tol_bond, maxiter=maxiter_bond,
                verbose=verbose)
            row.update(
                docc_bpk=rb['docc_bpk'], docc_2=rb['docc_2'],
                docc_1=rb['docc_1'], hop=rb['hop'],
                dmu=rb['dmu'], res=rb['res'], iters_bond=rb['iters'])
            # Persist converged bond variables for next temperature
            if rb['res'] < 1e-4:
                bond_state = {
                    'etab': rb['etab'].copy(),
                    'Bh': rb['Bh'].copy(),
                    'epsb': rb['epsb'].copy(),
                    'Bg': rb['Bg'].copy(),
                    'dmu': rb['dmu'],
                }

        dt = time.time() - t0
        line = f'  {Tv:6.4f}  {Tv / D:7.4f}  {ss["docc"]:10.6f}'
        if mode in ('bond', 'both') and rb is not None:
            line += (f'  {rb["docc_bpk"]:10.6f}  {rb["docc_2"]:8.6f}'
                     f'  {rb["hop"]:8.5f}  {rb["res"]:8.2e}'
                     f'  {rb["iters"]:4d}')
        line += f'  ({dt:.1f}s)'
        print(line)
        sys.stdout.flush()

        results.append(row)
        # Warm-start SS parameters
        eta0, W0 = ss['eta'], ss['W']
        eps0, V0 = ss['eps'], ss['V']

    return results, D


# ═══════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════

def save_results(results, fname, mode):
    """Save sweep results to a .dat file matching the professor's format.

    Parameters
    ----------
    results : list of dicts
        From ``run_temperature_sweep``.
    fname : str
        Output filename.
    mode : str
        'ss', 'bond', or 'both'.
    """
    keys = ['T', 'ToverD', 'beta', 'docc_ss']
    if mode in ('bond', 'both'):
        keys += ['docc_bpk', 'docc_2', 'docc_1', 'hop', 'dmu', 'res']
    header = '  '.join(keys)
    rows = [[r.get(k, np.nan) for k in keys] for r in results]
    np.savetxt(fname, rows, header=header, fmt='%.8f')
    print(f'\nSaved: {fname}')
