"""Bond-scheme Ghost-DMFT solver and temperature sweep.

Corrected implementation matching professor's ghost_dmft_bond_new.py (March 2026).

Key differences from previous version:
  - Independent M per g-ghost family: M1g, M2g, Mbg (M1h=M2h=Mbh=1 fixed)
  - Direct matching conditions (not BPK combination)
  - gateway2: LOCAL h2/g2 per site, SHARED hb/gb bond orbitals
  - Alternating outer loop: fix impurity correlators -> solve gateway -> re-solve impurity
  - dmu bisection for half-filling in impurity2
  - gamma_k = eps_k/4 (corrected form factor)

Functions:
  1. ``solve_singlesite`` — sequential h/g-sector matching
  2. ``solve_bond`` — alternating loop with direct matching
  3. ``run_temperature_sweep`` — warm-started sweep from high to low T
  4. ``save_results`` — .dat output
"""

import sys
import time

import numpy as np
from scipy.optimize import least_squares

from .lattice import make_square_lattice, lattice_statics
from .gateway import gateway1_statics, gateway2_statics
from .bond_ed import impurity1_statics, impurity2_statics


# ═══════════════════════════════════════════════════════════
# Single-site solver
# ═══════════════════════════════════════════════════════════

def solve_singlesite(beta, M1g, U, t, EPS, GAM, EPS_W,
                     eta0=None, W0=None, eps0=None, V0=None,
                     mix=0.5, tol=1e-8, maxiter=200):
    """Sequential single-site self-consistency.

    Alternates:
      Step 2: vary (eps1, V1) to match gateway1 h-sector = lattice h1-sector
      Step 3: compute impurity1 g-correlators
      Step 4: vary (eta1, W1) to match gateway1 g-sector = impurity1 g-sector

    Parameters
    ----------
    beta : float
    M1g : int
        Number of g1-ghost poles.
    U, t : float
    EPS, GAM, EPS_W : arrays
        Lattice dispersion, form factor, weights.
    eta0, W0 : float or None
        Initial h1-ghost guess.
    eps0, V0 : array or None
        Initial g1-ghost guess.
    mix : float
    tol : float
    maxiter : int

    Returns
    -------
    dict with keys: eta, W, eps, V, docc, iters.
    """
    eta = 0.0 if eta0 is None else float(np.atleast_1d(eta0)[0])
    W = 0.3 if W0 is None else float(np.atleast_1d(W0)[0])
    eps = np.zeros(M1g) if eps0 is None else np.asarray(eps0, float).copy()
    V = 0.4 * np.ones(M1g) if V0 is None else np.asarray(V0, float).copy()

    shift = 0.0
    docc = 0.0

    for it in range(maxiter):
        # Lattice: only d + h1 (M1h=1, M2h=0, Mbh=0)
        nh1_l, dh1_l, _, _, _, _, _ = lattice_statics(
            beta, [eta], [W], [0.], [0.], [0.], [0.],
            1, 0, 0, EPS, GAM, EPS_W, shift)
        nh1_l = float(nh1_l[0])
        dh1_l = float(dh1_l[0])

        # Step 2: find eps, V from gateway1 h-sector match
        def res2(p):
            e = p[:M1g]; v = p[M1g:]
            nh1_g, dh1_g, _, _, _ = gateway1_statics(
                beta, eta, W, e, v, M1g, shift)
            return np.array([nh1_g - nh1_l, dh1_g - dh1_l])

        sol2 = least_squares(res2, np.concatenate([eps, V]), method='trf',
                             ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=5000,
                             bounds=([-8] * (2*M1g), [8] * (2*M1g)))
        eps_new = sol2.x[:M1g]
        V_new = sol2.x[M1g:]

        # Step 3: impurity
        ng1, dg1, nd, docc = impurity1_statics(beta, eps_new, V_new, M1g, U, 0.0)

        # Step 4: find eta, W from gateway1 g-sector match
        def res4(p):
            _, _, ng1_g, dg1_g, _ = gateway1_statics(
                beta, p[0], p[1], eps_new, V_new, M1g, shift)
            return np.concatenate([ng1_g - ng1, dg1_g - dg1])

        sol4 = least_squares(res4, [eta, W], method='trf',
                             ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=5000,
                             bounds=([-8, -8], [8, 8]))
        eta_new = sol4.x[0]
        W_new = sol4.x[1]

        dp = (abs(eta_new - eta) + abs(W_new - W)
              + float(np.sum(np.abs(eps_new - eps) + np.abs(V_new - V))))
        eta = mix * eta_new + (1 - mix) * eta
        W = mix * W_new + (1 - mix) * W
        eps = mix * eps_new + (1 - mix) * eps
        V = mix * V_new + (1 - mix) * V

        if dp < tol:
            break

    return dict(eta=eta, W=W, eps=eps, V=V, docc=docc, iters=it)


# ═══════════════════════════════════════════════════════════
# Bond solver — alternating loop with direct matching
# ═══════════════════════════════════════════════════════════

def solve_bond(beta, M1g, M2g, Mbg, M1h, M2h, Mbh, U, t,
               EPS, GAM, EPS_W, z=4,
               p0=None, mix=0.5, tol=1e-7, maxiter=50, verbose=True):
    """Alternating outer loop: fix impurity → solve gateway → re-solve impurity.

    Direct matching conditions (NOT BPK combination):
      h-sector: gateway correlators = lattice correlators
      g-sector: gateway correlators = impurity correlators
      half-filling: nd2_gateway = 0.5

    Variables (Np = 2*(M1h + M2h + Mbh + M1g + M2g + Mbg)):
      [eta1(M1h), W1(M1h), eta2(M2h), W2(M2h), etab(Mbh), Bh(Mbh),
       eps1(M1g), V1(M1g), eps2(M2g), V2(M2g), epsb(Mbg), Bg(Mbg)]

    Parameters
    ----------
    beta : float
    M1g, M2g, Mbg : int
        g-ghost counts per family.
    M1h, M2h, Mbh : int
        h-ghost counts per family (typically all 1).
    U, t : float
    EPS, GAM, EPS_W : arrays
        Lattice dispersion, form factor, weights.
    z : int
        Coordination number.
    p0 : array or None
        Warm-start parameter vector.
    mix : float
    tol : float
    maxiter : int
    verbose : bool

    Returns
    -------
    dict with converged parameters and observables.
    """
    n1h = M1h; n2h = M2h; nbh = Mbh
    n1g = M1g; n2g = M2g; nbg = Mbg
    Np = 2 * (n1h + n2h + nbh + n1g + n2g + nbg)

    def unpack(p):
        i = 0
        eta1 = p[i:i+n1h]; i += n1h
        W1   = p[i:i+n1h]; i += n1h
        eta2 = p[i:i+n2h]; i += n2h
        W2   = p[i:i+n2h]; i += n2h
        etab = p[i:i+nbh]; i += nbh
        Bh   = p[i:i+nbh]; i += nbh
        eps1 = p[i:i+n1g]; i += n1g
        V1   = p[i:i+n1g]; i += n1g
        eps2 = p[i:i+n2g]; i += n2g
        V2   = p[i:i+n2g]; i += n2g
        epsb = p[i:i+nbg]; i += nbg
        Bg   = p[i:i+nbg]; i += nbg
        return eta1, W1, eta2, W2, etab, Bh, eps1, V1, eps2, V2, epsb, Bg

    if p0 is None:
        p = np.zeros(Np)
        i0 = 2 * (n1h + n2h + nbh)
        p[n1h:2*n1h] = 0.3                           # W1
        p[2*n1h+n2h:2*n1h+2*n2h] = 0.3               # W2
        p[i0+n1g:i0+2*n1g] = 0.4                      # V1
        p[i0+2*n1g+n2g:i0+2*n1g+2*n2g] = 0.4          # V2
    else:
        p = np.asarray(p0, dtype=float).copy()
        assert len(p) == Np, f'p0 has length {len(p)}, expected {Np}'

    shift = 0.0

    # Initialize impurity correlators (updated after each outer iteration)
    ng1_i = np.zeros(n1g); dg1_i = np.zeros(n1g)
    ng2_i = np.zeros(n2g); dg2_i = np.zeros(n2g)
    ngb_i = np.zeros(nbg); dgb_i = np.zeros(nbg)
    docc1 = docc2 = hop = nd_total = docc_bpk = 0.0
    PENALTY = 1e4

    for it in range(1, maxiter + 1):
        eta1, W1, eta2, W2, etab, Bh, eps1, V1, eps2, V2, epsb, Bg = unpack(p)

        # Solve all gateway conditions with frozen impurity correlators
        def residuals(p_):
            et1, W1_, et2, W2_, etb, Bh_, e1, v1, e2, v2, eb, bg = unpack(p_)

            nh1_l, dh1_l, nh2_l, dh2_l, nhb_l, dhb_l, _ = lattice_statics(
                beta, et1, W1_, et2, W2_, etb, Bh_,
                M1h, M2h, Mbh, EPS, GAM, EPS_W, shift)

            nh1_g, dh1_g, ng1_g, dg1_g, _ = gateway1_statics(
                beta, et1, W1_, e1, v1, n1g, shift)
            nh2_g, dh2_g, nhb_g, dhb_g, ng2_g, dg2_g, ngb_g, dgb_g, nd2_g = \
                gateway2_statics(
                    beta, et2, W2_, etb, Bh_, e2, v2, eb, bg, n2g, nbg, t, shift)

            r = np.concatenate([
                np.atleast_1d(nh1_g - nh1_l),     # h1: gw=lat
                np.atleast_1d(dh1_g - dh1_l),
                np.atleast_1d(nh2_g - nh2_l),     # h2: gw=lat
                np.atleast_1d(dh2_g - dh2_l),
                np.atleast_1d(nhb_g - nhb_l),     # hb: gw=lat
                np.atleast_1d(dhb_g - dhb_l),
                ng1_g - ng1_i,                     # g1: gw=imp
                dg1_g - dg1_i,
                ng2_g - ng2_i,                     # g2: gw=imp
                dg2_g - dg2_i,
                ngb_g - ngb_i,                     # gb: gw=imp
                dgb_g - dgb_i,
                [nd2_g - 0.5],                     # half-filling
            ])
            if docc1 <= 0 or docc2 <= 0 or docc_bpk <= 0:
                r *= PENALTY
            return r

        sol = least_squares(residuals, p, method='trf',
                            ftol=1e-10, xtol=1e-10, gtol=1e-10,
                            max_nfev=50000,
                            bounds=([-8] * Np, [8] * Np))

        p_new = sol.x
        dp = float(np.linalg.norm(p_new - p))
        p = mix * p_new + (1 - mix) * p
        eta1, W1, eta2, W2, etab, Bh, eps1, V1, eps2, V2, epsb, Bg = unpack(p)

        # Re-solve impurities with dmu bisection for half-filling
        ng1_i, dg1_i, nd1_i, docc1 = impurity1_statics(
            beta, eps1, V1, n1g, U, 0.0)

        dmu2 = 0.0
        for _ in range(30):
            ng2_i, dg2_i, ngb_i, dgb_i, nd2_i, docc2, hop = \
                impurity2_statics(beta, eps2, V2, epsb, Bg, n2g, nbg, U, t, dmu2)
            err = nd2_i - 0.5
            if abs(err) < 1e-6:
                break
            dmu2 -= err / (beta * nd2_i * (1 - nd2_i) * 2 + 1e-6)
            dmu2 = float(np.clip(dmu2, -2.0, 2.0))

        nd_total = (1 - z) * nd1_i + z * nd2_i
        docc_bpk = (1 - z) * docc1 + z * docc2

        rnorm = float(np.linalg.norm(sol.fun))

        if verbose:
            physical = docc1 > 0 and docc2 > 0 and docc_bpk > 0 and abs(nd_total - 0.5) < 0.05
            if physical:
                print(f'  it={it:3d}  dp={dp:.2e}  |r|={rnorm:.2e}  '
                      f'nd={nd_total:.4f}  docc1={docc1:.5f}  '
                      f'docc2={docc2:.5f}  docc_bpk={docc_bpk:.5f}  hop={hop:.5f}')
            else:
                print(f'  it={it:3d}  dp={dp:.2e}  |r|={rnorm:.2e}  '
                      f'[unphysical: nd={nd_total:.4f}  docc1={docc1:.4f}  '
                      f'docc2={docc2:.4f}  docc_bpk={docc_bpk:.4f}]')
            sys.stdout.flush()

        if dp < tol and rnorm < tol:
            if verbose:
                print(f'  Converged at it={it}')
            break

    return dict(
        p=p, eta1=eta1, W1=W1, eta2=eta2, W2=W2, etab=etab, Bh=Bh,
        eps1=eps1, V1=V1, eps2=eps2, V2=V2, epsb=epsb, Bg=Bg,
        docc1=docc1, docc2=docc2, docc_bpk=docc_bpk,
        hop=hop, nd_total=nd_total, iters=it, dp=dp, rnorm=rnorm)


# ═══════════════════════════════════════════════════════════
# Temperature sweep
# ═══════════════════════════════════════════════════════════

def run_temperature_sweep(U=1.3, t=0.5, M1g=1, M2g=1, Mbg=1,
                          n_k=20, nT=10, Tmin=0.1, Tmax=1.0, T_vals=None,
                          mix_ss=0.5, mix_bond=0.5,
                          tol_ss=1e-8, tol_bond=1e-7,
                          maxiter_ss=200, maxiter_bond=100,
                          verbose=False, tag=''):
    """Run bond-scheme DMFT over a range of temperatures.

    M1h = M2h = Mbh = 1 fixed. Only M1g, M2g, Mbg are free.

    Parameters
    ----------
    U, t : float
    M1g, M2g, Mbg : int
        g-ghost counts per family.
    n_k : int
    nT : int
    Tmin, Tmax : float
    T_vals : array, optional
        Explicit T values (overrides nT/Tmin/Tmax).
    mix_ss, mix_bond : float
    tol_ss, tol_bond : float
    maxiter_ss, maxiter_bond : int
    verbose : bool
    tag : str

    Returns
    -------
    results : list of dicts
    """
    M1h = M2h = Mbh = 1   # fixed

    EPS, GAM, EPS_W, D, z = make_square_lattice(t, n_k=n_k)

    if T_vals is None:
        T_vals = np.linspace(Tmax, Tmin, nT)

    # Check square system constraint
    assert M1h == M1g, f'Need M1h={M1g} (=M1g) for square system, got M1h={M1h}'
    assert M2h + Mbh == M2g + Mbg, \
        f'Need M2h+Mbh={M2g+Mbg} (=M2g+Mbg) for square system, got {M2h+Mbh}'

    print(f'\nGhost-DMFT bond  M1g={M1g} M2g={M2g} Mbg={Mbg}  '
          f'M1h={M1h} M2h={M2h} Mbh={Mbh}  U={U}  t={t}  z={z}')
    print(f'{"T":>6}  {"docc_ss":>9}  {"docc_bpk":>10}  '
          f'{"docc1":>8}  {"docc2":>8}  {"hop":>8}  {"nd":>7}  {"its":>4}')
    print('-' * 85)

    p0 = None   # warm-start parameter vector
    results = []

    for T in T_vals:
        beta = 1.0 / T
        t0 = time.time()

        # Single-site reference
        ss = solve_singlesite(
            beta, M1g, U, t, EPS, GAM, EPS_W,
            mix=mix_ss, tol=tol_ss, maxiter=maxiter_ss)

        if verbose:
            print(f'\nT={T:.4f}  beta={beta:.2f}  ss docc={ss["docc"]:.6f}')

        # Build initial p0 from single-site solution
        if p0 is None:
            Np = 2 * (M1h + M2h + Mbh + M1g + M2g + Mbg)
            p0 = np.zeros(Np)
            i = 0
            p0[i:i+M1h] = ss['eta']; i += M1h    # eta1
            p0[i:i+M1h] = ss['W'];   i += M1h    # W1
            p0[i:i+M2h] = ss['eta']; i += M2h    # eta2
            p0[i:i+M2h] = ss['W'];   i += M2h    # W2
            i += 2 * Mbh                           # etab=0, Bh=0
            i += M1g                               # eps1=0 initially
            p0[i:i+M1g] = ss['V'];   i += M1g    # V1
            i += M2g                               # eps2=0 initially
            p0[i:i+M2g] = ss['V'][0] * np.ones(M2g); i += M2g  # V2
            # epsb=0, Bg=0 initially

        rb = solve_bond(
            beta, M1g, M2g, Mbg, M1h, M2h, Mbh, U, t,
            EPS, GAM, EPS_W, z=z,
            p0=p0, mix=mix_bond, tol=tol_bond,
            maxiter=maxiter_bond, verbose=verbose)

        dt = time.time() - t0
        d1 = rb['docc1']; d2 = rb['docc2']; dbpk = rb['docc_bpk']
        nd = rb['nd_total']; hp = rb['hop']
        physical = d1 > 0 and d2 > 0 and dbpk > 0 and abs(nd - 0.5) < 0.05

        if physical:
            print(f'  {T:6.4f}  {ss["docc"]:9.6f}  {dbpk:10.6f}  '
                  f'{d1:8.6f}  {d2:8.6f}  {hp:8.5f}  {nd:7.4f}  '
                  f'{rb["iters"]:4d}  ({dt:.1f}s)')
            p0 = rb['p']  # warm start next T
        else:
            print(f'  {T:6.4f}  {ss["docc"]:9.6f}  [unphysical]'
                  f'  iters={rb["iters"]}  |r|={rb["rnorm"]:.2e}  ({dt:.1f}s)')
        sys.stdout.flush()

        results.append(dict(
            T=T, ToverD=T / D, beta=beta,
            docc_ss=ss['docc'], docc_bpk=dbpk,
            docc_1=d1, docc_2=d2, hop=hp,
            nd_total=nd, res=rb['rnorm'], iters_bond=rb['iters']))

        # Save after every T
        fname = f'bond_M1g{M1g}M2g{M2g}Mbg{Mbg}_U{U}'
        if tag:
            fname += f'_{tag}'
        fname += '.dat'
        save_results(results, fname)

    return results, D


# ═══════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════

def save_results(results, fname):
    """Save sweep results to a .dat file."""
    keys = ['T', 'docc_ss', 'docc_bpk', 'docc_1', 'docc_2', 'hop']
    header = '  '.join(keys)
    rows = [[r.get(k, np.nan) for k in keys] for r in results]
    np.savetxt(fname, rows, header=header, fmt='%.8f')
