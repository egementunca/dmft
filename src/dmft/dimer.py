"""Dimer ghost-DMFT solver: self-consistency loop, temperature sweep, (U,n) study.

Unified for half-filling (n_target=2.0) and arbitrary doping.

4-step sequential scheme per iteration:
  Step 1: Lattice → h-targets
  Step 2: Gateway least_squares fit (h-sector: gateway = lattice)
  Step 3: Impurity ED → g-targets; if doped: update Sigma_inf, bisect mu
  Step 4: Gateway least_squares fit (g-sector: gateway = impurity)
"""

import sys
import time

import numpy as np
from scipy.optimize import least_squares, brentq

from .dimer_ed import dimer_impurity_obs
from .dimer_gateway import dimer_gateway_obs
from .dimer_lattice import dimer_square_lattice_kgrid, dimer_lattice_obs


# ═══════════════════════════════════════════════════════════
# Single-temperature solver
# ═══════════════════════════════════════════════════════════

def solve_T(T, x0, Uval=1.3, t_d=0.5, t_b=0.3,
            M=1, hop=False, n_target=2.0, nk=20,
            mix=0.5, tol=1e-8, maxiter=300, verbose=False):
    """Dimer ghost-DMFT self-consistency at one temperature.

    x0 layout (length 6M or 6M+2):
      [eta_h(M), W_h(M), t_h(M), t_g(M), eps_g(M), V_g(M)]
      For doped: optionally append [mu, Sigma_inf].

    Parameters
    ----------
    T : float
    x0 : array
    Uval, t_d, t_b : float
    M : int
    hop : bool
    n_target : float — target filling (2.0 = half-filling)
    nk : int
    mix, tol : float
    maxiter : int
    verbose : bool

    Returns
    -------
    dict with converged parameters and observables.
    """
    beta = 1.0 / T
    half_filling = abs(n_target - 2.0) < 1e-10
    eps_k, wk = dimer_square_lattice_kgrid(t_d, nk=nk)

    x0 = np.array(x0, dtype=float)
    eta_h = x0[0*M:1*M].copy()
    W_h   = x0[1*M:2*M].copy()
    t_h   = x0[2*M:3*M].copy()
    t_g   = x0[3*M:4*M].copy()
    eps_g = x0[4*M:5*M].copy()
    V_g   = x0[5*M:6*M].copy()

    if half_filling:
        mu = Uval / 2.0
        Sigma_inf = Uval / 2.0
    else:
        mu        = float(x0[6*M])   if len(x0) > 6*M   else Uval / 2.0
        Sigma_inf = float(x0[6*M+1]) if len(x0) > 6*M+1 else Uval / 2.0

    lsq = dict(method='trf', ftol=1e-11, xtol=1e-11, gtol=1e-11,
               max_nfev=5000, bounds=(-15., 15.))
    docc = 0.25
    n_dimer_imp = n_target

    for it in range(1, maxiter + 1):

        # Step 1: Lattice → h-targets
        lat = dimer_lattice_obs(beta, mu, Sigma_inf, t_b, M,
                                eta_h, W_h, t_h, hop, eps_k, wk)
        n_hA_lat = lat['n_hA']
        d_hA_lat = lat['d_hA']
        hhop_lat = lat['hhop']

        # Step 2: Gateway fit — h-sector (solve for eps_g, V_g [, t_h])
        p2 = np.concatenate([eps_g, V_g] + ([t_h] if hop else []))

        def r2(p):
            eg_ = p[0:M]; Vg_ = p[M:2*M]
            th_ = p[2*M:3*M] if hop else t_h
            gw = dimer_gateway_obs(beta, mu, t_b, M, eg_, Vg_, t_g,
                                   eta_h, W_h, th_, hop, Sigma_inf)
            res = list(gw['n_hA'] - n_hA_lat) + list(gw['d_hA'] - d_hA_lat)
            if hop:
                res += list(gw['hhop'] - hhop_lat)
            return res

        sol2 = least_squares(r2, p2, **lsq)
        eps_g_n = sol2.x[0:M].copy()
        V_g_n   = sol2.x[M:2*M].copy()
        t_h_n   = sol2.x[2*M:3*M].copy() if hop else t_h.copy()

        # Step 3: Impurity ED → g-targets
        imp = dimer_impurity_obs(beta, mu, Uval, t_b, M,
                                 eps_g_n, V_g_n, t_g, hop)
        docc        = imp['docc']
        n_dimer_imp = imp['n_dimer_imp']
        n_g_imp     = imp['n_g']
        d_g_imp     = imp['d_g']
        ghop_imp    = imp['ghop']

        # For doped: update Sigma_inf and bisect mu
        if half_filling:
            Sigma_inf_n = Uval / 2.0
            mu_n = Uval / 2.0
        else:
            Sigma_inf_n = Uval * n_dimer_imp / 4.0

            def filling(mu_val):
                l = dimer_lattice_obs(beta, mu_val, Sigma_inf_n, t_b, M,
                                      eta_h, W_h, t_h_n, hop, eps_k, wk)
                return l['n_dimer_lat'] - n_target

            try:
                mu_n = brentq(filling, mu - 4.0, mu + 4.0,
                              xtol=1e-8, maxiter=100)
            except ValueError:
                mu_n = brentq(filling, -10.0, 10.0,
                              xtol=1e-8, maxiter=200)

        # Step 4: Gateway fit — g-sector (solve for eta_h, W_h [, t_g])
        p4 = np.concatenate([eta_h, W_h] + ([t_g] if hop else []))

        def r4(p):
            eh_ = p[0:M]; Wh_ = p[M:2*M]
            tg_ = p[2*M:3*M] if hop else t_g
            gw = dimer_gateway_obs(beta, mu_n, t_b, M, eps_g_n, V_g_n, tg_,
                                   eh_, Wh_, t_h_n, hop, Sigma_inf_n)
            res = list(gw['n_gA'] - n_g_imp) + list(gw['d_gA'] - d_g_imp)
            if hop:
                res += list(gw['ghop'] - ghop_imp)
            return res

        sol4 = least_squares(r4, p4, **lsq)
        eta_h_n = sol4.x[0:M].copy()
        W_h_n   = sol4.x[M:2*M].copy()
        t_g_n   = sol4.x[2*M:3*M].copy() if hop else t_g.copy()

        # Clip
        for arr in [eta_h_n, W_h_n, eps_g_n, V_g_n]:
            np.clip(arr, -10., 10., out=arr)

        # Mix ghost parameters; mu and Sigma_inf set directly
        x_new = np.concatenate([eta_h_n, W_h_n, t_h_n, t_g_n, eps_g_n, V_g_n])
        x_old = np.concatenate([eta_h, W_h, t_h, t_g, eps_g, V_g])
        dp = float(np.linalg.norm(np.concatenate(
            [x_new - x_old, [mu_n - mu, Sigma_inf_n - Sigma_inf]])))
        xm = mix * x_new + (1.0 - mix) * x_old

        eta_h = xm[0*M:1*M]; W_h = xm[1*M:2*M]
        t_h   = xm[2*M:3*M]; t_g = xm[3*M:4*M]
        eps_g = xm[4*M:5*M]; V_g = xm[5*M:6*M]
        Sigma_inf = Sigma_inf_n
        mu = mu_n

        if verbose:
            print(f'  it={it:3d}  dp={dp:.2e}  D={docc:.5f}'
                  f'  n_dimer={n_dimer_imp:.4f}'
                  f'  mu={mu:.4f}  Sinf={Sigma_inf:.4f}'
                  f'  n_lat={lat["n_dimer_lat"]:.4f}')
        if dp < tol:
            break

    x_out = np.concatenate([eta_h, W_h, t_h, t_g, eps_g, V_g, [mu, Sigma_inf]])
    return dict(T=T, iters=it, dp=dp, docc=docc,
                mu=mu, Sigma_inf=Sigma_inf,
                n_dimer_lat=lat['n_dimer_lat'],
                n_dimer_imp=n_dimer_imp,
                n_target=n_target,
                eps_g=eps_g.copy(), V_g=V_g.copy(),
                eta_h=eta_h.copy(), W_h=W_h.copy(),
                t_g=t_g.copy(), t_h=t_h.copy(),
                x=x_out)


# ═══════════════════════════════════════════════════════════
# Temperature sweep
# ═══════════════════════════════════════════════════════════

def _default_x0(M, Uval, n_target):
    """Default initial guess for the dimer solver."""
    if M == 1:
        x0 = np.array([0.01, 0.20, 0.01, 0.01, -0.01, 0.20])
    elif M == 2:
        x0 = np.array([-0.30, 0.30, 0.20, 0.20, 0.01, 0.01,
                         0.01, 0.01, -0.30, 0.30, 0.20, 0.20])
    else:
        sp = np.linspace(-0.4, 0.4, M)
        x0 = np.concatenate([sp, np.full(M, 0.2), np.full(M, 0.01),
                              np.full(M, 0.01), sp, np.full(M, 0.2)])
    if abs(n_target - 2.0) >= 1e-10:
        x0 = np.concatenate([x0, [Uval / 2.0, Uval / 2.0]])
    return x0


def run_sweep(Uval=1.3, t_d=0.5, t_b=0.3, M=1, hop=False,
              n_target=2.0, nk=20, nT=20, T_max=5.0, T_min=0.1,
              mix=0.5, tol=1e-8, maxiter=300, verbose=False, x0=None):
    """Temperature sweep with warm-starting.

    Unified for half-filling and doped.
    """
    Ts = np.logspace(np.log10(T_max), np.log10(T_min), nT)
    if x0 is None:
        x0 = _default_x0(M, Uval, n_target)
    x0 = np.array(x0, dtype=float)

    mode = 'hop' if hop else 'no-hop'
    filling = f'n={n_target:.2f}' if abs(n_target - 2.0) >= 1e-10 else 'half-fill'
    print(f'\nDimer ghost-DMFT  M={M}  [{mode}]  {filling}'
          f'  U={Uval}  t_d={t_d}  t_b={t_b}  nk={nk}')
    print(f'Impurity: Norb={2+2*M}, sectors up to '
          f'{max(1, int(np.math.comb(2+2*M, (2+2*M)//2))**2)} states')

    cols = ['T', 'D', 'mu', 'Sigma_inf', 'n_dimer_lat', 'n_dimer_imp',
            'eps_g[0]', 'V_g[0]', 'eta_h[0]', 'W_h[0]', 'iters', 'dp']
    hdr = '  '.join(f'{c:>12}' for c in cols)
    print(hdr); print('-' * len(hdr))

    results = []; xp = None; x2 = None
    for T in Ts:
        if   x2 is not None: xi = np.clip(2 * xp - x2, -5., 5.)
        elif xp is not None: xi = xp.copy()
        else:                xi = x0.copy()

        r = solve_T(T, xi, Uval=Uval, t_d=t_d, t_b=t_b,
                    M=M, hop=hop, n_target=n_target, nk=nk,
                    mix=mix, tol=tol, maxiter=maxiter, verbose=verbose)

        vals = [T, r['docc'], r['mu'], r['Sigma_inf'],
                r['n_dimer_lat'], r['n_dimer_imp'],
                r['eps_g'][0], r['V_g'][0], r['eta_h'][0], r['W_h'][0]]
        row = '  '.join(f'{v:12.5f}' for v in vals)
        print(row + f'  {r["iters"]:8d}  {r["dp"]:9.2e}')
        sys.stdout.flush()
        results.append(r); x2 = xp; xp = r['x'].copy()

    return results


# ═══════════════════════════════════════════════════════════
# (U, n) phase diagram scan
# ═══════════════════════════════════════════════════════════

def Z_scalar(W_h, eta_h):
    """k-averaged quasiparticle weight from lowest ghost pole."""
    W = W_h[0]; e = eta_h[0]
    lam0 = (e - np.sqrt(e**2 + 4 * W**2)) / 2.0
    return float(1.0 / (1.0 + W**2 / (lam0 - e)**2))


def run_study(M=1, hop=True, U_list=None, n_list=None,
              t_d=0.5, t_b=0.3, nk=32, nT=24,
              T_max=4.0, T_min=0.03,
              mix=0.5, tol=1e-8, maxiter=600, quick=False):
    """Full (U, n) grid scan for phase diagram.

    Saves .npy files after each (U, n) point.
    """
    if quick:
        U_list = U_list or [2.0, 3.0, 4.0]
        n_list = n_list or [2.0, 1.8, 1.6, 1.4]
        nk = min(nk, 20); nT = min(nT, 8); T_min = max(T_min, 0.10)
    else:
        U_list = U_list or [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        n_list = n_list or [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.2, 1.0]

    summary = dict(
        M=M, U_list=U_list, n_list=n_list, T_min=T_min,
        D_lowT=np.full((len(U_list), len(n_list)), np.nan),
        Z_lowT=np.full((len(U_list), len(n_list)), np.nan),
        mu_lowT=np.full((len(U_list), len(n_list)), np.nan),
        Si_lowT=np.full((len(U_list), len(n_list)), np.nan),
        dp_lowT=np.full((len(U_list), len(n_list)), np.nan),
        T_actual=np.full((len(U_list), len(n_list)), np.nan),
    )

    print(f'\n{"=" * 65}')
    print(f'Dimer ghost-DMFT study (internal)  M={M}  hop={hop}')
    print(f'U: {U_list}')
    print(f'n: {n_list}  (dimer electrons, half-fill=2)')
    print(f'T: {T_max} -> {T_min}  nT={nT}  nk={nk}')
    print(f'{"=" * 65}')

    for iU, U in enumerate(U_list):
        print(f'\n{"=" * 65}')
        print(f'U = {U}')
        print(f'{"=" * 65}')

        x_seed = None

        for in_, n in enumerate(n_list):
            t0 = time.time()
            half = abs(n - 2.0) < 1e-10

            if half:
                print(f'\n  n=2.0 (half-filling) U={U}')
                x0 = _default_x0(M, U, 2.0)
            else:
                print(f'\n  n={n:.2f}  U={U}')
                x0 = x_seed if x_seed is not None else _default_x0(M, U, n)

            res = run_sweep(Uval=U, t_d=t_d, t_b=t_b, M=M, hop=hop,
                            n_target=n, nk=nk, nT=nT,
                            T_max=T_max, T_min=T_min,
                            mix=mix, tol=tol, maxiter=maxiter, x0=x0)

            tag = 'halffill' if half else f'doped_n{n:.2f}'
            fname = f'study_M{M}_{tag}_U{U:.1f}_internal.npy'
            np.save(fname, np.array(res, dtype=object))

            rl = res[-1]
            Z = Z_scalar(rl['W_h'], rl['eta_h'])
            mu_val = U / 2.0 if half else rl['mu']
            Si_val = U / 2.0 if half else rl['Sigma_inf']

            # Seed next filling
            if half:
                x_seed = np.concatenate([rl['x'][:6*M], [U / 2., U / 2.]])
            else:
                x_seed = rl['x'].copy()

            summary['D_lowT'][iU, in_] = rl['docc']
            summary['Z_lowT'][iU, in_] = Z
            summary['mu_lowT'][iU, in_] = mu_val
            summary['Si_lowT'][iU, in_] = Si_val
            summary['dp_lowT'][iU, in_] = rl['dp']
            summary['T_actual'][iU, in_] = rl['T']
            np.save(f'study_M{M}_summary_internal.npy', summary)

            print(f'  --> SAVED {fname}')
            print(f'      D={rl["docc"]:.5f}  Z={Z:.4f}'
                  f'  mu={mu_val:.4f}  Sinf={Si_val:.4f}'
                  f'  dp={rl["dp"]:.1e}  t={time.time()-t0:.1f}s')
            sys.stdout.flush()

    print(f'\n{"=" * 65}')
    print(f'COMPLETE. Summary: study_M{M}_summary_internal.npy')
    print(f'{"=" * 65}')
    return summary


# ═══════════════════════════════════════════════════════════
# Sanity checks
# ═══════════════════════════════════════════════════════════

def check_atomic_limit(Uval=1.3):
    """Atomic limit: t_b=0, V_g=0 → closed-form double occupancy."""
    beta = 10.0; mu = Uval / 2.0
    Z = 1 + 2 * np.exp(beta * mu) + np.exp(beta * (2 * mu - Uval))
    D_exact = np.exp(beta * (2 * mu - Uval)) / Z
    imp = dimer_impurity_obs(beta, mu, Uval, 0.0, 1,
                             np.zeros(1), np.zeros(1), np.zeros(1), False)
    ok = abs(imp['docc'] - D_exact) < 1e-8
    print(f'Atomic limit (U={Uval}): D_exact={D_exact:.8f}  '
          f'D_imp={imp["docc"]:.8f}  {"PASSED" if ok else "FAILED"}')
    return ok


def check_halffill(Uval=1.3, T=1.0):
    """Doped solver at n_target=2.0 should recover mu=U/2."""
    print(f'Half-filling check: n_target=2, U={Uval}, T={T}')
    x0 = np.array([0.01, 0.20, 0.01, 0.01, -0.01, 0.20,
                    Uval / 2., Uval / 2.])
    r = solve_T(T, x0, Uval=Uval, n_target=2.0, nk=20,
                mix=0.5, tol=1e-8, maxiter=300)
    print(f'  mu={r["mu"]:.6f}         (should be {Uval/2.:.6f})')
    print(f'  Sigma_inf={r["Sigma_inf"]:.6f}  (should be {Uval/2.:.6f})')
    print(f'  n_dimer_lat={r["n_dimer_lat"]:.6f}  (should be 2.000000)')
    print(f'  n_dimer_imp={r["n_dimer_imp"]:.6f}  (should be 2.000000)')
    print(f'  D={r["docc"]:.6f}')
    ok = (abs(r['mu'] - Uval / 2.) < 1e-4 and
          abs(r['Sigma_inf'] - Uval / 2.) < 1e-4 and
          abs(r['n_dimer_lat'] - 2.0) < 1e-4 and
          abs(r['n_dimer_imp'] - 2.0) < 1e-4)
    print(f'  {"PASSED" if ok else "FAILED"}')
    return ok
