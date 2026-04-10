"""Ghost Nested Cluster scheme for the square lattice (z=4).

Combines single-site (1) and two-site (2) impurities via BPK:
  (1-z)*<O>_1 + z*<O>_2 = <O>_lattice

Components:
  - Impurity 1: d + M g1-ghosts (single-site, interacting)
  - Impurity 2: dA,dB + M g2-ghosts/site + t_g inter-site hopping (two-site, interacting)
  - Gateway 1: d + M g1-ghosts + M h1-ghosts (single-site, quadratic)
  - Gateway 2: dA,dB + M g2/site + M h2/site + t_g,t_h (two-site, quadratic)
  - Lattice: dA,dB + M h/site + M h2/site + t_h (square lattice k-sum)

Self-consistency (per iteration):
  Step 1: Lattice -> h-targets (n_h, d_h, n_h2, d_h2, h2hop)
  Step 2a: Fit V1, V2 via BPK: (1-z)*gw1 + z*gw2 = lattice (h-sector)
  Step 2b: Fit t_h from h2hop
  Step 3: Impurity1 + Impurity2 -> g-targets
  Step 4a: Fit eta1,W1,eta2,W2,W via g-sector matching + moment conditions
  Step 4b: Fit t_g from g2hop
"""

import sys

import numpy as np
from scipy.optimize import least_squares

from .bond_ed import impurity1_statics
from .dimer_ed import dimer_impurity_obs
from .dimer_lattice import _fermi_lat


# ═══════════════════════════════════════════════════════════
# Lattice: dA,dB + hA,hB + h2A,h2B (size 2+4M per k)
# ═══════════════════════════════════════════════════════════

def nc_lattice_obs(beta, mu, M, eta, W, eta2, W2, t_h, nk=20, t=0.5):
    """Square lattice k-sum for the nested cluster scheme.

    Orbital layout per k: dA=0, dB=1,
      hA[M], hB[M], h2A[M], h2B[M] — size 2+4M.
    h2 ghosts have inter-site hopping t_h.
    Sigma_inf = mu at half-filling → d-level shift = 0.
    """
    k = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    kx_2d, ky_2d = np.meshgrid(k, k, indexing='ij')
    eps_k = (-2.0 * t * (np.cos(kx_2d) + np.cos(ky_2d))).ravel()
    wk = np.ones(len(eps_k)) / len(eps_k)

    sz = 2 + 4 * M
    dA = 0; dB = 1
    hA  = list(range(2,       2 + M))
    hB  = list(range(2 + M,   2 + 2*M))
    h2A = list(range(2 + 2*M, 2 + 3*M))
    h2B = list(range(2 + 3*M, 2 + 4*M))

    Nk = len(eps_k)
    H = np.zeros((Nk, sz, sz))
    H[:, dA, dA] = eps_k   # Sigma_inf - mu = 0 at half-filling
    H[:, dB, dB] = 0.0
    H[:, dA, dB] = H[:, dB, dA] = -t
    for a in range(M):
        H[:, hA[a], hA[a]] = eta[a];   H[:, hB[a], hB[a]] = eta[a]
        H[:, dA, hA[a]] = H[:, hA[a], dA] = W[a]
        H[:, dB, hB[a]] = H[:, hB[a], dB] = W[a]
        H[:, h2A[a], h2A[a]] = eta2[a]; H[:, h2B[a], h2B[a]] = eta2[a]
        H[:, dA, h2A[a]] = H[:, h2A[a], dA] = W2[a]
        H[:, dB, h2B[a]] = H[:, h2B[a], dB] = W2[a]
        H[:, h2A[a], h2B[a]] = H[:, h2B[a], h2A[a]] = -t_h[a]

    e, U = np.linalg.eigh(H)
    f = _fermi_lat(e, beta)
    rho = np.einsum('kin,kn,kjn->kij', U, f, U)

    n_h = np.zeros(M); d_h = np.zeros(M)
    n_h2 = np.zeros(M); d_h2 = np.zeros(M); h2hop = np.zeros(M)
    for a in range(M):
        n_h[a]   = np.dot(wk, rho[:, hA[a], hA[a]])
        d_h[a]   = np.dot(wk, rho[:, dA, hA[a]])
        n_h2[a]  = np.dot(wk, rho[:, h2A[a], h2A[a]])
        d_h2[a]  = np.dot(wk, rho[:, dA, h2A[a]])
        h2hop[a] = np.dot(wk, rho[:, h2A[a], h2B[a]])

    return dict(n_h=n_h, d_h=d_h, n_h2=n_h2, d_h2=d_h2, h2hop=h2hop)


# ═══════════════════════════════════════════════════════════
# Gateway 1: d + M g1 + M h1 (size 1+2M)
# ═══════════════════════════════════════════════════════════

def nc_gateway1_obs(beta, mu, M, eps1, V1, eta1, W1):
    """Single-site quadratic gateway."""
    sz = 1 + 2 * M
    d = 0
    g1 = list(range(1,     1 + M))
    h1 = list(range(1 + M, 1 + 2*M))
    H = np.zeros((sz, sz))
    for a in range(M):
        H[g1[a], g1[a]] = eps1[a]
        H[d, g1[a]] = H[g1[a], d] = V1[a]
        H[h1[a], h1[a]] = eta1[a]
        H[d, h1[a]] = H[h1[a], d] = W1[a]
    e, U = np.linalg.eigh(H)
    f = _fermi_lat(e, beta)
    rho = (U * f[None, :]) @ U.T
    res = {}
    for a in range(M):
        res[f'n_g1_{a}'] = rho[g1[a], g1[a]]
        res[f'd_g1_{a}'] = rho[d, g1[a]]
        res[f'n_h1_{a}'] = rho[h1[a], h1[a]]
        res[f'd_h1_{a}'] = rho[d, h1[a]]
    return res


# ═══════════════════════════════════════════════════════════
# Gateway 2: dA,dB + M g2/site + M h2/site (size 2+4M)
# ═══════════════════════════════════════════════════════════

def nc_gateway2_obs(beta, mu, M, eps2, V2, t_g, eta2, W2, t_h, t=0.5):
    """Two-site quadratic gateway."""
    sz = 2 + 4 * M
    dA = 0; dB = 1
    g2A = list(range(2,       2 + M))
    g2B = list(range(2 + M,   2 + 2*M))
    h2A = list(range(2 + 2*M, 2 + 3*M))
    h2B = list(range(2 + 3*M, 2 + 4*M))
    H = np.zeros((sz, sz))
    H[dA, dB] = H[dB, dA] = -t
    for a in range(M):
        H[g2A[a], g2A[a]] = eps2[a]; H[g2B[a], g2B[a]] = eps2[a]
        H[dA, g2A[a]] = H[g2A[a], dA] = V2[a]
        H[dB, g2B[a]] = H[g2B[a], dB] = V2[a]
        H[g2A[a], g2B[a]] = H[g2B[a], g2A[a]] = -t_g[a]
        H[h2A[a], h2A[a]] = eta2[a]; H[h2B[a], h2B[a]] = eta2[a]
        H[dA, h2A[a]] = H[h2A[a], dA] = W2[a]
        H[dB, h2B[a]] = H[h2B[a], dB] = W2[a]
        H[h2A[a], h2B[a]] = H[h2B[a], h2A[a]] = -t_h[a]
    e, U = np.linalg.eigh(H)
    f = _fermi_lat(e, beta)
    rho = (U * f[None, :]) @ U.T
    res = {}
    for a in range(M):
        res[f'n_g2_{a}']  = rho[g2A[a], g2A[a]]
        res[f'd_g2_{a}']  = rho[dA, g2A[a]]
        res[f'g2hop_{a}'] = rho[g2A[a], g2B[a]]
        res[f'n_h2_{a}']  = rho[h2A[a], h2A[a]]
        res[f'd_h2_{a}']  = rho[dA, h2A[a]]
        res[f'h2hop_{a}'] = rho[h2A[a], h2B[a]]
    return res


# ═══════════════════════════════════════════════════════════
# Impurity 1 wrapper (uses bond_ed.impurity1_statics)
# ═══════════════════════════════════════════════════════════

def nc_impurity1_obs(beta, mu, U, M, eps1, V1):
    """Single-site impurity via sector-blocked ED (bond_ed).

    Returns dict with docc, n_g1 (array M, per-spin), d_g1 (array M, per-spin).
    """
    ng1, dg1, nd, docc = impurity1_statics(beta, eps1, V1, M, U, 0.0)
    return dict(docc=docc, n_g1=ng1, d_g1=dg1)


# ═══════════════════════════════════════════════════════════
# Impurity 2 wrapper (uses dimer_ed.dimer_impurity_obs)
# ═══════════════════════════════════════════════════════════

def nc_impurity2_obs(beta, mu, U, M, eps2, V2, t_g, t=0.5):
    """Two-site impurity via sector-blocked ED (dimer_ed).

    Returns dict with docc, n_g2, d_g2, g2hop (arrays M, per-spin).
    """
    imp = dimer_impurity_obs(beta, mu, U, t, M, eps2, V2, t_g, hop=True)
    return dict(docc=imp['docc'], n_g2=imp['n_g'], d_g2=imp['d_g'],
                g2hop=imp['ghop'])


# ═══════════════════════════════════════════════════════════
# Moment condition: derive lattice (eta, W) from BPK combination
# ═══════════════════════════════════════════════════════════

def solve_moments(M, z, eta1, W1, eta2, W2):
    """Derive lattice ghost parameters from BPK moment matching."""
    moments = np.zeros(2 * M)
    for k in range(2 * M):
        moments[k] = (1 - z) * np.sum(W1**2 * eta1**k) + z * np.sum(W2**2 * eta2**k)

    if M == 1:
        W_sq = max(moments[0], 0.0)
        return np.array([0.0]), np.array([np.sqrt(W_sq)])

    H_mat = np.array([[moments[i+j] for j in range(M)] for i in range(M)])
    H_shift = np.array([[moments[i+j+1] for j in range(M)] for i in range(M)])
    try:
        eta_new = np.sort(np.linalg.eigvals(np.linalg.solve(H_mat, H_shift)).real)
    except np.linalg.LinAlgError:
        eta_new = np.zeros(M)
    V_mat = np.vander(eta_new, M, increasing=True).T
    try:
        W_new = np.sqrt(np.abs(np.linalg.solve(V_mat, moments[:M])))
    except np.linalg.LinAlgError:
        W_new = np.ones(M) * 0.01
    return eta_new, W_new


# ═══════════════════════════════════════════════════════════
# Solver
# ═══════════════════════════════════════════════════════════

def solve_T(T, x0, Uval=1.3, z=4.0, M=1, nquad=200,
            mix=0.4, tol=1e-9, maxiter=300, verbose=False):
    """Nested cluster self-consistency at one temperature.

    x0 layout (8M): [eta(M), W(M), eta2(M), W2(M), t_h(M), eta1(M), W1(M), t_g(M)]
    """
    beta = 1.0 / T
    mu = Uval / 2.0
    lsq = dict(method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
    PENALTY = 1e4
    b_V = 100.0; b_W = 100.0

    x = np.array(x0, dtype=float)
    eta  = x[0*M:1*M].copy(); W    = x[1*M:2*M].copy()
    eta2 = x[2*M:3*M].copy(); W2   = x[3*M:4*M].copy()
    t_h  = x[4*M:5*M].copy()
    eta1 = x[5*M:6*M].copy(); W1   = x[6*M:7*M].copy()
    t_g  = x[7*M:8*M].copy()

    eps1 = np.full(M, -0.01); V1 = np.full(M, 0.01)
    eps2 = np.full(M, -0.01); V2 = np.full(M, 0.01)
    docc = 0.25

    for it in range(1, maxiter + 1):

        # Step 1: Lattice → h-targets
        lat = nc_lattice_obs(beta, mu, M, eta, W, eta2, W2, t_h, nk=nquad)
        n_h = lat['n_h'];  d_h = lat['d_h']
        n_h2 = lat['n_h2']; d_h2 = lat['d_h2']
        h2hop = lat['h2hop']

        # Step 2a: Fit V1, V2 via BPK h-sector matching
        def r2a(p):
            V1_ = p[0:M]; V2_ = p[M:2*M]
            gw1 = nc_gateway1_obs(beta, mu, M, eps1, V1_, eta1, W1)
            gw2 = nc_gateway2_obs(beta, mu, M, eps2, V2_, t_g, eta2, W2, t_h)
            res = []
            for a in range(M):
                nh1 = gw1[f'n_h1_{a}']; nh2 = gw2[f'n_h2_{a}']
                dh1 = gw1[f'd_h1_{a}']; dh2 = gw2[f'd_h2_{a}']
                lhs_hh = (1 - z) * nh1 + z * nh2
                res.append(lhs_hh - n_h[a])
                res.append((1 - z) * dh1 + z * dh2 - d_h[a])
                res.append(nh2 - n_h2[a])
                res.append(dh2 - d_h2[a])
                if lhs_hh < 0:
                    res[0] += PENALTY * lhs_hh
            return res

        sol = least_squares(r2a, np.concatenate([V1, V2]),
                            bounds=([-b_V] * 2*M, [b_V] * 2*M), **lsq)
        V1_new = sol.x[0:M]; V2_new = sol.x[M:2*M]
        eps1_new = eps1.copy(); eps2_new = eps2.copy()

        # Step 2b: Fit t_h from h2hop
        def r2b(p):
            gw2 = nc_gateway2_obs(beta, mu, M, eps2_new, V2_new, t_g,
                                  eta2, W2, p[:M])
            return [gw2[f'h2hop_{a}'] - h2hop[a] for a in range(M)]

        sol = least_squares(r2b, t_h, **lsq)
        t_h_new = sol.x[:M]

        # Step 3: Impurity targets
        imp1 = nc_impurity1_obs(beta, mu, Uval, M, eps1_new, V1_new)
        imp2 = nc_impurity2_obs(beta, mu, Uval, M, eps2_new, V2_new, t_g)
        docc = imp2['docc']
        n_g1 = imp1['n_g1']; d_g1 = imp1['d_g1']
        n_g2 = imp2['n_g2']; d_g2 = imp2['d_g2']
        g2hop = imp2['g2hop']

        # Step 4a: Fit eta1, W1, eta2, W2, W via g-sector + moment conditions
        if M == 1:
            def r4(p):
                W1_ = p[0:1]; W2_ = p[1:2]; W_ = p[2:3]
                gw1 = nc_gateway1_obs(beta, mu, M, eps1_new, V1_new,
                                      np.zeros(M), W1_)
                gw2 = nc_gateway2_obs(beta, mu, M, eps2_new, V2_new, t_g,
                                      np.zeros(M), W2_, t_h_new)
                res = [gw1['n_g1_0'] - n_g1[0], gw1['d_g1_0'] - d_g1[0],
                       gw2['n_g2_0'] - n_g2[0], gw2['d_g2_0'] - d_g2[0]]
                W_sq = (1 - z) * W1_[0]**2 + z * W2_[0]**2
                res.append(W_sq - W_[0]**2)
                return res

            sol = least_squares(r4, np.concatenate([W1, W2, W]),
                                bounds=([-b_W] * 3, [b_W] * 3), **lsq)
            W1_new = sol.x[0:1]; W2_new = sol.x[1:2]; W_new = sol.x[2:3]
            eta1_new = np.zeros(M); eta2_new = np.zeros(M); eta_new = np.zeros(M)
        else:
            # M=2: antisymmetric eta, fit eta0_1, W1, eta0_2, W2, W
            def r4(p):
                eta0_1 = p[0]; W1_ = p[1:3]
                eta0_2 = p[3]; W2_ = p[4:6]; W_ = p[6:8]
                e1_ = np.array([-eta0_1, eta0_1])
                e2_ = np.array([-eta0_2, eta0_2])
                gw1 = nc_gateway1_obs(beta, mu, M, eps1_new, V1_new, e1_, W1_)
                gw2 = nc_gateway2_obs(beta, mu, M, eps2_new, V2_new, t_g,
                                      e2_, W2_, t_h_new)
                res = []
                for a in range(M):
                    res.append(gw1[f'n_g1_{a}'] - n_g1[a])
                    res.append(gw1[f'd_g1_{a}'] - d_g1[a])
                    res.append(gw2[f'n_g2_{a}'] - n_g2[a])
                    res.append(gw2[f'd_g2_{a}'] - d_g2[a])
                for a in range(M):
                    W_sq = (1 - z) * W1_[a]**2 + z * W2_[a]**2
                    res.append(W_sq - W_[a]**2)
                return res

            b_eta0 = 2.0
            eta0_1_seed = np.clip(abs(eta1[1]) if M > 1 else 0.1, 0.05, b_eta0)
            eta0_2_seed = np.clip(abs(eta2[1]) if M > 1 else 0.1, 0.05, b_eta0)
            p0 = np.array([eta0_1_seed, W1[0], W1[1],
                           eta0_2_seed, W2[0], W2[1], W[0], W[1]])
            blo = [0.05, -b_W, -b_W, 0.05, -b_W, -b_W, -b_W, -b_W]
            bhi = [b_eta0, b_W, b_W, b_eta0, b_W, b_W, b_W, b_W]
            sol = least_squares(r4, p0, bounds=(blo, bhi), **lsq)
            eta0_1_new = sol.x[0]; W1_new = sol.x[1:3]
            eta0_2_new = sol.x[3]; W2_new = sol.x[4:6]; W_new = sol.x[6:8]
            eta1_new = np.array([-eta0_1_new, eta0_1_new])
            eta2_new = np.array([-eta0_2_new, eta0_2_new])
            eta_new, _ = solve_moments(M, z, eta1_new, W1_new, eta2_new, W2_new)

        # Step 4b: Fit t_g from g2hop
        def r4d(p):
            gw2 = nc_gateway2_obs(beta, mu, M, eps2_new, V2_new, p[:M],
                                  eta2_new, W2_new, t_h_new)
            return [gw2[f'g2hop_{a}'] - g2hop[a] for a in range(M)]

        sol = least_squares(r4d, t_g, **lsq)
        t_g_new = sol.x[:M]

        # Convergence & mixing
        x_new = np.concatenate([W_new, W1_new, W2_new, t_h_new, t_g_new])
        x_old = np.concatenate([W, W1, W2, t_h, t_g])
        dp = float(np.linalg.norm(x_new - x_old))

        W    = mix * W_new    + (1 - mix) * W
        W1   = mix * W1_new   + (1 - mix) * W1
        W2   = mix * W2_new   + (1 - mix) * W2
        eta  = mix * eta_new  + (1 - mix) * eta
        eta1 = mix * eta1_new + (1 - mix) * eta1
        eta2 = mix * eta2_new + (1 - mix) * eta2
        t_h = t_h_new; t_g = t_g_new
        eps1 = eps1_new; V1 = V1_new
        eps2 = eps2_new; V2 = V2_new

        if verbose:
            print(f'  it={it:3d}  dp={dp:.2e}  docc={docc:.8f}  '
                  f'docc1={imp1["docc"]:.6f}  docc2={imp2["docc"]:.6f}')
        if dp < tol:
            break

    return dict(T=T, iters=it, dp=dp, docc=docc,
                docc1=imp1['docc'], docc2=imp2['docc'],
                x=np.concatenate([eta, W, eta2, W2, t_h, eta1, W1, t_g]))


# ═══════════════════════════════════════════════════════════
# Temperature sweep
# ═══════════════════════════════════════════════════════════

def run_sweep(Uval=1.3, z=4.0, M=1, nquad=200, nk=20,
              nT=20, T_max=5.0, T_min=0.1,
              mix=0.4, tol=1e-9, maxiter=300, verbose=False):
    """Temperature sweep with warm-starting."""
    Ts = np.logspace(np.log10(T_max), np.log10(T_min), nT)

    if M == 1:
        x0 = np.array([0.0, 0.3, 0.0, 0.3, 0.05, 0.0, 0.3, 0.05])
    elif M == 2:
        x0 = np.zeros(16)
        x0[1] = x0[3] = 0.3      # W, W2
        x0[4] = x0[5] = 0.05     # t_h
        x0[6] = x0[7] = 0.3      # W1 (index 6M=6 for M=1... need 5M for M=2)
        # For M=2: [eta(2), W(2), eta2(2), W2(2), t_h(2), eta1(2), W1(2), t_g(2)]
        x0 = np.array([0.0, 0.0, 0.3, 0.3,     # eta, W
                        0.0, 0.0, 0.3, 0.3,     # eta2, W2
                        0.05, 0.05,              # t_h
                        0.0, 0.0, 0.3, 0.3,     # eta1, W1
                        0.05, 0.05])             # t_g
    else:
        x0 = np.zeros(8 * M)
        x0[M:2*M] = 0.3     # W
        x0[3*M:4*M] = 0.3   # W2
        x0[4*M:5*M] = 0.05  # t_h
        x0[6*M:7*M] = 0.3   # W1
        x0[7*M:8*M] = 0.05  # t_g

    print(f'\nGhost Nested Cluster (internal)  M={M}  U={Uval}  z={z}  nk={nquad}')
    print(f'{"T":>8}  {"docc":>10}  {"docc1":>10}  {"docc2":>10}  {"iters":>6}  {"dp":>10}')
    print('-' * 60)

    results = []; xp = None; x2 = None
    for T in Ts:
        if   x2 is not None: xi = np.clip(2 * xp - x2, -5., 5.)
        elif xp is not None: xi = xp.copy()
        else:                xi = x0.copy()

        r = solve_T(T, xi, Uval=Uval, z=z, M=M, nquad=nquad,
                    mix=mix, tol=tol, maxiter=maxiter, verbose=verbose)

        print(f'{T:8.4f}  {r["docc"]:10.8f}  {r["docc1"]:10.6f}  '
              f'{r["docc2"]:10.6f}  {r["iters"]:6d}  {r["dp"]:10.2e}')
        sys.stdout.flush()
        results.append(r); x2 = xp; xp = r['x'].copy()

    return results
