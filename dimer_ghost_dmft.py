#!/usr/bin/env python3
"""
dimer_ghost_dmft.py
====================
Dimer ghost-DMFT on the 2D square lattice.  Arbitrary M ghost families.
Optional inter-site ghost hopping (--hop).  No bond ghosts.
Standalone: numpy + scipy only.

Notation
--------
Dimer unit cell has two physical sites A and B.
Ghost orbitals carry TWO independent labels:

  Superscript = SITE:   g^(A)_m  or  g^(B)_m   (which dimer site)
  Subscript   = FAMILY: m = 1 .. M              (which ghost channel)

By the A<->B symmetry of the dimer at half-filling the parameters are
site-independent:
  eps_g[m]  -- on-site energy of g^(A)_m and g^(B)_m
  V_g[m]    -- hopping  d^(A) <-> g^(A)_m  (and d^(B) <-> g^(B)_m)
  t_g[m]    -- hopping  g^(A)_m <-> g^(B)_m   (--hop only)
  eta_h[m]  -- on-site energy of h^(A)_m and h^(B)_m
  W_h[m]    -- hopping  d^(A) <-> h^(A)_m  (and d^(B) <-> h^(B)_m)
  t_h[m]    -- hopping  h^(A)_m <-> h^(B)_m   (--hop only)

Matching conditions (sequential scheme, per iteration)
-------------------------------------------------------
Step 1  Lattice -> h-targets (k-sum, A-site):
          n_hA[m]  = < h^(A)+ _m  h^(A)_m >_lat
          d_hA[m]  = < d^(A)+     h^(A)_m >_lat
          hhop[m]  = < h^(A)+ _m  h^(B)_m >_lat    (if --hop)

Step 2  Gateway fit -- h-sector: gateway = lattice
        Fit (eps_g[m], V_g[m]) [and t_h[m] if --hop] so that:
          < h^(A)+ _m  h^(A)_m >_gw  =  n_hA[m]
          < d^(A)+     h^(A)_m >_gw  =  d_hA[m]
          < h^(A)+ _m  h^(B)_m >_gw  =  hhop[m]    (if --hop)

Step 3  Impurity -> g-targets (ED, A-site):
          n_g[m]   = < g^(A)+ _m  g^(A)_m >_imp
          d_g[m]   = < d^(A)+     g^(A)_m >_imp
          ghop[m]  = < g^(A)+ _m  g^(B)_m >_imp    (if --hop)

Step 4  Gateway fit -- g-sector: gateway = impurity
        Fit (eta_h[m], W_h[m]) [and t_g[m] if --hop] so that:
          < g^(A)+ _m  g^(A)_m >_gw  =  n_g[m]
          < d^(A)+     g^(A)_m >_gw  =  d_g[m]
          < g^(A)+ _m  g^(B)_m >_gw  =  ghop[m]    (if --hop)

Orbital bases
-------------
  H_imp:  d^(A)=0, d^(B)=1,
          g^(A)_1..g^(A)_M = 2..M+1,
          g^(B)_1..g^(B)_M = M+2..2M+1        NORB = 2+2M

  H_gw:   d^(A)=0, d^(B)=1,
          g^(A)_1..g^(A)_M = 2..M+1,
          g^(B)_1..g^(B)_M = M+2..2M+1,
          h^(A)_1..h^(A)_M = 2M+2..3M+1,
          h^(B)_1..h^(B)_M = 3M+2..4M+1      size = 2+4M

  H_lat:  d^(A)=0, d^(B)=1,
          h^(A)_1..h^(A)_M = 2..M+1,
          h^(B)_1..h^(B)_M = M+2..2M+1       size = 2+2M

x0 / x_out layout (length 6M):
  eta_h[0..M-1], W_h[0..M-1], t_h[0..M-1], t_g[0..M-1],
  eps_g[0..M-1], V_g[0..M-1]
"""

import numpy as np
from functools import lru_cache
from scipy.optimize import least_squares
from scipy.linalg import eigh as scipy_eigh
import argparse


# =============================================================================
# 1.  Fermi function
# =============================================================================

def fermi(x, beta):
    x = np.asarray(x, dtype=float)
    y = beta * x
    out = np.empty_like(y)
    out[y >  60] = 0.0
    out[y < -60] = 1.0
    m = (y >= -60) & (y <= 60)
    out[m] = 1.0 / (np.exp(y[m]) + 1.0)
    return out


# =============================================================================
# 2.  Generic Fock-space engine
# =============================================================================

def make_index(norb):
    return np.arange(4**norb, dtype=np.int32)


def c_action(state, site, spin, dag):
    bit = 2*site + spin
    occupied = (state >> bit) & 1
    if dag:
        if occupied: return -1, 0
        sign = (-1)**bin(state & ((1 << bit) - 1)).count('1')
        return state | (1 << bit), sign
    else:
        if not occupied: return -1, 0
        sign = (-1)**bin(state & ((1 << bit) - 1)).count('1')
        return state ^ (1 << bit), sign


def build_H(norb, onsite, hoppings, U_terms):
    dim   = 4**norb
    basis = np.arange(dim, dtype=np.int32)
    idx   = make_index(norb)
    H     = np.zeros((dim, dim), dtype=float)
    diag  = np.zeros(dim, dtype=float)
    for site, eps in onsite:
        nu = (basis >> (2*site))     & 1
        nd = (basis >> (2*site + 1)) & 1
        diag += eps * (nu + nd)
    for site, U in U_terms:
        nu = (basis >> (2*site))     & 1
        nd = (basis >> (2*site + 1)) & 1
        diag += U * nu * nd
    np.fill_diagonal(H, diag)
    for (si, sj, amp) in hoppings:
        if abs(amp) < 1e-14:
            continue
        for spin in range(2):
            bit_j = 2*sj + spin
            bit_i = 2*si + spin
            occ_j  = (basis >> bit_j) & 1
            emp_i  = 1 - ((basis >> bit_i) & 1)
            active = np.where(occ_j & emp_i)[0]
            if len(active) == 0:
                continue
            states = basis[active]
            sgn1 = np.where(
                np.array([bin(int(s) & ((1 << bit_j) - 1)).count('1') % 2
                          for s in states]), -1.0, 1.0)
            s1   = states ^ (1 << bit_j)
            sgn2 = np.where(
                np.array([bin(int(s) & ((1 << bit_i) - 1)).count('1') % 2
                          for s in s1]), -1.0, 1.0)
            s2   = s1 | (1 << bit_i)
            H[idx[s2], active] += amp * sgn1 * sgn2
            H[active, idx[s2]] += amp * sgn1 * sgn2
    return H


@lru_cache(maxsize=None)
def _get_NSz_blocks(norb):
    dim     = 4**norb
    basis   = np.arange(dim, dtype=np.int32)
    N_up    = np.zeros(dim, dtype=np.int32)
    N_dn    = np.zeros(dim, dtype=np.int32)
    for site in range(norb):
        N_up += (basis >> (2*site))     & 1
        N_dn += (basis >> (2*site + 1)) & 1
    sectors = {}
    N_arr   = N_up + N_dn
    Sz2_arr = N_up - N_dn
    for N in range(2*norb + 1):
        for Sz2 in range(-N, N + 1, 2):
            mask = np.where((N_arr == N) & (Sz2_arr == Sz2))[0]
            if len(mask):
                sectors[(N, Sz2)] = mask
    return sectors


def thermal_obs(H, beta, diag_ops, off_ops):
    dim    = H.shape[0]
    norb   = int(round(np.log(dim) / np.log(4)))
    sectors = _get_NSz_blocks(norb)
    all_E = []; all_V = []
    for mask in sectors.values():
        Hblk = H[np.ix_(mask, mask)]
        try:    e, v = np.linalg.eigh(Hblk)
        except: e, v = scipy_eigh(Hblk)
        all_E.append(e); all_V.append((mask, v))
    E_flat = np.concatenate(all_E) - min(e.min() for e in all_E)
    bw = np.exp(-beta * E_flat); p = bw / bw.sum()
    result = {}
    for name, diag in diag_ops.items():
        val = 0.; ptr = 0
        for (mask, v) in all_V:
            n = len(mask)
            Oeig = np.einsum('in,i,in->n', v, diag[mask], v)
            val += np.dot(p[ptr:ptr+n], Oeig); ptr += n
        result[name] = float(val)
    for name, op in off_ops.items():
        val = 0.; ptr = 0
        for (mask, v) in all_V:
            n = len(mask)
            Oeig = np.diag(v.T @ op[np.ix_(mask, mask)] @ v)
            val += np.dot(p[ptr:ptr+n], Oeig.real); ptr += n
        result[name] = float(val)
    return result


@lru_cache(maxsize=None)
def _occ_op_cached(norb, site):
    dim = 4**norb; basis = np.arange(dim)
    return (((basis >> (2*site)) & 1) +
            ((basis >> (2*site + 1)) & 1)).astype(float)

@lru_cache(maxsize=None)
def _docc_op_cached(norb, site):
    dim = 4**norb; basis = np.arange(dim)
    return (((basis >> (2*site)) & 1) *
            ((basis >> (2*site + 1)) & 1)).astype(float)

@lru_cache(maxsize=None)
def _cdag_c_op_cached(norb, si, sj):
    dim = 4**norb; basis = np.arange(dim, dtype=np.int32)
    idx = make_index(norb); op = np.zeros((dim, dim), dtype=float)
    for m, state in enumerate(basis):
        for spin in range(2):
            s1, sg1 = c_action(state, sj, spin, dag=False)
            if s1 < 0: continue
            s2, sg2 = c_action(s1, si, spin, dag=True)
            if s2 < 0: continue
            op[idx[s2], m] += sg1 * sg2
    return op

def occ_op(norb, site):        return _occ_op_cached(norb, site)
def docc_op(norb, site):       return _docc_op_cached(norb, site)
def cdag_c_op(norb, si, sj):  return _cdag_c_op_cached(norb, si, sj)


# =============================================================================
# 3.  Impurity
# =============================================================================

def impurity_obs(beta, mu, U, t_b, M, eps_g, V_g, t_g, hop):
    dA = 0;  dB = 1
    gA = [2 + m     for m in range(M)]
    gB = [2 + M + m for m in range(M)]
    norb = 2 + 2*M

    onsite   = [(dA, -mu), (dB, -mu)]
    hoppings = [(dA, dB, t_b)]
    for m in range(M):
        onsite   += [(gA[m], eps_g[m]), (gB[m], eps_g[m])]
        hoppings += [(dA, gA[m], V_g[m]),
                     (dB, gB[m], V_g[m])]
        if hop:
            hoppings += [(gA[m], gB[m], t_g[m])]

    H = build_H(norb, onsite, hoppings, [(dA, U), (dB, U)])

    diag_ops = {'docc': docc_op(norb, dA)}
    off_ops  = {}
    for m in range(M):
        diag_ops[f'n_gA_{m}'] = occ_op(norb, gA[m])
        off_ops [f'd_gA_{m}'] = cdag_c_op(norb, dA, gA[m])
        if hop:
            off_ops[f'ghop_{m}'] = cdag_c_op(norb, gA[m], gB[m])

    res = thermal_obs(H, beta, diag_ops, off_ops)

    out = {'docc': res['docc']}
    out['n_g']  = np.array([res[f'n_gA_{m}'] / 2.0 for m in range(M)])
    out['d_g']  = np.array([res[f'd_gA_{m}'] / 2.0 for m in range(M)])
    out['ghop'] = np.array([res[f'ghop_{m}'] / 2.0 for m in range(M)]) \
                  if hop else np.zeros(M)
    return out


# =============================================================================
# 4.  Gateway
# =============================================================================

def gateway_obs(beta, mu, t_b, M, eps_g, V_g, t_g, eta_h, W_h, t_h, hop):
    dA = 0;  dB = 1
    gA = [2 + m        for m in range(M)]
    gB = [2 + M + m    for m in range(M)]
    hA = [2 + 2*M + m  for m in range(M)]
    hB = [2 + 3*M + m  for m in range(M)]
    sz = 2 + 4*M

    H1b = np.zeros((sz, sz))
    H1b[dA, dB] = H1b[dB, dA] = -t_b
    for m in range(M):
        H1b[gA[m], gA[m]] = eps_g[m]
        H1b[gB[m], gB[m]] = eps_g[m]
        H1b[dA, gA[m]]    = H1b[gA[m], dA] = V_g[m]
        H1b[dB, gB[m]]    = H1b[gB[m], dB] = V_g[m]

        H1b[hA[m], hA[m]] = eta_h[m]
        H1b[hB[m], hB[m]] = eta_h[m]
        H1b[dA, hA[m]]    = H1b[hA[m], dA] = W_h[m]
        H1b[dB, hB[m]]    = H1b[hB[m], dB] = W_h[m]

        if hop:
            H1b[gA[m], gB[m]] = H1b[gB[m], gA[m]] = -t_g[m]
            H1b[hA[m], hB[m]] = H1b[hB[m], hA[m]] = -t_h[m]

    e, U = np.linalg.eigh(H1b)
    rho  = (U * fermi(e, beta)[None, :]) @ U.T

    out = {}
    out['n_gA'] = np.array([rho[gA[m], gA[m]] for m in range(M)])
    out['d_gA'] = np.array([rho[dA,    gA[m]] for m in range(M)])
    out['ghop'] = np.array([rho[gA[m], gB[m]] for m in range(M)])
    out['n_hA'] = np.array([rho[hA[m], hA[m]] for m in range(M)])
    out['d_hA'] = np.array([rho[dA,    hA[m]] for m in range(M)])
    out['hhop'] = np.array([rho[hA[m], hB[m]] for m in range(M)])
    return out


# =============================================================================
# 5.  Lattice
# =============================================================================

def square_lattice_kgrid(t_d, nk=20):
    k = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    eps_k = (-2.0 * t_d * (np.cos(kx) + np.cos(ky))).ravel()
    wk    = np.ones(eps_k.size) / eps_k.size
    return eps_k, wk


def lattice_obs(beta, mu, t_b, M, eta_h, W_h, t_h, hop, eps_k, wk):
    dA = 0;  dB = 1
    hA = [2 + m     for m in range(M)]
    hB = [2 + M + m for m in range(M)]
    sz = 2 + 2*M;  Nk = len(eps_k)

    H = np.zeros((Nk, sz, sz))
    H[:, dA, dA] = eps_k
    H[:, dB, dB] = 0.0
    H[:, dA, dB] = H[:, dB, dA] = -t_b
    for m in range(M):
        H[:, hA[m], hA[m]] = eta_h[m]
        H[:, hB[m], hB[m]] = eta_h[m]
        H[:, dA, hA[m]] = H[:, hA[m], dA] = W_h[m]
        H[:, dB, hB[m]] = H[:, hB[m], dB] = W_h[m]
        if hop:
            H[:, hA[m], hB[m]] = H[:, hB[m], hA[m]] = -t_h[m]

    e, U = np.linalg.eigh(H)
    y = beta * e
    f = np.where(y > 60, 0.,
        np.where(y < -60, 1., 1.0 / (np.exp(np.clip(y, -60, 60)) + 1.0)))
    rho = np.einsum('kin,kn,kjn->kij', U, f, U)

    n_hA = np.zeros(M);  d_hA = np.zeros(M);  hhop = np.zeros(M)
    for m in range(M):
        n_hA[m] = np.dot(wk, rho[:, hA[m], hA[m]])
        d_hA[m] = np.dot(wk, rho[:, dA,    hA[m]])
        hhop[m] = np.dot(wk, rho[:, hA[m], hB[m]])

    return dict(n_hA=n_hA, d_hA=d_hA, hhop=hhop)


# =============================================================================
# 6.  Self-consistency loop
# =============================================================================

def solve_T(T, x0, Uval=1.3, t_d=0.5, t_b=0.3, mu=None,
            M=1, hop=False,
            nk=20, mix=0.5, tol=1e-8, maxiter=300, verbose=False):
    beta = 1.0 / T
    if mu is None:
        mu = Uval / 2.0

    eps_k, wk = square_lattice_kgrid(t_d, nk=nk)

    x0    = np.array(x0, dtype=float)
    eta_h = x0[0*M:1*M].copy()
    W_h   = x0[1*M:2*M].copy()
    t_h   = x0[2*M:3*M].copy()
    t_g   = x0[3*M:4*M].copy()
    eps_g = x0[4*M:5*M].copy()
    V_g   = x0[5*M:6*M].copy()

    lsq  = dict(method='trf', ftol=1e-11, xtol=1e-11, gtol=1e-11,
                max_nfev=5000, bounds=(-15., 15.))
    docc = 0.25

    for it in range(1, maxiter + 1):

        lat      = lattice_obs(beta, mu, t_b, M, eta_h, W_h, t_h, hop,
                               eps_k, wk)
        n_hA_lat = lat['n_hA']
        d_hA_lat = lat['d_hA']
        hhop_lat = lat['hhop']

        p2_0 = np.concatenate([eps_g, V_g])
        if hop:
            p2_0 = np.concatenate([p2_0, t_h])

        def r2(p):
            eg_ = p[0:M];  Vg_ = p[M:2*M]
            th_ = p[2*M:3*M] if hop else t_h
            gw  = gateway_obs(beta, mu, t_b, M, eg_, Vg_, t_g,
                              eta_h, W_h, th_, hop)
            res = list(gw['n_hA'] - n_hA_lat) + list(gw['d_hA'] - d_hA_lat)
            if hop:
                res += list(gw['hhop'] - hhop_lat)
            return res

        sol2    = least_squares(r2, p2_0, **lsq)
        eps_g_n = sol2.x[0:M].copy()
        V_g_n   = sol2.x[M:2*M].copy()
        t_h_n   = sol2.x[2*M:3*M].copy() if hop else t_h.copy()

        imp      = impurity_obs(beta, mu, Uval, t_b, M,
                                eps_g_n, V_g_n, t_g, hop)
        docc     = imp['docc']
        n_g_imp  = imp['n_g']
        d_g_imp  = imp['d_g']
        ghop_imp = imp['ghop']

        p4_0 = np.concatenate([eta_h, W_h])
        if hop:
            p4_0 = np.concatenate([p4_0, t_g])

        def r4(p):
            eh_ = p[0:M];  Wh_ = p[M:2*M]
            tg_ = p[2*M:3*M] if hop else t_g
            gw  = gateway_obs(beta, mu, t_b, M, eps_g_n, V_g_n, tg_,
                              eh_, Wh_, t_h_n, hop)
            res = list(gw['n_gA'] - n_g_imp) + list(gw['d_gA'] - d_g_imp)
            if hop:
                res += list(gw['ghop'] - ghop_imp)
            return res

        sol4    = least_squares(r4, p4_0, **lsq)
        eta_h_n = sol4.x[0:M].copy()
        W_h_n   = sol4.x[M:2*M].copy()
        t_g_n   = sol4.x[2*M:3*M].copy() if hop else t_g.copy()

        eta_h_n = np.clip(eta_h_n, -10., 10.)
        W_h_n   = np.clip(W_h_n,   -10., 10.)
        eps_g_n = np.clip(eps_g_n, -10., 10.)
        V_g_n   = np.clip(V_g_n,   -10., 10.)

        x_new = np.concatenate([eta_h_n, W_h_n, t_h_n, t_g_n, eps_g_n, V_g_n])
        x_old = np.concatenate([eta_h,   W_h,   t_h,   t_g,   eps_g,   V_g  ])
        dp    = float(np.linalg.norm(x_new - x_old))
        xm    = mix * x_new + (1.0 - mix) * x_old

        eta_h = xm[0*M:1*M]
        W_h   = xm[1*M:2*M]
        t_h   = xm[2*M:3*M]
        t_g   = xm[3*M:4*M]
        eps_g = xm[4*M:5*M]
        V_g   = xm[5*M:6*M]

        if verbose:
            eg_s = ' '.join(f'{v:+.4f}' for v in eps_g)
            Vg_s = ' '.join(f'{v:+.4f}' for v in V_g)
            eh_s = ' '.join(f'{v:+.4f}' for v in eta_h)
            Wh_s = ' '.join(f'{v:+.4f}' for v in W_h)
            print(f'  it={it:3d}  dp={dp:.2e}  D={docc:.6f}'
                  f'  eps_g=[{eg_s}]  V_g=[{Vg_s}]'
                  f'  eta_h=[{eh_s}]  W_h=[{Wh_s}]')

        if dp < tol:
            break

    x_out = np.concatenate([eta_h, W_h, t_h, t_g, eps_g, V_g])
    return dict(T=T, iters=it, dp=dp, docc=docc,
                eps_g=eps_g.copy(), V_g=V_g.copy(),
                eta_h=eta_h.copy(), W_h=W_h.copy(),
                t_g=t_g.copy(), t_h=t_h.copy(),
                x=x_out)


# =============================================================================
# 7.  Temperature sweep
# =============================================================================

def run_sweep(Uval=1.3, t_d=0.5, t_b=0.3, M=1, hop=False,
              nk=20, nT=20, T_max=5., T_min=0.1,
              mix=0.5, tol=1e-8, maxiter=300, verbose=False):
    Ts = np.logspace(np.log10(T_max), np.log10(T_min), nT)

    if M == 1:
        x0 = np.array([ 0.01,  0.20,  0.01,  0.01, -0.01,  0.20])
    elif M == 2:
        x0 = np.array([-0.30,  0.30,
                         0.20,  0.20,
                         0.01,  0.01,
                         0.01,  0.01,
                        -0.30,  0.30,
                         0.20,  0.20])
    else:
        sp = np.linspace(-0.4, 0.4, M)
        x0 = np.concatenate([sp, np.full(M,0.2), np.full(M,0.01),
                              np.full(M,0.01), sp, np.full(M,0.2)])

    mode = 'hop' if hop else 'no-hop'
    print(f'\nDimer ghost-DMFT  M={M}  [{mode}]'
          f'  U={Uval}  t_d={t_d}  t_b={t_b}  nk={nk}')
    print(f'Impurity dim = {4**(2+2*M)}')

    cols  = ['T', 'D']
    cols += [f'eps_g[{m}]' for m in range(M)]
    cols += [f'V_g[{m}]'   for m in range(M)]
    cols += [f'eta_h[{m}]' for m in range(M)]
    cols += [f'W_h[{m}]'   for m in range(M)]
    if hop:
        cols += [f't_g[{m}]' for m in range(M)]
        cols += [f't_h[{m}]' for m in range(M)]
    cols += ['iters', 'dp']
    hdr = '  '.join(f'{c:>10}' for c in cols)
    print(hdr); print('-'*len(hdr))

    results = []; xp = None; x2 = None
    for T in Ts:
        if   x2 is not None: xi = np.clip(2*xp - x2, -5., 5.)
        elif xp is not None: xi = xp.copy()
        else:                xi = x0.copy()

        r = solve_T(T, xi, Uval=Uval, t_d=t_d, t_b=t_b, mu=None,
                    M=M, hop=hop, nk=nk, mix=mix, tol=tol,
                    maxiter=maxiter, verbose=verbose)

        vals  = [T, r['docc']]
        vals += list(r['eps_g'])
        vals += list(r['V_g'])
        vals += list(r['eta_h'])
        vals += list(r['W_h'])
        if hop:
            vals += list(r['t_g'])
            vals += list(r['t_h'])
        row = '  '.join(f'{v:10.5f}' for v in vals)
        print(row + f'  {r["iters"]:8d}  {r["dp"]:9.2e}')
        results.append(r); x2 = xp; xp = r['x'].copy()

    return results


# =============================================================================
# 8.  Sanity check
# =============================================================================

def check_atomic_limit(Uval=1.3):
    beta = 10.; mu = Uval / 2.
    Z       = 1 + 2*np.exp(beta*mu) + np.exp(beta*(2*mu - Uval))
    D_exact = np.exp(beta*(2*mu - Uval)) / Z
    imp = impurity_obs(beta, mu, Uval, 0., 1,
                       np.zeros(1), np.zeros(1), np.zeros(1), False)
    ok  = abs(imp['docc'] - D_exact) < 1e-8
    print(f'Atomic limit (U={Uval}): D_exact={D_exact:.8f}  '
          f'D_imp={imp["docc"]:.8f}  {"PASSED" if ok else "FAILED"}')
    return ok


# =============================================================================
# 9.  CLI
# =============================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Dimer ghost-DMFT')
    p.add_argument('--U',        type=float, default=1.3)
    p.add_argument('--t_d',      type=float, default=0.5)
    p.add_argument('--t_b',      type=float, default=0.3)
    p.add_argument('--M',        type=int,   default=1)
    p.add_argument('--nk',       type=int,   default=20)
    p.add_argument('--nT',       type=int,   default=20)
    p.add_argument('--T_max',    type=float, default=5.0)
    p.add_argument('--T_min',    type=float, default=0.1)
    p.add_argument('--mix',      type=float, default=0.5)
    p.add_argument('--tol',      type=float, default=1e-8)
    p.add_argument('--maxiter',  type=int,   default=300)
    p.add_argument('--hop',      action='store_true')
    p.add_argument('--verbose',  action='store_true')
    p.add_argument('--no_check', action='store_true')
    args = p.parse_args()

    if not args.no_check:
        print('Sanity check:')
        check_atomic_limit(Uval=args.U)
        print()

    run_sweep(Uval=args.U, t_d=args.t_d, t_b=args.t_b,
              M=args.M, hop=args.hop,
              nk=args.nk, nT=args.nT,
              T_max=args.T_max, T_min=args.T_min,
              mix=args.mix, tol=args.tol, maxiter=args.maxiter,
              verbose=args.verbose)
