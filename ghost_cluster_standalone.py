#!/usr/bin/env python3
"""
ghost_cluster_standalone.py
Combination of dimer_ghost_dmft_faster.py + ghost_nested_cluster_v2.py
All original code preserved. Bug fix: n_g2* -> n_g2_ in impurity2/gateway2.
"""

import numpy as np
from functools import lru_cache
from scipy.optimize import least_squares
from scipy.linalg import eigh as scipy_eigh
import argparse

INV_SQRT2 = 1.0 / np.sqrt(2.0)


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
        sign = (-1)**bin(state & ((1<<bit)-1)).count('1')
        return state | (1<<bit), sign
    else:
        if not occupied: return -1, 0
        sign = (-1)**bin(state & ((1<<bit)-1)).count('1')
        return state ^ (1<<bit), sign

def build_H(norb, onsite, hoppings, U_terms):
    dim   = 4**norb
    basis = np.arange(dim, dtype=np.int32)
    idx   = make_index(norb)
    H     = np.zeros((dim, dim), dtype=float)
    diag  = np.zeros(dim, dtype=float)
    for site, eps in onsite:
        nu = (basis >> (2*site))   & 1
        nd = (basis >> (2*site+1)) & 1
        diag += eps * (nu + nd)
    for site, U in U_terms:
        nu = (basis >> (2*site))   & 1
        nd = (basis >> (2*site+1)) & 1
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
                np.array([bin(int(s) & ((1<<bit_j)-1)).count('1') % 2
                          for s in states]), -1.0, 1.0)
            s1   = states ^ (1 << bit_j)
            sgn2 = np.where(
                np.array([bin(int(s) & ((1<<bit_i)-1)).count('1') % 2
                          for s in s1]), -1.0, 1.0)
            s2   = s1 | (1 << bit_i)
            H[idx[s2], active] += amp * sgn1 * sgn2
            H[active, idx[s2]] += amp * sgn1 * sgn2
    return H

@lru_cache(maxsize=None)
def _get_NSz_blocks(norb):
    dim   = 4**norb
    basis = np.arange(dim, dtype=np.int32)
    N_up  = np.zeros(dim, dtype=np.int32)
    N_dn  = np.zeros(dim, dtype=np.int32)
    for site in range(norb):
        N_up += (basis >> (2*site))   & 1
        N_dn += (basis >> (2*site+1)) & 1
    sectors = {}
    N_arr   = N_up + N_dn
    Sz2_arr = N_up - N_dn
    for N in range(2*norb+1):
        for Sz2 in range(-N, N+1, 2):
            mask = np.where((N_arr==N)&(Sz2_arr==Sz2))[0]
            if len(mask): sectors[(N,Sz2)] = mask
    return sectors

def thermal_obs(H, beta, diag_ops, off_ops):
    dim  = H.shape[0]
    norb = int(round(np.log(dim) / np.log(4)))
    sectors = _get_NSz_blocks(norb)
    all_E = []; all_V = []
    for mask in sectors.values():
        Hblk = H[np.ix_(mask, mask)]
        try:    e, v = np.linalg.eigh(Hblk)
        except: e, v = scipy_eigh(Hblk)
        all_E.append(e); all_V.append((mask, v))
    E_flat = np.concatenate(all_E) - min(e.min() for e in all_E)
    bw = np.exp(-beta*E_flat); p = bw/bw.sum()
    result = {}
    for name, diag in diag_ops.items():
        val=0.; ptr=0
        for (mask,v) in all_V:
            n=len(mask)
            Oeig=np.einsum('in,i,in->n',v,diag[mask],v)
            val+=np.dot(p[ptr:ptr+n],Oeig); ptr+=n
        result[name]=float(val)
    for name, op in off_ops.items():
        val=0.; ptr=0
        for (mask,v) in all_V:
            n=len(mask)
            Oeig=np.diag(v.T@op[np.ix_(mask,mask)]@v)
            val+=np.dot(p[ptr:ptr+n],Oeig.real); ptr+=n
        result[name]=float(val)
    return result

@lru_cache(maxsize=None)
def _occ_op_cached(norb, site):
    dim = 4**norb; basis = np.arange(dim)
    return (((basis >> (2*site)) & 1) +
            ((basis >> (2*site+1)) & 1)).astype(float)

@lru_cache(maxsize=None)
def _docc_op_cached(norb, site):
    dim = 4**norb; basis = np.arange(dim)
    return (((basis >> (2*site)) & 1) *
            ((basis >> (2*site+1)) & 1)).astype(float)

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

def occ_op(norb, site):       return _occ_op_cached(norb, site)
def docc_op(norb, site):      return _docc_op_cached(norb, site)
def cdag_c_op(norb, si, sj): return _cdag_c_op_cached(norb, si, sj)


# =============================================================================
# Nested Cluster: Lattice
# =============================================================================

def lattice_obs(beta, mu, M, eta, W, eta2, W2, t_h, nk=20, t=0.5):
    Sigma_inf = mu
    k = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    kx_2d, ky_2d = np.meshgrid(k, k, indexing='ij')
    kx_arr = kx_2d.ravel()
    eps_k  = (-2.*t*(np.cos(kx_2d)+np.cos(ky_2d))).ravel()
    wk     = np.ones(len(eps_k)) / len(eps_k)

    sz = 2 + 4*M
    dA = 0; dB = 1
    hA  = list(range(2,     2+M))
    hB  = list(range(2+M,   2+2*M))
    h2A = list(range(2+2*M, 2+3*M))
    h2B = list(range(2+3*M, 2+4*M))

    n_h   = np.zeros(M); d_h   = np.zeros(M)
    n_h2  = np.zeros(M); d_h2  = np.zeros(M)
    h2hop = np.zeros(M)

    for ek, kx, w in zip(eps_k, kx_arr, wk):
        H = np.zeros((sz, sz))
        H[dA, dA] = ek + Sigma_inf - mu
        H[dB, dB] = 0. + Sigma_inf - mu
        H[dA, dB] = H[dB, dA] = -t
        for a in range(M):
            H[hA[a], hA[a]] = eta[a];  H[hB[a], hB[a]] = eta[a]
            H[dA,    hA[a]] = H[hA[a], dA] = W[a]
            H[dB,    hB[a]] = H[hB[a], dB] = W[a]
            H[h2A[a], h2A[a]] = eta2[a]; H[h2B[a], h2B[a]] = eta2[a]
            H[dA,     h2A[a]] = H[h2A[a], dA] = W2[a]
            H[dB,     h2B[a]] = H[h2B[a], dB] = W2[a]
            H[h2A[a], h2B[a]] = H[h2B[a], h2A[a]] = -t_h[a]

        e, U = np.linalg.eigh(H)
        f   = 1.0/(np.exp(beta*e)+1.0)
        rho = (U*f) @ U.T

        for a in range(M):
            n_h[a]   += w * rho[hA[a],  hA[a] ]
            d_h[a]   += w * rho[dA,     hA[a] ]
            n_h2[a]  += w * rho[h2A[a], h2A[a]]
            d_h2[a]  += w * rho[dA,     h2A[a]]
            h2hop[a] += w * rho[h2A[a], h2B[a]]

    return dict(n_h=n_h, d_h=d_h, n_h2=n_h2, d_h2=d_h2, h2hop=h2hop)


def gateway1_obs(beta, mu, M, eps1, V1, eta1, W1):
    sz = 1 + 2*M
    d  = 0
    g1 = list(range(1,   1+M))
    h1 = list(range(1+M, 1+2*M))
    H  = np.zeros((sz, sz))
    for a in range(M):
        H[g1[a], g1[a]] = eps1[a]
        H[d,     g1[a]] = V1[a];  H[g1[a], d]     = V1[a]
        H[h1[a], h1[a]] = eta1[a]
        H[d,     h1[a]] = W1[a];  H[h1[a], d]     = W1[a]
    e, U = np.linalg.eigh(H)
    f   = 1.0/(np.exp(beta*e)+1.0)
    rho = (U*f) @ U.T
    res = {}
    for a in range(M):
        res[f'n_g1_{a}'] = rho[g1[a], g1[a]]
        res[f'd_g1_{a}'] = rho[d,     g1[a]]
        res[f'n_h1_{a}'] = rho[h1[a], h1[a]]
        res[f'd_h1_{a}'] = rho[d,     h1[a]]
    return res


def gateway2_obs(beta, mu, M, eps2, V2, t_g, eta2, W2, t_h, t=0.5):
    sz  = 2 + 4*M
    dA=0; dB=1
    g2A = list(range(2,     2+M))
    g2B = list(range(2+M,   2+2*M))
    h2A = list(range(2+2*M, 2+3*M))
    h2B = list(range(2+3*M, 2+4*M))
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
    f = 1.0/(np.exp(beta*e)+1.0)
    rho = (U*f) @ U.T
    res = {}
    for a in range(M):
        res[f'n_g2_{a}']  = rho[g2A[a], g2A[a]]
        res[f'd_g2_{a}']  = rho[dA,     g2A[a]]
        res[f'g2hop_{a}'] = rho[g2A[a], g2B[a]]
        res[f'n_h2_{a}']  = rho[h2A[a], h2A[a]]
        res[f'd_h2_{a}']  = rho[dA,     h2A[a]]
        res[f'h2hop_{a}'] = rho[h2A[a], h2B[a]]
    return res


def impurity1_obs(beta, mu, U, M, eps1, V1):
    norb = 1 + M
    d    = 0
    g1   = list(range(1, 1+M))
    onsite   = [(d, -mu)] + [(g1[a], eps1[a]) for a in range(M)]
    hoppings = [(d, g1[a], V1[a]) for a in range(M)]
    H = build_H(norb, onsite, hoppings, [(d, U)])
    diag_ops = {'docc': docc_op(norb, d)}
    diag_ops.update({f'n_g1_{a}': occ_op(norb, g1[a]) for a in range(M)})
    off_ops  = {f'd_g1_{a}': cdag_c_op(norb, d, g1[a]) for a in range(M)}
    res = thermal_obs(H, beta, diag_ops, off_ops)
    res['n_g1'] = np.array([res[f'n_g1_{a}']/2. for a in range(M)])
    res['d_g1'] = np.array([res[f'd_g1_{a}']/2. for a in range(M)])
    return res


def impurity2_obs(beta, mu, U, M, eps2, V2, t_g, t=0.5):
    norb = 2 + 2*M
    dA=0; dB=1
    g2A = list(range(2,   2+M))
    g2B = list(range(2+M, 2+2*M))
    onsite   = [(dA,-mu),(dB,-mu)]
    onsite  += [(g2A[a],eps2[a]) for a in range(M)]
    onsite  += [(g2B[a],eps2[a]) for a in range(M)]
    hoppings  = [(dA, dB, -t)]
    hoppings += [(dA, g2A[a], V2[a]) for a in range(M)]
    hoppings += [(dB, g2B[a], V2[a]) for a in range(M)]
    hoppings += [(g2A[a], g2B[a], -t_g[a]) for a in range(M)]
    H = build_H(norb, onsite, hoppings, [(dA,U),(dB,U)])
    diag_ops = {'docc': docc_op(norb, dA)}
    diag_ops.update({f'n_g2_{a}': occ_op(norb, g2A[a]) for a in range(M)})
    off_ops  = {f'd_g2_{a}': cdag_c_op(norb, dA, g2A[a]) for a in range(M)}
    off_ops.update({f'g2hop_{a}': cdag_c_op(norb, g2A[a], g2B[a]) for a in range(M)})
    res = thermal_obs(H, beta, diag_ops, off_ops)
    return dict(docc=res['docc'],
                n_g2=np.array([res[f'n_g2_{a}']/2. for a in range(M)]),
                d_g2=np.array([res[f'd_g2_{a}']/2. for a in range(M)]),
                g2hop=np.array([res[f'g2hop_{a}']/2. for a in range(M)]))


def solve_moments(M, z, eta1, W1, eta2, W2):
    moments = np.zeros(2*M)
    for k in range(2*M):
        moments[k] = (1-z)*np.sum(W1**2*eta1**k) + z*np.sum(W2**2*eta2**k)
    if M == 1:
        W_sq    = max(moments[0], 0.0)
        W_new   = np.array([np.sqrt(W_sq)])
        eta_new = np.array([0.0])
        return eta_new, W_new
    H_mat   = np.array([[moments[i+j]   for j in range(M)] for i in range(M)])
    H_shift = np.array([[moments[i+j+1] for j in range(M)] for i in range(M)])
    try:
        eta_new = np.sort(np.linalg.eigvals(np.linalg.solve(H_mat,H_shift)).real)
    except:
        eta_new = np.zeros(M)
    V_mat = np.vander(eta_new, M, increasing=True).T
    try:
        W_new = np.sqrt(np.abs(np.linalg.solve(V_mat, moments[:M])))
    except:
        W_new = np.ones(M)*0.01
    return eta_new, W_new


# =============================================================================
# Nested Cluster: Self-consistency loop
# =============================================================================

def solve_T(T, x0, Uval=1.3, z=4.0, D=1.0, M=1, nquad=200,
            mix=0.4, tol=1e-9, maxiter=300, verbose=False):
    beta = 1.0/T
    mu   = Uval/2.0
    lsq  = dict(method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
    PENALTY = 1e4
    b_V = 100.0; b_W = 100.0

    x    = np.array(x0, dtype=float)
    eta  = x[0*M:1*M].copy();  W    = x[1*M:2*M].copy()
    eta2 = x[2*M:3*M].copy();  W2   = x[3*M:4*M].copy()
    t_h  = x[4*M:5*M].copy()
    eta1 = x[5*M:6*M].copy();  W1   = x[6*M:7*M].copy()
    t_g  = x[7*M:8*M].copy()

    eps1 = np.full(M,-0.01); V1 = np.full(M, 0.01)
    eps2 = np.full(M,-0.01); V2 = np.full(M, 0.01)
    docc = 0.25

    for it in range(1, maxiter+1):
        lat   = lattice_obs(beta, mu, M, eta, W, eta2, W2, t_h, nk=nquad, t=0.5)
        n_h   = lat['n_h'];   d_h   = lat['d_h']
        n_h2  = lat['n_h2'];  d_h2  = lat['d_h2']
        h2hop = lat['h2hop']

        # Step 2a: fit V1, V2
        def r2a(p):
            V1_=p[0:M]; V2_=p[M:2*M]
            gw1 = gateway1_obs(beta, mu, M, eps1, V1_, eta1, W1)
            gw2 = gateway2_obs(beta, mu, M, eps2, V2_, t_g, eta2, W2, t_h)
            res = []
            for a in range(M):
                nh1=gw1[f'n_h1_{a}']; nh2=gw2[f'n_h2_{a}']
                dh1=gw1[f'd_h1_{a}']; dh2=gw2[f'd_h2_{a}']
                lhs_hh = (1-z)*nh1 + z*nh2
                res.append(lhs_hh - n_h[a])
                res.append((1-z)*dh1 + z*dh2 - d_h[a])
                res.append(nh2 - n_h2[a])
                res.append(dh2 - d_h2[a])
                if lhs_hh < 0:
                    res[0] += PENALTY * lhs_hh
            return res
        sol = least_squares(r2a, np.concatenate([V1, V2]),
                            bounds=([-b_V]*2*M, [b_V]*2*M), **lsq)
        V1_new = sol.x[0:M]; V2_new = sol.x[M:2*M]
        eps1_new = eps1.copy(); eps2_new = eps2.copy()

        # Step 2b: fit t_h
        def r2b(p):
            gw2 = gateway2_obs(beta, mu, M, eps2_new, V2_new, t_g,
                               eta2, W2, p[:M])
            return [gw2[f'h2hop_{a}'] - h2hop[a] for a in range(M)]
        sol     = least_squares(r2b, t_h, **lsq)
        t_h_new = sol.x[:M]

        # Step 3: impurity targets
        imp1  = impurity1_obs(beta, mu, Uval, M, eps1_new, V1_new)
        imp2  = impurity2_obs(beta, mu, Uval, M, eps2_new, V2_new, t_g)
        docc  = imp2['docc']
        n_g1  = imp1['n_g1']; d_g1 = imp1['d_g1']
        n_g2  = imp2['n_g2']; d_g2 = imp2['d_g2']
        g2hop = imp2['g2hop']

        # Step 4: fit eta1, W1, eta2, W2, W
        if M == 1:
            def r4(p):
                W1_ = p[0:1]; W2_ = p[1:2]; W_ = p[2:3]
                gw1 = gateway1_obs(beta, mu, M, eps1_new, V1_new, np.zeros(M), W1_)
                gw2 = gateway2_obs(beta, mu, M, eps2_new, V2_new, t_g,
                                   np.zeros(M), W2_, t_h_new)
                res = [gw1['n_g1_0']-n_g1[0], gw1['d_g1_0']-d_g1[0],
                       gw2['n_g2_0']-n_g2[0], gw2['d_g2_0']-d_g2[0]]
                W_sq = (1-z)*W1_[0]**2 + z*W2_[0]**2
                res.append(W_sq - W_[0]**2)
                return res
            sol = least_squares(r4, np.concatenate([W1,W2,W]),
                                bounds=([-b_W]*3,[b_W]*3), **lsq)
            W1_new = sol.x[0:1]; W2_new = sol.x[1:2]; W_new = sol.x[2:3]
            eta1_new = np.zeros(M); eta2_new = np.zeros(M); eta_new = np.zeros(M)
        else:
            def r4(p):
                eta0_1=p[0]; W1_=p[1:3]
                eta0_2=p[3]; W2_=p[4:6]; W_=p[6:8]
                e1_ = np.array([-eta0_1, eta0_1])
                e2_ = np.array([-eta0_2, eta0_2])
                gw1 = gateway1_obs(beta, mu, M, eps1_new, V1_new, e1_, W1_)
                gw2 = gateway2_obs(beta, mu, M, eps2_new, V2_new, t_g,
                                   e2_, W2_, t_h_new)
                res = []
                for a in range(M):
                    res.append(gw1[f'n_g1_{a}']-n_g1[a])
                    res.append(gw1[f'd_g1_{a}']-d_g1[a])
                    res.append(gw2[f'n_g2_{a}']-n_g2[a])
                    res.append(gw2[f'd_g2_{a}']-d_g2[a])
                for a in range(M):
                    W_sq = (1-z)*W1_[a]**2 + z*W2_[a]**2
                    res.append(W_sq - W_[a]**2)
                return res
            eta0_1_seed = np.clip(abs(eta1[1]), 0.05, 2.0)
            eta0_2_seed = np.clip(abs(eta2[1]), 0.05, 2.0)
            p0 = np.array([eta0_1_seed, W1[0], W1[1],
                           eta0_2_seed, W2[0], W2[1],
                           W[0], W[1]])
            b_eta0 = 2.0
            blo = [0.05,-b_W,-b_W, 0.05,-b_W,-b_W, -b_W,-b_W]
            bhi = [b_eta0,b_W,b_W, b_eta0,b_W,b_W,  b_W, b_W]
            sol = least_squares(r4, p0, bounds=(blo,bhi), **lsq)
            eta0_1_new = sol.x[0]; W1_new = sol.x[1:3]
            eta0_2_new = sol.x[3]; W2_new = sol.x[4:6]; W_new = sol.x[6:8]
            eta1_new = np.array([-eta0_1_new, eta0_1_new])
            eta2_new = np.array([-eta0_2_new, eta0_2_new])
            eta_new,_ = solve_moments(M, z, eta1_new, W1_new, eta2_new, W2_new)

        # Step 4d: fit t_g
        def r4d(p):
            gw2 = gateway2_obs(beta, mu, M, eps2_new, V2_new, p[:M],
                               eta2_new, W2_new, t_h_new)
            return [gw2[f'g2hop_{a}']-g2hop[a] for a in range(M)]
        sol     = least_squares(r4d, t_g, **lsq)
        t_g_new = sol.x[:M]

        # Convergence & mixing
        x_new = np.concatenate([W_new, W1_new, W2_new, t_h_new, t_g_new])
        x_old = np.concatenate([W,     W1,     W2,     t_h,     t_g    ])
        dp    = float(np.linalg.norm(x_new - x_old))

        W  = mix*W_new  + (1-mix)*W
        W1 = mix*W1_new + (1-mix)*W1
        W2 = mix*W2_new + (1-mix)*W2
        eta  = mix*eta_new  + (1-mix)*eta
        eta1 = mix*eta1_new + (1-mix)*eta1
        eta2 = mix*eta2_new + (1-mix)*eta2
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
                x=np.concatenate([eta,W,eta2,W2,t_h,eta1,W1,t_g]))


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--M',       type=int,   default=1)
    parser.add_argument('--U',       type=float, default=1.3)
    parser.add_argument('--z',       type=float, default=4.0)
    parser.add_argument('--T',       type=float, default=1.0)
    parser.add_argument('--nquad',   type=int,   default=50)
    parser.add_argument('--mix',     type=float, default=0.1)
    parser.add_argument('--tol',     type=float, default=1e-9)
    parser.add_argument('--maxiter', type=int,   default=5000)
    parser.add_argument('--nT',      type=int,   default=100)
    parser.add_argument('--T_max',   type=float, default=2.0)
    parser.add_argument('--T_min',   type=float, default=0.05)
    parser.add_argument('--sweep',   action="store_true")
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()
    M = args.M; U = args.U; z = args.z

    if M == 1:
        x0 = np.array([0.0,0.3,  0.0,0.3, 0.05,  0.0,0.3, 0.05])
    else:
        x0 = np.array([0.0,0.0,0.2,0.2, 0.0,0.0,0.2,0.2,
                        0.05,0.05, 0.0,0.0,0.2,0.2, 0.05,0.05])

    if args.sweep:
        T_vals = np.logspace(np.log10(args.T_max), np.log10(args.T_min), args.nT)
        print()
        print("Ghost Nested Cluster  M=%d  U=%.2f  z=%.1f  nquad=%d  nT=%d  mix=%.2f  maxiter=%d"
              % (M,U,z,args.nquad,args.nT,args.mix,args.maxiter))
        print("%8s  %10s  %10s  %6s  %8s" % ("T","docc1","docc2","iters","dp"))
        print("-"*52)
        xp=None; x2=None
        for T in T_vals:
            if   x2 is not None: xi = np.clip(2*xp-x2, -10., 10.)
            elif xp is not None: xi = xp.copy()
            else:                xi = x0.copy()
            r = solve_T(T, xi, Uval=U, z=z, M=M, nquad=args.nquad,
                        mix=args.mix, tol=args.tol, maxiter=args.maxiter,
                        verbose=args.verbose)
            print("%8.4f  %10.6f  %10.6f  %6d  %8.2e"
                  % (T,r["docc1"],r["docc2"],r["iters"],r["dp"]))
            sys.stdout.flush()
            x2=xp; xp=r["x"].copy()
    else:
        r = solve_T(args.T, x0, Uval=U, z=z, M=M, nquad=args.nquad,
                    mix=args.mix, tol=args.tol, maxiter=args.maxiter,
                    verbose=True)
        print("docc=%.8f  docc1=%.8f  docc2=%.8f  iters=%d  dp=%.2e"
              % (r["docc"],r["docc1"],r["docc2"],r["iters"],r["dp"]))
