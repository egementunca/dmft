#!/usr/bin/env python3
"""
Ghost-DMFT: single-site and bond scheme, Mg=Mh=M (M=1 or 2).
Square lattice, half-filling, finite temperature.
No frozen targets -- all parameters solved simultaneously.

Key structure:
  H_lat:  d + M site-local h-ghosts + M bond hb-ghosts
  H_imp:  d + M site-local g-ghosts (interacting)
  H2:     d0,d1 + M site-local g-ghosts per site + M bond gb-ghosts (interacting)
  H_gw:   d + M h-ghosts + M g-ghosts (single-site gateway)
  H2_gw:  d0,d1 + M site-local h-ghosts + M hb-ghosts
               + M site-local g-ghosts + M gb-ghosts (bond gateway)

Mode layout per spin block in H2:
  d0, d1,
  g0_s0, g0_s1, ..., g{M-1}_s0, g{M-1}_s1,   (2M site-local g-ghosts)
  gb0, gb1, ..., gb{M-1}                         (M bond gb-ghosts)
  => nps = 2 + 3M

Mode layout in H2_gw (8x8 for M=1, 14x14 for M=2):
  d0, d1,
  h0_s0, h0_s1, ..., h{M-1}_s0, h{M-1}_s1,   (2M site-local h-ghosts)
  hb0, ..., hb{M-1},                            (M bond hb-ghosts)
  g0_s0, g0_s1, ..., g{M-1}_s0, g{M-1}_s1,   (2M site-local g-ghosts)
  gb0, ..., gb{M-1}                             (M bond gb-ghosts)
  => n_gw = 2 + 6M

Matching conditions (4M+4M+1 = 8M+1 total):
  h-sector (4M): lattice <-> bond gateway
  g-sector (4M): BPK combination of gateways = BPK combination of imp+H2
  half-filling (1): n_site = 0.5

Usage:
  python ghost_dmft_bond.py --M 1 --U 1.3 --mode both
  python ghost_dmft_bond.py --M 2 --U 1.3 --mode both
  python ghost_dmft_bond.py --M 2 --U 1.3 --mode ss
"""

import numpy as np
from numpy.linalg import eigh
from scipy.optimize import least_squares
from itertools import combinations
from functools import lru_cache
from math import comb
import argparse, sys, time

# ═══════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════
def fermi(e, beta):
    x = beta * np.asarray(e, dtype=float)
    out = np.empty_like(x)
    out[x >  500] = 0.0
    out[x < -500] = 1.0
    m = (x >= -500) & (x <= 500)
    out[m] = 1.0 / (np.exp(x[m]) + 1.0)
    return out

def hop_element(s, i_o, j_o):
    s = int(s)
    if not ((s >> j_o) & 1) or ((s >> i_o) & 1): return 0, 0
    sgn = 1
    for k in range(min(i_o, j_o) + 1, max(i_o, j_o)):
        if (s >> k) & 1: sgn *= -1
    return (s ^ (1 << j_o)) ^ (1 << i_o), sgn

def make_fock_basis(n_modes, n_particles):
    states = np.array([sum(1 << b for b in bits)
                       for bits in combinations(range(n_modes), n_particles)],
                      dtype=np.int32)
    sidx = {int(s): i for i, s in enumerate(states)}
    return states, sidx

def popcount(x):
    xi = int(x)
    return xi.bit_count() if hasattr(xi, "bit_count") else bin(xi).count("1")

def c_op(dim, mode):
    C = np.zeros((dim, dim))
    for s in range(dim):
        if not ((s >> mode) & 1): continue
        sign = -1.0 if (popcount(s >> (mode + 1)) % 2) else 1.0
        C[s ^ (1 << mode), s] = sign
    return C


@lru_cache(maxsize=None)
def get_impurity_ops(M):
    Norb  = 1 + M
    Nmode = 2 * Norb
    dim   = 1 << Nmode
    C  = [c_op(dim, mode) for mode in range(Nmode)]
    Cd = [op.T for op in C]
    n_ops = [Cd[m] @ C[m] for m in range(Nmode)]
    return Cd, C, n_ops

@lru_cache(maxsize=None)
def get_H2_cache(M):
    nps = 2 + 3*M
    Nmode = 2*nps
    states, sidx = make_fock_basis(Nmode, nps)
    D = len(states)

    occ = np.zeros((Nmode, D), dtype=float)
    for m in range(Nmode):
        occ[m] = ((states >> m) & 1).astype(float)

    def trans(src, dst):
        rows=[]; cols=[]; signs=[]
        for i,s in enumerate(states):
            s2, sgn = hop_element(s, dst, src)
            if s2 and s2 in sidx:
                rows.append(i); cols.append(sidx[s2]); signs.append(sgn)
        return (np.array(rows,dtype=np.int32), np.array(cols,dtype=np.int32), np.array(signs,dtype=float))

    ham_transitions = []
    obs_transitions = {}
    for sp in range(2):
        base = sp * nps
        ham_transitions.append((base+0, base+1, 't'))
        ham_transitions.append((base+1, base+0, 't'))
        for l in range(M):
            for site in range(2):
                d_orb = base + site
                g_orb = base + 2 + 2*l + site
                ham_transitions.append((d_orb, g_orb, ('V', l)))
                ham_transitions.append((g_orb, d_orb, ('V', l)))
            gb_orb = base + 2 + 2*M + l
            for site in range(2):
                d_orb = base + site
                ham_transitions.append((d_orb, gb_orb, ('Bg', l)))
                ham_transitions.append((gb_orb, d_orb, ('Bg', l)))
    # precompute distinct transition tuples needed in H and observables
    transition_map = {}
    needed = set((src,dst) for src,dst,_ in ham_transitions)
    for l in range(M):
        needed.add((0*nps+2+2*l, 0*nps+0))
        needed.add((0*nps+2+2*M+l, 0*nps+0))
        needed.add((0*nps+2+2*M+l, 0*nps+1))
    needed.add((0*nps+0, 0*nps+1))
    for src,dst in needed:
        transition_map[(src,dst)] = trans(src,dst)
    return nps, occ, ham_transitions, transition_map

@lru_cache(maxsize=None)
def get_H2_sector_cache(M, nup):
    nps = 2 + 3*M
    ndn = nps - nup
    if ndn < 0 or ndn > nps: return nps, np.zeros((2*nps,0)), [], {}
    up_states = np.array([sum(1 << b for b in bits)
                          for bits in combinations(range(nps), nup)], dtype=np.int32)
    dn_states = np.array([sum(1 << b for b in bits)
                          for bits in combinations(range(nps), ndn)], dtype=np.int32)
    D = len(up_states) * len(dn_states)
    states = np.empty(D, dtype=np.int32)
    k = 0
    for su in up_states:
        for sd in dn_states:
            states[k] = int(su) | (int(sd) << nps); k += 1
    sidx = {int(s): i for i, s in enumerate(states)}

    occ = np.zeros((2*nps, D), dtype=float)
    for m in range(2*nps):
        occ[m] = ((states >> m) & 1).astype(float)

    def trans(src, dst):
        rows=[]; cols=[]; signs=[]
        for i,s in enumerate(states):
            s2, sgn = hop_element(s, dst, src)
            if s2 and s2 in sidx:
                rows.append(i); cols.append(sidx[s2]); signs.append(sgn)
        return (np.array(rows,dtype=np.int32), np.array(cols,dtype=np.int32), np.array(signs,dtype=float))

    ham_transitions = []
    for sp in range(2):
        base = sp * nps
        ham_transitions.append((base+0, base+1, 't'))
        ham_transitions.append((base+1, base+0, 't'))
        for l in range(M):
            for site in range(2):
                d_orb = base + site; g_orb = base + 2 + 2*l + site
                ham_transitions.append((d_orb, g_orb, ('V', l)))
                ham_transitions.append((g_orb, d_orb, ('V', l)))
            gb_orb = base + 2 + 2*M + l
            for site in range(2):
                d_orb = base + site
                ham_transitions.append((d_orb, gb_orb, ('Bg', l)))
                ham_transitions.append((gb_orb, d_orb, ('Bg', l)))
    needed = set((src,dst) for src,dst,_ in ham_transitions)
    for l in range(M):
        needed.add((0*nps+2+2*l, 0*nps+0))
        needed.add((0*nps+2+2*M+l, 0*nps+0))
        needed.add((0*nps+2+2*M+l, 0*nps+1))
    needed.add((0*nps+0, 0*nps+1))
    transition_map = {pair: trans(*pair) for pair in needed}
    return nps, occ, ham_transitions, transition_map

# ═══════════════════════════════════════════════════════════
# Lattice setup: Bethe or square
# ═══════════════════════════════════════════════════════════
def make_bethe_lattice(t, n_quad=700):
    """
    Bethe lattice, semicircular DOS, D=2t.
    Gauss-Chebyshev quadrature (from reference code).
    No bond scheme (gamma_k not defined).
    Returns EPS, GAM=None, W, D, z=None
    """
    D = 2 * t
    k = np.arange(1, n_quad + 1)
    x = np.cos(k * np.pi / (n_quad + 1))
    w = 2.0 / (n_quad + 1) * np.sin(k * np.pi / (n_quad + 1))**2
    eps = D * x
    return eps, None, w, D, None

def make_square_lattice(t, n_k=30):
    kx = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    KX, KY = np.meshgrid(kx, ky)
    EPS = (-2 * t * (np.cos(KX) + np.cos(KY))).ravel()
    GAM = (0.5 * (np.cos(KX) + np.cos(KY))).ravel()
    W   = np.ones(n_k**2) / n_k**2
    D   = 4 * t
    z   = 4
    return EPS, GAM, W, D, z

# ═══════════════════════════════════════════════════════════
# Single-site statics (general M)
# ═══════════════════════════════════════════════════════════
def lattice_statics(beta, eta, W, M, EPS, EPS_W, shift):
    """BZ sum: d + M h-ghosts. Returns nh[M], dh[M]."""
    N = len(EPS); n = 1 + M
    H = np.zeros((N, n, n))
    H[:, 0, 0] = EPS + shift
    for l in range(M):
        H[:, 1+l, 1+l] = eta[l]
        H[:, 0, 1+l] = H[:, 1+l, 0] = W[l]
    ev, Uv = np.linalg.eigh(H); f = fermi(ev, beta)
    nh = np.array([float(np.dot(EPS_W, np.sum(Uv[:,1+l,:]*f*Uv[:,1+l,:], axis=1)))
                   for l in range(M)])
    dh = np.array([float(np.dot(EPS_W, np.sum(Uv[:,0,:]*f*Uv[:,1+l,:], axis=1)))
                   for l in range(M)])
    return nh, dh

def gateway_statics(beta, eta, W, eps, V, M, shift):
    """Single-site gateway: d + M g-ghosts + M h-ghosts. Returns nh,dh,ng,dg."""
    n = 1 + 2*M
    H = np.zeros((n, n)); H[0, 0] = shift
    for l in range(M):
        H[1+l,   1+l  ] = eps[l]; H[0, 1+l  ] = H[1+l,   0] = V[l]
        H[1+M+l, 1+M+l] = eta[l]; H[0, 1+M+l] = H[1+M+l, 0] = W[l]
    ev, Uv = eigh(H); f = fermi(ev, beta); rho = (Uv * f) @ Uv.T
    ng = np.array([float(rho[1+l,   1+l  ]) for l in range(M)])
    dg = np.array([float(rho[0,     1+l  ]) for l in range(M)])
    nh = np.array([float(rho[1+M+l, 1+M+l]) for l in range(M)])
    dh = np.array([float(rho[0,     1+M+l]) for l in range(M)])
    return nh, dh, ng, dg

def impurity_statics(beta, eps, V, M, U, mu, ed=0.0):
    """
    Interacting impurity: d + M g-ghosts.
    Mode ordering: mode = 2*orb + spin, orb=0 is d, orb=1..M are g-ghosts.
    Returns ng[M], dg[M], docc.
    """
    Cd, C, n_ops = get_impurity_ops(M)

    def n(orb, spin): return n_ops[2*orb + spin]

    ed_eff = ed - mu
    H = ed_eff*(n(0,0) + n(0,1)) + U*(n(0,0) @ n(0,1))
    for l in range(M):
        orb = 1 + l
        H += eps[l] * (n(orb,0) + n(orb,1))
        H += V[l] * (Cd[2*orb+0] @ C[0] + Cd[0] @ C[2*orb+0])
        H += V[l] * (Cd[2*orb+1] @ C[1] + Cd[1] @ C[2*orb+1])

    ev, evec = eigh(H); E0 = ev.min()
    w = np.exp(-beta*(ev-E0)); prob = w/w.sum()
    def avg(O): return float(np.sum(prob * np.diag(evec.T @ O @ evec)))

    ng   = np.array([avg(n(1+l, 0)) for l in range(M)])
    dg   = np.array([avg(Cd[0] @ C[2*(1+l)]) for l in range(M)])
    docc = avg(n(0,0) @ n(0,1))
    return ng, dg, docc

# ═══════════════════════════════════════════════════════════
# Bond scheme statics (general M)
# ═══════════════════════════════════════════════════════════
def bond_lattice_statics(beta, eta, W, etab, Bh, M, EPS, GAM, EPS_W, shift):
    """
    BZ sum: d + M site-local h-ghosts + M bond hb-ghosts.
    Returns nh[M], dh[M], nhb[M], dhb[M].
    """
    N = len(EPS); n = 1 + 2*M
    H = np.zeros((N, n, n)); H[:, 0, 0] = EPS + shift
    for l in range(M):
        H[:, 1+l,   1+l  ] = eta[l];  H[:, 0, 1+l  ] = H[:, 1+l,   0] = W[l]
        H[:, 1+M+l, 1+M+l] = etab[l]; H[:, 0, 1+M+l] = H[:, 1+M+l, 0] = Bh[l]*GAM
    ev, Uv = np.linalg.eigh(H); f = fermi(ev, beta)
    nh  = np.array([float(np.dot(EPS_W, np.sum(Uv[:,1+l,  :]*f*Uv[:,1+l,  :], axis=1))) for l in range(M)])
    dh  = np.array([float(np.dot(EPS_W, np.sum(Uv[:,0,    :]*f*Uv[:,1+l,  :], axis=1))) for l in range(M)])
    nhb = np.array([float(np.dot(EPS_W, np.sum(Uv[:,1+M+l,:]*f*Uv[:,1+M+l,:], axis=1))) for l in range(M)])
    dhb = np.array([float(np.dot(EPS_W, np.sum(Uv[:,0,    :]*f*Uv[:,1+M+l,:], axis=1)*GAM)) for l in range(M)])
    return nh, dh, nhb, dhb

def bond_gateway_statics(beta, eta, W, eps, V, etab, Bh, epsb, Bg, M, t, shift):
    """Two-site quadratic gateway."""
    n_gw = 2 + 6*M
    H = np.zeros((n_gw, n_gw))
    H[0,0] = shift; H[1,1] = shift; H[0,1] = H[1,0] = -t

    for l in range(M):
        H[2+l,   2+l  ] = eta[l];  H[0, 2+l  ] = H[2+l,   0] = W[l]
        H[2+M+l, 2+M+l] = eta[l];  H[1, 2+M+l] = H[2+M+l, 1] = W[l]
        H[2+2*M+l, 2+2*M+l] = etab[l]
        H[0, 2+2*M+l] = H[2+2*M+l, 0] = Bh[l]
        H[1, 2+2*M+l] = H[2+2*M+l, 1] = Bh[l]
        H[2+3*M+l, 2+3*M+l] = eps[l];  H[0, 2+3*M+l] = H[2+3*M+l, 0] = V[l]
        H[2+4*M+l, 2+4*M+l] = eps[l];  H[1, 2+4*M+l] = H[2+4*M+l, 1] = V[l]
        H[2+5*M+l, 2+5*M+l] = epsb[l]
        H[0, 2+5*M+l] = H[2+5*M+l, 0] = Bg[l]
        H[1, 2+5*M+l] = H[2+5*M+l, 1] = Bg[l]

    ev, Uv = eigh(H); f = fermi(ev, beta); rho = (Uv * f) @ Uv.T
    nh   = np.array([0.5*(rho[2+l,   2+l  ] + rho[2+M+l, 2+M+l]) for l in range(M)])
    dh   = np.array([0.5*(rho[0,     2+l  ] + rho[1,     2+M+l]) for l in range(M)])
    nhb  = np.array([rho[2+2*M+l, 2+2*M+l]                        for l in range(M)])
    dhb  = np.array([0.5*(rho[0, 2+2*M+l] + rho[1, 2+2*M+l])     for l in range(M)])
    ng   = np.array([0.5*(rho[2+3*M+l, 2+3*M+l] + rho[2+4*M+l, 2+4*M+l]) for l in range(M)])
    dg   = np.array([0.5*(rho[0, 2+3*M+l] + rho[1, 2+4*M+l])     for l in range(M)])
    ngb  = np.array([rho[2+5*M+l, 2+5*M+l]                        for l in range(M)])
    dgb  = np.array([rho[0, 2+5*M+l] + rho[1, 2+5*M+l]           for l in range(M)])
    n_site = float(rho[0, 0])
    return nh, dh, nhb, dhb, ng, dg, ngb, dgb, n_site

def build_H2(beta, eps, V, epsb, Bg, dmu, M, U, mu, t, ed=0.0):
    """
    Two-site interacting cluster ED -- block diagonalized by (nup, ndn) sectors.
    Largest block for M=2: C(8,4)^2 = 4900 instead of C(16,8) = 12870.
    ~7x faster than full diagonalization.
    """
    nps   = 2 + 3*M
    ed_eff = ed - mu - dmu

    global_E0 = None
    block_data = []

    for nup in range(nps + 1):
        nps_b, occ, ham_transitions, transition_map = get_H2_sector_cache(M, nup)
        D_ = occ.shape[1]
        if D_ == 0: continue

        diag = np.zeros(D_)
        for sp in range(2):
            base = sp * nps_b
            diag += ed_eff * (occ[base+0] + occ[base+1])
            for l in range(M):
                diag += eps[l]  * (occ[base+2+2*l] + occ[base+2+2*l+1])
                diag += epsb[l] * occ[base+2+2*M+l]
        diag += U * occ[0*nps_b+0] * occ[1*nps_b+0]
        diag += U * occ[0*nps_b+1] * occ[1*nps_b+1]

        H = np.diag(diag)
        for src, dst, ampinfo in ham_transitions:
            rows, cols, signs = transition_map[(src,dst)]
            if len(rows) == 0: continue
            amp = -t if not isinstance(ampinfo, tuple) else (V[ampinfo[1]] if ampinfo[0]=='V' else Bg[ampinfo[1]])
            H[rows, cols] += amp * signs

        ev, evec = eigh(H)
        E0 = float(ev.min())
        if global_E0 is None or E0 < global_E0: global_E0 = E0
        block_data.append((ev, evec, occ, transition_map, nps_b))

    Z = 0.0
    num_ng=np.zeros(M); num_ngb=np.zeros(M)
    num_dg=np.zeros(M); num_dgb=np.zeros(M)
    num_docc=0.0; num_hop=0.0; num_nsite=0.0

    for ev, evec, occ, transition_map, nps_b in block_data:
        w = np.exp(-beta*(ev - global_E0))
        Zb = float(w.sum()); Z += Zb
        psi2w = (evec**2) @ w

        def avg_diag(vec): return float(vec @ psi2w)
        def avg_hop(src, dst):
            rows, cols, signs = transition_map[(src,dst)]
            if len(rows)==0: return 0.0
            # vectorized: sum_n w[n] * signs . (evec[rows,n] * evec[cols,n])
            return float((signs @ (evec[rows,:] * evec[cols,:])) @ w)

        for l in range(M):
            num_ng[l]  += avg_diag(occ[0*nps_b+2+2*l])
            num_ngb[l] += avg_diag(occ[0*nps_b+2+2*M+l])
            num_dg[l]  += avg_hop(0*nps_b+2+2*l, 0*nps_b+0)
            num_dgb[l] += avg_hop(0*nps_b+2+2*M+l, 0*nps_b+0) + avg_hop(0*nps_b+2+2*M+l, 0*nps_b+1)
        num_docc  += avg_diag(occ[0*nps_b+0] * occ[1*nps_b+0])
        num_hop   += avg_hop(0*nps_b+0, 0*nps_b+1)
        num_nsite += avg_diag(occ[0*nps_b+0] + occ[1*nps_b+0])

    if Z == 0.0: raise np.linalg.LinAlgError('zero partition function in H2')
    return num_ng/Z, num_dg/Z, num_ngb/Z, num_dgb/Z, num_docc/Z, num_hop/Z, num_nsite/Z

# ═══════════════════════════════════════════════════════════
# Single-site solver (general M)
# ═══════════════════════════════════════════════════════════
def solve_singlesite(beta, eta0, W0, eps0, V0, M, U, t, mu, shift,
                     EPS, EPS_W, mix=0.5, tol=1e-9, maxiter=200, verbose=False):
    """Sequential 2M-equation TRF fits (h-sector then g-sector)."""
    eta = np.array(eta0, dtype=float); W   = np.array(W0,   dtype=float)
    eps = np.array(eps0, dtype=float); V   = np.array(V0,   dtype=float)

    for it in range(1, maxiter+1):
        nh_lat, dh_lat = lattice_statics(beta, eta, W, M, EPS, EPS_W, shift)

        def r_h(p):
            nh_gw,dh_gw,_,_ = gateway_statics(beta,eta,W,p[:M],p[M:],M,shift)
            return np.concatenate([nh_gw-nh_lat, dh_gw-dh_lat])
        sol_h = least_squares(r_h, np.concatenate([eps,V]), method='trf',
                              ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=5000)
        eps_new=sol_h.x[:M]; V_new=sol_h.x[M:]

        ng_imp,dg_imp,_ = impurity_statics(beta,eps_new,V_new,M,U,mu)

        def r_g(p):
            _,_,ng_gw,dg_gw = gateway_statics(beta,p[:M],p[M:],eps_new,V_new,M,shift)
            return np.concatenate([ng_gw-ng_imp, dg_gw-dg_imp])
        sol_g = least_squares(r_g, np.concatenate([eta,W]), method='trf',
                              ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=5000)
        eta_new=sol_g.x[:M]; W_new=sol_g.x[M:]

        dp = float(np.linalg.norm(
            np.concatenate([eta_new-eta,W_new-W,eps_new-eps,V_new-V])))
        eta=mix*eta_new+(1-mix)*eta; W=mix*W_new+(1-mix)*W
        eps=mix*eps_new+(1-mix)*eps; V=mix*V_new+(1-mix)*V

        if verbose:
            _,_,docc=impurity_statics(beta,eps,V,M,U,mu)
            print(f'  SS it={it:3d}  dp={dp:.2e}  docc={docc:.6f}')
            sys.stdout.flush()
        if dp < tol: break

    _,_,docc = impurity_statics(beta,eps,V,M,U,mu)
    return dict(eta=eta,W=W,eps=eps,V=V,docc=docc,iters=it)

# ═══════════════════════════════════════════════════════════
# Bond solver -- NO FROZEN TARGETS (general M)
# ═══════════════════════════════════════════════════════════
def solve_bond(beta, ss, M, U, t, mu, shift, EPS, GAM, EPS_W, z,
               mix=0.3, tol=1e-9, maxiter=200, verbose=False):
    """
    Fully simultaneous (8M+1)-variable solve.
    NO frozen targets -- all correlators computed fresh at every evaluation.

    Variables: eta[M], W[M], etab[M], Bh[M], eps[M], V[M], epsb[M], Bg[M], dmu
    Total: 8M+1

    Matching conditions (8M+1):
      h-sector (4M): lattice <-> bond gateway
      g-sector (4M): BPK[gateways] = BPK[imp+H2]
      half-filling (1)
    """
    eta  = ss['eta'].copy(); W    = ss['W'].copy()
    eps  = ss['eps'].copy(); V    = ss['V'].copy()
    etab = ss.get('etab', np.zeros(M)).copy()
    Bh   = ss.get('Bh',   np.zeros(M)).copy()
    epsb = ss.get('epsb',  np.zeros(M)).copy()
    Bg   = ss.get('Bg',   np.zeros(M)).copy()
    dmu  = float(ss.get('dmu', 0.0))

    def residuals(p):
        eta_  = p[0*M:1*M]; W_    = p[1*M:2*M]
        etab_ = p[2*M:3*M]; Bh_   = p[3*M:4*M]
        eps_  = p[4*M:5*M]; V_    = p[5*M:6*M]
        epsb_ = p[6*M:7*M]; Bg_   = p[7*M:8*M]
        dmu_  = float(p[8*M])

        try:
            # Lattice
            nh_l,  dh_l  = lattice_statics(beta, eta_, W_, M, EPS, EPS_W, shift)
            nh_lb, dh_lb, nhb_l, dhb_l = bond_lattice_statics(
                beta, eta_, W_, etab_, Bh_, M, EPS, GAM, EPS_W, shift)

            # Single-site gateway
            nh1, dh1, ng1_, dg1_ = gateway_statics(
                beta, eta_, W_, eps_, V_, M, shift)

            # Two-site gateway
            nh2, dh2, nhb2, dhb2, ng2_, dg2_, ngb2_, dgb2_, ns = bond_gateway_statics(
                beta, eta_, W_, eps_, V_, etab_, Bh_, epsb_, Bg_, M, t, shift)

            # Impurity and H2 -- fresh, no freezing
            ng_imp, dg_imp, _ = impurity_statics(beta, eps_, V_, M, U, mu)
            ng_H2, dg_H2, ngb_H2, dgb_H2, _, _, _ = build_H2(
                beta, eps_, V_, epsb_, Bg_, dmu_, M, U, mu, t)

        except np.linalg.LinAlgError:
            return np.ones(8*M+1) * 1e6

        r = []
        for l in range(M):
            # h-sector: lattice <-> bond gateway
            r.append((dh2[l]-dh1[l]) - (dh_l[l]-dh1[l])/z)
            r.append((nh2[l]-nh1[l]) - (nh_l[l]-nh1[l])/z)
            r.append(nhb2[l] - nhb_l[l])
            r.append(dhb2[l] - dhb_l[l])
            # g-sector: BPK[gateways] = BPK[imp+H2]
            r.append((1-z)*dg1_[l] + z*dg2_[l] - ((1-z)*dg_imp[l] + z*dg_H2[l]))
            r.append((1-z)*ng1_[l] + z*ng2_[l] - ((1-z)*ng_imp[l] + z*ng_H2[l]))
            r.append(ngb2_[l] - ngb_H2[l])
            r.append(dgb2_[l] - dgb_H2[l])
        # half-filling
        r.append(ns - 0.5)
        return np.array(r)

    x0 = np.concatenate([eta, W, etab, Bh, eps, V, epsb, Bg, [dmu]])

    for it in range(1, maxiter+1):
        xc = np.clip(x0, -7.9, 7.9); xc[-1] = np.clip(xc[-1], -2.9, 2.9)

        sol = least_squares(residuals, xc, method='trf',
                            ftol=1e-12, xtol=1e-12, gtol=1e-12,
                            max_nfev=3000,
                            bounds=([-8]*(8*M)+[-3], [8]*(8*M)+[3]))

        dp       = float(np.linalg.norm(sol.x - xc))
        res_norm = float(np.linalg.norm(sol.fun))

        x0 = mix*sol.x + (1-mix)*xc

        if verbose:
            _,_,_,_,docc_2,hop,_ = build_H2(
                beta, x0[4*M:5*M], x0[5*M:6*M],
                x0[6*M:7*M], x0[7*M:8*M], float(x0[8*M]), M, U, mu, t)
            print(f'  it={it:3d}  |res|={res_norm:.2e}  dp={dp:.2e}'
                  f'  docc_2={docc_2:.5f}  nfev={sol.nfev}')
            sys.stdout.flush()

        if dp < tol and res_norm < tol: break

    eta  = x0[0*M:1*M]; W    = x0[1*M:2*M]
    etab = x0[2*M:3*M]; Bh   = x0[3*M:4*M]
    eps  = x0[4*M:5*M]; V    = x0[5*M:6*M]
    epsb = x0[6*M:7*M]; Bg   = x0[7*M:8*M]
    dmu  = float(x0[8*M])

    # Final BPK double occupancy
    _,_,docc_1_f = impurity_statics(beta, eps, V, M, U, mu)
    _,_,_,_,docc_2_f,hop_f,_ = build_H2(beta, eps, V, epsb, Bg, dmu, M, U, mu, t)
    docc_bpk = (1-z)*docc_1_f + z*docc_2_f

    return dict(eta=eta,W=W,etab=etab,Bh=Bh,eps=eps,V=V,epsb=epsb,Bg=Bg,
                dmu=dmu,docc_1=docc_1_f,docc_2=docc_2_f,docc_bpk=docc_bpk,
                hop=hop_f,res=res_norm,iters=it)

# ═══════════════════════════════════════════════════════════
# Temperature sweep
# ═══════════════════════════════════════════════════════════
def run_sweep(U=1.3, t=0.5, M=1, mode='both', n_k=30, T_vals=None,
              mix_ss=0.5, mix_bond=0.3,
              tol_ss=1e-9, tol_bond=1e-9,
              maxiter_ss=200, maxiter_bond=200,
              verbose=False):

    mu    = U / 2.0
    shift = 0.0
    EPS, GAM, EPS_W, D, z = make_square_lattice(t, n_k=n_k)

    if T_vals is None:
        T_vals = np.array([1.0,0.8,0.667,0.5,0.4,0.333,0.25,0.2,
                           0.167,0.143,0.125,0.111,0.1,0.091,0.083,
                           0.071,0.063,0.056,0.05])

    # Initial guesses
    if M == 1:
        eta0=np.array([-0.3]);        W0=np.array([0.2])
        eps0=np.array([-0.1]);        V0=np.array([0.4])
    else:
        eta0=np.array([-0.3,-0.1]);   W0=np.array([0.2,0.1])
        eps0=np.array([-0.1,-0.05]);  V0=np.array([0.4,0.3])

    results = []
    # Persist bond-only warm-start state across temperatures.
    bond_state = {}

    print(f'\nGhost-DMFT  M={M}  U={U}  t={t}  D={D:.1f}  mode={mode}')
    print(f'  {8*M+1} equations, {8*M+1} unknowns')
    print(f'  H2: nps={2+3*M}, states={comb(2*(2+3*M), 2+3*M)}')

    hdr = f'{"T":>8}  {"T/D":>7}  {"docc_ss":>10}'
    if mode in ('bond','both'):
        hdr += f'  {"docc_BPK":>10}  {"docc_2":>8}  {"hop":>8}  {"res":>8}  {"its":>4}'
    print(hdr); print('-'*(len(hdr)+8)); sys.stdout.flush()

    for Tv in T_vals:
        beta=1.0/Tv; t0=time.time()

        ss = solve_singlesite(beta, eta0, W0, eps0, V0, M, U, t, mu, shift,
                              EPS, EPS_W, mix=mix_ss, tol=tol_ss,
                              maxiter=maxiter_ss)

        row = dict(T=Tv, ToverD=Tv/D, beta=beta, docc_ss=ss['docc'])

        if mode in ('bond','both'):
            # Inject previous converged bond state into current single-site seed.
            ss.update(bond_state)
            rb = solve_bond(beta, ss, M, U, t, mu, shift, EPS, GAM, EPS_W, z,
                            mix=mix_bond, tol=tol_bond,
                            maxiter=maxiter_bond, verbose=verbose)
            row.update(docc_bpk=rb['docc_bpk'], docc_2=rb['docc_2'],
                       docc_1=rb['docc_1'], hop=rb['hop'],
                       dmu=rb['dmu'], res=rb['res'], iters_bond=rb['iters'])
            if rb['res'] < 1e-4:
                bond_state = {
                    'etab': rb['etab'].copy(),
                    'Bh': rb['Bh'].copy(),
                    'epsb': rb['epsb'].copy(),
                    'Bg': rb['Bg'].copy(),
                    'dmu': rb['dmu'],
                }

        dt=time.time()-t0
        line=f'  {Tv:6.4f}  {Tv/D:7.4f}  {ss["docc"]:10.6f}'
        if mode in ('bond','both'):
            line+=(f'  {rb["docc_bpk"]:10.6f}  {rb["docc_2"]:8.6f}'
                   f'  {rb["hop"]:8.5f}  {rb["res"]:8.2e}  {rb["iters"]:4d}')
        line+=f'  ({dt:.1f}s)'
        print(line); sys.stdout.flush()

        results.append(row)
        # warm-start SS parameters
        eta0,W0,eps0,V0 = ss['eta'],ss['W'],ss['eps'],ss['V']
    return results, D

# ═══════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════
def save_results(results, fname, mode):
    keys = ['T','ToverD','beta','docc_ss']
    if mode in ('bond','both'):
        keys += ['docc_bpk','docc_2','docc_1','hop','dmu','res']
    header = '  '.join(keys)
    rows = [[r.get(k,np.nan) for k in keys] for r in results]
    np.savetxt(fname, rows, header=header, fmt='%.8f')
    print(f'\nSaved: {fname}')

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ghost-DMFT M=1 or 2, square lattice')
    parser.add_argument('--M',       type=int,   default=1,        help='ghost poles (1 or 2)')
    parser.add_argument('--U',       type=float, default=1.3,      help='Hubbard U')
    parser.add_argument('--t',       type=float, default=0.5,      help='hopping t')
    parser.add_argument('--mode',    type=str,   default='both',   help='ss, bond, or both')
    parser.add_argument('--nk',      type=int,   default=30,       help='k-grid size')
    parser.add_argument('--verbose', action='store_true',           help='verbose bond iters')
    parser.add_argument('--out',     type=str,   default=None,     help='output filename')
    args = parser.parse_args()

    results, D = run_sweep(
        U=args.U, t=args.t, M=args.M,
        mode=args.mode, n_k=args.nk,
        verbose=args.verbose)

    if args.out is None:
        args.out = f'ghost_dmft_square_M{args.M}_U{args.U}_t{args.t}_{args.mode}.dat'
    save_results(results, args.out, args.mode)
