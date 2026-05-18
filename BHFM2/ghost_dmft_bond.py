#!/usr/bin/env python3
"""
Ghost-DMFT with FULL BOND SCHEME on 2D square lattice (Mh=Mb=Mg=Mbg=M).

Five sectors:
  - lattice (k-space, free):    [d, h, h^b_x, h^b_y]   per k        (size 1+M+2*Mb)
  - 1-site gateway (free):       [d, g, h]                          (size 1+M+M)
  - 2-site gateway (free):       [d1,d2, g1,g2, h1,h2, h^b_x,h^b_y,
                                  g^b_x, g^b_y]                     (size 2+2M+2M+2Mb+2Mbg)
  - 1-site impurity (ED, U):     [d, g] x 2 spins                   (Fock dim 2^(2(1+M)))
  - 2-site impurity (ED, U):     [d1,d2, g1,g2, g^b_x,g^b_y] x 2    (Fock dim 2^(2(2+2M+2Mbg)))

PH half-filling: eps_d=0, mu=U/2, Sigma_inf=U/2, all eta/eps=0 (M=1, PH-symmetric).
Free parameters at M=1: W, B_h, V, B_g  (4 scalars).

Matching (paper table):
  h:      <h†h>_lat = (1-z)<h†h>_gw1 + z<h†h>_gw2;  <d†h>_lat = ditto
  bond-h: <h^b†h^b>_lat = <h^b†h^b>_gw2; <h^b†(d1+d2)>_lat = <h^b†(d1+d2)>_gw2
  g:      (1-z)<g†g>_imp1 + z<g†g>_imp2 = (1-z)<g†g>_gw1 + z<g†g>_gw2; <d†g> ditto
  bond-g: <g^b†g^b>_imp2 = <g^b†g^b>_gw2; <g^b†(d1+d2)>_imp2 = <g^b†(d1+d2)>_gw2

Lattice correlators are computed with cos(k·δ/2) weighting per the bond-ghost
Fourier structure (paper Eq. for Sigma_b,lat).
"""

import numpy as np
from numpy.linalg import eigh
from scipy.optimize import least_squares
from itertools import combinations
import time

# =============================================================
# Fock-space ED helpers (sector-blocked by N_up, N_dn)
# =============================================================

def popcount(x: int) -> int:
    return int(x).bit_count()


def basis_for_N(N_orb: int, N_e: int):
    return [sum(1 << i for i in combo) for combo in combinations(range(N_orb), N_e)]


def build_sector(N_orb, N_up, N_dn):
    """Return basis (np.int64 array of (up_bits<<N_orb)|dn_bits) and index map."""
    ups = basis_for_N(N_orb, N_up)
    dns = basis_for_N(N_orb, N_dn)
    basis = np.array([(u << N_orb) | d for u in ups for d in dns], dtype=np.int64)
    idx = {int(s): i for i, s in enumerate(basis)}
    return basis, idx


def _sign(state, target):
    """JW sign: +1 if even number of occupied modes below `target`."""
    mask = (1 << target) - 1
    return 1.0 if (popcount(state & mask) % 2 == 0) else -1.0


def build_H_sector(h1: np.ndarray, U: float, U_orbs, N_orb, basis, idx):
    """Build dense Hamiltonian on sector with basis `basis`.
    h1: (N_orb,N_orb) one-body matrix (real symmetric, applied to both spins).
    U on each orbital in U_orbs: U * n_{c,up} n_{c,dn}.
    """
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=float)
    Nmask = (1 << N_orb) - 1

    # One-body part: sum_{a,b,sigma} h1[a,b] c_{a sigma}^dag c_{b sigma}
    for i, s in enumerate(basis):
        up = (s >> N_orb) & Nmask
        dn = s & Nmask
        for a in range(N_orb):
            for b in range(N_orb):
                t_ab = h1[a, b]
                if t_ab == 0.0:
                    continue
                for spin, bits in ((0, up), (1, dn)):
                    if not (bits & (1 << b)):
                        continue
                    if a != b and (bits & (1 << a)):
                        continue
                    # Apply c_b
                    nb = bits ^ (1 << b)
                    sgn = _sign(bits, b)
                    # Apply c_a^dag
                    if nb & (1 << a):
                        continue   # blocked (only if a!=b)
                    if a == b:
                        # n operator: same state, sign cancels
                        new_bits = bits
                        sgn_total = 1.0
                    else:
                        sgn2 = _sign(nb, a)
                        new_bits = nb | (1 << a)
                        sgn_total = sgn * sgn2
                    if spin == 0:
                        new_s = (new_bits << N_orb) | dn
                    else:
                        new_s = (up << N_orb) | new_bits
                    j = idx.get(int(new_s))
                    if j is None:
                        continue
                    H[j, i] += t_ab * sgn_total

    # Hubbard U
    for c in U_orbs:
        mask_up = 1 << c
        mask_dn = 1 << c
        for i, s in enumerate(basis):
            up = (s >> N_orb) & Nmask
            dn = s & Nmask
            if (up & mask_up) and (dn & mask_dn):
                H[i, i] += U

    return H


def expect_n_orb_sector(orb, psi, basis, N_orb):
    """<psi| n_{orb,up} + n_{orb,dn} |psi>."""
    Nmask = (1 << N_orb) - 1
    mask = 1 << orb
    val = 0.0
    p2 = (psi.conj() * psi).real if np.iscomplexobj(psi) else psi * psi
    for i, s in enumerate(basis):
        up = (s >> N_orb) & Nmask
        dn = s & Nmask
        # Cast to int to avoid numpy.bool + numpy.bool = numpy.bool (logical OR) bug
        cnt = int((up & mask) != 0) + int((dn & mask) != 0)
        if cnt:
            val += cnt * p2[i]
    return val


def expect_double_orb(c, psi, basis, N_orb):
    """<n_up n_dn> on orbital c."""
    Nmask = (1 << N_orb) - 1
    mask = 1 << c
    val = 0.0
    p2 = (psi.conj() * psi).real if np.iscomplexobj(psi) else psi * psi
    for i, s in enumerate(basis):
        up = (s >> N_orb) & Nmask
        dn = s & Nmask
        if (up & mask) and (dn & mask):
            val += p2[i]
    return val


def expect_cdag_c(a, b, psi, basis, idx, N_orb):
    """sum over both spins of <psi| c_a^dag c_b |psi>. Real for real psi."""
    if a == b:
        return expect_n_orb_sector(a, psi, basis, N_orb)
    Nmask = (1 << N_orb) - 1
    val = 0.0
    for i, s in enumerate(basis):
        if psi[i] == 0:
            continue
        up = (s >> N_orb) & Nmask
        dn = s & Nmask
        for spin, bits in ((0, up), (1, dn)):
            if not (bits & (1 << b)):
                continue
            if bits & (1 << a):
                continue
            sgn = _sign(bits, b)
            nb = bits ^ (1 << b)
            sgn2 = _sign(nb, a)
            new_bits = nb | (1 << a)
            if spin == 0:
                new_s = (new_bits << N_orb) | dn
            else:
                new_s = (up << N_orb) | new_bits
            j = idx.get(int(new_s))
            if j is None:
                continue
            val += psi[j] * psi[i] * sgn * sgn2
    return val


def thermal_average_observables(H_builder, N_orb, observables, beta,
                                 N_filter=None):
    """Run ED over all (N_up, N_dn) sectors, return Boltzmann-weighted observables.

    observables: list of callables  obs(psi, basis, idx) -> float
    N_filter: if given, only sectors with N_up + N_dn in N_filter are kept.

    Returns (values_array, E0_global)
    """
    all_states = []   # (E, psi, basis, idx)
    for N_up in range(N_orb + 1):
        for N_dn in range(N_orb + 1):
            if N_filter is not None and (N_up + N_dn) not in N_filter:
                continue
            basis, idx = build_sector(N_orb, N_up, N_dn)
            if len(basis) == 0:
                continue
            H = H_builder(N_up, N_dn, basis, idx)
            H = 0.5 * (H + H.T)
            evals, evecs = eigh(H)
            for j in range(len(evals)):
                all_states.append((float(evals[j]), evecs[:, j], basis, idx))

    E0 = min(s[0] for s in all_states)
    weights = np.array([np.exp(-beta * (s[0] - E0)) for s in all_states])
    Z = weights.sum()
    weights /= Z

    out = np.zeros(len(observables))
    for (E, psi, basis, idx), w in zip(all_states, weights):
        if w < 1e-14:
            continue
        for k, obs in enumerate(observables):
            out[k] += w * obs(psi, basis, idx)
    return out, E0


# =============================================================
# Free-fermion sector helpers
# =============================================================

def fermi(e, beta):
    x = beta * np.asarray(e)
    out = np.empty_like(x, dtype=float)
    out[x > 50] = 0.0
    out[x < -50] = 1.0
    m = (x <= 50) & (x >= -50)
    out[m] = 1.0 / (np.exp(x[m]) + 1.0)
    return out


def free_density_matrix(H, beta):
    evals, U = eigh(H)
    f = fermi(evals, beta)
    return (U * f[None, :]) @ U.T


# =============================================================
# Sector observables
# =============================================================

def lattice_observables(model, params, beta):
    """Lattice = k-resolved free-fermion problem with [d, h, h^b_x, h^b_y]."""
    M, Mb = model['M'], model['Mb']
    cosx, cosy = model['cosx'], model['cosy']
    eps_k = model['eps_k']     # already includes -2t(cos+cos)
    shift = model['ed'] + model['Sigma_inf'] - model['mu']

    eta = params['eta']        # length M
    W = params['W']
    eta_b = params['eta_b']    # length Mb
    B_h = params['B_h']

    Nk = len(eps_k)
    D = 1 + M + 2 * Mb         # d, h_1..h_M, h^b_x_1..Mb, h^b_y_1..Mb
    nh_acc = np.zeros(M)
    Cdh_acc = np.zeros(M)
    nhb_acc = np.zeros(Mb)
    Cdhb_acc = np.zeros(Mb)
    nd_acc = 0.0

    # cos(k·δ/2) factors -- δ = (1,0) or (0,1) in lattice units
    coskx2 = np.cos(np.arccos(cosx) / 2.0)   # equivalent to cos(kx/2) up to sign
    cosky2 = np.cos(np.arccos(cosy) / 2.0)
    # Note: we have only cos kx (in [-1,1]); k itself in (-pi, pi). cos(k/2) >= 0 for k in (-pi,pi).
    # Use sqrt((1+cos k)/2) which equals |cos(k/2)| -- and cos(k/2) >= 0 on (-pi,pi).
    coskx2 = np.sqrt(np.maximum(0.0, 0.5 * (1.0 + cosx)))
    cosky2 = np.sqrt(np.maximum(0.0, 0.5 * (1.0 + cosy)))

    wk = 1.0 / Nk

    for ik in range(Nk):
        H = np.zeros((D, D), dtype=float)
        H[0, 0] = eps_k[ik] + shift
        # h-ghosts couple constant W
        for a in range(M):
            i_h = 1 + a
            H[i_h, i_h] = eta[a]
            H[0, i_h] = W[a]
            H[i_h, 0] = W[a]
        # bond-h: x and y, coupling B_h * cos(k_δ/2)
        for b in range(Mb):
            ix = 1 + M + b
            iy = 1 + M + Mb + b
            H[ix, ix] = eta_b[b]
            H[iy, iy] = eta_b[b]
            cx = B_h[b] * coskx2[ik]
            cy = B_h[b] * cosky2[ik]
            H[0, ix] = cx; H[ix, 0] = cx
            H[0, iy] = cy; H[iy, 0] = cy

        rho = free_density_matrix(H, beta)
        nd_acc += wk * rho[0, 0]
        for a in range(M):
            nh_acc[a] += wk * rho[1 + a, 1 + a]
            Cdh_acc[a] += wk * rho[0, 1 + a]
        for b in range(Mb):
            ix = 1 + M + b; iy = 1 + M + Mb + b
            # average over the two bond directions
            nhb_acc[b] += wk * 0.5 * (rho[ix, ix] + rho[iy, iy])
            # Match the ORIGINAL convention for the lattice bond-h off-diagonal:
            #   Cx = <cos(kx/2) · rho[ix,0]>_k,  Cy analogous,
            #   C_dhb = 0.5*(Cx + Cy).  No extra factor of 2.
            Cdhb_acc[b] += wk * 0.5 * (
                coskx2[ik] * rho[ix, 0]
                + cosky2[ik] * rho[iy, 0]
            )

    spin = 2.0
    return {
        'n_h':   spin * nh_acc,
        'C_dh':  spin * Cdh_acc,
        'n_hb':  spin * nhb_acc,
        'C_dhb': spin * Cdhb_acc,
        'n_d':   spin * nd_acc,
    }


def gw1_observables(model, params, beta):
    """Single-site gateway: [d, g_1..g_M, h_1..h_M]."""
    M = model['M']
    shift = model['ed'] + model['Sigma_inf'] - model['mu']

    eps = params['eps']; V = params['V']
    eta = params['eta']; W = params['W']

    D = 1 + M + M
    H = np.zeros((D, D), dtype=float)
    H[0, 0] = shift
    for a in range(M):
        ig = 1 + a; ih = 1 + M + a
        H[ig, ig] = eps[a]; H[0, ig] = V[a]; H[ig, 0] = V[a]
        H[ih, ih] = eta[a]; H[0, ih] = W[a]; H[ih, 0] = W[a]

    rho = free_density_matrix(H, beta)
    spin = 2.0
    return {
        'n_g':  spin * np.diag(rho[1:1+M, 1:1+M]),
        'C_dg': spin * rho[0, 1:1+M],
        'n_h':  spin * np.diag(rho[1+M:, 1+M:]),
        'C_dh': spin * rho[0, 1+M:],
        'n_d':  spin * rho[0, 0],
    }


def gw2_observables(model, params, beta):
    """Two-site gateway:
       [d1, d2, g1_1..M, g2_1..M, h1_1..M, h2_1..M,
        h^b_x_1..Mb, h^b_y_1..Mb, g^b_x_1..Mbg, g^b_y_1..Mbg].
    """
    M, Mb, Mbg = model['M'], model['Mb'], model['Mbg']
    shift = model['ed'] + model['Sigma_inf'] - model['mu']
    t = model['t']

    eps = params['eps']; V = params['V']
    eta = params['eta']; W = params['W']
    eta_b = params['eta_b']; B_h = params['B_h']
    eps_bg = params['eps_bg']; B_g = params['B_g']

    off_d = 0
    off_g1 = 2
    off_g2 = off_g1 + M
    off_h1 = off_g2 + M
    off_h2 = off_h1 + M
    off_hbx = off_h2 + M
    off_hby = off_hbx + Mb
    off_gbx = off_hby + Mb
    off_gby = off_gbx + Mbg
    D = off_gby + Mbg

    H = np.zeros((D, D), dtype=float)
    H[0, 0] = shift; H[1, 1] = shift
    H[0, 1] = -t; H[1, 0] = -t

    for a in range(M):
        ig1 = off_g1 + a; ig2 = off_g2 + a
        ih1 = off_h1 + a; ih2 = off_h2 + a
        H[ig1, ig1] = eps[a]; H[ig2, ig2] = eps[a]
        H[ih1, ih1] = eta[a]; H[ih2, ih2] = eta[a]
        H[0, ig1] = V[a]; H[ig1, 0] = V[a]
        H[1, ig2] = V[a]; H[ig2, 1] = V[a]
        H[0, ih1] = W[a]; H[ih1, 0] = W[a]
        H[1, ih2] = W[a]; H[ih2, 1] = W[a]

    # bond-h on x,y links: couples to (d_1 + d_2) with coupling B_h/2 per leg
    # (i.e., bond-h spans the bond, see paper Eq. "(d_i + d_{i+δ}) ... B/2")
    for b in range(Mb):
        ix = off_hbx + b; iy = off_hby + b
        H[ix, ix] = eta_b[b]; H[iy, iy] = eta_b[b]
        cup = B_h[b] / 2.0
        H[0, ix] = cup; H[ix, 0] = cup
        H[1, ix] = cup; H[ix, 1] = cup
        H[0, iy] = cup; H[iy, 0] = cup
        H[1, iy] = cup; H[iy, 1] = cup

    for b in range(Mbg):
        ix = off_gbx + b; iy = off_gby + b
        H[ix, ix] = eps_bg[b]; H[iy, iy] = eps_bg[b]
        cup = B_g[b] / 2.0
        H[0, ix] = cup; H[ix, 0] = cup
        H[1, ix] = cup; H[ix, 1] = cup
        H[0, iy] = cup; H[iy, 0] = cup
        H[1, iy] = cup; H[iy, 1] = cup

    rho = free_density_matrix(H, beta)
    spin = 2.0

    out = {}
    # h-ghost: average over the two sites
    n_h = np.zeros(M); C_dh = np.zeros(M)
    for a in range(M):
        ih1 = off_h1 + a; ih2 = off_h2 + a
        n_h[a] = 0.5 * (rho[ih1, ih1] + rho[ih2, ih2])
        C_dh[a] = 0.5 * (rho[ih1, 0] + rho[ih2, 1])
    out['n_h'] = spin * n_h
    out['C_dh'] = spin * C_dh

    n_g = np.zeros(M); C_dg = np.zeros(M)
    for a in range(M):
        ig1 = off_g1 + a; ig2 = off_g2 + a
        n_g[a] = 0.5 * (rho[ig1, ig1] + rho[ig2, ig2])
        C_dg[a] = 0.5 * (rho[ig1, 0] + rho[ig2, 1])
    out['n_g'] = spin * n_g
    out['C_dg'] = spin * C_dg

    n_hb = np.zeros(Mb); C_hbd = np.zeros(Mb)
    for b in range(Mb):
        ix = off_hbx + b; iy = off_hby + b
        n_hb[b] = 0.5 * (rho[ix, ix] + rho[iy, iy])
        # <h^b†(d1+d2)> averaged over directions
        Cx = rho[ix, 0] + rho[ix, 1]
        Cy = rho[iy, 0] + rho[iy, 1]
        C_hbd[b] = 0.5 * (Cx + Cy)
    out['n_hb'] = spin * n_hb
    out['C_hbd'] = spin * C_hbd

    n_gb = np.zeros(Mbg); C_gbd = np.zeros(Mbg)
    for b in range(Mbg):
        ix = off_gbx + b; iy = off_gby + b
        n_gb[b] = 0.5 * (rho[ix, ix] + rho[iy, iy])
        Cx = rho[ix, 0] + rho[ix, 1]
        Cy = rho[iy, 0] + rho[iy, 1]
        C_gbd[b] = 0.5 * (Cx + Cy)
    out['n_gb'] = spin * n_gb
    out['C_gbd'] = spin * C_gbd

    out['n_d'] = spin * 0.5 * (rho[0, 0] + rho[1, 1])
    return out


def imp1_observables(model, params, beta):
    """Single-site Anderson impurity: [d, g_1..g_M] + spin, U on d.
    Full thermal averaging over all (N_up, N_dn) sectors."""
    M = model['M']
    U = model['U']
    shift = model['ed'] - model['mu']
    eps = params['eps']; V = params['V']

    N_orb = 1 + M
    h1 = np.zeros((N_orb, N_orb), dtype=float)
    h1[0, 0] = shift
    for a in range(M):
        h1[1 + a, 1 + a] = eps[a]
        h1[0, 1 + a] = V[a]
        h1[1 + a, 0] = V[a]

    def Hb(N_up, N_dn, basis, idx):
        return build_H_sector(h1, U, [0], N_orb, basis, idx)

    obs_list = [
        lambda psi, basis, idx: expect_n_orb_sector(0, psi, basis, N_orb),
        lambda psi, basis, idx: expect_double_orb(0, psi, basis, N_orb),
    ]
    for a in range(M):
        obs_list.append(lambda psi, basis, idx, a=a: expect_n_orb_sector(1+a, psi, basis, N_orb))
        obs_list.append(lambda psi, basis, idx, a=a: expect_cdag_c(1+a, 0, psi, basis, idx, N_orb))

    vals, E0 = thermal_average_observables(Hb, N_orb, obs_list, beta)
    n_d = vals[0]; D = vals[1]
    n_g = np.array([vals[2 + 2*a] for a in range(M)])
    C_dg = np.array([vals[2 + 2*a + 1] for a in range(M)])
    return {'n_d': n_d, 'double': D, 'n_g': n_g, 'C_dg': C_dg, 'E0': E0}


def imp2_observables(model, params, beta):
    """Two-site Anderson impurity:
       orbitals [d1, d2, g1_1..M, g2_1..M, g^b_x_1..Mbg, g^b_y_1..Mbg], U on d1, d2.
       Full thermal averaging.
    """
    M, Mbg = model['M'], model['Mbg']
    U = model['U']
    t = model['t']
    shift = model['ed'] - model['mu']
    eps = params['eps']; V = params['V']
    eps_bg = params['eps_bg']; B_g = params['B_g']

    off_d = 0
    off_g1 = 2
    off_g2 = off_g1 + M
    off_gbx = off_g2 + M
    off_gby = off_gbx + Mbg
    N_orb = off_gby + Mbg

    h1 = np.zeros((N_orb, N_orb), dtype=float)
    h1[0, 0] = shift; h1[1, 1] = shift
    h1[0, 1] = -t; h1[1, 0] = -t
    for a in range(M):
        i1 = off_g1 + a; i2 = off_g2 + a
        h1[i1, i1] = eps[a]; h1[i2, i2] = eps[a]
        h1[0, i1] = V[a]; h1[i1, 0] = V[a]
        h1[1, i2] = V[a]; h1[i2, 1] = V[a]
    for b in range(Mbg):
        ix = off_gbx + b; iy = off_gby + b
        h1[ix, ix] = eps_bg[b]; h1[iy, iy] = eps_bg[b]
        cup = B_g[b] / 2.0
        h1[ix, 0] = cup; h1[0, ix] = cup
        h1[ix, 1] = cup; h1[1, ix] = cup
        h1[iy, 0] = cup; h1[0, iy] = cup
        h1[iy, 1] = cup; h1[1, iy] = cup

    def Hb(N_up, N_dn, basis, idx):
        return build_H_sector(h1, U, [0, 1], N_orb, basis, idx)

    # Observables we need:
    obs_list = []
    # 0: n_d (averaged over sites 0,1)
    obs_list.append(lambda psi, basis, idx:
        0.5*(expect_n_orb_sector(0, psi, basis, N_orb) +
             expect_n_orb_sector(1, psi, basis, N_orb)))
    # 1: double occupancy averaged over sites 0,1
    obs_list.append(lambda psi, basis, idx:
        0.5*(expect_double_orb(0, psi, basis, N_orb) +
             expect_double_orb(1, psi, basis, N_orb)))
    # n_g, C_dg per ghost a (averaged over sites)
    for a in range(M):
        i1 = off_g1 + a; i2 = off_g2 + a
        obs_list.append(lambda psi, basis, idx, i1=i1, i2=i2:
            0.5*(expect_n_orb_sector(i1, psi, basis, N_orb) +
                 expect_n_orb_sector(i2, psi, basis, N_orb)))
        obs_list.append(lambda psi, basis, idx, i1=i1, i2=i2:
            0.5*(expect_cdag_c(i1, 0, psi, basis, idx, N_orb) +
                 expect_cdag_c(i2, 1, psi, basis, idx, N_orb)))
    # n_gb, C_gbd per bond-g (averaged over directions)
    for b in range(Mbg):
        ix = off_gbx + b; iy = off_gby + b
        obs_list.append(lambda psi, basis, idx, ix=ix, iy=iy:
            0.5*(expect_n_orb_sector(ix, psi, basis, N_orb) +
                 expect_n_orb_sector(iy, psi, basis, N_orb)))
        obs_list.append(lambda psi, basis, idx, ix=ix, iy=iy:
            0.5*((expect_cdag_c(ix, 0, psi, basis, idx, N_orb) +
                  expect_cdag_c(ix, 1, psi, basis, idx, N_orb)) +
                 (expect_cdag_c(iy, 0, psi, basis, idx, N_orb) +
                  expect_cdag_c(iy, 1, psi, basis, idx, N_orb))))

    vals, E0 = thermal_average_observables(Hb, N_orb, obs_list, beta)
    out = {'n_d': vals[0], 'double': vals[1], 'E0': E0}
    out['n_g'] = np.array([vals[2 + 2*a] for a in range(M)])
    out['C_dg'] = np.array([vals[2 + 2*a + 1] for a in range(M)])
    base = 2 + 2*M
    out['n_gb'] = np.array([vals[base + 2*b] for b in range(Mbg)])
    out['C_gbd'] = np.array([vals[base + 2*b + 1] for b in range(Mbg)])
    return out


# =============================================================
# SCF: alternating substeps with deferred mixing
# =============================================================

def _sign_penalty(lhs, rhs, scale=10.0):
    """Penalty that vanishes when sign(lhs)==sign(rhs), grows when they oppose.
    Returns a residual contribution; safe for use inside a least-squares vector."""
    prod = lhs * rhs
    return scale * np.maximum(0.0, -prod)


def alt_scf_full(model, params, beta, mix=0.05, tol=1e-7, maxiter=80,
                 verbose=True, bound_W=3.0, sign_penalty=10.0,
                 enforce_PH=True):
    """Alternating SCF that turns on bond ghosts.

    Bounds & penalties:
      * couplings (W, V, B_h, B_g) bounded to [-bound_W, +bound_W]
      * if enforce_PH (default True), eta/eps fixed to 0 (PH symmetry, M=1)
      * sign-product penalty: hybridization correlators on the two sides of
        each matching equation should have the same sign; we add
        sign_penalty * max(0, -lhs*rhs) to the residual.
    """
    M = model['M']
    z = model['z']

    if enforce_PH:
        for key in ['eta', 'eps', 'eta_b', 'eps_bg']:
            params[key] = np.zeros_like(params[key])

    lo = np.array([-bound_W])
    hi = np.array([+bound_W])

    history = []
    t_total = time.time()

    for it in range(1, maxiter + 1):
        t_iter = time.time()

        W_old   = params['W'].copy(); V_old  = params['V'].copy()
        Bh_old  = params['B_h'].copy(); Bg_old = params['B_g'].copy()

        # --- Lattice statics ---
        lat = lattice_observables(model, params, beta)
        nh_lat = lat['n_h']; Cdh_lat = lat['C_dh']
        nhb_lat = lat['n_hb']; Cdhb_lat = lat['C_dhb']

        # --- Substep (a): solve V so gateway h-sector matches lat ---
        # h-sector matching:  lat = (1-z)*gw1 + z*gw2
        def res_h(x):
            params['V'] = x.copy()
            gw1 = gw1_observables(model, params, beta)
            gw2 = gw2_observables(model, params, beta)
            gw_n = (1-z)*gw1['n_h']  + z*gw2['n_h']
            gw_C = (1-z)*gw1['C_dh'] + z*gw2['C_dh']
            r_n  = gw_n - nh_lat
            r_C  = gw_C - Cdh_lat
            pen  = _sign_penalty(gw_C, Cdh_lat, sign_penalty)
            return np.concatenate([r_n, r_C, pen])

        sol_a = least_squares(res_h, x0=params['V'].copy(),
                              bounds=(lo, hi), method='trf',
                              ftol=1e-12, xtol=1e-12, max_nfev=2000)
        V_new_unmixed = sol_a.x.copy()
        params['V'] = V_new_unmixed

        # --- Impurity statics with new V ---
        imp1 = imp1_observables(model, params, beta)
        imp2 = imp2_observables(model, params, beta)
        target_n_g = (1-z)*imp1['n_g']  + z*imp2['n_g']
        target_C_g = (1-z)*imp1['C_dg'] + z*imp2['C_dg']

        # --- Substep (b): solve W so gateway g-sector matches impurity ---
        def res_g(x):
            params['W'] = x.copy()
            gw1 = gw1_observables(model, params, beta)
            gw2 = gw2_observables(model, params, beta)
            gw_n = (1-z)*gw1['n_g']  + z*gw2['n_g']
            gw_C = (1-z)*gw1['C_dg'] + z*gw2['C_dg']
            r_n  = gw_n - target_n_g
            r_C  = gw_C - target_C_g
            pen  = _sign_penalty(gw_C, target_C_g, sign_penalty)
            return np.concatenate([r_n, r_C, pen])

        sol_b = least_squares(res_g, x0=params['W'].copy(),
                              bounds=(lo, hi), method='trf',
                              ftol=1e-12, xtol=1e-12, max_nfev=2000)
        W_new_unmixed = sol_b.x.copy()
        params['W'] = W_new_unmixed

        # --- Substep (c): solve B_g so gw2 bond-h matches lat ---
        # bond-h matching: lat = gw2 directly
        lat = lattice_observables(model, params, beta)
        nhb_lat = lat['n_hb']; Cdhb_lat = lat['C_dhb']

        def res_bh(x):
            params['B_g'] = x.copy()
            gw2 = gw2_observables(model, params, beta)
            r_n = gw2['n_hb']  - nhb_lat
            r_C = gw2['C_hbd'] - Cdhb_lat
            pen = _sign_penalty(gw2['C_hbd'], Cdhb_lat, sign_penalty)
            return np.concatenate([r_n, r_C, pen])

        sol_c = least_squares(res_bh, x0=params['B_g'].copy(),
                              bounds=(lo, hi), method='trf',
                              ftol=1e-12, xtol=1e-12, max_nfev=2000)
        Bg_new_unmixed = sol_c.x.copy()
        params['B_g'] = Bg_new_unmixed

        # --- Substep (d): solve B_h so gw2 bond-g matches imp2 ---
        imp2 = imp2_observables(model, params, beta)
        ngb_imp = imp2['n_gb']; Cgbd_imp = imp2['C_gbd']

        def res_bg(x):
            params['B_h'] = x.copy()
            gw2 = gw2_observables(model, params, beta)
            r_n = gw2['n_gb']  - ngb_imp
            r_C = gw2['C_gbd'] - Cgbd_imp
            pen = _sign_penalty(gw2['C_gbd'], Cgbd_imp, sign_penalty)
            return np.concatenate


# =============================================================
# Main
# =============================================================

if __name__ == '__main__':
    M = 1
    U = 1.3
    t = 0.5
    Nk = 16          # small for first run -- bigger later if it converges
    T = 0.05         # start warm; the bond-ghost ED is the bottleneck
    beta = 1.0 / T

    cosx_arr, cosy_arr = (np.cos((np.arange(Nk)+0.5)*2*np.pi/Nk - np.pi),
                          np.cos((np.arange(Nk)+0.5)*2*np.pi/Nk - np.pi))
    # Build a flattened k-grid
    cosx, cosy = np.meshgrid(cosx_arr, cosy_arr, indexing='ij')
    cosx = cosx.ravel(); cosy = cosy.ravel()
    eps_k = -2.0 * t * (cosx + cosy)

    model = {
        'M': M, 'Mb': M, 'Mg': M, 'Mbg': M,
        'U': U, 't': t, 'z': 4,
        'ed': 0.0, 'mu': U/2.0, 'Sigma_inf': U/2.0,
        'cosx': cosx, 'cosy': cosy, 'eps_k': eps_k,
    }

    # Initial PH-symmetric parameters: all eta = 0, modest couplings
    params = {
        'eta':    np.array([0.0]),
        'W':      np.array([0.30]),
        'eta_b':  np.array([0.0]),
        'B_h':    np.array([0.10]),
        'eps':    np.array([0.0]),
        'V':      np.array([0.30]),
        'eps_bg': np.array([0.0]),
        'B_g':    np.array([0.10]),
    }

    print(f'M={M}, U={U}, t={t}, Nk={Nk} ({Nk*Nk} k-points), T={T} (beta={beta})')
    print(f'2-site impurity Fock dim = 2^{2*(2 + 2*M + 2*M)} = {1 << (2*(2+2*M+2*M))}')

    print('\n--- Initial single-shot observables ---')
    t0 = time.time()
    lat = lattice_observables(model, params, beta)
    print(f'lat: n_d={lat["n_d"]:.4f}  n_h={lat["n_h"]}  C_dh={lat["C_dh"]}  '
          f'n_hb={lat["n_hb"]}  C_dhb={lat["C_dhb"]}')
    gw1 = gw1_observables(model, params, beta)
    print(f'gw1: n_d={gw1["n_d"]:.4f}  n_h={gw1["n_h"]}  C_dh={gw1["C_dh"]}  '
          f'n_g={gw1["n_g"]}  C_dg={gw1["C_dg"]}')
    gw2 = gw2_observables(model, params, beta)
    print(f'gw2: n_d={gw2["n_d"]:.4f}  n_h={gw2["n_h"]}  C_dh={gw2["C_dh"]}  '
          f'n_g={gw2["n_g"]}  C_dg={gw2["C_dg"]}\n'
          f'     n_hb={gw2["n_hb"]}  C_hbd={gw2["C_hbd"]}  '
          f'n_gb={gw2["n_gb"]}  C_gbd={gw2["C_gbd"]}')
    print(f'(free-fermion sectors: {time.time()-t0:.1f}s)')

    t0 = time.time()
    imp1 = imp1_observables(model, params, beta)
    print(f'imp1: n_d={imp1["n_d"]:.4f}  D={imp1["double"]:.4f}  '
          f'n_g={imp1["n_g"]}  C_dg={imp1["C_dg"]}')
    print(f'(imp1 ED: {time.time()-t0:.1f}s)')

    t0 = time.time()
    imp2 = imp2_observables(model, params, beta)
    print(f'imp2: n_d={imp2["n_d"]:.4f}  D={imp2["double"]:.4f}  '
          f'n_g={imp2["n_g"]}  C_dg={imp2["C_dg"]}\n'
          f'     n_gb={imp2["n_gb"]}  C_gbd={imp2["C_gbd"]}')
    print(f'(imp2 ED: {time.time()-t0:.1f}s)')

    print('\n--- Running alternating SCF ---')
    params, hist = alt_scf_full(model, params, beta,
                                mix=0.5, tol=1e-7, maxiter=40, verbose=True)

    print('\n--- Converged ---')
    for k in ['eta', 'W', 'eta_b', 'B_h', 'eps', 'V', 'eps_bg', 'B_g']:
        print(f'  {k:8s} = {params[k]}')

    imp1 = imp1_observables(model, params, beta)
    imp2 = imp2_observables(model, params, beta)
    z = model['z']
    D_lat = (1-z) * imp1['double'] + z * imp2['double']
    print(f'\n  D_imp1   = {imp1["double"]:.5f}')
    print(f'  D_imp2   = {imp2["double"]:.5f}')
    print(f'  D_latt   = (1-z) D1 + z D2 = {D_lat:.5f}')
    print(f'  n_imp1   = {imp1["n_d"]:.5f}')
    print(f'  n_imp2   = {imp2["n_d"]:.5f}')
