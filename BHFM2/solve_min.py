#!/usr/bin/env python3
"""Minimal ghost-DMFT: bonding-only ghosts, no anti-bond.

Parameter count at M=1, Mb=1:
  Lattice (bond-h only): eta, W, eta_b, B_h                = 4
  Imp1:   eps1, V1, eta1, W1                                = 4
  Imp2:   eps2, V2, eta2, W2, eta_b2, B_h2, eps_b, B_g     = 8
  mu                                                        = 1
  TOTAL = 17

Residual dimension = 15 eqs at n_moments=4.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
from dataclasses import dataclass, replace
from numpy.linalg import eigh
from scipy.optimize import least_squares

from ghost_dmft_bond import build_sector
from ed_fast import (build_H_sector_fast, make_lookup,
                      expect_n_fast, expect_double_fast, expect_cdag_c_fast)


@dataclass
class ModelParamsMin:
    t: float = 0.5
    eps_d: float = 0.0
    U: float = 1.3
    z: float = 0.5
    filling_target: float = 1.0
    Sigma_inf: float = 0.65
    beta: float = 0.5
    Nk: int = 16
    use_y_bond: bool = True
    n_moments: int = 4


@dataclass
class SParMin:
    eta: np.ndarray
    W: np.ndarray
    eta_b: np.ndarray
    B_h: np.ndarray
    eps1: np.ndarray
    V1: np.ndarray
    eta1: np.ndarray
    W1: np.ndarray
    eps2: np.ndarray
    V2: np.ndarray
    eta2: np.ndarray
    W2: np.ndarray
    eta_b2: np.ndarray
    B_h2: np.ndarray
    eps_b: np.ndarray
    B_g: np.ndarray
    mu: float


class CodecMin:
    def __init__(self, M, Mb):
        self.M = M; self.Mb = Mb
    def _fields(self):
        M, Mb = self.M, self.Mb
        return [('eta', M), ('W', M), ('eta_b', Mb), ('B_h', Mb),
                ('eps1', M), ('V1', M), ('eta1', M), ('W1', M),
                ('eps2', M), ('V2', M), ('eta2', M), ('W2', M),
                ('eta_b2', Mb), ('B_h2', Mb),
                ('eps_b', Mb), ('B_g', Mb)]
    @property
    def size(self):
        return 10*self.M + 6*self.Mb + 1
    def pack(self, p):
        parts = []
        for name, _ in self._fields():
            parts.append(np.asarray(getattr(p, name), dtype=float))
        parts.append(np.array([p.mu], dtype=float))
        return np.concatenate(parts)
    def unpack(self, x):
        i = 0; kw = {}
        for name, n in self._fields():
            kw[name] = np.array(x[i:i+n], dtype=float)
            i += n
        return SParMin(mu=float(x[i]), **kw)


def init_min(M, Mb, W0=0.3, V0=0.3, B0=0.1, base_mu=0.65):
    return SParMin(
        eta=np.zeros(M), W=np.full(M, W0),
        eta_b=np.zeros(Mb), B_h=np.full(Mb, B0),
        eps1=np.zeros(M), V1=np.full(M, V0),
        eta1=np.zeros(M), W1=np.full(M, W0),
        eps2=np.zeros(M), V2=np.full(M, V0),
        eta2=np.zeros(M), W2=np.full(M, W0),
        eta_b2=np.zeros(Mb), B_h2=np.full(Mb, B0),
        eps_b=np.zeros(Mb), B_g=np.full(Mb, B0),
        mu=base_mu)


def _fermi(x, beta):
    xc = np.clip(beta*x, -700, 700)
    return 1.0/(np.exp(xc)+1.0)

def _dm(H, beta):
    e, U = eigh(H)
    return (U * _fermi(e, beta)[None,:]) @ U.T


def lat_obs(p, mp, M, Mb):
    """Lattice: [d, h(M), hbx(Mb), hby(Mb)]  — bonding only."""
    Nk = mp.Nk
    kvals = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    kx, ky = np.meshgrid(kvals, kvals, indexing='ij')
    eps_k = -2.0*mp.t*(np.cos(kx)+np.cos(ky))
    ckx = np.cos(kx/2.0); cky = np.cos(ky/2.0)
    wk = 1.0/(Nk*Nk)
    ndir = 2 if mp.use_y_bond else 1
    n_orb = 1 + M + Mb*ndir

    hh = np.zeros(M); dh = np.zeros(M)
    hbp_hbp = np.zeros(Mb); hbp_d = np.zeros(Mb)
    dens = 0.0; bond_x = 0.0

    for ix in range(Nk):
        for iy in range(Nk):
            H = np.zeros((n_orb, n_orb))
            # physical site: bare dispersion + static Hartree shift - mu
            H[0, 0] = eps_k[ix, iy] + mp.eps_d + mp.Sigma_inf - p.mu
            off = 1
            for a in range(M):
                # h-ghost on-site energy (pole position of local Sigma contribution)
                H[off+a, off+a] = p.eta[a]
                # h-ghost hybridization to physical site (amplitude of that pole)
                H[0, off+a] = p.W[a]; H[off+a, 0] = p.W[a]
            off += M
            for a in range(Mb):
                # bond ghost on-site energy (pole of non-local Sigma_nn)
                H[off+a, off+a] = p.eta_b[a]
                # bond ghost coupling: B_h * cos(kx/2) — structure factor from bond sitting between sites
                amp = p.B_h[a]*ckx[ix, iy]
                H[0, off+a] = amp; H[off+a, 0] = amp
            off += Mb
            if mp.use_y_bond:
                for a in range(Mb):
                    # same bond ghost, y-direction — shares eta_b with x channel (same pole)
                    H[off+a, off+a] = p.eta_b[a]
                    # y-direction structure factor cos(ky/2) restores square lattice symmetry
                    amp = p.B_h[a]*cky[ix, iy]
                    H[0, off+a] = amp; H[off+a, 0] = amp
                off += Mb
            # thermal one-particle density matrix C[i,j] = <c†_i c_j>
            C = _dm(H, mp.beta)
            spin = 2.0  # factor of 2 for spin degeneracy
            dens += spin * C[0, 0] * wk                        # total filling <n>
            for a in range(M):
                hh[a] += spin * C[1+a, 1+a] * wk              # h-ghost occupation
                dh[a] += spin * C[0, 1+a] * wk                # d–h coherence
            off2 = 1 + M
            for a in range(Mb):
                idx_x = off2 + a
                hbp_hbp[a] += spin * C[idx_x, idx_x] * wk / ndir          # bond ghost occupation
                # extra cos(kx/2) converts C[bond,d] back to real-space bond coherence
                hbp_d[a] += spin * 2.0 * ckx[ix, iy] * C[idx_x, 0] * wk / ndir
            if mp.use_y_bond:
                off2_y = 1 + M + Mb
                for a in range(Mb):
                    idx_y = off2_y + a
                    hbp_hbp[a] += spin * C[idx_y, idx_y] * wk / ndir      # y-bond ghost occupation
                    hbp_d[a] += spin * 2.0 * cky[ix, iy] * C[idx_y, 0] * wk / ndir
            # nearest-neighbor kinetic bond order <c†_i c_{i+x}> ~ sum_k cos(kx) n_k
            bond_x += spin * np.cos(kx[ix, iy]) * C[0, 0] * wk
    return dict(hh=hh, dh=dh, hbp_hbp=hbp_hbp, hbp_d=hbp_d,
                 dens=float(dens), bond_x=float(bond_x))


def gw1_obs(p, mp, M):
    """gw1: [d, g(M), h(M)] (single site, no bonds)"""
    # Ghost (mean-field) side of the single-site self-consistency — no U, no bond.
    # Sigma_inf included here as static Hartree shift; U is absent by construction.
    n_orb = 1 + 2*M
    H = np.zeros((n_orb, n_orb))
    H[0, 0] = mp.eps_d + mp.Sigma_inf - p.mu  # physical site with Hartree shift
    og = 1; oh = 1+M
    for a in range(M):
        H[og+a, og+a] = p.eps1[a]              # Anderson bath level (g-type)
        H[0, og+a] = p.V1[a]; H[og+a, 0] = p.V1[a]   # site-bath hybridization
        H[oh+a, oh+a] = p.eta1[a]              # h-ghost pole energy
        H[0, oh+a] = p.W1[a]; H[oh+a, 0] = p.W1[a]   # site-ghost hybridization
    C = _dm(H, mp.beta)
    spin = 2.0
    return dict(
        gg=np.array([spin*C[og+a, og+a] for a in range(M)]),   # g-bath occupation
        dg=np.array([spin*C[0, og+a] for a in range(M)]),       # d–g coherence
        hh=np.array([spin*C[oh+a, oh+a] for a in range(M)]),   # h-ghost occupation
        dh=np.array([spin*C[0, oh+a] for a in range(M)]),       # d–h coherence
        dens=float(spin*C[0, 0]))


def gw2_obs(p, mp, M, Mb):
    """gw2: [d1, d2, h2_1(M), h2_2(M), hb(Mb), g2_1(M), g2_2(M), gb(Mb)]
    bond couplings: (B/2)(d1+d2).  NO anti-bond."""
    # Ghost (mean-field) side of the bond self-consistency — two-site cluster, no U.
    # d1-d2 connected by -t; each site gets its own local bath/ghost plus shared bond bath/ghost.
    d1, d2 = 0, 1
    oh1 = 2; oh2 = oh1+M; ohb = oh2+M
    og1 = ohb+Mb; og2 = og1+M; ogb = og2+M
    n_orb = ogb + Mb
    H = np.zeros((n_orb, n_orb))
    shift = mp.eps_d + mp.Sigma_inf - p.mu  # Hartree shift included (ghost side)
    H[d1, d1] = shift; H[d2, d2] = shift
    H[d1, d2] = H[d2, d1] = -mp.t          # nearest-neighbor hopping between the two sites
    for a in range(M):
        # h-ghost: one per site, same pole energy eta2, couples locally via W2
        H[oh1+a, oh1+a] = p.eta2[a]; H[oh2+a, oh2+a] = p.eta2[a]
        H[d1, oh1+a] = p.W2[a]; H[oh1+a, d1] = p.W2[a]
        H[d2, oh2+a] = p.W2[a]; H[oh2+a, d2] = p.W2[a]
    for b in range(Mb):
        # bond h-ghost: single orbital coupled to (d1+d2)/2 — impurity mirror of lattice B_h
        H[ohb+b, ohb+b] = p.eta_b2[b]
        amp = p.B_h2[b]/2.0                 # /2 because coupling splits equally to d1 and d2
        H[d1, ohb+b] = amp; H[ohb+b, d1] = amp
        H[d2, ohb+b] = amp; H[ohb+b, d2] = amp
    for a in range(M):
        # Anderson bath: one per site, same energy eps2, couples locally via V2
        H[og1+a, og1+a] = p.eps2[a]; H[og2+a, og2+a] = p.eps2[a]
        H[d1, og1+a] = p.V2[a]; H[og1+a, d1] = p.V2[a]
        H[d2, og2+a] = p.V2[a]; H[og2+a, d2] = p.V2[a]
    for b in range(Mb):
        # bond Anderson bath: single orbital coupled to (d1+d2)/2 — bath for the bond channel
        H[ogb+b, ogb+b] = p.eps_b[b]
        amp = p.B_g[b]/2.0
        H[d1, ogb+b] = amp; H[ogb+b, d1] = amp
        H[d2, ogb+b] = amp; H[ogb+b, d2] = amp
    C = _dm(H, mp.beta)
    spin = 2.0
    hh = np.zeros(M); dh = np.zeros(M); gg = np.zeros(M); dg = np.zeros(M)
    for a in range(M):
        i1, i2 = oh1+a, oh2+a
        hh[a] = spin*0.5*(C[i1,i1] + C[i2,i2])   # h-ghost occupation averaged over sites
        dh[a] = spin*0.5*(C[d1,i1] + C[d2,i2])   # d–h coherence averaged over sites
        j1, j2 = og1+a, og2+a
        gg[a] = spin*0.5*(C[j1,j1] + C[j2,j2])   # g-bath occupation averaged over sites
        dg[a] = spin*0.5*(C[d1,j1] + C[d2,j2])   # d–g coherence averaged over sites
    hbp_hbp = np.zeros(Mb); hbp_dplus = np.zeros(Mb)
    gbp_gbp = np.zeros(Mb); gbp_dplus = np.zeros(Mb)
    for b in range(Mb):
        ip = ohb+b
        hbp_hbp[b] = spin*C[ip,ip]                         # bond h-ghost occupation
        hbp_dplus[b] = spin*(C[d1,ip] + C[d2,ip])          # bond h-ghost to bonding combination coherence
        jp = ogb+b
        gbp_gbp[b] = spin*C[jp,jp]                         # bond Anderson bath occupation
        gbp_dplus[b] = spin*(C[d1,jp] + C[d2,jp])          # bond bath to bonding combination coherence
    return dict(hh=hh, dh=dh, gg=gg, dg=dg,
                 hbp_hbp=hbp_hbp, hbp_dplus=hbp_dplus,
                 gbp_gbp=gbp_gbp, gbp_dplus=gbp_dplus,
                 dens_per_site=float(spin*0.5*(C[d1,d1]+C[d2,d2])),
                 bond_kinetic_per_bond=float(spin*C[d1,d2]))


def imp1_obs(p, mp, M):
    """imp1 ED: [d, g(M)] with U on d."""
    # Interacting single-site impurity solved by full ED.
    # No Sigma_inf here — U is treated exactly, not via static Hartree.
    # Counterpart to gw1_obs: same observables, but with correlations.
    N_orb = 1 + M
    h1 = np.zeros((N_orb, N_orb))
    h1[0, 0] = mp.eps_d - p.mu          # bare impurity level, no Hartree shift
    for a in range(M):
        h1[1+a, 1+a] = p.eps1[a]        # Anderson bath level
        h1[0, 1+a] = p.V1[a]; h1[1+a, 0] = p.V1[a]   # bath hybridization
    U_orbs = np.array([0], dtype=np.int64)  # U acts only on the physical site
    all_states = []
    # build all (N_up, N_dn) sectors, diagonalize each block separately
    for N_up in range(N_orb+1):
        for N_dn in range(N_orb+1):
            bnp, _ = build_sector(N_orb, N_up, N_dn)
            if len(bnp) == 0: continue
            basis = bnp.astype(np.int64)
            keys, vals = make_lookup(basis)
            H = build_H_sector_fast(h1, mp.U, U_orbs, N_orb, basis, keys, vals)
            e, ev = eigh(H)
            for j in range(len(e)):
                all_states.append((float(e[j]), ev[:, j], basis, keys, vals))
    E0 = min(s[0] for s in all_states)
    w = np.array([np.exp(-mp.beta*(s[0]-E0)) for s in all_states])
    w /= w.sum()   # canonical thermal weights
    n_d = 0.0; D = 0.0; gg = np.zeros(M); dg = np.zeros(M)
    for (E, psi, basis, keys, vals), wi in zip(all_states, w):
        if wi < 1e-14: continue
        n_d += wi * expect_n_fast(0, psi, basis, N_orb)               # impurity filling
        D += wi * expect_double_fast(0, psi, basis, N_orb)             # double occupancy
        for a in range(M):
            gg[a] += wi * expect_n_fast(1+a, psi, basis, N_orb)       # bath occupation
            dg[a] += wi * expect_cdag_c_fast(1+a, 0, psi, basis, keys, vals, N_orb)  # d–g coherence
    return dict(dens=n_d, double_occ=D, gg=gg, dg=dg)


def imp2_obs(p, mp, M, Mb):
    """imp2 ED: [d1, d2, g1(M), g2(M), gb(Mb)]  NO anti-bond g."""
    # Interacting two-site bond impurity solved by full ED.
    # Counterpart to gw2_obs: same cluster geometry (d1-d2 + baths), but U treated exactly.
    # Note: h-ghost orbitals absent here — only Anderson bath (g-type) and bond bath (gb).
    # h-ghosts are a ghost-DMFT device; they don't appear in the interacting ED problem.
    d1, d2 = 0, 1
    og1 = 2; og2 = og1+M; ogb = og2+M
    N_orb = ogb + Mb
    h1 = np.zeros((N_orb, N_orb))
    h1[d1, d1] = mp.eps_d - p.mu   # bare site energy, no Hartree (U explicit)
    h1[d2, d2] = mp.eps_d - p.mu
    h1[d1, d2] = -mp.t; h1[d2, d1] = -mp.t  # intra-cluster hopping
    for a in range(M):
        i1, i2 = og1+a, og2+a
        h1[i1, i1] = p.eps2[a]; h1[i2, i2] = p.eps2[a]   # local Anderson bath per site
        h1[d1, i1] = p.V2[a]; h1[i1, d1] = p.V2[a]       # site-bath hybridization
        h1[d2, i2] = p.V2[a]; h1[i2, d2] = p.V2[a]
    for b in range(Mb):
        ip = ogb+b
        h1[ip, ip] = p.eps_b[b]   # bond Anderson bath level
        amp = p.B_g[b]/2.0        # couples to bonding combination (d1+d2)/2
        h1[d1, ip] = amp; h1[ip, d1] = amp
        h1[d2, ip] = amp; h1[ip, d2] = amp
    U_orbs = np.array([d1, d2], dtype=np.int64)  # U acts on both physical sites
    all_states = []
    for N_up in range(N_orb+1):
        for N_dn in range(N_orb+1):
            bnp, _ = build_sector(N_orb, N_up, N_dn)
            if len(bnp) == 0: continue
            basis = bnp.astype(np.int64)
            keys, vals = make_lookup(basis)
            H = build_H_sector_fast(h1, mp.U, U_orbs, N_orb, basis, keys, vals)
            e, ev = eigh(H)
            for j in range(len(e)):
                all_states.append((float(e[j]), ev[:, j], basis, keys, vals))
    E0 = min(s[0] for s in all_states)
    w = np.array([np.exp(-mp.beta*(s[0]-E0)) for s in all_states])
    w /= w.sum()   # canonical thermal weights
    n_d1 = 0.0; n_d2 = 0.0; D_avg = 0.0
    gg = np.zeros(M); dg = np.zeros(M)
    gbp_gbp = np.zeros(Mb); gbp_dplus = np.zeros(Mb)
    bond_kin = 0.0
    for (E, psi, basis, keys, vals), wi in zip(all_states, w):
        if wi < 1e-14: continue
        n_d1 += wi * expect_n_fast(d1, psi, basis, N_orb)   # site 1 filling
        n_d2 += wi * expect_n_fast(d2, psi, basis, N_orb)   # site 2 filling
        D_avg += wi * 0.5 * (expect_double_fast(d1, psi, basis, N_orb) +
                               expect_double_fast(d2, psi, basis, N_orb))   # avg double occ
        bond_kin += wi * expect_cdag_c_fast(d1, d2, psi, basis, keys, vals, N_orb)  # <c†_d1 c_d2>
        for a in range(M):
            i1, i2 = og1+a, og2+a
            gg[a] += wi * 0.5*(expect_n_fast(i1, psi, basis, N_orb) +
                                 expect_n_fast(i2, psi, basis, N_orb))   # bath occupation avg
            dg[a] += wi * 0.5*(expect_cdag_c_fast(i1, d1, psi, basis, keys, vals, N_orb) +
                                 expect_cdag_c_fast(i2, d2, psi, basis, keys, vals, N_orb))  # d–g coherence avg
        for b in range(Mb):
            ip = ogb+b
            gbp_gbp[b] += wi * expect_n_fast(ip, psi, basis, N_orb)    # bond bath occupation
            gbp_dplus[b] += wi * (expect_cdag_c_fast(ip, d1, psi, basis, keys, vals, N_orb) +
                                    expect_cdag_c_fast(ip, d2, psi, basis, keys, vals, N_orb))  # bond bath to bonding combination
    return dict(n_d1=n_d1, n_d2=n_d2, dens_per_site=0.5*(n_d1+n_d2),
                 double_occ_per_site=D_avg, gg=gg, dg=dg,
                 gbp_gbp=gbp_gbp, gbp_dplus=gbp_dplus,
                 bond_kinetic_per_bond=bond_kin)


def mom_lat(p, m):
    # m-th moment of the lattice ghost self-energy: sum_a amplitude_a^2 * pole_a^m
    # Local h-ghost contribution + bond h-ghost contribution (both enter Sigma)
    s = float(np.sum(p.W**2 * p.eta**m))
    s += float(np.sum(p.B_h**2 * p.eta_b**m))
    return s

def mom_imp1(p, m):
    # m-th moment from single-site impurity ghost (h-channel only)
    return float(np.sum(p.W1**2 * p.eta1**m))

def mom_imp2(p, m):
    # m-th moment from bond impurity ghost (local h + bond h contributions)
    s = float(np.sum(p.W2**2 * p.eta2**m))
    s += float(np.sum(p.B_h2**2 * p.eta_b2**m))
    return s


def residual_min(p, mp_base, M, Mb):
    """Residual with bond-h and bond-g (bonding only, no anti-bond)."""
    z = mp_base.z
    imp1 = imp1_obs(p, mp_base, M)
    imp2 = imp2_obs(p, mp_base, M, Mb)
    n_avg = (1-z)*imp1['dens'] + z*imp2['dens_per_site']
    Sigma_inf_iter = mp_base.U * n_avg / 2.0
    mp = replace(mp_base, Sigma_inf=Sigma_inf_iter)
    lat = lat_obs(p, mp, M, Mb)
    gw1 = gw1_obs(p, mp, M)
    gw2 = gw2_obs(p, mp, M, Mb)

    r = []
    r.extend((lat['hh'] - ((1-z)*gw1['hh'] + z*gw2['hh'])).tolist())
    r.extend((lat['dh'] - ((1-z)*gw1['dh'] + z*gw2['dh'])).tolist())
    r.extend((lat['hbp_hbp'] - gw2['hbp_hbp']).tolist())
    r.extend((lat['hbp_d']   - gw2['hbp_dplus']).tolist())
    r.extend((imp1['gg'] - gw1['gg']).tolist())
    r.extend((imp1['dg'] - gw1['dg']).tolist())
    r.extend((imp2['gg'] - gw2['gg']).tolist())
    r.extend((imp2['dg'] - gw2['dg']).tolist())
    r.extend((imp2['gbp_gbp']   - gw2['gbp_gbp']).tolist())
    r.extend((imp2['gbp_dplus'] - gw2['gbp_dplus']).tolist())
    r.append(n_avg - mp_base.filling_target)
    r.append(lat['dens'] - mp_base.filling_target)
    r.append(lat['bond_x'] - imp2['bond_kinetic_per_bond'])
    for m in range(mp_base.n_moments):
        lhs = mom_lat(p, m)
        rhs = (1-z)*mom_imp1(p, m) + z*mom_imp2(p, m)
        r.append(lhs - rhs)
    return np.array(r, dtype=float)


def make_bounds_min(M, Mb, E=5.0, C=3.0, mu_lo=-5.0, mu_hi=5.0):
    fields = [
        ('eta', M, -E, E), ('W', M, 1e-8, C),
        ('eta_b', Mb, -E, E), ('B_h', Mb, 1e-8, C),
        ('eps1', M, -E, E), ('V1', M, 1e-8, C),
        ('eta1', M, -E, E), ('W1', M, 1e-8, C),
        ('eps2', M, -E, E), ('V2', M, 1e-8, C),
        ('eta2', M, -E, E), ('W2', M, 1e-8, C),
        ('eta_b2', Mb, -E, E), ('B_h2', Mb, 1e-8, C),
        ('eps_b', Mb, -E, E), ('B_g', Mb, 1e-8, C),
    ]
    lo = []; hi = []
    for _, n, l, h in fields:
        lo.extend([l]*n); hi.extend([h]*n)
    lo.append(mu_lo); hi.append(mu_hi)
    return np.array(lo), np.array(hi)


def solve_min(mp0, M, Mb, filling_target, p_init=None,
               max_nfev=400, ftol=1e-12, xtol=1e-12, verbose=False):
    mp = replace(mp0, filling_target=filling_target)
    codec = CodecMin(M, Mb)
    lo, hi = make_bounds_min(M, Mb)
    if p_init is None:
        mu0 = mp.U * filling_target / 2.0
        p_init = init_min(M, Mb, W0=0.3, V0=0.3, B0=0.1, base_mu=mu0)
    x0 = codec.pack(p_init)
    x0 = np.clip(x0, lo + 1e-8, hi - 1e-8)
    t0 = time.time()
    niter = [0]
    def fn(x):
        niter[0] += 1
        pp = codec.unpack(x)
        r = residual_min(pp, mp, M, Mb)
        if verbose and niter[0] % 30 == 1:
            print(f'    nfev={niter[0]:3d}  ||r||={np.linalg.norm(r):.3e}  '
                  f't={time.time()-t0:.0f}s', flush=True)
        return r
    sol = least_squares(fn, x0, method='trf', bounds=(lo, hi),
                        ftol=ftol, xtol=xtol, max_nfev=max_nfev, verbose=0)
    p = codec.unpack(sol.x)
    z = mp.z
    imp1 = imp1_obs(p, mp, M)
    imp2 = imp2_obs(p, mp, M, Mb)
    n_avg = (1-z)*imp1['dens'] + z*imp2['dens_per_site']
    Sigma_inf_final = mp.U * n_avg / 2.0
    mp_final = replace(mp, Sigma_inf=Sigma_inf_final)
    lat = lat_obs(p, mp_final, M, Mb)
    D_lat = (1-z)*imp1['double_occ'] + z*imp2['double_occ_per_site']
    info = dict(n_target=filling_target, n_avg=n_avg, n_lat=lat['dens'],
                 n_imp1=imp1['dens'], n_imp2=imp2['dens_per_site'],
                 D_imp1=imp1['double_occ'], D_imp2=imp2['double_occ_per_site'],
                 D_lat=D_lat, mu=p.mu, Sigma_inf=Sigma_inf_final,
                 resnorm=float(np.linalg.norm(sol.fun)),
                 nfev=sol.nfev, wall=time.time()-t0,
                 W=p.W[0], V1=p.V1[0], V2=p.V2[0],
                 W1=p.W1[0], W2=p.W2[0],
                 B_h=p.B_h[0], B_h2=p.B_h2[0], B_g=p.B_g[0],
                 eta=p.eta[0], eta_b=p.eta_b[0],
                 eps1=p.eps1[0], eps2=p.eps2[0], eps_b=p.eps_b[0],
                 eta1=p.eta1[0], eta2=p.eta2[0])
    return p, mp_final, info


if __name__ == '__main__':
    mp0 = ModelParamsMin(U=1.3, t=0.5, beta=0.5, Nk=16)
    print('Minimal solver test at U=1.3, T=2, half filling')
    print(f'  n_params = {CodecMin(1,1).size}, n_eqs = {len(residual_min(init_min(1,1), mp0, 1, 1))}')
    p, mp_f, info = solve_min(mp0, 1, 1, 1.0, max_nfev=300, verbose=True)
    print(f'\nResult: ||r||={info["resnorm"]:.3e}  wall={info["wall"]:.1f}s  nfev={info["nfev"]}')
    print(f'  n_avg={info["n_avg"]:.6f}  mu={info["mu"]:+.4f}  Sigma_inf={info["Sigma_inf"]:+.4f}')
    print(f'  D_imp1={info["D_imp1"]:.5f} D_imp2={info["D_imp2"]:.5f} D_lat={info["D_lat"]:.5f}')
    print(f'  W={info["W"]:.4f} V2={info["V2"]:.4f} B_h={info["B_h"]:.4f} B_g={info["B_g"]:.4f}')
