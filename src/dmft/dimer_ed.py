"""Sector-blocked ED for the dimer ghost-DMFT impurity.

Orbital layout per spin:
  0=dA, 1=dB, 2..M+1=gA_0..gA_{M-1}, M+2..2M+1=gB_0..gB_{M-1}
  Norb = 2 + 2*M

(nup, ndn) sector decomposition — never allocates full Fock-space matrix.
Reuses _eigh and _hop_element from bond_ed.
"""

import numpy as np
from numpy.linalg import eigh
from itertools import combinations
from functools import lru_cache

from .bond_ed import _eigh, _hop_element


def _fermi_ed(e, beta):
    x = beta * np.asarray(e, dtype=float)
    out = np.empty_like(x)
    out[x > 500] = 0.0
    out[x < -500] = 1.0
    m = (x >= -500) & (x <= 500)
    out[m] = 1.0 / (np.exp(x[m]) + 1.0)
    return out


@lru_cache(maxsize=None)
def _get_dimer_sector(Norb, nup, ndn):
    """Cached (nup, ndn) sector basis.

    Returns (states, sidx) where states is int32 array and sidx is dict.
    """
    if nup < 0 or ndn < 0 or nup > Norb or ndn > Norb:
        return np.zeros(0, dtype=np.int32), {}
    up_states = np.array([sum(1 << b for b in bits)
                          for bits in combinations(range(Norb), nup)],
                         dtype=np.int32)
    dn_states = np.array([sum(1 << b for b in bits)
                          for bits in combinations(range(Norb), ndn)],
                         dtype=np.int32)
    states = np.empty(len(up_states) * len(dn_states), dtype=np.int32)
    k = 0
    for su in up_states:
        for sd in dn_states:
            states[k] = int(su) | (int(sd) << Norb)
            k += 1
    sidx = {int(s): i for i, s in enumerate(states)}
    return states, sidx


def dimer_impurity_obs(beta, mu, U, t_b, M, eps_g, V_g, t_g, hop):
    """Sector-blocked ED for the dimer impurity.

    Parameters
    ----------
    beta : float
    mu : float — chemical potential (-mu on d-sites)
    U : float — Hubbard U on dA and dB
    t_b : float — dA-dB hopping
    M : int — ghost families per site
    eps_g : array (M,)
    V_g : array (M,)
    t_g : array (M,) — inter-site ghost hopping gA↔gB (used if hop)
    hop : bool

    Returns
    -------
    dict with:
      docc        : float   — <n_dA_up n_dA_dn>
      n_dA        : float   — <n_dA> (both spins)
      n_dimer_imp : float   — 2 * n_dA (by symmetry)
      n_g         : array (M,) — <n_gA_m> / 2 (per spin)
      d_g         : array (M,) — <dA† gA_m> / 2 (per spin)
      ghop        : array (M,) — <gA† gB> / 2 (per spin, if hop; else 0)
    """
    eps_g = np.asarray(eps_g, dtype=float)
    V_g = np.asarray(V_g, dtype=float)
    t_g = np.asarray(t_g, dtype=float)

    Norb = 2 + 2 * M
    dA = 0; dB = 1
    gA = [2 + m for m in range(M)]
    gB = [2 + M + m for m in range(M)]

    global_E0 = None
    block_data = []

    for nup in range(Norb + 1):
        for ndn in range(Norb + 1):
            states, sidx = _get_dimer_sector(Norb, nup, ndn)
            D_ = len(states)
            if D_ == 0:
                continue

            # Occupation vectors: occ[mode, state_idx]
            occ = np.zeros((2 * Norb, D_), dtype=float)
            for mode in range(2 * Norb):
                occ[mode] = ((states >> mode) & 1).astype(float)

            # Diagonal Hamiltonian
            diag = np.zeros(D_)
            for sp in range(2):
                base = sp * Norb
                diag += (-mu) * (occ[base + dA] + occ[base + dB])
                for m in range(M):
                    diag += eps_g[m] * (occ[base + gA[m]] + occ[base + gB[m]])
            # Hubbard U on dA and dB
            diag += U * occ[dA] * occ[Norb + dA]       # n_dA_up * n_dA_dn
            diag += U * occ[dB] * occ[Norb + dB]       # n_dB_up * n_dB_dn

            H = np.diag(diag)

            # Off-diagonal hopping
            def _add_hop(src_orb, dst_orb, amp, H=H, states=states, sidx=sidx):
                for j, s in enumerate(states):
                    s2, sgn = _hop_element(int(s), dst_orb, src_orb)
                    if s2 and s2 in sidx:
                        H[sidx[s2], j] += amp * sgn

            for sp in range(2):
                base = sp * Norb
                # dA-dB hopping
                _add_hop(base + dA, base + dB, t_b)
                _add_hop(base + dB, base + dA, t_b)
                # d-ghost hybridization
                for m in range(M):
                    _add_hop(base + dA, base + gA[m], V_g[m])
                    _add_hop(base + gA[m], base + dA, V_g[m])
                    _add_hop(base + dB, base + gB[m], V_g[m])
                    _add_hop(base + gB[m], base + dB, V_g[m])
                # inter-site ghost hopping
                if hop:
                    for m in range(M):
                        _add_hop(base + gA[m], base + gB[m], t_g[m])
                        _add_hop(base + gB[m], base + gA[m], t_g[m])

            ev, evec = _eigh(H)
            E0 = float(ev.min())
            if global_E0 is None or E0 < global_E0:
                global_E0 = E0
            block_data.append((ev, evec, states, sidx, occ))

    # Accumulate thermal averages
    Z = 0.0
    num_docc = 0.0
    num_n_dA = 0.0
    num_n_g = np.zeros(M)
    num_d_g = np.zeros(M)
    num_ghop = np.zeros(M)

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        for ev, evec, states, sidx, occ in block_data:
            w = np.exp(np.clip(-beta * (ev - global_E0), -700, 700))
            Z += float(w.sum())
            psi2w = (evec**2) @ w

            def avg_diag(vec):
                return float(vec @ psi2w)

            def avg_hop(src_orb, dst_orb):
                val = 0.0
                for j, s in enumerate(states):
                    s2, sgn = _hop_element(int(s), dst_orb, src_orb)
                    if s2 and s2 in sidx:
                        i = sidx[s2]
                        val += sgn * float((evec[i, :] * evec[j, :]) @ w)
                return val

            # docc on dA: n_dA_up * n_dA_dn
            num_docc += avg_diag(occ[dA] * occ[Norb + dA])
            # n_dA (both spins)
            num_n_dA += avg_diag(occ[dA] + occ[Norb + dA])

            # Ghost correlators (both spins, divide by 2 at the end)
            for m in range(M):
                num_n_g[m] += avg_diag(occ[gA[m]] + occ[Norb + gA[m]])
                # d_g: sum spin-up + spin-down hops
                num_d_g[m] += (avg_hop(gA[m], dA)
                               + avg_hop(Norb + gA[m], Norb + dA))
                if hop:
                    num_ghop[m] += (avg_hop(gB[m], gA[m])
                                    + avg_hop(Norb + gB[m], Norb + gA[m]))

    if Z == 0.0:
        raise np.linalg.LinAlgError('zero partition function in dimer impurity')

    docc = num_docc / Z
    n_dA = num_n_dA / Z
    n_dimer_imp = 2.0 * n_dA  # by A↔B symmetry

    return dict(
        docc=float(docc),
        n_dA=float(n_dA),
        n_dimer_imp=float(n_dimer_imp),
        n_g=num_n_g / (2.0 * Z),      # per spin
        d_g=num_d_g / (2.0 * Z),      # per spin
        ghop=num_ghop / (2.0 * Z) if hop else np.zeros(M),
    )
