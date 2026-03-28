"""Fock-space exact diagonalization for the bond-scheme Ghost-DMFT.

Contains two ED kernels used by the bond solver:

1. ``impurity_statics`` — single-site interacting impurity (d + M g-ghosts)
   for static correlators ⟨n_g⟩, ⟨d†g⟩, double occupancy.

2. ``build_H2`` — two-site interacting cluster (d0, d1 + 2M g-ghosts + M gb-ghosts)
   block-diagonalized by (n_up, n_down) particle sectors.

This module uses bitmask integer Fock-space encoding with ``@lru_cache``
for sector-specific basis and transition data.  It is deliberately kept
separate from ``solvers/ed.py`` (which uses tuple-based Fock basis and
Lehmann Green's functions for the single-site Variant A/B loops).
"""

import numpy as np
from numpy.linalg import eigh
from itertools import combinations
from functools import lru_cache

# ═══════════════════════════════════════════════════════════
# GPU support (CuPy / cuSOLVER)
# ═══════════════════════════════════════════════════════════
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

_gpu_state = [False]   # _gpu_state[0]: GPU active flag (mutable, set by _init_gpu)
_GPU_DIM_THRESHOLD = 256  # only dispatch sectors above this dimension


def _init_gpu(requested):
    """Attempt to enable GPU. Returns True if GPU is active.

    Call this from the entry-point script before running the sweep:
      from dmft.bond_ed import _init_gpu
      _init_gpu(not args.no_gpu)
    """
    if not requested:
        return False
    if not _CUPY_AVAILABLE:
        import sys
        print('WARNING: GPU requested but CuPy not found; falling back to CPU.',
              file=sys.stderr)
        return False
    try:
        cp.cuda.Device(0).use()
        _gpu_state[0] = True
        return True
    except cp.cuda.runtime.CUDARuntimeError as e:
        import sys
        print(f'WARNING: GPU init failed ({e}); falling back to CPU.', file=sys.stderr)
        return False


def _eigh(H):
    """Diagonalize a real symmetric matrix, routing to GPU for large sectors."""
    if _gpu_state[0] and H.shape[0] >= _GPU_DIM_THRESHOLD:
        H_gpu = cp.asarray(H)
        ev_gpu, evec_gpu = cp.linalg.eigh(H_gpu)
        return cp.asnumpy(ev_gpu), cp.asnumpy(evec_gpu)
    return eigh(H)


# ═══════════════════════════════════════════════════════════
# Low-level Fock-space utilities
# ═══════════════════════════════════════════════════════════

def _fermi(e, beta):
    """Numerically stable Fermi function."""
    x = beta * np.asarray(e, dtype=float)
    out = np.empty_like(x)
    out[x > 500] = 0.0
    out[x < -500] = 1.0
    m = (x >= -500) & (x <= 500)
    out[m] = 1.0 / (np.exp(x[m]) + 1.0)
    return out


def _hop_element(s, i_o, j_o):
    """Apply c†_i c_j to Fock state s (bitmask). Returns (new_state, sign)."""
    s = int(s)
    if not ((s >> j_o) & 1) or ((s >> i_o) & 1):
        return 0, 0
    sgn = 1
    for k in range(min(i_o, j_o) + 1, max(i_o, j_o)):
        if (s >> k) & 1:
            sgn *= -1
    return (s ^ (1 << j_o)) ^ (1 << i_o), sgn


def _popcount(x):
    return bin(int(x)).count('1')


def _c_op(dim, mode):
    """Full annihilation operator matrix for a given mode."""
    C = np.zeros((dim, dim))
    for s in range(dim):
        if not ((s >> mode) & 1):
            continue
        # Count occupied modes above 'mode' for Jordan-Wigner sign
        bits_above = (s >> (mode + 1))
        sign = (-1.0) ** _popcount(bits_above)
        C[s ^ (1 << mode), s] = sign
    return C


# ═══════════════════════════════════════════════════════════
# Cached operator sets
# ═══════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def _get_impurity_ops(M):
    """Build and cache impurity annihilation/creation/number operators."""
    Norb = 1 + M
    Nmode = 2 * Norb
    dim = 1 << Nmode
    C = [_c_op(dim, mode) for mode in range(Nmode)]
    Cd = [op.T for op in C]
    n_ops = [Cd[m] @ C[m] for m in range(Nmode)]
    return Cd, C, n_ops


@lru_cache(maxsize=None)
def _get_H2_sector_cache(M, nup):
    """Cache Fock basis, occupation arrays, and transition data for one
    (n_up) sector of the two-site cluster.

    The cluster has nps = 2 + 3M modes per spin.  We fix n_up spin-up
    particles and n_down = nps - n_up spin-down particles (half-filling).
    """
    nps = 2 + 3 * M
    ndn = nps - nup
    if ndn < 0 or ndn > nps:
        return nps, np.zeros((2 * nps, 0)), [], {}

    up_states = np.array([sum(1 << b for b in bits)
                          for bits in combinations(range(nps), nup)],
                         dtype=np.int32)
    dn_states = np.array([sum(1 << b for b in bits)
                          for bits in combinations(range(nps), ndn)],
                         dtype=np.int32)
    D = len(up_states) * len(dn_states)
    states = np.empty(D, dtype=np.int32)
    k = 0
    for su in up_states:
        for sd in dn_states:
            states[k] = int(su) | (int(sd) << nps)
            k += 1
    sidx = {int(s): i for i, s in enumerate(states)}

    occ = np.zeros((2 * nps, D), dtype=float)
    for m_idx in range(2 * nps):
        occ[m_idx] = ((states >> m_idx) & 1).astype(float)

    def trans(src, dst):
        rows, cols, signs = [], [], []
        for i, s in enumerate(states):
            s2, sgn = _hop_element(s, dst, src)
            if s2 and s2 in sidx:
                rows.append(i)
                cols.append(sidx[s2])
                signs.append(sgn)
        return (np.array(rows, dtype=np.int32),
                np.array(cols, dtype=np.int32),
                np.array(signs, dtype=float))

    # Hamiltonian transitions
    ham_transitions = []
    for sp in range(2):
        base = sp * nps
        ham_transitions.append((base + 0, base + 1, 't'))
        ham_transitions.append((base + 1, base + 0, 't'))
        for l in range(M):
            for site in range(2):
                d_orb = base + site
                g_orb = base + 2 + 2 * l + site
                ham_transitions.append((d_orb, g_orb, ('V', l)))
                ham_transitions.append((g_orb, d_orb, ('V', l)))
            gb_orb = base + 2 + 2 * M + l
            for site in range(2):
                d_orb = base + site
                ham_transitions.append((d_orb, gb_orb, ('Bg', l)))
                ham_transitions.append((gb_orb, d_orb, ('Bg', l)))

    # All transitions needed for Hamiltonian + observables
    needed = set((src, dst) for src, dst, _ in ham_transitions)
    for l in range(M):
        needed.add((0 * nps + 2 + 2 * l, 0 * nps + 0))
        needed.add((0 * nps + 2 + 2 * M + l, 0 * nps + 0))
        needed.add((0 * nps + 2 + 2 * M + l, 0 * nps + 1))
    needed.add((0 * nps + 0, 0 * nps + 1))
    transition_map = {pair: trans(*pair) for pair in needed}

    return nps, occ, ham_transitions, transition_map


# ═══════════════════════════════════════════════════════════
# Single-site interacting impurity (static correlators)
# ═══════════════════════════════════════════════════════════

def impurity_statics(beta, eps, V, M, U, mu, ed=0.0):
    """Exact diag of the single-site interacting impurity.

    H_imp: d + M g-ghosts with Hubbard U on d.
    Mode ordering: mode = 2*orb + spin, orb=0 is d, orb=1..M are g-ghosts.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    eps : array, shape (M,)
        g-ghost energies.
    V : array, shape (M,)
        g-ghost hybridizations.
    M : int
        Number of g-ghost poles.
    U : float
        Hubbard interaction.
    mu : float
        Chemical potential.
    ed : float
        Impurity level energy.

    Returns
    -------
    ng : array, shape (M,)
        ⟨g_l† g_l⟩ (spin-up).
    dg : array, shape (M,)
        ⟨d†_up g_l,up⟩.
    docc : float
        ⟨n_d,up n_d,down⟩.
    """
    Cd, C, n_ops = _get_impurity_ops(M)

    def n(orb, spin):
        return n_ops[2 * orb + spin]

    ed_eff = ed - mu
    H = ed_eff * (n(0, 0) + n(0, 1)) + U * (n(0, 0) @ n(0, 1))
    for l in range(M):
        orb = 1 + l
        H += eps[l] * (n(orb, 0) + n(orb, 1))
        H += V[l] * (Cd[2 * orb + 0] @ C[0] + Cd[0] @ C[2 * orb + 0])
        H += V[l] * (Cd[2 * orb + 1] @ C[1] + Cd[1] @ C[2 * orb + 1])

    ev, evec = eigh(H)
    E0 = ev.min()
    with np.errstate(over='ignore', invalid='ignore'):
        boltz = np.clip(-beta * (ev - E0), -700, 700)
        w = np.exp(boltz)
        prob = w / w.sum()

        def avg(O):
            return float(np.sum(prob * np.diag(evec.T @ O @ evec)))

        ng = np.array([avg(n(1 + l, 0)) for l in range(M)])
        dg = np.array([avg(Cd[0] @ C[2 * (1 + l)]) for l in range(M)])
        docc = avg(n(0, 0) @ n(0, 1))
    return ng, dg, docc


# ═══════════════════════════════════════════════════════════
# Two-site cluster ED (block-diagonalized)
# ═══════════════════════════════════════════════════════════

def build_H2(beta, eps, V, epsb, Bg, dmu, M, U, mu, t, ed=0.0):
    """Two-site interacting cluster, block-diagonalized by (n_up, n_down).

    The cluster has nps = 2 + 3M modes per spin:
      d0, d1, g0_s0, g0_s1, ..., g{M-1}_s0, g{M-1}_s1, gb0, ..., gb{M-1}

    Block diag largest block for M=2: C(8,4)^2 = 4900 vs full C(16,8) = 12870.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    eps : array, shape (M,)
        g-ghost energies.
    V : array, shape (M,)
        g-ghost hybridizations.
    epsb : array, shape (M,)
        Bond gb-ghost energies.
    Bg : array, shape (M,)
        Bond gb-ghost hybridizations.
    dmu : float
        Chemical potential shift for the two-site cluster.
    M : int
        Number of ghost poles.
    U : float
        Hubbard interaction.
    mu : float
        Chemical potential.
    t : float
        Hopping between d0 and d1.
    ed : float
        Impurity level energy.

    Returns
    -------
    ng : array, shape (M,)
        ⟨g_l† g_l⟩ on site 0 (spin-up).
    dg : array, shape (M,)
        ⟨d†_0 g_l⟩ on site 0.
    ngb : array, shape (M,)
        ⟨gb_l† gb_l⟩.
    dgb : array, shape (M,)
        ⟨d†_0 gb_l⟩ + ⟨d†_1 gb_l⟩.
    docc : float
        Double occupancy on site 0.
    hop : float
        ⟨d†_0 d_1⟩ hopping correlator.
    nsite : float
        ⟨n_d0_up + n_d1_up⟩ (spin-up site occupancy).
    """
    nps = 2 + 3 * M
    ed_eff = ed - mu - dmu

    global_E0 = None
    block_data = []

    for nup in range(nps + 1):
        nps_b, occ, ham_transitions, transition_map = _get_H2_sector_cache(M, nup)
        D_ = occ.shape[1]
        if D_ == 0:
            continue

        # Diagonal part
        diag = np.zeros(D_)
        for sp in range(2):
            base = sp * nps_b
            diag += ed_eff * (occ[base + 0] + occ[base + 1])
            for l in range(M):
                diag += eps[l] * (occ[base + 2 + 2 * l] + occ[base + 2 + 2 * l + 1])
                diag += epsb[l] * occ[base + 2 + 2 * M + l]
        diag += U * occ[0 * nps_b + 0] * occ[1 * nps_b + 0]
        diag += U * occ[0 * nps_b + 1] * occ[1 * nps_b + 1]

        H = np.diag(diag)
        for src, dst, ampinfo in ham_transitions:
            rows, cols, signs = transition_map[(src, dst)]
            if len(rows) == 0:
                continue
            if not isinstance(ampinfo, tuple):
                amp = -t
            elif ampinfo[0] == 'V':
                amp = V[ampinfo[1]]
            else:
                amp = Bg[ampinfo[1]]
            H[rows, cols] += amp * signs

        ev, evec = _eigh(H)   # GPU-accelerated for large sectors
        E0 = float(ev.min())
        if global_E0 is None or E0 < global_E0:
            global_E0 = E0
        block_data.append((ev, evec, occ, transition_map, nps_b))

    # Accumulate thermal averages across blocks
    Z = 0.0
    num_ng = np.zeros(M)
    num_ngb = np.zeros(M)
    num_dg = np.zeros(M)
    num_dgb = np.zeros(M)
    num_docc = 0.0
    num_hop = 0.0
    num_nsite = 0.0

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        for ev, evec, occ, transition_map, nps_b in block_data:
            boltz_arg = -beta * (ev - global_E0)
            boltz_arg = np.clip(boltz_arg, -700, 700)
            w = np.exp(boltz_arg)
            Z += float(w.sum())
            psi2w = (evec ** 2) @ w

            def avg_diag(vec):
                return float(vec @ psi2w)

            def avg_hop(src, dst):
                rows, cols, signs = transition_map[(src, dst)]
                if len(rows) == 0:
                    return 0.0
                return float((signs @ (evec[rows, :] * evec[cols, :])) @ w)

            for l in range(M):
                num_ng[l] += avg_diag(occ[0 * nps_b + 2 + 2 * l])
                num_ngb[l] += avg_diag(occ[0 * nps_b + 2 + 2 * M + l])
                num_dg[l] += avg_hop(0 * nps_b + 2 + 2 * l, 0 * nps_b + 0)
                num_dgb[l] += (avg_hop(0 * nps_b + 2 + 2 * M + l, 0 * nps_b + 0)
                               + avg_hop(0 * nps_b + 2 + 2 * M + l, 0 * nps_b + 1))
            num_docc += avg_diag(occ[0 * nps_b + 0] * occ[1 * nps_b + 0])
            num_hop += avg_hop(0 * nps_b + 0, 0 * nps_b + 1)
            num_nsite += avg_diag(occ[0 * nps_b + 0] + occ[1 * nps_b + 0])

    if Z == 0.0:
        raise np.linalg.LinAlgError('zero partition function in H2')

    return (num_ng / Z, num_dg / Z, num_ngb / Z, num_dgb / Z,
            num_docc / Z, num_hop / Z, num_nsite / Z)
