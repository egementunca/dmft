"""Exact Diagonalization (ED) solver for the Anderson impurity model.

Solves the interacting impurity problem by constructing the full many-body
Hamiltonian in Fock space and diagonalizing it exactly.

H_imp = (eps_d + sigma_inf - mu) sum_s n_{d,s} + U n_{d,up} n_{d,down}
        + sum_{l,s} eps_l n_{g_l,s} + sum_{l,s} (V_l d_s^dag g_{l,s} + h.c.)

The Hilbert space dimension is 4^{n_orb} where n_orb = 1 + M_g (per spin),
but block-diagonalization by (N_up, N_down) sectors makes this tractable
for M_g <= 4.

Green's function is computed via the Lehmann representation:
    G(iw) = (1/Z) sum_{i,j} |<i|d|j>|^2 / (iw - (E_j - E_i))
            * (exp(-beta E_i) + exp(-beta E_j))
"""

import numpy as np
from itertools import combinations
from .base import ImpuritySolver


class EDSolver(ImpuritySolver):
    """Exact Diagonalization impurity solver."""

    def solve(self, iw, mu, eps_d, U, V, eps, beta, sigma_inf):
        n_orb = 1 + len(eps)  # impurity + bath

        # Build one-body Hamiltonian (per spin, same for up and down)
        # Note: sigma_inf is NOT in H_imp — it's part of the self-energy output.
        # The impurity Hamiltonian has the bare eps_d level.
        h = np.zeros((n_orb, n_orb))
        h[0, 0] = eps_d - mu
        for l in range(len(eps)):
            h[1 + l, 1 + l] = eps[l]
            h[0, 1 + l] = V[l]
            h[1 + l, 0] = np.conj(V[l])

        # Build and diagonalize each (N_up, N_down) sector
        sectors = {}
        all_energies = []
        all_states = []
        all_sectors = []

        for n_up in range(n_orb + 1):
            for n_down in range(n_orb + 1):
                basis = _fock_basis(n_orb, n_up, n_down)
                if len(basis) == 0:
                    continue
                H_sector = _build_hamiltonian(h, U, n_orb, basis, n_up, n_down)
                eigvals, eigvecs = np.linalg.eigh(H_sector)

                sectors[(n_up, n_down)] = {
                    'basis': basis,
                    'energies': eigvals,
                    'states': eigvecs,
                }
                all_energies.extend(eigvals)
                for i in range(len(eigvals)):
                    all_states.append((n_up, n_down, i))
                    all_sectors.append((n_up, n_down))

        all_energies = np.array(all_energies)

        # Partition function
        E_min = np.min(all_energies)
        boltz = np.exp(-beta * (all_energies - E_min))
        Z = np.sum(boltz)

        # Lehmann representation for G_dd(iw) — annihilate spin-up d-electron
        G_imp = np.zeros(len(iw), dtype=complex)
        n_d_total = 0.0
        n_double = 0.0

        # Lehmann representation (annihilation operator only — complete):
        # G(iw) = (1/Z) sum_{m,n} |<m|c|n>|^2 (e^{-bE_n} + e^{-bE_m}) / (iw - (E_n - E_m))
        # where c = d_up, |n> is the source (has d_up), |m> is the target (d_up removed).

        for n_up in range(n_orb + 1):
            for n_down in range(n_orb + 1):
                if (n_up, n_down) not in sectors:
                    continue
                sec_src = sectors[(n_up, n_down)]
                E_src = sec_src['energies']
                boltz_src = np.exp(-beta * (E_src - E_min))

                # Lehmann: annihilate d_up from source -> target in (n_up-1, n_down)
                if n_up >= 1 and (n_up - 1, n_down) in sectors:
                    sec_tgt = sectors[(n_up - 1, n_down)]
                    mat = _annihilation_matrix(0, n_orb, sec_src['basis'],
                                                sec_tgt['basis'], 'up')
                    # Transform to eigenbasis: <tgt_a| c |src_b>
                    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                        mat_eig = sec_tgt['states'].T @ mat @ sec_src['states']
                    mat_eig = np.nan_to_num(mat_eig, nan=0.0)

                    E_tgt = sec_tgt['energies']
                    boltz_tgt = np.exp(-beta * (E_tgt - E_min))

                    for a in range(len(E_tgt)):
                        for b in range(len(E_src)):
                            weight = np.abs(mat_eig[a, b])**2
                            if weight < 1e-15:
                                continue
                            pole = E_src[b] - E_tgt[a]
                            G_imp += weight * (boltz_src[b] + boltz_tgt[a]) / (iw - pole)

                # Occupancy: <n_{d,up}> and double occupancy <n_{d,up} n_{d,down}>
                for b in range(len(E_src)):
                    n_d_up_val = _number_operator_expect(0, n_orb, sec_src['basis'],
                                                          sec_src['states'][:, b], 'up')
                    n_d_total += n_d_up_val * boltz_src[b]

                    n_d_down_val = _number_operator_expect(0, n_orb, sec_src['basis'],
                                                            sec_src['states'][:, b], 'down')
                    n_double += n_d_up_val * n_d_down_val * boltz_src[b]

        G_imp /= Z
        n_imp = n_d_total / Z
        n_double /= Z

        # Bath correlators: <g_l^dag g_l> and <d^dag g_l> for two-ghost matching
        n_bath = len(eps)
        bath_gg = np.zeros(n_bath)
        bath_dg = np.zeros(n_bath)

        for n_up in range(n_orb + 1):
            for n_down in range(n_orb + 1):
                if (n_up, n_down) not in sectors:
                    continue
                sec = sectors[(n_up, n_down)]
                E_sec = sec['energies']
                boltz_sec = np.exp(-beta * (E_sec - E_min))

                for b_idx in range(len(E_sec)):
                    state_vec = sec['states'][:, b_idx]
                    bw = boltz_sec[b_idx]
                    for l in range(n_bath):
                        g_orb = 1 + l
                        # <g_l^dag g_l> via number operator (spin up)
                        bath_gg[l] += bw * _number_operator_expect(
                            g_orb, n_orb, sec['basis'], state_vec, 'up')
                        # <d^dag g_l> via off-diagonal one-body density matrix
                        bath_dg[l] += bw * _one_body_rdm_element(
                            0, g_orb, n_orb, sec['basis'], state_vec, 'up')

        bath_gg /= Z
        bath_dg /= Z

        # Self-energy from Dyson: Sigma = G_0^{-1} - G^{-1}
        from ..greens_function import hybridization
        delta = hybridization(iw, V, eps)
        G0_inv = iw + mu - eps_d - delta
        Sigma_imp = G0_inv - 1.0 / G_imp

        return {
            'G_imp': G_imp,
            'Sigma_imp': Sigma_imp,
            'n_imp': float(np.real(n_imp)),
            'n_double': float(np.real(n_double)),
            'bath_gg': bath_gg,
            'bath_dg': bath_dg,
        }


def _fock_basis(n_orb, n_up, n_down):
    """Generate Fock basis states for (n_up, n_down) sector.

    Each state is encoded as a tuple (up_config, down_config) where
    up_config and down_config are tuples of occupied orbital indices.

    Returns list of (up_tuple, down_tuple).
    """
    if n_up > n_orb or n_down > n_orb:
        return []
    up_configs = list(combinations(range(n_orb), n_up))
    down_configs = list(combinations(range(n_orb), n_down))
    return [(u, d) for u in up_configs for d in down_configs]


def _build_hamiltonian(h, U, n_orb, basis, n_up, n_down):
    """Build the many-body Hamiltonian matrix in the Fock basis.

    H = sum_{ab,s} h_{ab} c_{a,s}^dag c_{b,s} + U n_{0,up} n_{0,down}
    """
    dim = len(basis)
    H = np.zeros((dim, dim))

    for i, (up_i, down_i) in enumerate(basis):
        up_set = set(up_i)
        down_set = set(down_i)

        # Diagonal: one-body + interaction
        for orb in up_i:
            H[i, i] += h[orb, orb]
        for orb in down_i:
            H[i, i] += h[orb, orb]

        # Hubbard U: only on orbital 0
        if 0 in up_set and 0 in down_set:
            H[i, i] += U

        # Off-diagonal: hopping terms c_a^dag c_b with a != b
        # Spin up
        for a in range(n_orb):
            for b in range(n_orb):
                if a == b or abs(h[a, b]) < 1e-15:
                    continue
                if b in up_set and a not in up_set:
                    # c_a^dag c_b |up_i> : remove b, add a
                    new_up = tuple(sorted((up_set - {b}) | {a}))
                    sign = _fermionic_sign(up_i, b, a)
                    j = _find_state(basis, new_up, down_i)
                    if j >= 0:
                        H[j, i] += h[a, b] * sign

        # Spin down
        for a in range(n_orb):
            for b in range(n_orb):
                if a == b or abs(h[a, b]) < 1e-15:
                    continue
                if b in down_set and a not in down_set:
                    new_down = tuple(sorted((down_set - {b}) | {a}))
                    sign = _fermionic_sign(down_i, b, a)
                    j = _find_state(basis, up_i, new_down)
                    if j >= 0:
                        H[j, i] += h[a, b] * sign

    return H


def _fermionic_sign(config, remove_orb, add_orb):
    """Compute the fermionic sign for c_add^dag c_remove acting on config.

    Sign = (-1)^(number of occupied orbitals between remove and add positions).
    """
    config_list = list(config)
    # Count how many orbitals are between remove and add
    remove_pos = config_list.index(remove_orb)

    # Remove: sign from anticommuting past orbitals to the left
    sign_remove = (-1)**remove_pos

    # New config after removal
    new_config = list(config)
    new_config.remove(remove_orb)

    # Add: sign from anticommuting past orbitals to find insertion point
    insert_pos = 0
    for orb in new_config:
        if orb < add_orb:
            insert_pos += 1
        else:
            break
    sign_add = (-1)**insert_pos

    return sign_remove * sign_add


def _find_state(basis, up, down):
    """Find index of state (up, down) in basis. Returns -1 if not found."""
    target = (up, down)
    for i, state in enumerate(basis):
        if state == target:
            return i
    return -1


def _annihilation_matrix(orb, n_orb, basis_i, basis_j, spin):
    """Matrix elements <j| c_{orb,spin} |i>.

    |i> is in the (n_up, n_down) sector.
    |j> is in the (n_up-1, n_down) sector (for spin='up').
    """
    dim_i = len(basis_i)
    dim_j = len(basis_j)
    mat = np.zeros((dim_j, dim_i))

    for i, (up_i, down_i) in enumerate(basis_i):
        config = up_i if spin == 'up' else down_i
        other = down_i if spin == 'up' else up_i

        if orb not in config:
            continue

        config_list = list(config)
        pos = config_list.index(orb)
        sign = (-1)**pos

        new_config = tuple(sorted(set(config) - {orb}))
        new_state = (new_config, other) if spin == 'up' else (other, new_config)

        j = _find_state(basis_j, new_state[0], new_state[1])
        if j >= 0:
            mat[j, i] = sign

    return mat


def _creation_matrix(orb, n_orb, basis_i, basis_j, spin):
    """Matrix elements <j| c_{orb,spin}^dag |i>.

    |j> is in the (n_up+1, n_down) sector (for spin='up').
    """
    dim_i = len(basis_i)
    dim_j = len(basis_j)
    mat = np.zeros((dim_j, dim_i))

    for i, (up_i, down_i) in enumerate(basis_i):
        config = up_i if spin == 'up' else down_i
        other = down_i if spin == 'up' else up_i

        if orb in config:
            continue

        config_list = list(config)
        # Find insertion position
        insert_pos = 0
        for o in config_list:
            if o < orb:
                insert_pos += 1
            else:
                break
        sign = (-1)**insert_pos

        new_config = tuple(sorted(set(config) | {orb}))
        new_state = (new_config, other) if spin == 'up' else (other, new_config)

        j = _find_state(basis_j, new_state[0], new_state[1])
        if j >= 0:
            mat[j, i] = sign

    return mat


def _number_operator_expect(orb, n_orb, basis, state_vec, spin):
    """Expectation value <state| n_{orb,spin} |state>."""
    result = 0.0
    for i, (up_i, down_i) in enumerate(basis):
        config = up_i if spin == 'up' else down_i
        if orb in config:
            result += np.abs(state_vec[i])**2
    return result


def _one_body_rdm_element(orb_a, orb_b, n_orb, basis, state_vec, spin):
    """Compute <state| c_{a,spin}^dag c_{b,spin} |state>.

    For a=b this reduces to the number operator.
    For a!=b this is the off-diagonal one-body density matrix element.
    """
    if orb_a == orb_b:
        return _number_operator_expect(orb_a, n_orb, basis, state_vec, spin)

    result = 0.0
    for i, (up_i, down_i) in enumerate(basis):
        config = up_i if spin == 'up' else down_i
        other = down_i if spin == 'up' else up_i

        if orb_b not in config or orb_a in config:
            continue

        sign = _fermionic_sign(config, orb_b, orb_a)
        new_config = tuple(sorted((set(config) - {orb_b}) | {orb_a}))
        new_state = (new_config, other) if spin == 'up' else (other, new_config)
        j = _find_state(basis, new_state[0], new_state[1])
        if j >= 0:
            result += sign * state_vec[j] * state_vec[i]

    return result
