"""Gateway (H_imp^(0)) quadratic model: impurity d + g-bath + h-ghosts.

This is the bridge model where both bath and ghost sectors coexist
but there is NO Hubbard U interaction. It is fully quadratic, so all
Green's functions are obtained by matrix inversion / Schur complement.

The inverse resolvent is a (1 + M_g + M_h) x (1 + M_g + M_h) matrix
with the d-orbital in position 0:

    G_hat^{-1} = [[A,     -V^*,    -W^*  ],
                  [-V,     D_g,     0     ],
                  [-W,     0,       D_h   ]]

where:
    A = iw + mu - eps_d - sigma_inf
    (D_g)_{ll} = iw - eps_l
    (D_h)_{ll} = iw - eta_l

The d-d Green's function via Schur complement:
    G_dd^(0) = 1 / (A - Delta(iw) - Sigma^(h)(iw))
            = 1 / (iw + mu - eps_d - sigma_inf - Delta(iw) - Sigma^(h)(iw))
"""

import numpy as np
from .matsubara import (
    fermi_function,
    matsubara_sum_numerical,
    matsubara_sum_convergence,
    matsubara_sum_pair_numerical,
    matsubara_sum_pair_convergence,
)


def gateway_greens_functions(iw: np.ndarray, mu: float, eps_d: float,
                              sigma_inf: float,
                              V: np.ndarray, eps: np.ndarray,
                              W: np.ndarray, eta: np.ndarray) -> dict:
    """Compute all gateway model Green's function blocks.

    Parameters
    ----------
    iw : array, shape (N,)
        Imaginary frequencies (1j * w_n).
    mu, eps_d, sigma_inf : float
        Chemical potential, impurity level, static self-energy.
    V, eps : arrays, shape (M_g,)
        Bath hybridization amplitudes and energies.
    W, eta : arrays, shape (M_h,)
        Ghost hybridization amplitudes and energies.

    Returns
    -------
    dict with keys:
        'dd': array (N,) — G_dd^(0)
        'gd': array (M_g, N) — G_{g_l, d}^(0)
        'hd': array (M_h, N) — G_{h_l, d}^(0)
        'gg': array (M_g, N) — G_{g_l, g_l}^(0) (diagonal)
        'hh': array (M_h, N) — G_{h_l, h_l}^(0) (diagonal)
    """
    N = len(iw)
    M_g = len(eps)
    M_h = len(eta)

    # Bath and ghost propagators
    # denom_g[l, n] = 1 / (iw_n - eps_l)
    denom_g = 1.0 / (iw[None, :] - eps[:, None])  # (M_g, N)
    denom_h = 1.0 / (iw[None, :] - eta[:, None])  # (M_h, N)

    # Hybridization and ghost self-energy
    delta = np.sum(np.abs(V[:, None])**2 * denom_g, axis=0)  # (N,)
    sigma_h = np.sum(np.abs(W[:, None])**2 * denom_h, axis=0)  # (N,)

    # d-d Green's function
    G_dd = 1.0 / (iw + mu - eps_d - sigma_inf - delta - sigma_h)  # (N,)

    # Off-diagonal: G_{g_l, d} = V_l / (iw - eps_l) * G_dd
    G_gd = V[:, None] * denom_g * G_dd[None, :]  # (M_g, N)

    # Off-diagonal: G_{h_l, d} = W_l / (iw - eta_l) * G_dd
    G_hd = W[:, None] * denom_h * G_dd[None, :]  # (M_h, N)

    # Diagonal bath: G_{g_l, g_l} = 1/(iw - eps_l) + |V_l|^2/(iw - eps_l)^2 * G_dd
    G_gg = denom_g + (np.abs(V)**2)[:, None] * denom_g**2 * G_dd[None, :]  # (M_g, N)

    # Diagonal ghost: G_{h_l, h_l} = 1/(iw - eta_l) + |W_l|^2/(iw - eta_l)^2 * G_dd
    G_hh = denom_h + (np.abs(W)**2)[:, None] * denom_h**2 * G_dd[None, :]  # (M_h, N)

    return {
        'dd': G_dd,
        'gd': G_gd, 'hd': G_hd,
        'gg': G_gg, 'hh': G_hh,
    }


def gateway_onebody_matrix(mu: float, eps_d: float, sigma_inf: float,
                            V: np.ndarray, eps: np.ndarray,
                            W: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Build the one-body grand-canonical matrix K for the gateway model.

    K = H - mu*N in the orbital basis [d, g_1,...,g_M_g, h_1,...,h_M_h].

    The eigenvalues of K determine all correlators via Fermi functions:
        <c_b^dag c_a> = [f(K)]_{ab} = sum_j U_{a,j}* U_{b,j} f(e_j)

    Returns
    -------
    ndarray, shape (1+M_g+M_h, 1+M_g+M_h)
    """
    M_g = len(eps)
    M_h = len(eta)
    dim = 1 + M_g + M_h
    K = np.zeros((dim, dim), dtype=complex)

    # d-orbital: on-site energy relative to mu
    K[0, 0] = eps_d + sigma_inf - mu

    # g-bath levels
    for l in range(M_g):
        K[1 + l, 1 + l] = eps[l]
        K[0, 1 + l] = np.conj(V[l])
        K[1 + l, 0] = V[l]

    # h-ghost levels
    for l in range(M_h):
        K[1 + M_g + l, 1 + M_g + l] = eta[l]
        K[0, 1 + M_g + l] = np.conj(W[l])
        K[1 + M_g + l, 0] = W[l]

    return K


def gateway_correlators(mu: float, eps_d: float,
                         sigma_inf: float,
                         V: np.ndarray, eps: np.ndarray,
                         W: np.ndarray, eta: np.ndarray,
                         beta: float) -> dict:
    """Exact equal-time gateway correlators via diagonalization.

    For a quadratic model, <c_b^dag c_a> = [f(K)]_{ab} where
    f(K) = U @ diag(f(eigenvalues)) @ U^dag is the matrix Fermi function.

    This is exact (no Matsubara truncation) and numerically robust.

    Parameters
    ----------
    mu, eps_d, sigma_inf : float
    V, eps : arrays, shape (M_g,)
    W, eta : arrays, shape (M_h,)
    beta : float

    Returns
    -------
    dict with:
        'hh': array (M_h,) — <h_l^dag h_l>_0
        'dh': array (M_h,) — <d^dag h_l>_0
        'gg': array (M_g,) — <g_l^dag g_l>_0
        'dg': array (M_g,) — <d^dag g_l>_0
    """
    M_g = len(eps)
    M_h = len(eta)

    K = gateway_onebody_matrix(mu, eps_d, sigma_inf, V, eps, W, eta)
    eigvals, U = np.linalg.eigh(K)

    # Matrix Fermi function: f_matrix = U @ diag(f(e)) @ U^dag
    f_e = fermi_function(eigvals, beta)
    f_matrix = U @ np.diag(f_e) @ U.conj().T

    # Extract correlators
    # Orbital ordering: [d, g_1,...,g_Mg, h_1,...,h_Mh]
    d_idx = 0
    g_idx = np.arange(1, 1 + M_g)
    h_idx = np.arange(1 + M_g, 1 + M_g + M_h)

    hh_corr = np.array([f_matrix[i, i] for i in h_idx], dtype=complex)
    # <d^dag h_l> = [f(K)]_{h_l,d}
    dh_corr = np.array([f_matrix[i, d_idx] for i in h_idx], dtype=complex)
    gg_corr = np.array([f_matrix[i, i] for i in g_idx], dtype=complex)
    # <d^dag g_l> = [f(K)]_{g_l,d}
    dg_corr = np.array([f_matrix[i, d_idx] for i in g_idx], dtype=complex)

    return {
        'hh': np.real_if_close(hh_corr),
        'dh': np.real_if_close(dh_corr),
        'gg': np.real_if_close(gg_corr),
        'dg': np.real_if_close(dg_corr),
    }


def gateway_correlators_from_matsubara(
    iw: np.ndarray,
    mu: float,
    eps_d: float,
    sigma_inf: float,
    V: np.ndarray,
    eps: np.ndarray,
    W: np.ndarray,
    eta: np.ndarray,
    beta: float,
    return_diagnostics: bool = False,
    diagnostic_n_values: np.ndarray = None,
) -> dict:
    """Gateway correlators from Matsubara sums of Green's-function blocks.

    This is an independent route to equal-time correlators, useful for
    validating truncation errors and checking consistency with the exact
    diagonalization-based `gateway_correlators()`.

    Parameters
    ----------
    iw : array, shape (N,)
        Positive Matsubara frequencies (1j*w_n).
    mu, eps_d, sigma_inf : float
    V, eps : arrays, shape (M_g,)
    W, eta : arrays, shape (M_h,)
    beta : float
    return_diagnostics : bool
        If True, return convergence history versus n_matsubara.
    diagnostic_n_values : array, optional
        Explicit n_matsubara values for diagnostics.

    Returns
    -------
    dict with keys: 'hh', 'dh', 'gg', 'dg', and optional 'diagnostics'.
    """
    gf = gateway_greens_functions(iw, mu, eps_d, sigma_inf, V, eps, W, eta)
    M_g = len(eps)
    M_h = len(eta)

    hh_corr = np.zeros(M_h)
    dh_corr = np.zeros(M_h, dtype=complex)
    gg_corr = np.zeros(M_g)
    dg_corr = np.zeros(M_g, dtype=complex)

    diagnostics = None
    if return_diagnostics:
        n_values = np.asarray(
            matsubara_sum_convergence(
                np.zeros_like(iw), beta, n_values=diagnostic_n_values
            )['n_matsubara']
        )
        diagnostics = {
            'n_matsubara': n_values,
            'hh': np.zeros((M_h, len(n_values))),
            'dh': np.zeros((M_h, len(n_values)), dtype=complex),
            'gg': np.zeros((M_g, len(n_values))),
            'dg': np.zeros((M_g, len(n_values)), dtype=complex),
        }

    for l in range(M_h):
        # G_hh = 1/(iw-eta) + dynamic part
        hh_dynamic = gf['hh'][l] - 1.0 / (iw - eta[l])
        hh_corr[l] = (
            fermi_function(np.array([eta[l]]), beta)[0]
            + matsubara_sum_numerical(hh_dynamic, beta).real
        )

        # <d^dag h_l> = (1/beta) sum_n G_{h_l,d}(iw_n)
        G_hd = gf['hd'][l]
        G_dh = np.conj(W[l]) * gf['dd'] / (iw - eta[l])
        dh_corr[l] = matsubara_sum_pair_numerical(
            G_hd, G_dh, beta, tail_c2_ab=W[l], tail_c2_ba=np.conj(W[l])
        )

        if return_diagnostics:
            hh_seq = matsubara_sum_convergence(
                hh_dynamic, beta, n_values=diagnostic_n_values
            )['sum']
            dh_seq = matsubara_sum_pair_convergence(
                G_hd, G_dh, beta,
                tail_c2_ab=W[l], tail_c2_ba=np.conj(W[l]),
                n_values=diagnostic_n_values
            )['sum']
            diagnostics['hh'][l] = fermi_function(np.array([eta[l]]), beta)[0] + hh_seq.real
            diagnostics['dh'][l] = dh_seq

    for l in range(M_g):
        gg_dynamic = gf['gg'][l] - 1.0 / (iw - eps[l])
        gg_corr[l] = (
            fermi_function(np.array([eps[l]]), beta)[0]
            + matsubara_sum_numerical(gg_dynamic, beta).real
        )

        # <d^dag g_l> = (1/beta) sum_n G_{g_l,d}(iw_n)
        G_gd = gf['gd'][l]
        G_dg = np.conj(V[l]) * gf['dd'] / (iw - eps[l])
        dg_corr[l] = matsubara_sum_pair_numerical(
            G_gd, G_dg, beta, tail_c2_ab=V[l], tail_c2_ba=np.conj(V[l])
        )

        if return_diagnostics:
            gg_seq = matsubara_sum_convergence(
                gg_dynamic, beta, n_values=diagnostic_n_values
            )['sum']
            dg_seq = matsubara_sum_pair_convergence(
                G_gd, G_dg, beta,
                tail_c2_ab=V[l], tail_c2_ba=np.conj(V[l]),
                n_values=diagnostic_n_values
            )['sum']
            diagnostics['gg'][l] = fermi_function(np.array([eps[l]]), beta)[0] + gg_seq.real
            diagnostics['dg'][l] = dg_seq

    out = {
        'hh': np.real_if_close(hh_corr),
        'dh': np.real_if_close(dh_corr),
        'gg': np.real_if_close(gg_corr),
        'dg': np.real_if_close(dg_corr),
    }
    if diagnostics is not None:
        out['diagnostics'] = diagnostics
    return out


# ═══════════════════════════════════════════════════════════
# Static correlator functions for bond-scheme DMFT
# ═══════════════════════════════════════════════════════════

def gateway_statics(beta: float, eta, W_ghost, eps, V, M: int, shift: float):
    """Single-site static gateway correlators via diagonalization.

    H_gw: d + M g-ghosts + M h-ghosts (quadratic, 1 + 2M orbitals).

    Parameters
    ----------
    beta : float
        Inverse temperature.
    eta : array, shape (M,)
        h-ghost energies.
    W_ghost : array, shape (M,)
        h-ghost hybridizations.
    eps : array, shape (M,)
        g-ghost (bath) energies.
    V : array, shape (M,)
        g-ghost hybridizations.
    M : int
        Number of ghost poles.
    shift : float
        On-site shift for d-orbital.

    Returns
    -------
    nh : array, shape (M,)
    dh : array, shape (M,)
    ng : array, shape (M,)
    dg : array, shape (M,)
    """
    from numpy.linalg import eigh as _eigh
    n = 1 + 2 * M
    H = np.zeros((n, n))
    H[0, 0] = shift
    for l in range(M):
        H[1 + l, 1 + l] = eps[l]
        H[0, 1 + l] = H[1 + l, 0] = V[l]
        H[1 + M + l, 1 + M + l] = eta[l]
        H[0, 1 + M + l] = H[1 + M + l, 0] = W_ghost[l]
    ev, Uv = _eigh(H)
    f = _fermi_gw(ev, beta)
    rho = (Uv * f) @ Uv.T
    ng = np.array([float(rho[1 + l, 1 + l]) for l in range(M)])
    dg = np.array([float(rho[0, 1 + l]) for l in range(M)])
    nh = np.array([float(rho[1 + M + l, 1 + M + l]) for l in range(M)])
    dh = np.array([float(rho[0, 1 + M + l]) for l in range(M)])
    return nh, dh, ng, dg


def bond_gateway_statics(beta: float, eta, W_ghost, eps, V,
                          etab, Bh, epsb, Bg, M: int, t: float, shift: float):
    """Two-site quadratic bond gateway correlators.

    H2_gw: d0, d1 + 2M site-local h-ghosts + M bond hb-ghosts
           + 2M site-local g-ghosts + M bond gb-ghosts.
    Matrix dimension: n_gw = 2 + 6M.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    eta, W_ghost : arrays, shape (M,)
        Site-local h-ghost parameters.
    eps, V : arrays, shape (M,)
        Site-local g-ghost parameters.
    etab, Bh : arrays, shape (M,)
        Bond hb-ghost parameters.
    epsb, Bg : arrays, shape (M,)
        Bond gb-ghost parameters.
    M : int
        Number of ghost poles.
    t : float
        Hopping between the two sites.
    shift : float
        On-site shift for d-orbitals.

    Returns
    -------
    nh, dh : arrays, shape (M,)
        Site-local h-ghost correlators (averaged over two sites).
    nhb, dhb : arrays, shape (M,)
        Bond hb-ghost correlators.
    ng, dg : arrays, shape (M,)
        Site-local g-ghost correlators (averaged over two sites).
    ngb, dgb : arrays, shape (M,)
        Bond gb-ghost correlators.
    n_site : float
        d-orbital occupancy on site 0.
    """
    from numpy.linalg import eigh as _eigh
    n_gw = 2 + 6 * M
    H = np.zeros((n_gw, n_gw))
    H[0, 0] = shift
    H[1, 1] = shift
    H[0, 1] = H[1, 0] = -t

    for l in range(M):
        # site-local h-ghosts: h on site 0 at 2+l, h on site 1 at 2+M+l
        H[2 + l, 2 + l] = eta[l]
        H[0, 2 + l] = H[2 + l, 0] = W_ghost[l]
        H[2 + M + l, 2 + M + l] = eta[l]
        H[1, 2 + M + l] = H[2 + M + l, 1] = W_ghost[l]
        # bond hb-ghosts at 2+2M+l
        H[2 + 2 * M + l, 2 + 2 * M + l] = etab[l]
        H[0, 2 + 2 * M + l] = H[2 + 2 * M + l, 0] = Bh[l]
        H[1, 2 + 2 * M + l] = H[2 + 2 * M + l, 1] = Bh[l]
        # site-local g-ghosts: g on site 0 at 2+3M+l, g on site 1 at 2+4M+l
        H[2 + 3 * M + l, 2 + 3 * M + l] = eps[l]
        H[0, 2 + 3 * M + l] = H[2 + 3 * M + l, 0] = V[l]
        H[2 + 4 * M + l, 2 + 4 * M + l] = eps[l]
        H[1, 2 + 4 * M + l] = H[2 + 4 * M + l, 1] = V[l]
        # bond gb-ghosts at 2+5M+l
        H[2 + 5 * M + l, 2 + 5 * M + l] = epsb[l]
        H[0, 2 + 5 * M + l] = H[2 + 5 * M + l, 0] = Bg[l]
        H[1, 2 + 5 * M + l] = H[2 + 5 * M + l, 1] = Bg[l]

    ev, Uv = _eigh(H)
    f = _fermi_gw(ev, beta)
    rho = (Uv * f) @ Uv.T

    nh = np.array([0.5 * (rho[2 + l, 2 + l] + rho[2 + M + l, 2 + M + l])
                   for l in range(M)])
    dh = np.array([0.5 * (rho[0, 2 + l] + rho[1, 2 + M + l])
                   for l in range(M)])
    nhb = np.array([rho[2 + 2 * M + l, 2 + 2 * M + l] for l in range(M)])
    dhb = np.array([0.5 * (rho[0, 2 + 2 * M + l] + rho[1, 2 + 2 * M + l])
                    for l in range(M)])
    ng = np.array([0.5 * (rho[2 + 3 * M + l, 2 + 3 * M + l]
                          + rho[2 + 4 * M + l, 2 + 4 * M + l])
                   for l in range(M)])
    dg = np.array([0.5 * (rho[0, 2 + 3 * M + l] + rho[1, 2 + 4 * M + l])
                   for l in range(M)])
    ngb = np.array([rho[2 + 5 * M + l, 2 + 5 * M + l] for l in range(M)])
    dgb = np.array([rho[0, 2 + 5 * M + l] + rho[1, 2 + 5 * M + l]
                    for l in range(M)])
    n_site = float(rho[0, 0])
    return nh, dh, nhb, dhb, ng, dg, ngb, dgb, n_site


def _fermi_gw(e, beta: float):
    """Numerically stable Fermi function for gateway diagonalization."""
    x = beta * np.asarray(e, dtype=float)
    out = np.empty_like(x)
    out[x > 500] = 0.0
    out[x < -500] = 1.0
    m = (x >= -500) & (x <= 500)
    out[m] = 1.0 / (np.exp(x[m]) + 1.0)
    return out


# ═══════════════════════════════════════════════════════════
# Corrected bond-scheme gateway functions
# (professor's ghost_dmft_bond_new.py, March 2026)
# ═══════════════════════════════════════════════════════════

def gateway1_statics(beta: float, eta1, W1, eps1, V1, M1g: int, shift: float):
    """Single-site gateway correlators (d + 1 h1-ghost + M1g g1-ghosts).

    Orbital layout: 0=d, 1=h1, 2..2+M1g-1=g1_0..g1_{M1g-1}

    Parameters
    ----------
    beta : float
    eta1, W1 : float
        Single h1-ghost energy and hybridization (M1h=1 fixed).
    eps1, V1 : arrays, shape (M1g,)
        g1-ghost energies and hybridizations.
    M1g : int
    shift : float
        On-site d-orbital shift.

    Returns
    -------
    nh1, dh1 : float
    ng1, dg1 : arrays, shape (M1g,)
    nd : float
    """
    eta1 = float(np.atleast_1d(eta1)[0])
    W1   = float(np.atleast_1d(W1)[0])
    eps1 = np.atleast_1d(np.asarray(eps1, dtype=float))
    V1   = np.atleast_1d(np.asarray(V1,   dtype=float))
    Norb = 2 + M1g   # d, h1, g1_0..g1_{M1g-1}
    H = np.zeros((Norb, Norb))
    H[0, 0] = shift
    H[1, 1] = eta1;  H[0, 1] = H[1, 0] = W1
    for l in range(M1g):
        H[2 + l, 2 + l] = eps1[l]
        H[0, 2 + l] = H[2 + l, 0] = V1[l]
    ev, Uv = np.linalg.eigh(H)
    f = _fermi_gw(ev, beta)
    rho = (Uv * f) @ Uv.T
    nh1 = float(rho[1, 1])
    dh1 = float(rho[0, 1])
    ng1 = np.array([float(rho[2 + l, 2 + l]) for l in range(M1g)])
    dg1 = np.array([float(rho[0, 2 + l])     for l in range(M1g)])
    nd  = float(rho[0, 0])
    return nh1, dh1, ng1, dg1, nd


def gateway2_statics(beta: float, eta2, W2, etab, Bh,
                     eps2, V2, epsb, Bg, M2g: int, Mbg: int,
                     t: float, shift: float):
    """Two-site gateway correlators with CORRECT LOCAL/SHARED coupling.

    Orbital layout (M2h=Mbh=1 fixed):
      0=d1, 1=d2,
      2=h2_site1 (LOCAL, d1 only),   3=h2_site2 (LOCAL, d2 only),
      4=hb       (SHARED, both d1 and d2),
      5..5+M2g-1         = g2_site1 (LOCAL, d1 only),
      5+M2g..5+2*M2g-1   = g2_site2 (LOCAL, d2 only),
      5+2*M2g..5+2*M2g+Mbg-1 = gb   (SHARED, both d1 and d2)

    Parameters
    ----------
    beta : float
    eta2, W2 : float   (M2h=1 fixed)
    etab, Bh : float   (Mbh=1 fixed)
    eps2, V2 : arrays, shape (M2g,)
    epsb, Bg : arrays, shape (Mbg,)
    M2g, Mbg : int
    t : float          d1-d2 hopping
    shift : float

    Returns
    -------
    nh2, dh2 : float   (averaged over site1/site2)
    nhb, dhb : float   (dhb = <d1†hb> + <d2†hb>)
    ng2 : array (M2g,)
    dg2 : array (M2g,)
    ngb : array (Mbg,)
    dgb : array (Mbg,)  (= <d1†gb_l> + <d2†gb_l>)
    nd  : float         average d occupancy
    """
    eta2 = float(np.atleast_1d(eta2)[0])
    W2   = float(np.atleast_1d(W2)[0])
    etab = float(np.atleast_1d(etab)[0])
    Bh   = float(np.atleast_1d(Bh)[0])
    eps2 = np.atleast_1d(np.asarray(eps2, dtype=float))
    V2   = np.atleast_1d(np.asarray(V2,   dtype=float))
    epsb = np.atleast_1d(np.asarray(epsb, dtype=float))
    Bg   = np.atleast_1d(np.asarray(Bg,   dtype=float))

    M2h = 1; Mbh = 1   # fixed
    Norb = 2 + 2*M2h + Mbh + 2*M2g + Mbg
    H = np.zeros((Norb, Norb))

    # d1, d2 on-site + hopping
    H[0, 0] = shift;  H[1, 1] = shift
    H[0, 1] = H[1, 0] = -t

    # h2: LOCAL per site
    for l in range(M2h):
        i1 = 2 + l;  i2 = 2 + M2h + l
        H[i1, i1] = eta2;  H[i2, i2] = eta2
        H[0, i1] = H[i1, 0] = W2   # d1 – h2_site1
        H[1, i2] = H[i2, 1] = W2   # d2 – h2_site2

    # hb: SHARED bond orbital
    for l in range(Mbh):
        i = 2 + 2*M2h + l
        H[i, i] = etab
        H[0, i] = H[i, 0] = Bh   # d1 – hb
        H[1, i] = H[i, 1] = Bh   # d2 – hb

    # g2: LOCAL per site
    off = 2 + 2*M2h + Mbh
    for l in range(M2g):
        i1 = off + l;  i2 = off + M2g + l
        H[i1, i1] = eps2[l];  H[i2, i2] = eps2[l]
        H[0, i1] = H[i1, 0] = V2[l]   # d1 – g2_site1
        H[1, i2] = H[i2, 1] = V2[l]   # d2 – g2_site2

    # gb: SHARED bond orbital
    off2 = off + 2*M2g
    for l in range(Mbg):
        i = off2 + l
        H[i, i] = epsb[l]
        H[0, i] = H[i, 0] = Bg[l]   # d1 – gb
        H[1, i] = H[i, 1] = Bg[l]   # d2 – gb

    ev, Uv = np.linalg.eigh(H)
    f = _fermi_gw(ev, beta)
    rho = (Uv * f) @ Uv.T

    # h2: average over site1 and site2
    nh2 = float(np.mean([rho[2 + l, 2 + l] for l in range(M2h)]))
    dh2 = float(np.mean([rho[0, 2 + l]     for l in range(M2h)]))
    # hb: shared — dhb = <d1†hb> + <d2†hb>
    nhb = float(np.mean([rho[2 + 2*M2h + l, 2 + 2*M2h + l] for l in range(Mbh)]))
    dhb = float(np.mean([rho[0, 2 + 2*M2h + l] + rho[1, 2 + 2*M2h + l]
                         for l in range(Mbh)]))
    # g2: average over site1 and site2
    ng2 = np.array([0.5*(rho[off + l, off + l] + rho[off + M2g + l, off + M2g + l])
                    for l in range(M2g)])
    dg2 = np.array([0.5*(rho[0, off + l] + rho[1, off + M2g + l])
                    for l in range(M2g)])
    # gb: shared — dgb = <d1†gb> + <d2†gb>
    ngb = np.array([rho[off2 + l, off2 + l]            for l in range(Mbg)])
    dgb = np.array([rho[0, off2 + l] + rho[1, off2 + l] for l in range(Mbg)])

    nd = 0.5*(rho[0, 0] + rho[1, 1])
    return nh2, dh2, nhb, dhb, ng2, dg2, ngb, dgb, nd
