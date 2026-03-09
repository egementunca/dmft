"""Pole fitting and correlator matching for two-ghost DMFT.

Standard DMFT bath discretization: given Delta(iw) on the Matsubara grid,
find {eps_l, V_l} such that Delta(iw) ~ sum |V_l|^2 / (iw - eps_l).

Correlator matching (Variant B): match equal-time correlators between
lattice/impurity and the gateway model to determine pole parameters.
"""

import numpy as np
from scipy.optimize import least_squares


def fit_hybridization_poles(delta_iw: np.ndarray, iw: np.ndarray,
                             M_g: int, symmetric: bool = True) -> tuple:
    """Fit hybridization to a pole expansion.

    Delta(iw) = sum_{l=1}^{M_g} |V_l|^2 / (iw - eps_l)

    Parameters
    ----------
    delta_iw : array, shape (N,)
        Hybridization function on Matsubara grid.
    iw : array, shape (N,)
        Imaginary frequencies (1j * w_n).
    M_g : int
        Number of bath poles.
    symmetric : bool
        If True, enforce particle-hole symmetry: eps_l = -eps_{M-l+1},
        V_l = V_{M-l+1}. Halves the number of parameters.

    Returns
    -------
    V : array, shape (M_g,)
    eps : array, shape (M_g,)
    """
    if symmetric and M_g > 1:
        return _fit_symmetric(delta_iw, iw, M_g)
    else:
        return _fit_general(delta_iw, iw, M_g)


def _fit_general(delta_iw, iw, M_g):
    """General (no symmetry) pole fit."""
    # Initial guess: spread poles across bandwidth
    eps0 = np.linspace(-0.8, 0.8, M_g)
    V0 = np.full(M_g, 0.3)
    x0 = np.concatenate([eps0, V0])

    def residual(x):
        eps = x[:M_g]
        V = x[M_g:]
        delta_fit = np.sum(V[:, None]**2 / (iw[None, :] - eps[:, None]), axis=0)
        r = delta_fit - delta_iw
        return np.concatenate([r.real, r.imag])

    result = least_squares(residual, x0, method='lm', max_nfev=5000)
    eps = result.x[:M_g]
    V = result.x[M_g:]
    return V, eps


def _fit_symmetric(delta_iw, iw, M_g):
    """Particle-hole symmetric pole fit.

    For even M_g: pairs (eps_l, -eps_l) with same |V|.
    For odd M_g: one pole at eps=0, plus pairs.
    """
    n_pairs = M_g // 2
    has_center = M_g % 2 == 1

    # Parameters: [eps_1, ..., eps_{n_pairs}, V_pair_1, ..., V_pair_{n_pairs}, (V_center)]
    n_params = n_pairs + n_pairs + (1 if has_center else 0)
    x0 = np.zeros(n_params)
    # Initial eps > 0 for pairs
    x0[:n_pairs] = np.linspace(0.2, 0.8, n_pairs) if n_pairs > 0 else []
    # Initial V
    x0[n_pairs:2*n_pairs] = 0.3
    if has_center:
        x0[-1] = 0.3

    def residual(x):
        eps_pos = x[:n_pairs]
        V_pair = x[n_pairs:2*n_pairs]
        V_center = x[-1] if has_center else None

        # Build full arrays
        eps_full = np.concatenate([-eps_pos[::-1], [0.0] if has_center else [], eps_pos])
        V_full = np.concatenate([V_pair[::-1], [V_center] if has_center else [], V_pair])

        delta_fit = np.sum(V_full[:, None]**2 / (iw[None, :] - eps_full[:, None]), axis=0)
        r = delta_fit - delta_iw
        return np.concatenate([r.real, r.imag])

    result = least_squares(residual, x0, method='lm', max_nfev=5000)

    eps_pos = result.x[:n_pairs]
    V_pair = result.x[n_pairs:2*n_pairs]
    V_center = result.x[-1] if has_center else None

    eps_full = np.concatenate([-eps_pos[::-1], [0.0] if has_center else [], eps_pos])
    V_full = np.concatenate([V_pair[::-1], [V_center] if has_center else [], V_pair])

    return V_full, eps_full


def fit_self_energy_poles(sigma_iw: np.ndarray, iw: np.ndarray,
                           sigma_inf: float, M_h: int,
                           symmetric: bool = True) -> tuple:
    """Fit self-energy dynamic part to a pole expansion.

    Sigma(iw) - sigma_inf = sum_{l=1}^{M_h} |W_l|^2 / (iw - eta_l)

    Parameters
    ----------
    sigma_iw : array, shape (N,)
        Self-energy on Matsubara grid.
    iw : array, shape (N,)
        Imaginary frequencies (1j * w_n).
    sigma_inf : float
        Static tail (high-frequency limit).
    M_h : int
        Number of ghost poles.
    symmetric : bool
        If True, enforce particle-hole symmetry.

    Returns
    -------
    W : array, shape (M_h,)
    eta : array, shape (M_h,)
    """
    sigma_dynamic = sigma_iw - sigma_inf
    # Reuse the hybridization fitter — same functional form
    return fit_hybridization_poles(sigma_dynamic, iw, M_h, symmetric=symmetric)


# ---------------------------------------------------------------------------
# Correlator matching for two-ghost DMFT (Variant B)
# ---------------------------------------------------------------------------

def match_h_correlators(target_hh: np.ndarray, target_dh: np.ndarray,
                         mu: float, eps_d: float, sigma_inf: float,
                         W: np.ndarray, eta: np.ndarray,
                         M_g: int, beta: float,
                         eps0: np.ndarray = None, V0: np.ndarray = None,
                         symmetric: bool = True) -> tuple:
    """Find bath parameters {eps, V} by matching h-sector correlators.

    Lattice <-> gateway matching: adjust bath poles so the gateway model's
    h-sector correlators match the lattice h-sector correlators.

    Solves:
        gateway_correlators(...)['hh'] = target_hh
        gateway_correlators(...)['dh'] = target_dh

    Parameters
    ----------
    target_hh : array, shape (M_h,)
        Target <h_l^dag h_l> from lattice.
    target_dh : array, shape (M_h,)
        Target <d^dag h_l> from lattice.
    mu, eps_d, sigma_inf : float
        Fixed parameters for the gateway model.
    W, eta : arrays, shape (M_h,)
        Current ghost pole parameters (fixed during this matching).
    M_g : int
        Number of bath poles to determine.
    beta : float
        Inverse temperature.
    eps0, V0 : arrays, optional
        Initial guess for bath parameters. If None, use defaults.
    symmetric : bool
        Enforce particle-hole symmetry.

    Returns
    -------
    V : array, shape (M_g,)
    eps : array, shape (M_g,)
    """
    from .gateway import gateway_correlators

    target = np.real(np.concatenate([target_hh, target_dh]))

    if symmetric and M_g > 1:
        return _match_h_symmetric(target, mu, eps_d, sigma_inf,
                                   W, eta, M_g, beta, eps0, V0)
    else:
        return _match_h_general(target, mu, eps_d, sigma_inf,
                                 W, eta, M_g, beta, eps0, V0)


def _match_h_general(target, mu, eps_d, sigma_inf, W, eta, M_g, beta, eps0, V0):
    """General (no symmetry) h-correlator matching."""
    from .gateway import gateway_correlators

    if eps0 is None:
        eps0 = np.linspace(-0.8, 0.8, M_g)
    if V0 is None:
        V0 = np.full(M_g, 0.3)
    x0 = np.real(np.concatenate([eps0, V0]))

    def residual(x):
        eps_x = x[:M_g]
        V_x = x[M_g:]
        corr = gateway_correlators(mu, eps_d, sigma_inf,
                                    V_x, eps_x, W, eta, beta)
        pred = np.real(np.concatenate([corr['hh'], corr['dh']]))
        return pred - target

    result = least_squares(residual, x0, method='lm', max_nfev=5000)
    return result.x[M_g:], result.x[:M_g]  # V, eps


def _match_h_symmetric(target, mu, eps_d, sigma_inf, W, eta, M_g, beta,
                        eps0, V0):
    """PH-symmetric h-correlator matching."""
    from .gateway import gateway_correlators

    n_pairs = M_g // 2
    has_center = M_g % 2 == 1
    n_params = n_pairs + n_pairs + (1 if has_center else 0)

    # Initial guess
    x0 = np.zeros(n_params)
    if eps0 is not None and V0 is not None:
        # Extract from full arrays
        x0[:n_pairs] = np.abs(eps0[M_g // 2 + (1 if has_center else 0):
                                    M_g // 2 + (1 if has_center else 0) + n_pairs])
        x0[n_pairs:2*n_pairs] = np.abs(V0[-n_pairs:]) if n_pairs > 0 else []
        if has_center:
            x0[-1] = np.abs(V0[M_g // 2])
    else:
        x0[:n_pairs] = np.linspace(0.2, 0.8, n_pairs) if n_pairs > 0 else []
        x0[n_pairs:2*n_pairs] = 0.3
        if has_center:
            x0[-1] = 0.3

    def _unpack(x):
        eps_pos = x[:n_pairs]
        V_pair = x[n_pairs:2*n_pairs]
        V_center = x[-1] if has_center else None
        eps_full = np.concatenate([-eps_pos[::-1],
                                    [0.0] if has_center else [],
                                    eps_pos])
        V_full = np.concatenate([V_pair[::-1],
                                  [V_center] if has_center else [],
                                  V_pair])
        return eps_full, V_full

    def residual(x):
        eps_full, V_full = _unpack(x)
        corr = gateway_correlators(mu, eps_d, sigma_inf,
                                    V_full, eps_full, W, eta, beta)
        pred = np.real(np.concatenate([corr['hh'], corr['dh']]))
        return pred - target

    result = least_squares(residual, x0, method='lm', max_nfev=5000)
    eps_full, V_full = _unpack(result.x)
    return V_full, eps_full


def match_g_correlators(target_gg: np.ndarray, target_dg: np.ndarray,
                         mu: float, eps_d: float, sigma_inf: float,
                         V: np.ndarray, eps: np.ndarray,
                         M_h: int, beta: float,
                         eta0: np.ndarray = None, W0: np.ndarray = None,
                         symmetric: bool = True) -> tuple:
    """Find ghost parameters {eta, W} by matching g-sector correlators.

    Impurity <-> gateway matching: adjust ghost poles so the gateway model's
    g-sector correlators match the impurity g-sector correlators.

    Solves:
        gateway_correlators(...)['gg'] = target_gg
        gateway_correlators(...)['dg'] = target_dg

    Parameters
    ----------
    target_gg : array, shape (M_g,)
        Target <g_l^dag g_l> from impurity.
    target_dg : array, shape (M_g,)
        Target <d^dag g_l> from impurity.
    mu, eps_d, sigma_inf : float
        Fixed parameters for the gateway model.
    V, eps : arrays, shape (M_g,)
        Current bath pole parameters (fixed during this matching).
    M_h : int
        Number of ghost poles to determine.
    beta : float
    eta0, W0 : arrays, optional
        Initial guess for ghost parameters.
    symmetric : bool
        Enforce particle-hole symmetry.

    Returns
    -------
    W : array, shape (M_h,)
    eta : array, shape (M_h,)
    """
    from .gateway import gateway_correlators

    target = np.real(np.concatenate([target_gg, target_dg]))

    if symmetric and M_h > 1:
        return _match_g_symmetric(target, mu, eps_d, sigma_inf,
                                   V, eps, M_h, beta, eta0, W0)
    else:
        return _match_g_general(target, mu, eps_d, sigma_inf,
                                 V, eps, M_h, beta, eta0, W0)


def _match_g_general(target, mu, eps_d, sigma_inf, V, eps, M_h, beta,
                      eta0, W0):
    """General (no symmetry) g-correlator matching."""
    from .gateway import gateway_correlators

    if eta0 is None:
        eta0 = np.linspace(-0.8, 0.8, M_h)
    if W0 is None:
        W0 = np.full(M_h, 0.3)
    x0 = np.real(np.concatenate([eta0, W0]))

    def residual(x):
        eta_x = x[:M_h]
        W_x = x[M_h:]
        corr = gateway_correlators(mu, eps_d, sigma_inf,
                                    V, eps, W_x, eta_x, beta)
        pred = np.real(np.concatenate([corr['gg'], corr['dg']]))
        return pred - target

    result = least_squares(residual, x0, method='lm', max_nfev=5000)
    return result.x[M_h:], result.x[:M_h]  # W, eta


def _match_g_symmetric(target, mu, eps_d, sigma_inf, V, eps, M_h, beta,
                        eta0, W0):
    """PH-symmetric g-correlator matching."""
    from .gateway import gateway_correlators

    n_pairs = M_h // 2
    has_center = M_h % 2 == 1
    n_params = n_pairs + n_pairs + (1 if has_center else 0)

    # Initial guess
    x0 = np.zeros(n_params)
    if eta0 is not None and W0 is not None:
        x0[:n_pairs] = np.abs(eta0[M_h // 2 + (1 if has_center else 0):
                                    M_h // 2 + (1 if has_center else 0) + n_pairs])
        x0[n_pairs:2*n_pairs] = np.abs(W0[-n_pairs:]) if n_pairs > 0 else []
        if has_center:
            x0[-1] = np.abs(W0[M_h // 2])
    else:
        x0[:n_pairs] = np.linspace(0.3, 1.0, n_pairs) if n_pairs > 0 else []
        x0[n_pairs:2*n_pairs] = 0.3
        if has_center:
            x0[-1] = 0.3

    def _unpack(x):
        eta_pos = x[:n_pairs]
        W_pair = x[n_pairs:2*n_pairs]
        W_center = x[-1] if has_center else None
        eta_full = np.concatenate([-eta_pos[::-1],
                                    [0.0] if has_center else [],
                                    eta_pos])
        W_full = np.concatenate([W_pair[::-1],
                                  [W_center] if has_center else [],
                                  W_pair])
        return eta_full, W_full

    def residual(x):
        eta_full, W_full = _unpack(x)
        corr = gateway_correlators(mu, eps_d, sigma_inf,
                                    V, eps, W_full, eta_full, beta)
        pred = np.real(np.concatenate([corr['gg'], corr['dg']]))
        return pred - target

    result = least_squares(residual, x0, method='lm', max_nfev=5000)
    eta_full, W_full = _unpack(result.x)
    return W_full, eta_full
