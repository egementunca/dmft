# TWO_GHOST_RESIDUAL_DIAGNOSTICS

> Note: This is a historical diagnostics snapshot from before the latest
> residual-conditioning fixes. For current behavior, read
> `docs/reports/RESIDUAL_FIX_NOTES.md`.

## Scope
- Conventions kept fixed (Bethe D=1, t=0.5, Option A, symmetric=True).
- Solver: ED. Variant B default mode: `ghost_update_mode="correlator"`; `fit` only for comparison.
- Milestone settings baseline: `beta=50`, `mix=0.05`, `tol=1e-4`, `h_reg=g_reg=1e-2`.

## A) Code Excerpts (Exact)

### A1. matching.py residuals/scaling/regularization/solver [/Users/egementunca/dmft/src/dmft/matching.py:141-402]
```python
def match_h_correlators(target_hh: np.ndarray, target_dh: np.ndarray,
                         mu: float, eps_d: float, sigma_inf: float,
                         W: np.ndarray, eta: np.ndarray,
                         M_g: int, beta: float,
                         eps0: np.ndarray = None, V0: np.ndarray = None,
                         symmetric: bool = True,
                         reg_strength: float = 0.0) -> tuple:
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
    scale = np.maximum(np.abs(target), 1e-3)

    if symmetric and M_g > 1:
        return _match_h_symmetric(target, mu, eps_d, sigma_inf,
                                   W, eta, M_g, beta, eps0, V0,
                                   reg_strength, scale)
    else:
        return _match_h_general(target, mu, eps_d, sigma_inf,
                                 W, eta, M_g, beta, eps0, V0,
                                 reg_strength, scale)


def _match_h_general(target, mu, eps_d, sigma_inf, W, eta, M_g, beta, eps0, V0,
                     reg_strength, scale):
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
        r = (pred - target) / scale
        if reg_strength > 0.0:
            r_reg = np.sqrt(reg_strength) * (x - x0)
            return np.concatenate([r, r_reg])
        return r

    result = least_squares(residual, x0, method='trf', max_nfev=5000)
    return result.x[M_g:], result.x[:M_g]  # V, eps


def _match_h_symmetric(target, mu, eps_d, sigma_inf, W, eta, M_g, beta,
                        eps0, V0, reg_strength, scale):
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
        r = (pred - target) / scale
        if reg_strength > 0.0:
            r_reg = np.sqrt(reg_strength) * (x - x0)
            return np.concatenate([r, r_reg])
        return r

    result = least_squares(residual, x0, method='trf', max_nfev=5000)
    eps_full, V_full = _unpack(result.x)
    return V_full, eps_full


def match_g_correlators(target_gg: np.ndarray, target_dg: np.ndarray,
                         mu: float, eps_d: float, sigma_inf: float,
                         V: np.ndarray, eps: np.ndarray,
                         M_h: int, beta: float,
                         eta0: np.ndarray = None, W0: np.ndarray = None,
                         symmetric: bool = True,
                         reg_strength: float = 0.0) -> tuple:
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
    scale = np.maximum(np.abs(target), 1e-3)

    if symmetric and M_h > 1:
        return _match_g_symmetric(target, mu, eps_d, sigma_inf,
                                   V, eps, M_h, beta, eta0, W0,
                                   reg_strength, scale)
    else:
        return _match_g_general(target, mu, eps_d, sigma_inf,
                                 V, eps, M_h, beta, eta0, W0,
                                 reg_strength, scale)


def _match_g_general(target, mu, eps_d, sigma_inf, V, eps, M_h, beta,
                      eta0, W0, reg_strength, scale):
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
        r = (pred - target) / scale
        if reg_strength > 0.0:
            r_reg = np.sqrt(reg_strength) * (x - x0)
            return np.concatenate([r, r_reg])
        return r

    result = least_squares(residual, x0, method='trf', max_nfev=5000)
    return result.x[M_h:], result.x[:M_h]  # W, eta


def _match_g_symmetric(target, mu, eps_d, sigma_inf, V, eps, M_h, beta,
                        eta0, W0, reg_strength, scale):
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
        r = (pred - target) / scale
        if reg_strength > 0.0:
            r_reg = np.sqrt(reg_strength) * (x - x0)
            return np.concatenate([r, r_reg])
        return r

    result = least_squares(residual, x0, method='trf', max_nfev=5000)
```

### A2. dmft_loop.py convergence criterion + causality/backtracking [/Users/egementunca/dmft/src/dmft/dmft_loop.py:398-534]
```python
        # 8) Rebuild Sigma from updated poles and assess convergence
        Sigma_next = self_energy_poles(iw, poles.W, poles.eta, poles.sigma_inf)

        if sigma_mix > 0.0:
            Sigma_blend = sigma_mix * Sigma_next + (1.0 - sigma_mix) * Sigma
            W_fit, eta_fit = fit_self_energy_poles(
                Sigma_blend, iw, poles.sigma_inf, params.M_h, symmetric=symmetric
            )
            poles.W, poles.eta = _canonicalize_real_poles(
                W_fit, eta_fit, symmetric=symmetric
            )
            poles.W, poles.eta = _clip_poles(
                poles.W, poles.eta, coupling_max=ghost_coupling_max, energy_max=ghost_energy_max
            )
            Sigma_next = self_energy_poles(iw, poles.W, poles.eta, poles.sigma_inf)

        G_loc_next = bethe_local_gf(iw, params.mu, params.eps_d, Sigma_next, params.t)

        backtrack_alpha = 1.0
        if not _causality_ok(G_loc_next, causality_tol):
            found = False
            for k in range(max_causality_backtrack):
                alpha = 0.5 ** (k + 1)
                W_try = _mix_parameters(old_W, poles.W, alpha)
                eta_try = _mix_parameters(old_eta, poles.eta, alpha)
                W_try, eta_try = _canonicalize_real_poles(
                    W_try, eta_try, symmetric=symmetric
                )
                W_try, eta_try = _clip_poles(
                    W_try, eta_try,
                    coupling_max=ghost_coupling_max,
                    energy_max=ghost_energy_max,
                )
                Sigma_try = self_energy_poles(iw, W_try, eta_try, poles.sigma_inf)
                G_try = bethe_local_gf(iw, params.mu, params.eps_d, Sigma_try, params.t)
                if _causality_ok(G_try, causality_tol):
                    poles.W, poles.eta = W_try, eta_try
                    Sigma_next = Sigma_try
                    G_loc_next = G_try
                    backtrack_alpha = alpha
                    found = True
                    break
            if not found:
                poles.W, poles.eta = old_W, old_eta
                Sigma_next = self_energy_poles(iw, poles.W, poles.eta, poles.sigma_inf)
                G_loc_next = bethe_local_gf(iw, params.mu, params.eps_d, Sigma_next, params.t)
                backtrack_alpha = 0.0

        sigma_diff = _relative_change(Sigma_next, Sigma)
        gloc_diff = _relative_change(G_loc_next, G_loc)
        diff = sigma_diff if convergence_metric == 'sigma' else gloc_diff

        causality_ok = _causality_ok(G_loc_next, causality_tol)
        max_imag_gloc = float(np.max(G_loc_next.imag))
        ph_symmetric_ok = _ph_symmetry_ok(
            poles.V, poles.eps, poles.W, poles.eta
        ) if symmetric else True
        pole_collision = (
            _has_pole_collision(poles.eps, pole_collision_tol)
            or _has_pole_collision(poles.eta, pole_collision_tol)
        )
        noisy_impurity = (
            np.any(~np.isfinite(np.asarray(Sigma_imp)))
            or np.any(~np.isfinite(np.asarray(imp_gg)))
            or np.any(~np.isfinite(np.asarray(imp_dg)))
        )
        h_resid_growth = (
            prev_h_resid is not None and h_resid > residual_growth_factor * prev_h_resid
        )
        g_resid_growth = (
            prev_g_resid is not None and g_resid > residual_growth_factor * prev_g_resid
        )

        Z = _quasiparticle_weight(Sigma_next, wn)
        info = {
            'iteration': iteration,
            'diff': diff,
            'metric': convergence_metric,
            'sigma_diff': sigma_diff,
            'gloc_diff': gloc_diff,
            'Z': Z,
            'n_imp': n_imp,
            'sigma_inf': poles.sigma_inf,
            'h_resid': h_resid,
            'g_resid': g_resid,
            'ghost_update_mode': ghost_update_mode,
            'causality_ok': causality_ok,
            'max_imag_gloc': max_imag_gloc,
            'causality_backtrack_alpha': backtrack_alpha,
            'ph_symmetric_ok': ph_symmetric_ok,
            'pole_collision': pole_collision,
            'h_resid_growth': h_resid_growth,
            'g_resid_growth': g_resid_growth,
            'underdetermined_h': underdetermined_h,
            'underdetermined_g': underdetermined_g,
            'impurity_noisy': bool(noisy_impurity),
            'bath_poles': {
                'eps': poles.eps.copy(),
                'V': poles.V.copy(),
            },
            'ghost_poles': {
                'eta': poles.eta.copy(),
                'W': poles.W.copy(),
            },
        }
        history.append(info)

        if verbose:
            print(
                f"  iter {iteration:3d}: {convergence_metric}_diff={diff:.2e} "
                f"(Sigma={sigma_diff:.2e}, G={gloc_diff:.2e})  "
                f"h_res={h_resid:.2e} g_res={g_resid:.2e}  "
                f"n={n_imp:.4f} sigma_inf={poles.sigma_inf:.4f} Z={Z:.4f}  "
                f"bath[{_poles_brief(poles.eps, poles.V)}] "
                f"ghost[{_poles_brief(poles.eta, poles.W)}]"
            )
            if (not causality_ok) or pole_collision or h_resid_growth or g_resid_growth:
                print(
                    "    flags:"
                    f" causality_ok={causality_ok}"
                    f" max_imag_gloc={max_imag_gloc:.2e}"
                    f" backtrack_alpha={backtrack_alpha:.3f}"
                    f" pole_collision={pole_collision}"
                    f" h_resid_growth={h_resid_growth}"
                    f" g_resid_growth={g_resid_growth}"
                    f" impurity_noisy={bool(noisy_impurity)}"
                )

        Sigma = Sigma_next
        prev_h_resid = h_resid
        prev_g_resid = g_resid
        if diff < params.tol and causality_ok:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations.")
            break
        if diff < params.tol and (not causality_ok) and verbose:
            print("    diff below tol but causality check failed; continuing.")
```

### A3. lattice_correlators formulas and diagnostics [/Users/egementunca/dmft/src/dmft/lattice.py:128-208]
```python
def lattice_correlators(iw: np.ndarray, G_dd_lat: np.ndarray,
                        W: np.ndarray, eta: np.ndarray,
                        beta: float,
                        return_diagnostics: bool = False,
                        diagnostic_n_values: np.ndarray = None) -> dict:
    """Equal-time lattice correlators from Matsubara sums.

    Computes:
        <h_l^dag h_l>_lat  from G_lat^{h_l, h_l}
        <d^dag h_l>_lat    from G_lat^{h_l, d}  (note: <d^dag h_l> = G_{h_l,d})

    For the h-ghost GFs, we decompose into pole contributions that
    can be summed exactly, plus a remainder involving G_dd_lat.

    Parameters
    ----------
    iw, G_dd_lat, W, eta, beta : as above.

    Returns
    -------
    dict with:
        'hh': array, shape (M_h,) — <h_l^dag h_l>_lat
        'dh': array, shape (M_h,) — <d^dag h_l>_lat
        optional 'diagnostics': convergence curves vs n_matsubara
    """
    gf = lattice_h_sector_gf(iw, G_dd_lat, W, eta)
    M_h = len(eta)

    hh_corr = np.zeros(M_h)
    dh_corr = np.zeros(M_h, dtype=complex)

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
        }

    for l in range(M_h):
        # <h_l^dag h_l> = (1/beta) sum_n G^{hh}_l(iw_n) over all n
        # G^{hh}_l = 1/(iw - eta_l) + |W_l|^2/(iw - eta_l)^2 * G_dd

        # First term: 1/(iw - eta_l) -> occupancy sum with convergence factor = f(eta_l)
        term1 = fermi_function(np.array([eta[l]]), beta)[0]

        # Second term: dynamic part summed numerically with robust tail handling.
        F_hh = np.abs(W[l])**2 * G_dd_lat / (iw - eta[l])**2
        term2 = matsubara_sum_numerical(F_hh, beta).real

        hh_corr[l] = term1 + term2

        # <d^dag h_l> = (1/beta) sum_n G^{hd}_l(iw_n)
        # G^{hd}_l = W_l / (iw - eta_l) * G_dd  (for our matrix convention).
        F_dh = W[l] * G_dd_lat / (iw - eta[l])              # G_{h,d}
        F_hd = np.conj(W[l]) * G_dd_lat / (iw - eta[l])     # G_{d,h}
        dh_corr[l] = matsubara_sum_pair_numerical(
            F_dh, F_hd, beta, tail_c2_ab=W[l], tail_c2_ba=np.conj(W[l])
        )

        if return_diagnostics:
            hh_seq = matsubara_sum_convergence(
                F_hh, beta, n_values=diagnostic_n_values
            )['sum']
            dh_seq = matsubara_sum_pair_convergence(
                F_dh, F_hd, beta,
                tail_c2_ab=W[l], tail_c2_ba=np.conj(W[l]),
                n_values=diagnostic_n_values
            )['sum']
            diagnostics['hh'][l] = term1 + hh_seq.real
            diagnostics['dh'][l] = dh_seq

    out = {'hh': np.real_if_close(hh_corr), 'dh': np.real_if_close(dh_corr)}
    if diagnostics is not None:
        out['diagnostics'] = diagnostics
    return out
```

### A4. impurity_g_correlators formulas and diagnostics [/Users/egementunca/dmft/src/dmft/observables.py:106-187]
```python
def impurity_g_correlators(iw: np.ndarray, G_imp: np.ndarray,
                            V: np.ndarray, eps: np.ndarray,
                            beta: float,
                            return_diagnostics: bool = False,
                            diagnostic_n_values: np.ndarray = None) -> dict:
    """Compute impurity bath (g-sector) correlators via Matsubara sums.

    Uses Schur complement relations for the interacting impurity model:
        G_imp^{g_l,g_l}(iw) = 1/(iw - eps_l) + |V_l|^2/(iw - eps_l)^2 * G_imp(iw)
        G_imp^{g_l,d}(iw) = V_l/(iw - eps_l) * G_imp(iw)

    Then equal-time correlators from Matsubara sums.

    Parameters
    ----------
    iw : array, shape (N,)
        Positive Matsubara frequencies (1j * w_n).
    G_imp : array, shape (N,)
        Impurity Green's function on positive Matsubara frequencies.
    V, eps : arrays, shape (M_g,)
        Bath hybridization amplitudes and energies.
    beta : float
        Inverse temperature.

    Returns
    -------
    dict with:
        'gg': array (M_g,) — <g_l^dag g_l>_imp
        'dg': array (M_g,) — <d^dag g_l>_imp
        optional 'diagnostics': convergence curves vs n_matsubara
    """
    M_g = len(eps)
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
            'gg': np.zeros((M_g, len(n_values))),
            'dg': np.zeros((M_g, len(n_values)), dtype=complex),
        }

    for l in range(M_g):
        # <g_l^dag g_l>: first term 1/(iw - eps_l)
        # The occupancy sum (with convergence factor e^{iw 0+}) gives f(eps_l)
        term1 = fermi_function(np.array([eps[l]]), beta)[0]

        # Second term: |V_l|^2 * G_imp / (iw - eps_l)^2
        F_gg = np.abs(V[l])**2 * G_imp / (iw - eps[l])**2
        term2 = matsubara_sum_numerical(F_gg, beta).real

        gg_corr[l] = term1 + term2

        # <d^dag g_l>: sum_n G_{g_l,d}(iw_n), and G_{g_l,d}=V_l/(iw-eps_l)*G_imp
        F_dg = V[l] * G_imp / (iw - eps[l])              # G_{g,d}
        F_gd = np.conj(V[l]) * G_imp / (iw - eps[l])     # G_{d,g}
        dg_corr[l] = matsubara_sum_pair_numerical(
            F_dg, F_gd, beta, tail_c2_ab=V[l], tail_c2_ba=np.conj(V[l])
        )

        if return_diagnostics:
            gg_seq = matsubara_sum_convergence(
                F_gg, beta, n_values=diagnostic_n_values
            )['sum']
            dg_seq = matsubara_sum_pair_convergence(
                F_dg, F_gd, beta,
                tail_c2_ab=V[l], tail_c2_ba=np.conj(V[l]),
                n_values=diagnostic_n_values
            )['sum']
            diagnostics['gg'][l] = term1 + gg_seq.real
            diagnostics['dg'][l] = dg_seq

    out = {'gg': np.real_if_close(gg_corr), 'dg': np.real_if_close(dg_corr)}
    if diagnostics is not None:
        out['diagnostics'] = diagnostics
    return out
```

### A5. gateway_correlators conventions and orientation [/Users/egementunca/dmft/src/dmft/gateway.py:132-187]
```python
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
```

### A6. paired Matsubara off-diagonal sum logic [/Users/egementunca/dmft/src/dmft/matsubara.py:207-272]
```python
def matsubara_sum_pair_numerical(
    G_ab_positive: np.ndarray,
    G_ba_positive: np.ndarray,
    beta: float,
    tail_c2_ab: complex = 0.0,
    tail_c2_ba: complex = 0.0,
) -> complex:
    """Matsubara sum for off-diagonal correlators using paired blocks.

    For <c_b^dag c_a>, the needed sum is:
        (1/beta) sum_{all n} G_{ab}(iw_n)
      = (1/beta) sum_{n>=0} [G_{ab}(iw_n) + G_{ba}(iw_n)^*]

    This avoids assuming G_{ab}(-iw) = G_{ab}(iw)^*, which is generally false
    for off-diagonal blocks with complex couplings.

    The paired sum has a 1/(iw)^2 tail coefficient:
        c2_pair = c2_ab + c2_ba^*
    and for positive frequencies only:
        (1/beta) sum_{n>=0} 1/(iw_n)^2 = -beta/8.
    """
    if len(G_ab_positive) != len(G_ba_positive):
        raise ValueError("G_ab_positive and G_ba_positive must have same length.")

    n_w = len(G_ab_positive)
    wn = matsubara_frequencies(n_w, beta)
    iw = 1j * wn

    pair = G_ab_positive + np.conj(G_ba_positive)
    c2_pair = tail_c2_ab + np.conj(tail_c2_ba)
    pair_sub = pair - c2_pair / iw**2

    numerical = (1.0 / beta) * np.sum(pair_sub)
    analytic = c2_pair * (-beta / 8.0)
    return numerical + analytic


def matsubara_sum_pair_convergence(
    G_ab_positive: np.ndarray,
    G_ba_positive: np.ndarray,
    beta: float,
    tail_c2_ab: complex = 0.0,
    tail_c2_ba: complex = 0.0,
    n_values: np.ndarray = None,
) -> dict:
    """Convergence diagnostics for `matsubara_sum_pair_numerical`."""
    n_w = len(G_ab_positive)
    if n_w != len(G_ba_positive):
        raise ValueError("G_ab_positive and G_ba_positive must have same length.")

    base = matsubara_sum_convergence(
        np.zeros_like(G_ab_positive), beta, n_values=n_values
    )['n_matsubara']

    sums = np.array([
        matsubara_sum_pair_numerical(
            G_ab_positive[:n], G_ba_positive[:n], beta,
            tail_c2_ab=tail_c2_ab, tail_c2_ba=tail_c2_ba
        )
        for n in base
    ], dtype=complex)

    return {
        'n_matsubara': base,
        'sum': sums,
    }
```

## B) Runtime Diagnostics

### B0. Milestone script run
- Command run: `PYTHONPATH=src python3 scripts/two_ghost_milestones.py`.
- Values below are from direct sweeps with the same baseline settings.

### B1. Fixed U=2, beta=50, M in {1,2,3}, n_iw in {512,1024,2048}
| M | n_iw | iters | conv | diff | causal | h_resid | g_resid | sigma_inf | n_imp | Z | eps | V | eta | W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 105 | True | 3.775e-05 | True | 1.436e-02 | 2.540e-01 | 1.000000 | 0.500000 | 0.0043 | [0.] | [0.3466] | [0.] | [0.9569] |
| 1 | 1024 | 144 | True | 5.401e-05 | True | 9.233e-03 | 2.461e-01 | 1.000000 | 0.500000 | 0.0048 | [0.] | [0.3061] | [0.] | [0.9022] |
| 1 | 2048 | 105 | True | 6.794e-05 | True | 3.236e+02 | 2.425e-01 | 1.000000 | 0.500000 | 0.0051 | [0.] | [0.2932] | [0.] | [0.8805] |
| 2 | 512 | 113 | True | 1.053e-05 | True | 6.615e+00 | 7.427e-02 | 1.000000 | 0.500000 | 0.9354 | [-0.0881  0.0881] | [1.3062 1.3062] | [-0.2993  0.2993] | [0.0568 0.0568] |
| 2 | 1024 | 182 | True | 4.329e-05 | True | 2.688e-02 | 6.996e-02 | 1.000000 | 0.500000 | 0.9796 | [-0.092  0.092] | [1.3787 1.3787] | [-0.3866  0.3866] | [0.0399 0.0399] |
| 2 | 2048 | 109 | True | 4.938e-05 | True | 8.004e-02 | 7.026e-02 | 1.000000 | 0.500000 | 0.8391 | [-0.0869  0.0869] | [1.4178 1.4178] | [-0.3164  0.3164] | [0.0999 0.0999] |
| 3 | 512 | 184 | True | 3.058e-05 | True | 1.482e-01 | 1.961e-01 | 1.000000 | 0.500000 | 0.1484 | [-0.2792  0.      0.2792] | [0.6991 0.1541 0.6991] | [-0.7458  0.      0.7458] | [0.0022 0.1505 0.0022] |
| 3 | 1024 | 237 | True | 3.087e-05 | True | 1.206e-01 | 2.111e-01 | 1.000000 | 0.500000 | 0.2172 | [-0.226  0.     0.226] | [0.5813 0.1485 0.5813] | [-0.6206  0.      0.6206] | [0.0008 0.1193 0.0008] |
| 3 | 2048 | 149 | True | 9.276e-05 | True | 2.608e+01 | 5.590e-02 | 1.000000 | 0.500000 | 0.4772 | [-0.2819  0.      0.2819] | [1.468  0.2256 1.468 ] | [-0.7396  0.      0.7396] | [0.0302 0.0657 0.0302] |

### B2. Sensitivity at U=2, M=2, n_iw=512
- Sweep: `h_reg in {0,1e-4,1e-3,1e-2,1e-1}`, `mix_bath in {0.02,0.05,0.1}`, `mix_ghost in {0.02,0.05,0.1}`.
- `g_reg` fixed at `1e-2`.
| h_reg | mix_bath | mix_ghost | iters | conv | diff | causal | h_resid | g_resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0e+00 | 0.02 | 0.02 | 274 | True | 9.562e-05 | True | 2.901e+01 | 3.457e-02 |
| 0.0e+00 | 0.02 | 0.05 | 195 | True | 1.872e-05 | True | 1.781e+01 | 3.415e-02 |
| 0.0e+00 | 0.02 | 0.1 | 141 | True | 9.858e-05 | True | 9.217e-01 | 3.578e-02 |
| 0.0e+00 | 0.05 | 0.02 | 88 | True | 8.377e-05 | True | 3.583e+01 | 3.012e-02 |
| 0.0e+00 | 0.05 | 0.05 | 119 | True | 4.697e-05 | True | 3.060e+01 | 2.977e-02 |
| 0.0e+00 | 0.05 | 0.1 | 103 | True | 8.467e-06 | True | 2.870e+01 | 3.025e-02 |
| 0.0e+00 | 0.1 | 0.02 | 73 | True | 7.817e-05 | True | 3.741e+01 | 2.956e-02 |
| 0.0e+00 | 0.1 | 0.05 | 52 | True | 8.024e-05 | True | 3.068e+01 | 3.088e-02 |
| 0.0e+00 | 0.1 | 0.1 | 56 | True | 5.306e-05 | True | 2.946e+01 | 3.046e-02 |
| 1.0e-04 | 0.02 | 0.02 | 207 | True | 8.911e-05 | True | 2.919e+01 | 3.387e-02 |
| 1.0e-04 | 0.02 | 0.05 | 163 | True | 5.966e-05 | True | 8.577e+00 | 3.551e-02 |
| 1.0e-04 | 0.02 | 0.1 | 183 | True | 7.719e-05 | True | 7.686e+00 | 3.743e-02 |
| 1.0e-04 | 0.05 | 0.02 | 122 | True | 7.832e-05 | True | 3.530e+01 | 3.222e-02 |
| 1.0e-04 | 0.05 | 0.05 | 72 | True | 7.924e-05 | True | 2.977e+01 | 3.073e-02 |
| 1.0e-04 | 0.05 | 0.1 | 63 | True | 8.477e-05 | True | 1.481e+01 | 3.438e-02 |
| 1.0e-04 | 0.1 | 0.02 | 61 | True | 4.074e-05 | True | 3.309e+01 | 3.196e-02 |
| 1.0e-04 | 0.1 | 0.05 | 55 | True | 1.608e-05 | True | 2.483e+01 | 3.461e-02 |
| 1.0e-04 | 0.1 | 0.1 | 46 | True | 9.582e-05 | True | 2.820e+01 | 2.828e-02 |
| 1.0e-03 | 0.02 | 0.02 | 318 | True | 3.992e-05 | True | 2.412e+01 | 4.377e-02 |
| 1.0e-03 | 0.02 | 0.05 | 177 | True | 5.853e-06 | True | 9.369e+00 | 4.843e-02 |
| 1.0e-03 | 0.02 | 0.1 | 135 | True | 9.882e-05 | True | 1.529e-02 | 2.646e-02 |
| 1.0e-03 | 0.05 | 0.02 | 194 | True | 1.671e-05 | True | 3.027e+01 | 3.792e-02 |
| 1.0e-03 | 0.05 | 0.05 | 200 | True | 2.167e-05 | True | 1.254e+01 | 4.556e-02 |
| 1.0e-03 | 0.05 | 0.1 | 80 | True | 6.844e-05 | True | 1.127e-01 | 4.947e-02 |
| 1.0e-03 | 0.1 | 0.02 | 54 | True | 9.319e-05 | True | 3.274e+01 | 3.908e-02 |
| 1.0e-03 | 0.1 | 0.05 | 54 | True | 7.440e-05 | True | 3.075e+01 | 3.876e-02 |
| 1.0e-03 | 0.1 | 0.1 | 80 | True | 6.558e-05 | True | 5.012e-02 | 4.326e-02 |
| 1.0e-02 | 0.02 | 0.02 | 310 | True | 9.176e-05 | True | 6.878e+00 | 7.027e-02 |
| 1.0e-02 | 0.02 | 0.05 | 188 | True | 9.598e-05 | True | 1.781e-02 | 7.932e-02 |
| 1.0e-02 | 0.02 | 0.1 | 500 | False | 1.785e-03 | True | 1.989e-01 | 1.699e-01 |
| 1.0e-02 | 0.05 | 0.02 | 234 | True | 9.009e-05 | True | 6.568e+00 | 5.370e-02 |
| 1.0e-02 | 0.05 | 0.05 | 113 | True | 1.053e-05 | True | 6.615e+00 | 7.427e-02 |
| 1.0e-02 | 0.05 | 0.1 | 78 | True | 9.972e-05 | True | 1.024e-01 | 5.863e-02 |
| 1.0e-02 | 0.1 | 0.02 | 225 | True | 5.917e-05 | True | 1.039e+01 | 5.593e-02 |
| 1.0e-02 | 0.1 | 0.05 | 113 | True | 3.165e-06 | True | 7.408e+00 | 5.982e-02 |
| 1.0e-02 | 0.1 | 0.1 | 95 | True | 8.553e-05 | True | 2.461e-02 | 6.673e-02 |
| 1.0e-01 | 0.02 | 0.02 | 353 | True | 9.889e-05 | True | 1.679e-02 | 8.808e-02 |
| 1.0e-01 | 0.02 | 0.05 | 240 | True | 9.224e-05 | True | 1.252e-02 | 1.229e-01 |
| 1.0e-01 | 0.02 | 0.1 | 165 | True | 7.841e-05 | True | 1.569e-01 | 2.091e-01 |
| 1.0e-01 | 0.05 | 0.02 | 295 | True | 9.084e-05 | True | 3.664e+00 | 8.093e-02 |
| 1.0e-01 | 0.05 | 0.05 | 188 | True | 6.149e-06 | True | 2.187e+00 | 1.047e-01 |
| 1.0e-01 | 0.05 | 0.1 | 500 | False | 1.552e-15 | False | 4.764e+00 | 1.490e-01 |
| 1.0e-01 | 0.1 | 0.02 | 307 | True | 7.854e-05 | True | 4.756e+00 | 8.239e-02 |
| 1.0e-01 | 0.1 | 0.05 | 143 | True | 8.911e-05 | True | 1.943e+00 | 9.772e-02 |
| 1.0e-01 | 0.1 | 0.1 | 82 | True | 8.477e-05 | True | 1.018e+00 | 1.388e-01 |

### B3. ghost_update_mode comparison at U=2, M=2, n_iw=512
| mode | iters | conv | diff | causal | h_resid | g_resid | backtrack_count | backtrack_fail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correlator | 113 | True | 1.053e-05 | True | 6.615e+00 | 7.427e-02 | 54 | 54 |
| fit | 205 | True | 7.967e-05 | True | 1.892e-02 | 2.845e-01 | 115 | 115 |

## C) Interpretation
- n_iw trend: M=1: h_resid 1.436e-02->3.236e+02, g_resid 2.540e-01->2.425e-01; M=2: h_resid 6.615e+00->8.004e-02, g_resid 7.427e-02->7.026e-02; M=3: h_resid 1.482e-01->2.608e+01, g_resid 1.961e-01->5.590e-02
- n_iw quantification: M=1: Δh=+2253230.0% Δg=-4.5% (512->2048); M=2: Δh=-98.8% Δg=-5.4% (512->2048); M=3: Δh=+17504.6% Δg=-71.5% (512->2048)
- Residual component dominance (representative U=2, M=2, n_iw=512): h-sector dominated by dh (l2 hh=1.674e-02, dh=6.615e+00); g-sector dominated by gg (l2 gg=7.267e-02, dg=1.531e-02)
- Conditioning/scaling signal: Average residuals at h_reg=0: h=2.671e+01, g=3.173e-02, backtrack_fail=62.2; at h_reg=1e-2: h=4.245e+00, g=7.651e-02, backtrack_fail=102.2.
- Mixing tradeoff signal: mix_bath=0.02: mean h=8.933e+00, mean g=7.130e-02; mix_bath=0.05: mean h=1.612e+01, mean g=5.611e-02; mix_bath=0.1: mean h=1.818e+01, mean g=5.389e-02; mix_ghost=0.02: mean h=2.128e+01, mean g=4.963e-02; mix_ghost=0.05: mean h=1.421e+01, mean g=5.780e-02; mix_ghost=0.1: mean h=7.748e+00, mean g=7.387e-02
- Mode comparison signal: correlator: h=6.615e+00, g=7.427e-02, backtrack_fail=54; fit: h=1.892e-02, g=2.845e-01, backtrack_fail=115
- Diagnosis verdict: Primary issue is mixed: (a) convergence/stationarity mismatch (diff+causality stop can pass with large residuals), (b) conditioning sensitivity (strong dependence on h_reg and frequent backtrack failures), and (c) finite-M projection floor (notably M=1 g_resid staying O(1e-1) across n_iw).
- Convergence metric mismatch: stopping uses `diff` + causality only; residual norms are monitored but not part of the stopping condition.
- Practical implication: runs can satisfy `diff` while retaining large h/g residuals (projection mismatch not ruled out by current stop rule).

### Per-component residual breakdown (representative U=2, M=2, n_iw=512)
| component | l2 | linf | mean_abs | values |
| --- | --- | --- | --- | --- |
| hh | 1.674e-02 | 1.183e-02 | 1.183e-02 | [-0.01183  0.01183] |
| dh | 6.615e+00 | 4.678e+00 | 4.678e+00 | [-4.67783 -4.67783] |
| gg | 7.267e-02 | 5.437e-02 | 5.129e-02 | [-0.04822 -0.05437] |
| dg | 1.531e-02 | 1.474e-02 | 9.440e-03 | [0.01474 0.00414] |

## D) Ranked Recommendations
1. Add residual-aware stopping: require `diff < tol_diff` AND `(h_resid < tol_h)` AND `(g_resid < tol_g)` for true stationarity-converged runs.
2. Keep dimensionless scaling in matching (already present) but report weighted residual components explicitly (`hh`, `dh`, `gg`, `dg`) each iteration.
3. Add explicit bounds in least-squares (`least_squares(..., bounds=...)`) for pole energies/amplitudes; current clipping happens outside optimizer.
4. For robustness, regularize pole movement with separate strengths for energies and couplings, not a single isotropic penalty on packed vector `x`.
5. If residual floors persist with increasing n_iw and stable conditioning, treat as finite-M projection error and increase M or add extra moments/constraints.
6. Use `ghost_update_mode="fit"` only as debug baseline; do not judge stationarity quality from fit-mode residuals.

## E) Plots
![residual_vs_niw.png](diagnostics/residual_vs_niw.png)
![residual_vs_iter_M2_niw512.png](diagnostics/residual_vs_iter_M2_niw512.png)

## Repro Commands
```bash
PYTHONPATH=src python3 scripts/generate_two_ghost_residual_diagnostics.py
PYTHONPATH=src python3 scripts/two_ghost_milestones.py
```
