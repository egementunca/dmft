"""Main DMFT self-consistency loop.

Variant A (standard): Uses Bethe lattice self-consistency + pole bath fitting.
1. Initialize Sigma = sigma_inf (constant)
2. G_loc from Bethe formula with Sigma
3. Delta_cav = t^2 * G_loc (Bethe self-consistency)
4. Fit Delta_cav to poles -> {eps_l, V_l}
5. Solve impurity -> G_imp, Sigma_imp, n_imp
6. sigma_inf <- U * n_imp (tail constraint for pole representation)
7. New Sigma from Dyson: Sigma = G_0^{-1} - G_imp^{-1}
8. Mix Sigma and refit ghost poles to track mixed self-energy
9. Check convergence, iterate

Variant B (two-ghost): Uses correlator matching between lattice/impurity
and the gateway model.
1. Initialize pole parameters {eps, V, eta, W, sigma_inf}
2. Build Sigma from poles -> G_loc (Bethe) -> lattice h-correlators
3. Match h-correlators (lattice <-> gateway) -> update {eps, V}
4. Solve impurity with {eps, V} -> g-correlators
5. Match g-correlators (impurity <-> gateway) -> update {eta, W}
6. Update sigma_inf = U * n_imp (tail constraint)
7. Check convergence, iterate
"""

import numpy as np
from .config import DMFTParams, PoleParams
from .matsubara import matsubara_frequencies
from .lattice import bethe_local_gf, bethe_self_consistency, lattice_correlators
from .greens_function import hybridization, self_energy_poles
from .gateway import gateway_correlators
from .matching import (
    fit_hybridization_poles,
    fit_self_energy_poles,
    match_h_correlators,
    match_g_correlators,
)
from .observables import impurity_g_correlators
from .solvers.base import ImpuritySolver


def dmft_loop(params: DMFTParams, solver: ImpuritySolver,
              initial_poles: PoleParams = None,
              verbose: bool = True) -> dict:
    """Run the DMFT self-consistency loop (Variant A).

    Convention used in this codebase:
    `sigma_inf` is treated as a self-energy tail constraint in the outer loop
    and pole representation. The impurity Hamiltonian itself is unshifted
    (Option A), so the solver is called with `sigma_inf=0`.

    Practical ED stabilization in Variant A:
    - use conservative Sigma mixing to avoid oscillatory ED updates

    Parameters
    ----------
    params : DMFTParams
        Physical and numerical parameters.
    solver : ImpuritySolver
        Impurity solver instance.
    initial_poles : PoleParams, optional
        Initial pole parameters. If None, use symmetric defaults.
    verbose : bool
        Print iteration info.

    Returns
    -------
    dict with:
        'G_loc': converged local Green's function
        'Sigma': converged self-energy
        'poles': converged PoleParams
        'Z': quasiparticle weight
        'n_imp': impurity occupancy
        'history': list of dicts per iteration
    """
    wn = matsubara_frequencies(params.n_matsubara, params.beta)
    iw = 1j * wn

    # Initialize
    if initial_poles is None:
        initial_poles = PoleParams.initial_symmetric(
            params.M_g, params.M_h, params.U, params.t
        )
    poles = initial_poles.copy()

    is_ed_solver = solver.__class__.__name__ == "EDSolver"
    mix_sigma = float(np.clip(params.mix, 0.0, 1.0))
    if is_ed_solver:
        # ED Sigma updates are much stiffer than IPT; aggressive mixing
        # can drive the loop into nonphysical fixed points.
        mix_sigma = min(mix_sigma, 0.01)

    Sigma = self_energy_poles(iw, poles.W, poles.eta, poles.sigma_inf)

    history = []

    for iteration in range(params.max_iter):
        # 1. Lattice step: G_loc from Bethe lattice
        G_loc = bethe_local_gf(iw, params.mu, params.eps_d, Sigma, params.t)

        # 2. Bethe self-consistency: new hybridization
        Delta_new = bethe_self_consistency(G_loc, params.t)

        # 3. Fit hybridization to poles
        V_new, eps_new = fit_hybridization_poles(
            Delta_new, iw, params.M_g, symmetric=True
        )
        poles.V = V_new
        poles.eps = eps_new

        # 4. Build the Weiss field for the impurity solver
        Delta_pole = hybridization(iw, poles.V, poles.eps)

        # 5. Solve impurity
        result = solver.solve(
            iw, params.mu, params.eps_d, params.U,
            poles.V, poles.eps, params.beta, 0.0
        )

        G_imp = result['G_imp']
        Sigma_new = result['Sigma_imp']
        n_imp = result['n_imp']

        # 6. Update sigma_inf
        poles.sigma_inf = params.U * n_imp

        # 7. Mix self-energy
        Sigma_mixed = mix_sigma * Sigma_new + (1.0 - mix_sigma) * Sigma

        # 8. Keep the ghost pole representation synchronized with Sigma.
        # This is needed for pole-based observables (e.g. real-axis A(w)).
        W_new, eta_new = fit_self_energy_poles(
            Sigma_mixed, iw, poles.sigma_inf, params.M_h, symmetric=True
        )
        poles.W = W_new
        poles.eta = eta_new

        # 9. Convergence check
        diff = np.max(np.abs(Sigma_mixed - Sigma)) / max(np.max(np.abs(Sigma)), 1e-10)

        # Store history
        Z = _quasiparticle_weight(Sigma_new, wn)
        info = {
            'iteration': iteration,
            'diff': diff,
            'Z': Z,
            'n_imp': n_imp,
            'mix_sigma': mix_sigma,
        }
        history.append(info)

        if verbose:
            print(f"  iter {iteration:3d}: diff={diff:.2e}  Z={Z:.4f}  n={n_imp:.4f}")

        Sigma = Sigma_mixed

        if diff < params.tol:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations.")
            break
    else:
        if verbose:
            print(f"  WARNING: Not converged after {params.max_iter} iterations.")

    return {
        'G_loc': G_loc,
        'Sigma': Sigma,
        'poles': poles,
        'Z': Z,
        'n_imp': n_imp,
        'history': history,
        'iw': iw,
        'wn': wn,
    }


def dmft_loop_two_ghost(params: DMFTParams, solver: ImpuritySolver,
                         initial_poles: PoleParams = None,
                         verbose: bool = True,
                         use_correlator_matching: bool = True,
                         ghost_update_mode: str = None,
                         symmetric: bool = True,
                         bath_mix: float = None,
                         ghost_mix: float = None,
                         sigma_mix: float = 0.0,
                         h_reg_strength: float = 1e-3,
                         g_reg_strength: float = 1e-3,
                         h_scale_floor_hh: float = 5e-2,
                         h_scale_floor_dh: float = 1e-2,
                         g_scale_floor_gg: float = 5e-2,
                         g_scale_floor_dg: float = 1e-2,
                         tol_h: float = 1e-2,
                         tol_g: float = 1e-2,
                         strict_stationarity: bool = False,
                         polish_iters: int = 20,
                         polish_sigma_mix: float = 0.0,
                         convergence_metric: str = 'sigma',
                         pole_collision_tol: float = 1e-8,
                         residual_growth_factor: float = 1.05,
                         causality_tol: float = 1e-8,
                         max_causality_backtrack: int = 8) -> dict:
    """Run the two-ghost DMFT loop (Variant B).

    Notes-faithful Variant B:
    both bath poles {eps, V} and ghost poles {eta, W} are updated from
    Hellmann-Feynman correlator matching with the quadratic gateway model
    H_imp^(0). A debug fallback can update ghosts by pole-fitting Sigma_imp.

    Convention used in this codebase:
    `sigma_inf` is an outer-loop tail constraint (Option A). The interacting
    impurity Hamiltonian is unshifted, so impurity solving and impurity↔gateway
    g-sector matching both use `sigma_inf=0` on the impurity/g-matching side.

    Algorithm:
    1. Build Sigma(iw) = sigma_inf + sum_l |W_l|^2 / (iw - eta_l)
    2. Lattice step: G_loc on Bethe from Sigma
    3. Match lattice h-correlators to gateway h-correlators -> update {eps, V}
    4. Solve interacting impurity with updated bath -> impurity g-correlators
    5. sigma_inf <- U * n_imp (tail constraint)
    6. Match impurity g-correlators to gateway g-correlators -> update {eta, W}
       (or debug fallback: fit Sigma_imp poles)
    7. Rebuild Sigma from updated poles, check convergence, iterate.

    Parameters
    ----------
    params : DMFTParams
    solver : ImpuritySolver
    initial_poles : PoleParams, optional
    verbose : bool
    use_correlator_matching : bool
        Backward-compatible switch. Ignored when `ghost_update_mode` is set.
    ghost_update_mode : {'correlator', 'fit'}, optional
        Ghost update method. Default resolves from `use_correlator_matching`.
        `'correlator'` is the notes-faithful default.
    symmetric : bool
        Enforce real PH-symmetric matching constraints.
    bath_mix, ghost_mix : float, optional
        Mixing factors (0..1) for bath and ghost parameters. Defaults to
        `params.mix`.
    sigma_mix : float
        Optional extra mixing directly on Sigma, followed by pole refit.
        Kept at 0 by default to preserve parameter-space iteration.
    h_reg_strength, g_reg_strength : float
        Drift regularization strengths for the h- and g-matching least squares.
    h_scale_floor_hh, h_scale_floor_dh, g_scale_floor_gg, g_scale_floor_dg : float
        Component-wise scale floors for matching residual normalization.
    tol_h, tol_g : float
        Stationarity tolerances on scaled h/g residual norms.
    strict_stationarity : bool
        If True, require `diff`, causality, and h/g residual thresholds to stop.
    polish_iters : int
        Two-stage stationarity polish iterations after diff+causality convergence.
    polish_sigma_mix : float
        Sigma mixing used in polish stage (0 holds Sigma fixed).
    convergence_metric : {'sigma', 'gloc'}
        Scalar metric used for stopping criterion.
    pole_collision_tol : float
        Minimum spacing threshold to flag nearly duplicated poles.
    residual_growth_factor : float
        Residual-increase warning threshold from one iteration to the next.
    causality_tol : float
        Threshold for causal Matsubara Green's function check:
        max_n Im G_loc(iw_n) < causality_tol.
    max_causality_backtrack : int
        Max backtracking steps on ghost updates if causality is violated.

    Returns
    -------
    dict with G_loc, Sigma, poles, Z, n_imp, history, iw, wn.
    """
    wn = matsubara_frequencies(params.n_matsubara, params.beta)
    iw = 1j * wn

    if initial_poles is None:
        initial_poles = PoleParams.initial_symmetric(
            params.M_g, params.M_h, params.U, params.t
        )
    poles = initial_poles.copy()

    if ghost_update_mode is None:
        ghost_update_mode = 'correlator' if use_correlator_matching else 'fit'
    if ghost_update_mode not in {'correlator', 'fit'}:
        raise ValueError("ghost_update_mode must be 'correlator' or 'fit'")
    if convergence_metric not in {'sigma', 'gloc'}:
        raise ValueError("convergence_metric must be 'sigma' or 'gloc'")

    bath_mix = params.mix if bath_mix is None else bath_mix
    ghost_mix = params.mix if ghost_mix is None else ghost_mix
    bath_mix = float(np.clip(bath_mix, 0.0, 1.0))
    ghost_mix = float(np.clip(ghost_mix, 0.0, 1.0))
    sigma_mix = float(np.clip(sigma_mix, 0.0, 1.0))
    polish_sigma_mix = float(np.clip(polish_sigma_mix, 0.0, 1.0))

    h_unknowns = _count_match_unknowns(params.M_g, symmetric=symmetric)
    g_unknowns = _count_match_unknowns(params.M_h, symmetric=symmetric)
    h_constraints = 2 * params.M_h
    g_constraints = 2 * params.M_g
    underdetermined_h = h_constraints < h_unknowns
    underdetermined_g = g_constraints < g_unknowns

    if underdetermined_h and h_reg_strength <= 0.0:
        h_reg_strength = 1e-6
    if underdetermined_g and g_reg_strength <= 0.0:
        g_reg_strength = 1e-6

    bath_energy_max = max(6.0, 3.0 * abs(params.U), 6.0 * abs(params.t))
    ghost_energy_max = max(6.0, 3.5 * abs(params.U))
    bath_coupling_max = max(6.0, 4.0 * abs(params.t), 2.0 * np.sqrt(max(abs(params.U), 1e-12)))
    ghost_coupling_max = max(6.0, 3.0 * abs(params.U))

    if verbose and (underdetermined_h or underdetermined_g):
        print(
            "  matching geometry:"
            f" h_constraints={h_constraints}, h_unknowns={h_unknowns}"
            f" g_constraints={g_constraints}, g_unknowns={g_unknowns}"
            f" (regularized LS active)"
        )

    # Start with initial self-energy from poles
    Sigma = self_energy_poles(iw, poles.W, poles.eta, poles.sigma_inf)
    last_causal_sigma = Sigma.copy()
    last_causal_poles = poles.copy()
    history = []
    prev_h_resid = None
    prev_g_resid = None
    h_targets = g_targets = None
    polish_mode = False
    polish_remaining = 0
    Sigma_polish_ref = None

    for iteration in range(params.max_iter):
        # 1) Lattice step from current ghost self-energy (or fixed Sigma in polish)
        Sigma_iter = Sigma_polish_ref if polish_mode else Sigma
        G_loc = bethe_local_gf(iw, params.mu, params.eps_d, Sigma_iter, params.t)

        # 2) Lattice h-sector correlators
        lat_corr = lattice_correlators(
            iw, G_loc, poles.W, poles.eta, params.beta
        )
        h_targets = {
            'hh': np.real(np.asarray(lat_corr['hh'])),
            'dh': np.real(np.asarray(lat_corr['dh'])),
        }

        # 3) Match h-correlators: lattice <-> gateway, update {eps, V}
        V_new, eps_new = match_h_correlators(
            h_targets['hh'], h_targets['dh'],
            params.mu, params.eps_d, poles.sigma_inf,
            poles.W, poles.eta, params.M_g, params.beta,
            eps0=poles.eps, V0=poles.V, symmetric=symmetric,
            reg_strength=h_reg_strength,
            scale_floor_hh=h_scale_floor_hh,
            scale_floor_dh=h_scale_floor_dh,
            energy_max=bath_energy_max,
            coupling_max=bath_coupling_max,
        )
        V_new, eps_new = _canonicalize_real_poles(V_new, eps_new, symmetric=symmetric)
        V_new, eps_new = _clip_poles(
            V_new, eps_new, coupling_max=bath_coupling_max, energy_max=bath_energy_max
        )
        poles.eps = _mix_parameters(poles.eps, eps_new, bath_mix)
        poles.V = _mix_parameters(poles.V, V_new, bath_mix)
        poles.V, poles.eps = _canonicalize_real_poles(
            poles.V, poles.eps, symmetric=symmetric
        )
        poles.V, poles.eps = _clip_poles(
            poles.V, poles.eps, coupling_max=bath_coupling_max, energy_max=bath_energy_max
        )

        h_pred = gateway_correlators(
            params.mu, params.eps_d, poles.sigma_inf,
            poles.V, poles.eps, poles.W, poles.eta, params.beta
        )
        h_pred_hh = np.real(np.asarray(h_pred['hh']))
        h_pred_dh = np.real(np.asarray(h_pred['dh']))
        h_delta_hh = h_pred_hh - h_targets['hh']
        h_delta_dh = h_pred_dh - h_targets['dh']
        h_abs_hh = np.abs(h_delta_hh)
        h_abs_dh = np.abs(h_delta_dh)
        h_scale_hh = np.maximum(np.abs(h_targets['hh']), h_scale_floor_hh)
        h_scale_dh = np.maximum(np.abs(h_targets['dh']), h_scale_floor_dh)
        h_resid = _scaled_residual_norm(
            h_delta_hh, h_delta_dh, h_scale_hh, h_scale_dh
        )
        h_resid_abs = _residual_norm(
            h_targets['hh'], h_targets['dh'],
            h_pred_hh, h_pred_dh,
        )

        # 4) Solve interacting impurity with updated bath (Option A: unshifted)
        sigma_inf_impurity = 0.0
        result = solver.solve(
            iw, params.mu, params.eps_d, params.U,
            poles.V, poles.eps, params.beta, sigma_inf_impurity
        )
        Sigma_imp = result['Sigma_imp']
        n_imp = result['n_imp']

        # 5) Tail constraint
        sigma_inf_new = params.U * n_imp

        # 6) Impurity g-sector correlators (targets for ghost update)
        if 'bath_gg' in result and 'bath_dg' in result:
            imp_gg = np.asarray(result['bath_gg'])
            imp_dg = np.asarray(result['bath_dg'])
        else:
            imp_corr_dict = impurity_g_correlators(
                iw, result['G_imp'], poles.V, poles.eps, params.beta
            )
            imp_gg = np.asarray(imp_corr_dict['gg'])
            imp_dg = np.asarray(imp_corr_dict['dg'])

        g_targets = {
            'gg': np.real(imp_gg),
            'dg': np.real(imp_dg),
        }

        old_W = poles.W.copy()
        old_eta = poles.eta.copy()

        # 7) Update ghost poles {eta, W}
        if ghost_update_mode == 'correlator':
            W_new, eta_new = match_g_correlators(
                g_targets['gg'], g_targets['dg'],
                params.mu, params.eps_d, sigma_inf_impurity,
                poles.V, poles.eps, params.M_h, params.beta,
                eta0=poles.eta, W0=poles.W, symmetric=symmetric,
                reg_strength=g_reg_strength,
                scale_floor_gg=g_scale_floor_gg,
                scale_floor_dg=g_scale_floor_dg,
                energy_max=ghost_energy_max,
                coupling_max=ghost_coupling_max,
            )
        else:
            W_new, eta_new = fit_self_energy_poles(
                Sigma_imp, iw, sigma_inf_new, params.M_h, symmetric=symmetric
            )
        W_new, eta_new = _canonicalize_real_poles(W_new, eta_new, symmetric=symmetric)
        W_new, eta_new = _clip_poles(
            W_new, eta_new, coupling_max=ghost_coupling_max, energy_max=ghost_energy_max
        )
        poles.eta = _mix_parameters(poles.eta, eta_new, ghost_mix)
        poles.W = _mix_parameters(poles.W, W_new, ghost_mix)
        poles.W, poles.eta = _canonicalize_real_poles(
            poles.W, poles.eta, symmetric=symmetric
        )
        poles.W, poles.eta = _clip_poles(
            poles.W, poles.eta, coupling_max=ghost_coupling_max, energy_max=ghost_energy_max
        )
        poles.sigma_inf = sigma_inf_new

        g_pred = gateway_correlators(
            params.mu, params.eps_d, sigma_inf_impurity,
            poles.V, poles.eps, poles.W, poles.eta, params.beta
        )
        g_pred_gg = np.real(np.asarray(g_pred['gg']))
        g_pred_dg = np.real(np.asarray(g_pred['dg']))
        g_delta_gg = g_pred_gg - g_targets['gg']
        g_delta_dg = g_pred_dg - g_targets['dg']
        g_abs_gg = np.abs(g_delta_gg)
        g_abs_dg = np.abs(g_delta_dg)
        g_scale_gg = np.maximum(np.abs(g_targets['gg']), g_scale_floor_gg)
        g_scale_dg = np.maximum(np.abs(g_targets['dg']), g_scale_floor_dg)
        g_resid = _scaled_residual_norm(
            g_delta_gg, g_delta_dg, g_scale_gg, g_scale_dg
        )
        g_resid_abs = _residual_norm(
            g_targets['gg'], g_targets['dg'],
            g_pred_gg, g_pred_dg,
        )

        # 8) Rebuild Sigma from updated poles and assess convergence
        Sigma_next = self_energy_poles(iw, poles.W, poles.eta, poles.sigma_inf)

        if sigma_mix > 0.0:
            Sigma_blend = sigma_mix * Sigma_next + (1.0 - sigma_mix) * Sigma_iter
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
                poles.W = last_causal_poles.W.copy()
                poles.eta = last_causal_poles.eta.copy()
                poles.sigma_inf = sigma_inf_new
                Sigma_next = last_causal_sigma.copy()
                G_loc_next = bethe_local_gf(iw, params.mu, params.eps_d, Sigma_next, params.t)
                backtrack_alpha = 0.0

        if polish_mode:
            Sigma_next = (
                (1.0 - polish_sigma_mix) * Sigma_polish_ref
                + polish_sigma_mix * Sigma_next
            )
            G_loc_next = bethe_local_gf(iw, params.mu, params.eps_d, Sigma_next, params.t)

        sigma_diff = _relative_change(Sigma_next, Sigma_iter)
        gloc_diff = _relative_change(G_loc_next, G_loc)
        diff = sigma_diff if convergence_metric == 'sigma' else gloc_diff

        causality_ok = _causality_ok(G_loc_next, causality_tol)
        max_imag_gloc = float(np.max(G_loc_next.imag))
        if causality_ok:
            last_causal_sigma = Sigma_next.copy()
            last_causal_poles = poles.copy()
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
        stationarity_ok = (h_resid < tol_h) and (g_resid < tol_g)
        stable_step = backtrack_alpha > 0.0

        max_abs_dhh = float(np.max(h_abs_hh)) if len(h_abs_hh) else 0.0
        max_abs_ddh = float(np.max(h_abs_dh)) if len(h_abs_dh) else 0.0
        max_abs_dgg = float(np.max(g_abs_gg)) if len(g_abs_gg) else 0.0
        max_abs_ddg = float(np.max(g_abs_dg)) if len(g_abs_dg) else 0.0
        max_t_hh = float(np.max(np.abs(h_targets['hh']))) if len(h_targets['hh']) else 0.0
        max_t_dh = float(np.max(np.abs(h_targets['dh']))) if len(h_targets['dh']) else 0.0
        max_t_gg = float(np.max(np.abs(g_targets['gg']))) if len(g_targets['gg']) else 0.0
        max_t_dg = float(np.max(np.abs(g_targets['dg']))) if len(g_targets['dg']) else 0.0

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
            'h_resid_abs': h_resid_abs,
            'g_resid': g_resid,
            'g_resid_abs': g_resid_abs,
            'stationarity_ok': stationarity_ok,
            'tol_h': tol_h,
            'tol_g': tol_g,
            'polish_mode': bool(polish_mode),
            'polish_remaining': int(polish_remaining),
            'stable_step': stable_step,
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
            'max_abs_dhh': max_abs_dhh,
            'max_abs_ddh': max_abs_ddh,
            'max_abs_dgg': max_abs_dgg,
            'max_abs_ddg': max_abs_ddg,
            'max_target_hh': max_t_hh,
            'max_target_dh': max_t_dh,
            'max_target_gg': max_t_gg,
            'max_target_dg': max_t_dg,
            'h_match': {
                'target_hh': h_targets['hh'].copy(),
                'target_dh': h_targets['dh'].copy(),
                'pred_hh': h_pred_hh.copy(),
                'pred_dh': h_pred_dh.copy(),
                'delta_hh': h_delta_hh.copy(),
                'delta_dh': h_delta_dh.copy(),
                'abs_delta_hh': h_abs_hh.copy(),
                'abs_delta_dh': h_abs_dh.copy(),
                'scale_hh': h_scale_hh.copy(),
                'scale_dh': h_scale_dh.copy(),
            },
            'g_match': {
                'target_gg': g_targets['gg'].copy(),
                'target_dg': g_targets['dg'].copy(),
                'pred_gg': g_pred_gg.copy(),
                'pred_dg': g_pred_dg.copy(),
                'delta_gg': g_delta_gg.copy(),
                'delta_dg': g_delta_dg.copy(),
                'abs_delta_gg': g_abs_gg.copy(),
                'abs_delta_dg': g_abs_dg.copy(),
                'scale_gg': g_scale_gg.copy(),
                'scale_dg': g_scale_dg.copy(),
            },
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
                f"abs[max|dhh|={max_abs_dhh:.2e}, max|ddh|={max_abs_ddh:.2e}, "
                f"max|dgg|={max_abs_dgg:.2e}, max|ddg|={max_abs_ddg:.2e}]  "
                f"tgt[max|hh|={max_t_hh:.2e}, max|dh|={max_t_dh:.2e}, "
                f"max|gg|={max_t_gg:.2e}, max|dg|={max_t_dg:.2e}]  "
                f"n={n_imp:.4f} sigma_inf={poles.sigma_inf:.4f} Z={Z:.4f}  "
                f"stationarity={stationarity_ok} polish={polish_mode}  "
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
        if strict_stationarity:
            if diff < params.tol and causality_ok and stationarity_ok and stable_step:
                if verbose:
                    print(f"  Converged (strict stationarity) after {iteration + 1} iterations.")
                break
            if diff < params.tol and (not causality_ok) and verbose:
                print("    diff below tol but causality check failed; continuing.")
            if diff < params.tol and causality_ok and (not stable_step) and verbose:
                print("    diff below tol but update was rejected by causality backtracking; continuing.")
            continue

        if polish_mode:
            if stationarity_ok and causality_ok:
                if verbose:
                    print(f"  Converged after polish at iteration {iteration + 1}.")
                break
            polish_remaining -= 1
            if polish_remaining <= 0:
                if verbose:
                    print(
                        "  WARNING: polish stage exhausted without stationarity "
                        f"(h_res={h_resid:.2e}, g_res={g_resid:.2e})."
                    )
                break
            continue

        if diff < params.tol and causality_ok and stable_step:
            if stationarity_ok or polish_iters <= 0:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations.")
                break
            polish_mode = True
            polish_remaining = polish_iters
            Sigma_polish_ref = Sigma.copy()
            if verbose:
                print(
                    f"  Entering stationarity polish for up to {polish_iters} iterations "
                    f"(tol_h={tol_h:.1e}, tol_g={tol_g:.1e}, sigma_mix={polish_sigma_mix:.2f})."
                )
            continue
        if diff < params.tol and (not causality_ok) and verbose:
            print("    diff below tol but causality check failed; continuing.")
        if diff < params.tol and causality_ok and (not stable_step) and verbose:
            print("    diff below tol but update was rejected by causality backtracking; continuing.")
    else:
        if verbose:
            print(f"  WARNING: Not converged after {params.max_iter} iterations.")

    G_loc_final = bethe_local_gf(iw, params.mu, params.eps_d, Sigma, params.t)

    return {
        'G_loc': G_loc_final,
        'Sigma': Sigma,
        'poles': poles,
        'Z': Z,
        'n_imp': n_imp,
        'history': history,
        'iw': iw,
        'wn': wn,
        'matching': {
            'h_targets': h_targets,
            'g_targets': g_targets,
            'h_constraints': h_constraints,
            'g_constraints': g_constraints,
            'h_unknowns': h_unknowns,
            'g_unknowns': g_unknowns,
            'h_scale_floors': {'hh': h_scale_floor_hh, 'dh': h_scale_floor_dh},
            'g_scale_floors': {'gg': g_scale_floor_gg, 'dg': g_scale_floor_dg},
            'tol_h': tol_h,
            'tol_g': tol_g,
            'strict_stationarity': strict_stationarity,
            'polish_iters': polish_iters,
            'polish_sigma_mix': polish_sigma_mix,
        },
    }


def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
    """Relative max norm change."""
    denom = max(np.max(np.abs(old)), 1e-12)
    return float(np.max(np.abs(new - old)) / denom)


def _mix_parameters(old: np.ndarray, new: np.ndarray, mix: float) -> np.ndarray:
    """Linear parameter mixing with no shape changes."""
    return (1.0 - mix) * np.asarray(old) + mix * np.asarray(new)


def _canonicalize_real_poles(coupling: np.ndarray, energy: np.ndarray,
                              symmetric: bool) -> tuple:
    """Fix gauge/ordering for real PH-symmetric runs."""
    e = np.real(np.asarray(energy).copy())
    v = np.real(np.asarray(coupling).copy())
    order = np.argsort(e)
    e = e[order]
    v = np.abs(v[order])  # positive-amplitude gauge

    if symmetric:
        n = len(e)
        half = n // 2
        if n % 2 == 1:
            e[half] = 0.0
        for i in range(half):
            val = 0.5 * (abs(e[i]) + abs(e[-1 - i]))
            e[i] = -val
            e[-1 - i] = val
            amp = 0.5 * (v[i] + v[-1 - i])
            v[i] = amp
            v[-1 - i] = amp
    return v, e


def _clip_poles(coupling: np.ndarray, energy: np.ndarray,
                 coupling_max: float, energy_max: float) -> tuple:
    """Clip poles to a bounded physically reasonable range."""
    v = np.clip(np.real(np.asarray(coupling)), 0.0, coupling_max)
    e = np.clip(np.real(np.asarray(energy)), -energy_max, energy_max)
    return v, e


def _residual_norm(target_diag: np.ndarray, target_off: np.ndarray,
                    pred_diag: np.ndarray, pred_off: np.ndarray) -> float:
    """L2 norm of combined matching residuals."""
    r = np.concatenate([
        np.real(np.asarray(pred_diag) - np.asarray(target_diag)),
        np.real(np.asarray(pred_off) - np.asarray(target_off)),
    ])
    return float(np.linalg.norm(r))


def _scaled_residual_norm(delta_diag: np.ndarray, delta_off: np.ndarray,
                           scale_diag: np.ndarray, scale_off: np.ndarray) -> float:
    """L2 norm of scale-normalized residual components."""
    r = np.concatenate([
        np.real(np.asarray(delta_diag) / np.asarray(scale_diag)),
        np.real(np.asarray(delta_off) / np.asarray(scale_off)),
    ])
    return float(np.linalg.norm(r))


def _count_match_unknowns(M: int, symmetric: bool) -> int:
    """Number of real unknowns in matching optimization."""
    if not symmetric:
        return 2 * M
    n_pairs = M // 2
    has_center = M % 2
    return 2 * n_pairs + has_center


def _has_pole_collision(energy: np.ndarray, tol: float) -> bool:
    """Detect nearly duplicated poles."""
    e = np.sort(np.real(np.asarray(energy)))
    if len(e) <= 1:
        return False
    return bool(np.min(np.abs(np.diff(e))) < tol)


def _causality_ok(G_iw: np.ndarray, tol: float = 1e-8) -> bool:
    """Causality check on Matsubara axis for w_n > 0."""
    return bool(np.max(np.imag(G_iw)) < tol)


def _ph_symmetry_ok(V: np.ndarray, eps: np.ndarray,
                     W: np.ndarray, eta: np.ndarray,
                     tol: float = 5e-4) -> bool:
    """Check PH symmetry constraints on pole sets."""
    return (
        _array_pair_symmetry_ok(eps, tol=tol)
        and _array_pair_symmetry_ok(eta, tol=tol)
        and _array_reverse_equal_ok(np.abs(V), tol=tol)
        and _array_reverse_equal_ok(np.abs(W), tol=tol)
    )


def _array_pair_symmetry_ok(x: np.ndarray, tol: float) -> bool:
    y = np.real(np.asarray(x))
    n = len(y)
    half = n // 2
    if n % 2 == 1 and abs(y[half]) > tol:
        return False
    for i in range(half):
        if abs(y[i] + y[-1 - i]) > tol:
            return False
    return True


def _array_reverse_equal_ok(x: np.ndarray, tol: float) -> bool:
    y = np.real(np.asarray(x))
    n = len(y)
    for i in range(n // 2):
        if abs(y[i] - y[-1 - i]) > tol:
            return False
    return True


def _poles_brief(energies: np.ndarray, couplings: np.ndarray, max_items: int = 2) -> str:
    """Compact pole summary for iteration logs."""
    e = np.real(np.asarray(energies))
    v = np.real(np.asarray(couplings))
    m = min(max_items, len(e))
    parts = [f"({e[i]:+.3f},{v[i]:.3f})" for i in range(m)]
    if len(e) > m:
        parts.append("...")
    return " ".join(parts)


def _quasiparticle_weight(Sigma_iw, wn):
    """Z = [1 - Im Sigma(iw_0) / w_0]^{-1}."""
    w0 = wn[0]
    dSigma = np.imag(Sigma_iw[0]) / w0
    Z = 1.0 / (1.0 - dSigma)
    return max(0.0, min(Z, 1.0))  # clamp to [0, 1]
