"""Main DMFT self-consistency loop.

Variant A (standard): Uses Bethe lattice self-consistency + pole bath fitting.
1. Initialize Sigma = sigma_inf (constant)
2. G_loc from Bethe formula with Sigma
3. Delta_cav = t^2 * G_loc (Bethe self-consistency)
4. Fit Delta_cav to poles -> {eps_l, V_l}
5. Solve impurity -> G_imp, Sigma_imp, n_imp
6. sigma_inf <- U * n_imp
7. New Sigma from Dyson: Sigma = G_0^{-1} - G_imp^{-1}
8. Mix, check convergence, iterate

Variant B (two-ghost): Uses correlator matching between lattice/impurity
and the gateway model.
1. Initialize pole parameters {eps, V, eta, W, sigma_inf}
2. Build Sigma from poles -> G_loc (Bethe) -> lattice h-correlators
3. Match h-correlators (lattice <-> gateway) -> update {eps, V}
4. Solve impurity with {eps, V} -> g-correlators
5. Match g-correlators (impurity <-> gateway) -> update {eta, W}
6. Update sigma_inf = U * n_imp
7. Check convergence, iterate
"""

import numpy as np
from .config import DMFTParams, PoleParams
from .matsubara import matsubara_frequencies
from .lattice import bethe_local_gf, bethe_self_consistency, lattice_correlators
from .greens_function import hybridization, self_energy_poles
from .matching import (
    fit_hybridization_poles,
    match_h_correlators,
    match_g_correlators,
)
from .observables import impurity_g_correlators
from .solvers.base import ImpuritySolver


def dmft_loop(params: DMFTParams, solver: ImpuritySolver,
              initial_poles: PoleParams = None,
              verbose: bool = True) -> dict:
    """Run the DMFT self-consistency loop (Variant A).

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

    # Start with initial self-energy from poles
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
            poles.V, poles.eps, params.beta, poles.sigma_inf
        )

        G_imp = result['G_imp']
        Sigma_new = result['Sigma_imp']
        n_imp = result['n_imp']

        # 6. Update sigma_inf
        poles.sigma_inf = params.U * n_imp

        # 7. Mix self-energy
        Sigma_mixed = params.mix * Sigma_new + (1.0 - params.mix) * Sigma

        # 8. Convergence check
        diff = np.max(np.abs(Sigma_mixed - Sigma)) / max(np.max(np.abs(Sigma)), 1e-10)

        # Store history
        Z = _quasiparticle_weight(Sigma_new, wn)
        info = {
            'iteration': iteration,
            'diff': diff,
            'Z': Z,
            'n_imp': n_imp,
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
                         use_correlator_matching: bool = False) -> dict:
    """Run the two-ghost DMFT loop (Variant B).

    Both bath (g) and ghost (h) sectors are represented with finite pole
    expansions, giving explicit pole-form Delta(iw) and Sigma(iw).

    Algorithm:
    1. Build Sigma from ghost poles -> G_loc (Bethe) -> lattice h-correlators
    2. Match h-correlators (lattice <-> gateway) -> update bath poles {eps, V}
    3. Solve impurity with bath -> Sigma_new, n_imp, g-correlators
    4. Update ghost poles {eta, W}:
       - Default: pole fitting of Sigma_dynamic
       - Optional: g-correlator matching (Hellmann-Feynman stationarity)
    5. sigma_inf = U * n_imp, mix Sigma, iterate

    Parameters
    ----------
    params : DMFTParams
    solver : ImpuritySolver
    initial_poles : PoleParams, optional
    verbose : bool
    use_correlator_matching : bool
        If True, determine ghost poles via g-correlator matching instead
        of pole fitting. Experimental — may require careful tuning.

    Returns
    -------
    dict with G_loc, Sigma, poles, Z, n_imp, history, iw, wn, imp_corr.
    """
    from .matching import fit_self_energy_poles

    wn = matsubara_frequencies(params.n_matsubara, params.beta)
    iw = 1j * wn

    if initial_poles is None:
        initial_poles = PoleParams.initial_symmetric(
            params.M_g, params.M_h, params.U, params.t
        )
    poles = initial_poles.copy()

    # Start with initial self-energy
    Sigma = self_energy_poles(iw, poles.W, poles.eta, poles.sigma_inf)
    history = []
    imp_gg = imp_dg = None

    for iteration in range(params.max_iter):
        # 1. Lattice step
        G_loc = bethe_local_gf(iw, params.mu, params.eps_d, Sigma, params.t)

        # 2. Lattice <-> gateway h-sector matching: update bath poles
        lat_corr = lattice_correlators(
            iw, G_loc, poles.W, poles.eta, params.beta
        )
        V_new, eps_new = match_h_correlators(
            lat_corr['hh'], lat_corr['dh'],
            params.mu, params.eps_d, poles.sigma_inf,
            poles.W, poles.eta, params.M_g, params.beta,
            eps0=poles.eps, V0=poles.V, symmetric=True
        )
        poles.V = V_new
        poles.eps = eps_new

        # 3. Solve impurity
        sigma_inf_impurity = poles.sigma_inf
        result = solver.solve(
            iw, params.mu, params.eps_d, params.U,
            poles.V, poles.eps, params.beta, sigma_inf_impurity
        )
        Sigma_new = result['Sigma_imp']
        n_imp = result['n_imp']

        # 4. Update sigma_inf
        sigma_inf_new = params.U * n_imp

        # 5. Update ghost poles
        if use_correlator_matching:
            # Get g-correlators from impurity
            if 'bath_gg' in result and 'bath_dg' in result:
                imp_gg = result['bath_gg']
                imp_dg = result['bath_dg']
            else:
                imp_corr_dict = impurity_g_correlators(
                    iw, result['G_imp'], poles.V, poles.eps, params.beta)
                imp_gg = imp_corr_dict['gg']
                imp_dg = imp_corr_dict['dg']

            # Match g-correlators -> {eta, W}
            W_new, eta_new = match_g_correlators(
                imp_gg, imp_dg,
                params.mu, params.eps_d, sigma_inf_impurity,
                poles.V, poles.eps, params.M_h, params.beta,
                eta0=poles.eta, W0=poles.W, symmetric=True
            )
            Sigma_from_ghosts = self_energy_poles(
                iw, W_new, eta_new, sigma_inf_new)
        else:
            # Direct pole fitting of self-energy (robust default)
            W_new, eta_new = fit_self_energy_poles(
                Sigma_new, iw, sigma_inf_new, params.M_h, symmetric=True)
            Sigma_from_ghosts = self_energy_poles(
                iw, W_new, eta_new, sigma_inf_new)

            # Compute g-correlators for diagnostics
            if 'bath_gg' in result and 'bath_dg' in result:
                imp_gg = result['bath_gg']
                imp_dg = result['bath_dg']

        # 6. Mix self-energy
        Sigma_mixed = params.mix * Sigma_from_ghosts + (1.0 - params.mix) * Sigma

        # 7. Convergence check
        diff = np.max(np.abs(Sigma_mixed - Sigma)) / max(
            np.max(np.abs(Sigma)), 1e-10)

        # 8. Re-extract ghost poles from mixed self-energy
        poles.W, poles.eta = fit_self_energy_poles(
            Sigma_mixed, iw, sigma_inf_new, params.M_h, symmetric=True)
        poles.sigma_inf = sigma_inf_new
        Sigma = Sigma_mixed

        Z = _quasiparticle_weight(Sigma_new, wn)
        info = {
            'iteration': iteration,
            'diff': diff,
            'Z': Z,
            'n_imp': n_imp,
        }
        history.append(info)

        if verbose:
            print(f"  iter {iteration:3d}: diff={diff:.2e}  Z={Z:.4f}  n={n_imp:.4f}")

        if diff < params.tol:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations.")
            break
    else:
        if verbose:
            print(f"  WARNING: Not converged after {params.max_iter} iterations.")

    G_loc_final = bethe_local_gf(iw, params.mu, params.eps_d, Sigma,
                                  params.t)

    out = {
        'G_loc': G_loc_final,
        'Sigma': Sigma,
        'poles': poles,
        'Z': Z,
        'n_imp': n_imp,
        'history': history,
        'iw': iw,
        'wn': wn,
    }
    if imp_gg is not None:
        out['imp_corr'] = {'gg': imp_gg, 'dg': imp_dg}
    return out


def _quasiparticle_weight(Sigma_iw, wn):
    """Z = [1 - Im Sigma(iw_0) / w_0]^{-1}."""
    w0 = wn[0]
    dSigma = np.imag(Sigma_iw[0]) / w0
    Z = 1.0 / (1.0 - dSigma)
    return max(0.0, min(Z, 1.0))  # clamp to [0, 1]
