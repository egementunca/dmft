"""Tests for the DMFT self-consistency loop (Variants A and B)."""

import numpy as np
import warnings
import pytest
from dmft.config import DMFTParams
from dmft.solvers.ipt import IPTSolver
from dmft.dmft_loop import dmft_loop, dmft_loop_two_ghost

warnings.filterwarnings('ignore')


@pytest.fixture
def ipt_solver():
    return IPTSolver()


def _half_filling_params(**kwargs):
    defaults = dict(U=2.0, beta=50.0, n_matsubara=512, M_g=2, M_h=2)
    defaults.update(kwargs)
    p = DMFTParams.half_filling(**defaults)
    p.max_iter = 60
    p.mix = 0.3
    return p


# ---------------------------------------------------------------------------
# Variant A tests
# ---------------------------------------------------------------------------

def test_variant_a_converges(ipt_solver):
    """Variant A converges within max_iter."""
    p = _half_filling_params()
    r = dmft_loop(p, ipt_solver, verbose=False)
    assert r['history'][-1]['diff'] < p.tol


def test_variant_a_half_filling(ipt_solver):
    """n = 0.5 at half-filling."""
    p = _half_filling_params()
    r = dmft_loop(p, ipt_solver, verbose=False)
    assert abs(r['n_imp'] - 0.5) < 0.01


def test_variant_a_z_physical(ipt_solver):
    """Z in (0, 1) for metallic phase."""
    p = _half_filling_params(U=2.0)
    r = dmft_loop(p, ipt_solver, verbose=False)
    assert 0.1 < r['Z'] < 1.0


def test_variant_a_causality(ipt_solver):
    """Im G_loc < 0 for positive frequencies."""
    p = _half_filling_params()
    r = dmft_loop(p, ipt_solver, verbose=False)
    assert np.all(r['G_loc'].imag < 0)


def test_variant_a_z_decreases_with_u(ipt_solver):
    """Z(U=3) < Z(U=1): stronger correlations reduce Z."""
    p1 = _half_filling_params(U=1.0)
    p2 = _half_filling_params(U=3.0)
    r1 = dmft_loop(p1, ipt_solver, verbose=False)
    r2 = dmft_loop(p2, ipt_solver, verbose=False)
    assert r2['Z'] < r1['Z']


# ---------------------------------------------------------------------------
# Variant B (two-ghost) tests
# ---------------------------------------------------------------------------

def test_variant_b_converges(ipt_solver):
    """Variant B converges within max_iter."""
    p = _half_filling_params()
    p.max_iter = 90
    r = dmft_loop_two_ghost(p, ipt_solver, verbose=False)
    assert r['history'][-1]['diff'] < p.tol


def test_variant_b_half_filling(ipt_solver):
    """n = 0.5 at half-filling for Variant B."""
    p = _half_filling_params()
    r = dmft_loop_two_ghost(p, ipt_solver, verbose=False)
    assert abs(r['n_imp'] - 0.5) < 0.01


def test_variant_b_z_physical(ipt_solver):
    """Z in (0, 1) for metallic phase with Variant B."""
    p = _half_filling_params(U=2.0)
    r = dmft_loop_two_ghost(p, ipt_solver, verbose=False)
    assert 0.0 < r['Z'] < 1.0


def test_variant_b_poles_available(ipt_solver):
    """Variant B returns valid pole parameters."""
    p = _half_filling_params()
    r = dmft_loop_two_ghost(p, ipt_solver, verbose=False)
    poles = r['poles']
    assert len(poles.V) == p.M_g
    assert len(poles.eps) == p.M_g
    assert len(poles.W) == p.M_h
    assert len(poles.eta) == p.M_h


def test_variant_a_b_both_physical(ipt_solver):
    """Both variants should return causal Green's functions and physical Z."""
    p = _half_filling_params(U=2.0)
    r_a = dmft_loop(p, ipt_solver, verbose=False)
    r_b = dmft_loop_two_ghost(p, ipt_solver, verbose=False)
    assert 0.0 < r_a['Z'] < 1.0
    assert 0.0 < r_b['Z'] < 1.0
    assert np.all(r_a['G_loc'].imag < 0)
    assert np.all(r_b['G_loc'].imag < 0)


def test_variant_b_uses_h_matching_for_bath(monkeypatch):
    """Variant B bath update should come from h-sector matching."""
    p = _half_filling_params(n_matsubara=64)
    p.max_iter = 1

    expected_V = np.array([0.17, 0.17])
    expected_eps = np.array([-0.23, 0.23])
    called = {'h_match': False}

    def _fail_fit(*args, **kwargs):
        raise AssertionError("Variant B should not call fit_hybridization_poles")

    def _mock_h_match(*args, **kwargs):
        called['h_match'] = True
        return expected_V.copy(), expected_eps.copy()

    monkeypatch.setattr("dmft.dmft_loop.fit_hybridization_poles", _fail_fit)
    monkeypatch.setattr("dmft.dmft_loop.match_h_correlators", _mock_h_match)

    class _DummySolver:
        def solve(self, iw, mu, eps_d, U, V, eps, beta, sigma_inf):
            return {
                'G_imp': 1.0 / (iw + 1.0),
                'Sigma_imp': np.full_like(iw, U / 2.0, dtype=complex),
                'n_imp': 0.5,
            }

    r = dmft_loop_two_ghost(
        p, _DummySolver(), verbose=False, bath_mix=1.0, ghost_mix=1.0
    )
    assert called['h_match']
    np.testing.assert_allclose(r['poles'].V, expected_V)
    np.testing.assert_allclose(r['poles'].eps, expected_eps)


def test_variant_b_default_uses_g_matching(monkeypatch):
    """Default Variant B mode should use correlator matching for ghost updates."""
    p = _half_filling_params(n_matsubara=64)
    p.max_iter = 1

    called = {'g_match': False}

    def _mock_h_match(*args, **kwargs):
        return np.array([0.2, 0.2]), np.array([-0.3, 0.3])

    def _mock_g_match(*args, **kwargs):
        called['g_match'] = True
        return np.array([0.25, 0.25]), np.array([-0.45, 0.45])

    def _fail_fit(*args, **kwargs):
        raise AssertionError("Default Variant B should not call fit_self_energy_poles")

    monkeypatch.setattr("dmft.dmft_loop.match_h_correlators", _mock_h_match)
    monkeypatch.setattr("dmft.dmft_loop.match_g_correlators", _mock_g_match)
    monkeypatch.setattr("dmft.dmft_loop.fit_self_energy_poles", _fail_fit)

    class _DummySolver:
        def solve(self, iw, mu, eps_d, U, V, eps, beta, sigma_inf):
            return {
                'G_imp': 1.0 / (iw + 1.0),
                'Sigma_imp': np.full_like(iw, U / 2.0, dtype=complex),
                'n_imp': 0.5,
                'bath_gg': np.array([0.3, 0.3]),
                'bath_dg': np.array([0.02, -0.02]),
            }

    dmft_loop_two_ghost(
        p, _DummySolver(), verbose=False, bath_mix=1.0, ghost_mix=1.0
    )
    assert called['g_match']


def test_variant_b_fit_mode_uses_self_energy_fit(monkeypatch):
    """Debug fit mode should bypass g-correlator matching."""
    p = _half_filling_params(n_matsubara=64)
    p.max_iter = 1

    called = {'fit': False}

    def _mock_h_match(*args, **kwargs):
        return np.array([0.2, 0.2]), np.array([-0.3, 0.3])

    def _fail_g_match(*args, **kwargs):
        raise AssertionError("fit mode should not call match_g_correlators")

    def _mock_fit(*args, **kwargs):
        called['fit'] = True
        return np.array([0.22, 0.22]), np.array([-0.42, 0.42])

    monkeypatch.setattr("dmft.dmft_loop.match_h_correlators", _mock_h_match)
    monkeypatch.setattr("dmft.dmft_loop.match_g_correlators", _fail_g_match)
    monkeypatch.setattr("dmft.dmft_loop.fit_self_energy_poles", _mock_fit)

    class _DummySolver:
        def solve(self, iw, mu, eps_d, U, V, eps, beta, sigma_inf):
            return {
                'G_imp': 1.0 / (iw + 1.0),
                'Sigma_imp': np.full_like(iw, U / 2.0, dtype=complex),
                'n_imp': 0.5,
                'bath_gg': np.array([0.3, 0.3]),
                'bath_dg': np.array([0.02, -0.02]),
            }

    dmft_loop_two_ghost(
        p, _DummySolver(), verbose=False,
        ghost_update_mode='fit', bath_mix=1.0, ghost_mix=1.0
    )
    assert called['fit']


def test_variant_b_g_matching_uses_impurity_sigma_inf(monkeypatch):
    """g-correlator matching should use the same sigma_inf as impurity solve."""
    p = _half_filling_params(U=4.0, n_matsubara=64)
    p.max_iter = 1

    captured = {}

    def _mock_h_match(*args, **kwargs):
        return np.array([0.2, 0.2]), np.array([-0.3, 0.3])

    def _mock_g_match(target_gg, target_dg, mu, eps_d, sigma_inf, *args, **kwargs):
        captured['sigma_inf_match'] = sigma_inf
        return np.array([0.25, 0.25]), np.array([-0.45, 0.45])

    monkeypatch.setattr("dmft.dmft_loop.match_h_correlators", _mock_h_match)
    monkeypatch.setattr("dmft.dmft_loop.match_g_correlators", _mock_g_match)

    class _CaptureSolver:
        def __init__(self):
            self.sigma_inf_used = None
            self.n_imp = 0.2

        def solve(self, iw, mu, eps_d, U, V, eps, beta, sigma_inf):
            self.sigma_inf_used = sigma_inf
            return {
                'G_imp': 1.0 / (iw + 1.0),
                'Sigma_imp': np.full_like(iw, U / 2.0, dtype=complex),
                'n_imp': self.n_imp,
                'bath_gg': np.array([0.3, 0.3]),
                'bath_dg': np.array([0.02, -0.02]),
            }

    solver = _CaptureSolver()
    dmft_loop_two_ghost(
        p, solver, verbose=False,
        ghost_update_mode='correlator',
        bath_mix=1.0,
        ghost_mix=1.0
    )

    assert np.isclose(captured['sigma_inf_match'], solver.sigma_inf_used)
    assert not np.isclose(captured['sigma_inf_match'], p.U * solver.n_imp)


def test_variant_b_does_not_converge_if_causality_fails(monkeypatch):
    """Variant B should not stop early when diff is small but causality fails."""
    p = _half_filling_params(n_matsubara=64)
    p.max_iter = 3

    monkeypatch.setattr("dmft.dmft_loop._causality_ok", lambda *args, **kwargs: False)

    def _mock_h_match(*args, **kwargs):
        return np.array([0.2, 0.2]), np.array([-0.3, 0.3])

    def _mock_g_match(*args, **kwargs):
        return np.array([0.25, 0.25]), np.array([-0.45, 0.45])

    monkeypatch.setattr("dmft.dmft_loop.match_h_correlators", _mock_h_match)
    monkeypatch.setattr("dmft.dmft_loop.match_g_correlators", _mock_g_match)

    class _DummySolver:
        def solve(self, iw, mu, eps_d, U, V, eps, beta, sigma_inf):
            return {
                'G_imp': 1.0 / (iw + 1.0),
                'Sigma_imp': np.full_like(iw, U / 2.0, dtype=complex),
                'n_imp': 0.5,
                'bath_gg': np.array([0.3, 0.3]),
                'bath_dg': np.array([0.02, -0.02]),
            }

    r = dmft_loop_two_ghost(
        p, _DummySolver(), verbose=False, bath_mix=1.0, ghost_mix=1.0
    )
    assert len(r['history']) == p.max_iter
