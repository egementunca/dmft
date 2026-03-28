"""Tests for the bond-scheme solver (M=1 only for speed)."""

import numpy as np
import pytest
from dmft.bond import solve_singlesite, solve_bond, run_temperature_sweep
from dmft.lattice import make_square_lattice


@pytest.fixture
def lattice():
    """Square lattice at t=0.5."""
    return make_square_lattice(0.5, n_k=10)  # small grid for speed


class TestSolveSinglesite:
    """Tests for the single-site solver."""

    def test_convergence_M1_high_T(self, lattice):
        """Single-site solver should converge at high T."""
        EPS, GAM, W, D, z = lattice
        result = solve_singlesite(
            beta=1.0,  # T=1
            eta0=np.array([-0.3]), W0=np.array([0.2]),
            eps0=np.array([-0.1]), V0=np.array([0.4]),
            M=1, U=1.3, t=0.5, mu=0.65, shift=0.0,
            EPS=EPS, EPS_W=W)
        assert result['iters'] < 200
        assert 0.0 < result['docc'] < 0.25

    def test_output_keys(self, lattice):
        """Should return expected dict keys."""
        EPS, _, W, _, _ = lattice
        result = solve_singlesite(
            beta=1.0,
            eta0=np.array([-0.3]), W0=np.array([0.2]),
            eps0=np.array([-0.1]), V0=np.array([0.4]),
            M=1, U=1.3, t=0.5, mu=0.65, shift=0.0,
            EPS=EPS, EPS_W=W)
        for key in ['eta', 'W', 'eps', 'V', 'docc', 'iters']:
            assert key in result


class TestSolveBond:
    """Tests for the bond solver."""

    def test_bond_convergence_M1_high_T(self, lattice):
        """Bond solver should converge at high T with M=1."""
        EPS, GAM, W, D, z = lattice
        ss = solve_singlesite(
            beta=1.0,
            eta0=np.array([-0.3]), W0=np.array([0.2]),
            eps0=np.array([-0.1]), V0=np.array([0.4]),
            M=1, U=1.3, t=0.5, mu=0.65, shift=0.0,
            EPS=EPS, EPS_W=W)
        rb = solve_bond(
            beta=1.0, ss=ss, M=1, U=1.3, t=0.5, mu=0.65,
            shift=0.0, EPS=EPS, GAM=GAM, EPS_W=W, z=z,
            maxiter=100)
        assert rb['res'] < 1e-4
        assert 0.0 < rb['docc_bpk'] < 0.25

    def test_bond_output_keys(self, lattice):
        """Should return expected dict keys."""
        EPS, GAM, W, D, z = lattice
        ss = solve_singlesite(
            beta=2.0,
            eta0=np.array([-0.3]), W0=np.array([0.2]),
            eps0=np.array([-0.1]), V0=np.array([0.4]),
            M=1, U=1.3, t=0.5, mu=0.65, shift=0.0,
            EPS=EPS, EPS_W=W)
        rb = solve_bond(
            beta=2.0, ss=ss, M=1, U=1.3, t=0.5, mu=0.65,
            shift=0.0, EPS=EPS, GAM=GAM, EPS_W=W, z=z,
            maxiter=50)
        for key in ['eta', 'W', 'etab', 'Bh', 'eps', 'V', 'epsb', 'Bg',
                    'dmu', 'docc_1', 'docc_2', 'docc_bpk', 'hop', 'res',
                    'iters']:
            assert key in rb


class TestTemperatureSweep:
    """Tests for run_temperature_sweep (ss mode only for speed)."""

    def test_ss_mode_produces_results(self):
        """SS-only sweep should produce correct output structure."""
        results, D = run_temperature_sweep(
            U=1.3, t=0.5, M=1, mode='ss', n_k=10,
            T_vals=np.array([1.0, 0.5]))
        assert len(results) == 2
        assert D == pytest.approx(2.0)
        assert 'docc_ss' in results[0]
        assert 'T' in results[0]
