"""Tests for square lattice construction and static correlators."""

import numpy as np
import pytest
from dmft.lattice import (
    make_square_lattice,
    lattice_statics,
    bond_lattice_statics,
)


class TestMakeSquareLattice:
    """Tests for make_square_lattice."""

    def test_bandwidth(self):
        """Half-bandwidth should be D=4t."""
        t = 0.5
        EPS, GAM, W, D, z = make_square_lattice(t, n_k=20)
        assert D == pytest.approx(4 * t)
        assert z == 4

    def test_weights_sum_to_one(self):
        """BZ weights should sum to 1."""
        _, _, W, _, _ = make_square_lattice(0.5, n_k=30)
        np.testing.assert_allclose(W.sum(), 1.0, atol=1e-14)

    def test_dispersion_range(self):
        """Dispersion should be within [-D, D]."""
        t = 0.5
        EPS, _, _, D, _ = make_square_lattice(t, n_k=50)
        assert np.all(EPS >= -D - 1e-10)
        assert np.all(EPS <= D + 1e-10)

    def test_gamma_range(self):
        """Bond form factor should be in [-1, 1]."""
        _, GAM, _, _, _ = make_square_lattice(0.5, n_k=30)
        assert np.all(GAM >= -1.0 - 1e-10)
        assert np.all(GAM <= 1.0 + 1e-10)

    def test_output_shapes(self):
        """All k-dependent outputs should have n_k^2 elements."""
        n_k = 20
        EPS, GAM, W, _, _ = make_square_lattice(0.5, n_k=n_k)
        assert EPS.shape == (n_k**2,)
        assert GAM.shape == (n_k**2,)
        assert W.shape == (n_k**2,)


class TestLatticeStatics:
    """Tests for static BZ-summed correlators."""

    @pytest.fixture
    def lattice(self):
        EPS, GAM, W, D, z = make_square_lattice(0.5, n_k=20)
        return EPS, GAM, W, D, z

    def test_nh_occupancy_bounds(self, lattice):
        """h-ghost occupancies should be in [0, 1]."""
        EPS, _, W, _, _ = lattice
        nh, dh = lattice_statics(
            beta=5.0, eta=np.array([-0.3]),
            W_ghost=np.array([0.2]), M=1, EPS=EPS, EPS_W=W, shift=0.0)
        assert np.all(nh >= -1e-10) and np.all(nh <= 1.0 + 1e-10)

    def test_output_shapes(self, lattice):
        """Output shapes should match M."""
        EPS, _, W, _, _ = lattice
        M = 2
        nh, dh = lattice_statics(
            beta=5.0, eta=np.array([-0.3, 0.1]),
            W_ghost=np.array([0.2, 0.1]), M=M,
            EPS=EPS, EPS_W=W, shift=0.0)
        assert nh.shape == (M,)
        assert dh.shape == (M,)


class TestBondLatticeStatics:
    """Tests for bond-extended lattice static correlators."""

    def test_output_shapes(self):
        """All outputs should have shape (M,)."""
        EPS, GAM, W, _, _ = make_square_lattice(0.5, n_k=10)
        M = 1
        nh, dh, nhb, dhb = bond_lattice_statics(
            beta=5.0, eta=np.array([-0.3]),
            W_ghost=np.array([0.2]),
            etab=np.array([0.0]), Bh=np.array([0.1]),
            M=M, EPS=EPS, GAM=GAM, EPS_W=W, shift=0.0)
        assert nh.shape == (M,)
        assert nhb.shape == (M,)

    def test_nhb_occupancy_bounds(self):
        """Bond hb-ghost occupancies should be in [0, 1]."""
        EPS, GAM, W, _, _ = make_square_lattice(0.5, n_k=10)
        _, _, nhb, _ = bond_lattice_statics(
            beta=5.0, eta=np.array([-0.3]),
            W_ghost=np.array([0.2]),
            etab=np.array([0.0]), Bh=np.array([0.1]),
            M=1, EPS=EPS, GAM=GAM, EPS_W=W, shift=0.0)
        assert np.all(nhb >= -1e-10) and np.all(nhb <= 1.0 + 1e-10)
