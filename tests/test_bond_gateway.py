"""Tests for bond gateway static correlator functions."""

import numpy as np
import pytest
from dmft.gateway import gateway_statics, bond_gateway_statics


class TestGatewayStatics:
    """Tests for gateway_statics (single-site static correlators)."""

    def test_occupancy_bounds(self):
        """All correlator occupancies should be in [0, 1]."""
        nh, dh, ng, dg = gateway_statics(
            beta=10.0, eta=np.array([-0.3]),
            W_ghost=np.array([0.2]),
            eps=np.array([-0.1]), V=np.array([0.4]),
            M=1, shift=0.0)
        for arr in [nh, ng]:
            assert np.all(arr >= -1e-10), f"Occupancy below 0: {arr}"
            assert np.all(arr <= 1.0 + 1e-10), f"Occupancy above 1: {arr}"

    def test_output_shapes(self):
        """Output arrays should have shape (M,)."""
        M = 2
        nh, dh, ng, dg = gateway_statics(
            beta=5.0, eta=np.array([-0.3, 0.1]),
            W_ghost=np.array([0.2, 0.1]),
            eps=np.array([-0.1, 0.05]), V=np.array([0.4, 0.3]),
            M=M, shift=0.0)
        assert nh.shape == (M,)
        assert dg.shape == (M,)


class TestBondGatewayStatics:
    """Tests for bond_gateway_statics (two-site quadratic gateway)."""

    def test_matrix_dimension(self):
        """H2_gw should be (2+6M) x (2+6M)."""
        # Just check it runs without error for M=1 (n_gw=8)
        result = bond_gateway_statics(
            beta=5.0,
            eta=np.array([-0.3]), W_ghost=np.array([0.2]),
            eps=np.array([-0.1]), V=np.array([0.4]),
            etab=np.array([0.0]), Bh=np.array([0.1]),
            epsb=np.array([0.0]), Bg=np.array([0.1]),
            M=1, t=0.5, shift=0.0)
        assert len(result) == 9  # 8 arrays + n_site

    def test_n_site_range(self):
        """n_site should be in [0, 1]."""
        *_, n_site = bond_gateway_statics(
            beta=5.0,
            eta=np.array([-0.3]), W_ghost=np.array([0.2]),
            eps=np.array([-0.1]), V=np.array([0.4]),
            etab=np.array([0.0]), Bh=np.array([0.1]),
            epsb=np.array([0.0]), Bg=np.array([0.1]),
            M=1, t=0.5, shift=0.0)
        assert 0.0 <= n_site <= 1.0 + 1e-10

    def test_output_shapes_M2(self):
        """All ghost arrays should have shape (M,) for M=2."""
        M = 2
        (nh, dh, nhb, dhb, ng, dg, ngb, dgb, n_site
         ) = bond_gateway_statics(
            beta=5.0,
            eta=np.array([-0.3, 0.1]),
            W_ghost=np.array([0.2, 0.1]),
            eps=np.array([-0.1, 0.05]),
            V=np.array([0.4, 0.3]),
            etab=np.array([0.0, 0.0]),
            Bh=np.array([0.1, 0.05]),
            epsb=np.array([0.0, 0.0]),
            Bg=np.array([0.1, 0.05]),
            M=M, t=0.5, shift=0.0)
        for arr in [nh, dh, nhb, dhb, ng, dg, ngb, dgb]:
            assert arr.shape == (M,), f"Wrong shape: {arr.shape}"
