"""Tests for bond-scheme ED: impurity_statics and build_H2."""

import numpy as np
import pytest
from dmft.bond_ed import impurity_statics, build_H2, _get_H2_sector_cache
from math import comb


class TestImpurityStatics:
    """Tests for the single-site interacting impurity ED."""

    def test_docc_range_M1(self):
        """Double occupancy must be in [0, 0.25] at half-filling."""
        ng, dg, docc = impurity_statics(
            beta=10.0, eps=np.array([-0.1]), V=np.array([0.4]),
            M=1, U=1.3, mu=0.65)
        assert 0.0 <= docc <= 0.25 + 1e-12

    def test_docc_range_M2(self):
        """Double occupancy must be in [0, 0.25] for M=2."""
        ng, dg, docc = impurity_statics(
            beta=10.0, eps=np.array([-0.1, -0.05]),
            V=np.array([0.4, 0.3]),
            M=2, U=1.3, mu=0.65)
        assert 0.0 <= docc <= 0.25 + 1e-12

    def test_ng_occupancy_bounds(self):
        """g-ghost occupancies must be in [0, 1]."""
        ng, dg, docc = impurity_statics(
            beta=5.0, eps=np.array([-0.3]), V=np.array([0.2]),
            M=1, U=2.0, mu=1.0)
        assert np.all(ng >= -1e-12) and np.all(ng <= 1.0 + 1e-12)

    def test_finite_temperature_deterministic(self):
        """Same parameters should give identical results."""
        args = (10.0, np.array([-0.1]), np.array([0.4]), 1, 1.3, 0.65)
        r1 = impurity_statics(*args)
        r2 = impurity_statics(*args)
        np.testing.assert_allclose(r1[0], r2[0])
        np.testing.assert_allclose(r1[2], r2[2])

    def test_high_T_docc_approaches_quarter(self):
        """At very high T (beta→0), docc → 0.25 (uncorrelated)."""
        _, _, docc = impurity_statics(
            beta=0.01, eps=np.array([0.0]), V=np.array([0.0]),
            M=1, U=0.0, mu=0.0)
        assert abs(docc - 0.25) < 0.01


class TestBuildH2:
    """Tests for the two-site interacting cluster ED."""

    def test_nsite_range_M1(self):
        """nsite (per spin) should be in [0, 2] for two sites."""
        (ng, dg, ngb, dgb, docc, hop, nsite) = build_H2(
            beta=5.0, eps=np.array([-0.1]), V=np.array([0.4]),
            epsb=np.array([0.0]), Bg=np.array([0.1]),
            dmu=0.0, M=1, U=1.3, mu=0.65, t=0.5)
        assert 0.0 <= nsite <= 2.0 + 1e-12

    def test_docc_range_M1(self):
        """Double occupancy per site in [0, 0.5] for two sites."""
        (ng, dg, ngb, dgb, docc, hop, nsite) = build_H2(
            beta=5.0, eps=np.array([-0.1]), V=np.array([0.4]),
            epsb=np.array([0.0]), Bg=np.array([0.1]),
            dmu=0.0, M=1, U=1.3, mu=0.65, t=0.5)
        assert 0.0 <= docc <= 0.5 + 1e-12

    def test_ng_occupancy_bounds_M1(self):
        """g-ghost occupancies in [0, 1]."""
        (ng, dg, ngb, dgb, docc, hop, nsite) = build_H2(
            beta=5.0, eps=np.array([-0.1]), V=np.array([0.4]),
            epsb=np.array([0.0]), Bg=np.array([0.1]),
            dmu=0.0, M=1, U=1.3, mu=0.65, t=0.5)
        assert np.all(ng >= -1e-12) and np.all(ng <= 1.0 + 1e-12)
        assert np.all(ngb >= -1e-12) and np.all(ngb <= 1.0 + 1e-12)

    def test_output_shapes_M1(self):
        """Output arrays should have shape (M,) for M=1."""
        result = build_H2(
            beta=5.0, eps=np.array([-0.1]), V=np.array([0.4]),
            epsb=np.array([0.0]), Bg=np.array([0.1]),
            dmu=0.0, M=1, U=1.3, mu=0.65, t=0.5)
        ng, dg, ngb, dgb, docc, hop, nsite = result
        assert ng.shape == (1,)
        assert ngb.shape == (1,)


class TestSectorCache:
    """Tests for sector cache dimensions."""

    def test_sector_dimensions_M1(self):
        """For M=1 (nps=5), largest block at nup=2: C(5,2)*C(5,3)=100."""
        nps, occ, _, _ = _get_H2_sector_cache(M=1, nup=2)
        assert nps == 5
        expected = comb(5, 2) * comb(5, 3)
        assert occ.shape[1] == expected

    def test_sector_dimensions_M2(self):
        """For M=2 (nps=8), largest block at nup=4: C(8,4)*C(8,4)=4900."""
        nps, occ, _, _ = _get_H2_sector_cache(M=2, nup=4)
        assert nps == 8
        expected = comb(8, 4) * comb(8, 4)
        assert occ.shape[1] == expected

    def test_empty_sector(self):
        """Sector with nup > nps should return empty."""
        nps, occ, _, _ = _get_H2_sector_cache(M=1, nup=6)
        assert occ.shape[1] == 0
