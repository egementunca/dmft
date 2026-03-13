import numpy as np

from dmft.phase_scan import GhostDMFT_M, c_op, run_phase_scan


def test_c_op_canonical_anticommutation():
    dim = 16  # 4 fermionic modes
    nmode = int(np.log2(dim))
    C = [c_op(dim, m) for m in range(nmode)]
    I = np.eye(dim)

    for i in range(nmode):
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            anti = C[i] @ C[i].conj().T + C[i].conj().T @ C[i]
        np.testing.assert_allclose(anti, I, atol=1e-12)

    for i in range(nmode):
        for j in range(i + 1, nmode):
            with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                anti = C[i] @ C[j] + C[j] @ C[i]
            np.testing.assert_allclose(anti, 0.0, atol=1e-12)


def test_ghost_dmft_adapter_contract():
    model = GhostDMFT_M(U=2.0, t=0.5, M=1, nquad=20, n_matsubara=64, ed=0.0)
    res = model.ghost_dmft(
        beta=20.0,
        eta0=np.array([0.0]),
        W0=np.array([0.25]),
        eps0=np.array([0.0]),
        V0=np.array([0.35]),
        mix=0.05,
        tol=1e-3,
        maxiter=8,
        verbose=False,
        polish_iters=0,
    )

    for key in ["eta", "W", "eps", "V", "docc", "nd_imp", "Z", "sigma_inf", "raw"]:
        assert key in res

    assert res["eta"].shape == (1,)
    assert res["W"].shape == (1,)
    assert res["eps"].shape == (1,)
    assert res["V"].shape == (1,)
    assert np.isfinite(res["docc"])
    assert np.isfinite(res["nd_imp"])
    assert np.isfinite(res["Z"])


def test_run_phase_scan_smoke():
    df, bounds = run_phase_scan(
        U_vals=np.array([2.0, 2.2]),
        T_vals=np.array([0.12]),
        M=1,
        t=0.5,
        nquad=20,
        n_matsubara=64,
        mix=0.05,
        tol=1e-3,
        maxiter=8,
    )
    assert len(df) == 2
    assert {"U", "T", "stable_phase", "coexistence"}.issubset(df.columns)
    assert len(bounds) == 1
    assert {"T", "Uc1", "Uc2", "Uc_eq"}.issubset(bounds.columns)
