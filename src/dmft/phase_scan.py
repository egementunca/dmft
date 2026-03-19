"""Phase-scan utilities for two-ghost DMFT branch continuation.

This module adapts a professor-style ``GhostDMFT_M`` scan workflow to this
codebase's Variant B loop (`dmft_loop_two_ghost`) and conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - optional dependency at runtime
    pd = None  # type: ignore[assignment]
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

from .config import DMFTParams, PoleParams
from .dmft_loop import dmft_loop_two_ghost
from .matsubara import matsubara_frequencies
from .solvers.ed import EDSolver


def _plt():
    """Lazy matplotlib import (avoids cache warnings for non-plot workflows)."""
    import matplotlib.pyplot as plt

    return plt


def c_op(dim: int, mode: int) -> np.ndarray:
    """Return the fermionic annihilation operator for a Fock-space mode.

    Basis states are encoded as integers ``|b_{N-1}...b_0>`` with mode ``0``
    mapped to the least-significant bit.
    """
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError("dim must be a positive power of two")
    nmode = int(np.log2(dim))
    if mode < 0 or mode >= nmode:
        raise ValueError(f"mode must satisfy 0 <= mode < {nmode}")

    op = np.zeros((dim, dim), dtype=complex)
    mask = 1 << mode
    for ket in range(dim):
        if ket & mask:
            # Jordan-Wigner sign from occupied lower-index modes.
            n_lower = bin(ket & (mask - 1)).count("1")
            sign = -1.0 if n_lower % 2 else 1.0
            bra = ket ^ mask
            op[bra, ket] = sign
    return op


def _bethe_dos_quadrature(nquad: int, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature for the Bethe DOS with half-bandwidth D=2t."""
    if nquad <= 1:
        raise ValueError("nquad must be > 1")
    if t <= 0:
        raise ValueError("t must be positive")

    x, w = np.polynomial.legendre.leggauss(nquad)
    D = 2.0 * t
    eps = D * x
    rho = (2.0 / (np.pi * D**2)) * np.sqrt(np.clip(D**2 - eps**2, 0.0, None))
    weights = w * D * rho
    return eps.astype(float), weights.astype(float)


def _as_seed_array(x: Any, M: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if len(arr) != M:
        raise ValueError(f"{name} must have length {M}, got {len(arr)}")
    return arr.copy()


def _default_seed(M: int, branch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if branch == "metal":
        eta0 = np.array([-0.20] * M, dtype=float)
        W0 = np.full(M, 0.45)
        eps0 = np.array([-0.05] * M, dtype=float)
        V0 = np.full(M, 0.50)
        return eta0, W0, eps0, V0
    if branch == "insulator":
        eta0 = np.array([0.90] * M, dtype=float)
        W0 = np.full(M, 0.05)
        eps0 = np.array([0.40] * M, dtype=float)
        V0 = np.full(M, 0.08)
        return eta0, W0, eps0, V0
    raise ValueError("branch must be 'metal' or 'insulator'")


@dataclass
class GhostDMFT_M:
    """Professor-style adapter around this repository's two-ghost DMFT loop."""

    U: float
    t: float = 0.5
    M: int = 1
    nquad: int = 120
    n_matsubara: int = 512
    ed: float = 0.0

    def __post_init__(self):
        self.mu = 0.5 * float(self.U)
        self.Sigma_inf = 0.5 * float(self.U)
        self._solver = EDSolver()
        self.EPS_NODES, self.EPS_W = _bethe_dos_quadrature(self.nquad, self.t)

    def ghost_dmft(
        self,
        beta: float,
        eta0: np.ndarray,
        W0: np.ndarray,
        eps0: np.ndarray,
        V0: np.ndarray,
        mix: float = 0.05,
        tol: float = 1e-4,
        maxiter: int = 160,
        verbose: bool = False,
        ghost_update_mode: str = "correlator",
        h_reg: float = 1e-2,
        g_reg: float = 1e-2,
        h_floor_hh: float = 5e-2,
        h_floor_dh: float = 1e-2,
        g_floor_gg: float = 5e-2,
        g_floor_dg: float = 1e-2,
        tol_h: float = 1e-2,
        tol_g: float = 1e-2,
        strict_stationarity: bool = False,
        polish_iters: int = 20,
    ) -> dict:
        """Run Variant B with professor-compatible argument/return shape."""
        params = DMFTParams(
            U=float(self.U),
            beta=float(beta),
            mu=0.5 * float(self.U),
            eps_d=float(self.ed),
            t=float(self.t),
            n_matsubara=int(self.n_matsubara),
            M_g=int(self.M),
            M_h=int(self.M),
            mix=float(mix),
            tol=float(tol),
            max_iter=int(maxiter),
        )

        init = PoleParams(
            eps=_as_seed_array(eps0, self.M, "eps0"),
            V=np.abs(_as_seed_array(V0, self.M, "V0")),
            eta=_as_seed_array(eta0, self.M, "eta0"),
            W=np.abs(_as_seed_array(W0, self.M, "W0")),
            sigma_inf=float(self.Sigma_inf),
        )

        loop = dmft_loop_two_ghost(
            params,
            self._solver,
            initial_poles=init,
            verbose=verbose,
            ghost_update_mode=ghost_update_mode,
            symmetric=True,
            bath_mix=mix,
            ghost_mix=mix,
            sigma_mix=0.0,
            h_reg_strength=h_reg,
            g_reg_strength=g_reg,
            h_scale_floor_hh=h_floor_hh,
            h_scale_floor_dh=h_floor_dh,
            g_scale_floor_gg=g_floor_gg,
            g_scale_floor_dg=g_floor_dg,
            tol_h=tol_h,
            tol_g=tol_g,
            strict_stationarity=strict_stationarity,
            polish_iters=polish_iters,
            convergence_metric="sigma",
        )

        poles = loop["poles"]
        self.Sigma_inf = float(poles.sigma_inf)

        # docc/nd_imp from one final impurity solve at converged bath poles.
        iw = 1j * matsubara_frequencies(params.n_matsubara, params.beta)
        imp = self._solver.solve(
            iw, params.mu, params.eps_d, params.U,
            poles.V, poles.eps, params.beta, 0.0
        )
        nd_per_spin = float(imp["n_imp"])
        docc = float(imp["n_double"])

        history = loop.get("history", [])
        last = history[-1] if history else {}
        converged = bool(
            float(last.get("diff", np.inf)) < params.tol
            and bool(last.get("causality_ok", True))
        )

        return {
            "eta": np.real(np.asarray(poles.eta)).copy(),
            "W": np.real(np.asarray(poles.W)).copy(),
            "eps": np.real(np.asarray(poles.eps)).copy(),
            "V": np.real(np.asarray(poles.V)).copy(),
            "sigma_inf": self.Sigma_inf,
            "docc": docc,
            "nd_imp": 2.0 * nd_per_spin,
            "Z": float(loop["Z"]),
            "diff": float(last.get("diff", np.nan)),
            "h_resid": float(last.get("h_resid", np.nan)),
            "g_resid": float(last.get("g_resid", np.nan)),
            "iters": int(len(history)),
            "converged": converged,
            "raw": loop,
        }


def impurity_free_energy(model: GhostDMFT_M, beta: float,
                         eps_g: np.ndarray, V: np.ndarray) -> float:
    """Grand potential of the interacting impurity model (spinful)."""
    U = model.U
    M = model.M
    Norb = 1 + M
    Nmode = 2 * Norb
    dim = 1 << Nmode

    C = [c_op(dim, mode) for mode in range(Nmode)]
    Cd = [op.conj().T for op in C]
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        n_ops = [Cd[m] @ C[m] for m in range(Nmode)]

    def n(orb: int, spin: int) -> np.ndarray:
        return n_ops[2 * orb + spin]

    d_up = C[0]
    d_dn = C[1]
    ddag_up = d_up.conj().T
    ddag_dn = d_dn.conj().T

    ed_eff = model.ed - model.mu
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        H = ed_eff * (n(0, 0) + n(0, 1)) + U * (n(0, 0) @ n(0, 1))

        for l in range(M):
            orb = 1 + l
            H += eps_g[l] * (n(orb, 0) + n(orb, 1))
            g_up = C[2 * orb + 0]
            g_dn = C[2 * orb + 1]
            gdag_up = g_up.conj().T
            gdag_dn = g_dn.conj().T
            H += V[l] * (ddag_up @ g_up + gdag_up @ d_up)
            H += V[l] * (ddag_dn @ g_dn + gdag_dn @ d_dn)

    evals = np.linalg.eigvalsh(H)
    e0 = float(np.min(evals))
    Z = np.sum(np.exp(-beta * (evals - e0)))
    return float(e0 - (1.0 / beta) * np.log(Z))


def gateway_free_energy(model: GhostDMFT_M, beta: float,
                        eta: np.ndarray, W: np.ndarray,
                        eps_g: np.ndarray, V: np.ndarray,
                        sigma_inf: float | None = None) -> float:
    """Grand potential of the quadratic gateway model H_imp^(0)."""
    M = model.M
    sigma_tail = model.Sigma_inf if sigma_inf is None else float(sigma_inf)
    shift = model.ed + sigma_tail - model.mu
    n = 1 + 2 * M
    H = np.zeros((n, n), dtype=float)

    H[0, 0] = shift
    H[0, 1:1 + M] = V
    H[1:1 + M, 0] = V
    H[0, 1 + M:] = W
    H[1 + M:, 0] = W
    H[1:1 + M, 1:1 + M] = np.diag(eps_g)
    H[1 + M:, 1 + M:] = np.diag(eta)

    evals = np.linalg.eigvalsh(H)
    return float(-(2.0 / beta) * np.sum(np.log1p(np.exp(-beta * evals))))


def lattice_free_energy(model: GhostDMFT_M, beta: float,
                        eta: np.ndarray, W: np.ndarray,
                        sigma_inf: float | None = None) -> float:
    """Lattice contribution to the functional from DOS quadrature."""
    sigma_tail = model.Sigma_inf if sigma_inf is None else float(sigma_inf)
    shift = model.ed + sigma_tail - model.mu
    fsum = 0.0

    for eps, ww in zip(model.EPS_NODES, model.EPS_W):
        H = np.zeros((1 + model.M, 1 + model.M), dtype=float)
        H[0, 0] = eps + shift
        H[0, 1:] = W
        H[1:, 0] = W
        H[1:, 1:] = np.diag(eta)

        evals = np.linalg.eigvalsh(H)
        fsum += ww * np.sum(np.log1p(np.exp(-beta * evals)))

    return float(-(2.0 / beta) * fsum)


def total_free_energy(model: GhostDMFT_M, beta: float, res: dict) -> float:
    """Functional value F = Ω_lat + Ω_imp - Ω_imp^(0)."""
    sigma_inf = float(res.get("sigma_inf", model.Sigma_inf))
    return (
        lattice_free_energy(model, beta, res["eta"], res["W"], sigma_inf=sigma_inf)
        + impurity_free_energy(model, beta, res["eps"], res["V"])
        - gateway_free_energy(
            model, beta, res["eta"], res["W"], res["eps"], res["V"],
            sigma_inf=sigma_inf
        )
    )


def z_proxy(model: GhostDMFT_M, eta: np.ndarray, W: np.ndarray,
            sigma_inf: float | None = None) -> float:
    """Simple d-weight proxy in the lowest local (d,h) eigenmode."""
    sigma_tail = model.Sigma_inf if sigma_inf is None else float(sigma_inf)
    H = np.zeros((1 + model.M, 1 + model.M), dtype=float)
    H[0, 0] = model.ed + sigma_tail - model.mu
    H[0, 1:] = W
    H[1:, 0] = W
    H[1:, 1:] = np.diag(eta)

    _, Umat = np.linalg.eigh(H)
    return float(Umat[0, 0] ** 2)


def _branch_valid(res: dict, branch: str,
                  z_metal_min: float,
                  z_ins_max: float,
                  v_ins_max: float,
                  use_branch_filters: bool) -> bool:
    if not bool(res.get("converged", False)):
        return False
    if not use_branch_filters:
        return True

    z = float(res.get("Z", np.nan))
    v = float(np.max(np.abs(np.asarray(res.get("V", []))))) if len(res.get("V", [])) else np.nan
    if not np.isfinite(z) or not np.isfinite(v):
        return False
    if branch == "metal":
        return z >= z_metal_min
    if branch == "insulator":
        return (z <= z_ins_max) and (v <= v_ins_max)
    raise ValueError("branch must be 'metal' or 'insulator'")


def _sign_crossing_root(x: np.ndarray, y: np.ndarray) -> float:
    for i in range(len(x) - 1):
        if not np.isfinite(y[i]) or not np.isfinite(y[i + 1]):
            continue
        if y[i] == 0.0:
            return float(x[i])
        if y[i] * y[i + 1] < 0:
            return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float("nan")


def extract_phase_boundaries(df):
    """Extract Uc1(T), Uc2(T), and equilibrium Uc(T) from a scan table."""
    if pd is None:  # pragma: no cover - runtime guard
        raise ImportError("pandas is required for phase-scan tables") from _PANDAS_IMPORT_ERROR

    rows = []
    for T in sorted(df["T"].unique()):
        sub = df[np.isclose(df["T"], T)].sort_values("U")
        metal_valid = sub["metal_valid"].astype(bool).to_numpy()
        ins_valid = sub["ins_valid"].astype(bool).to_numpy()

        uc2 = float(sub.loc[metal_valid, "U"].max()) if np.any(metal_valid) else np.nan
        uc1 = float(sub.loc[ins_valid, "U"].min()) if np.any(ins_valid) else np.nan

        both = sub["coexistence"].astype(bool).to_numpy()
        x = sub.loc[both, "U"].to_numpy(dtype=float)
        y = sub.loc[both, "deltaF"].to_numpy(dtype=float)
        uc_eq = _sign_crossing_root(x, y) if len(x) >= 2 else np.nan

        rows.append({"T": float(T), "Uc1": uc1, "Uc2": uc2, "Uc_eq": uc_eq})

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def run_phase_scan(
    U_vals: np.ndarray,
    T_vals: np.ndarray,
    M: int = 1,
    t: float = 0.5,
    nquad: int = 120,
    n_matsubara: int = 512,
    mix: float = 0.5,
    tol: float = 1e-8,
    maxiter: int = 100,
    compat_mode: bool = True,
    require_converged_for_valid: bool = False,
    use_branch_filters: bool = False,
    z_metal_min: float = 0.12,
    z_ins_max: float = 0.08,
    v_ins_max: float = 0.20,
    ins_seed_v_clip: float | None = None,
    coexist_docc_tol: float = 1e-3,
    coexist_z_tol: float = 1e-3,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Branch-continuation scan with metallic and insulating seeds.

    When ``compat_mode=True`` (default), coexistence/stability logic follows
    the professor script exactly:
    - valid branch if finite free energies on both branches,
    - stable phase from sign of deltaF,
    - coexistence flag from either |D_m - D_i| > 1e-6 or |deltaF| > 1e-10.

    ``compat_mode=False`` enables stricter branch validity and optional
    filtering/tolerances for diagnostic studies.
    """
    if pd is None:  # pragma: no cover - runtime guard
        raise ImportError("pandas is required for run_phase_scan") from _PANDAS_IMPORT_ERROR

    U_vals = np.asarray(U_vals, dtype=float)
    T_vals = np.asarray(T_vals, dtype=float)
    if U_vals.ndim != 1 or T_vals.ndim != 1:
        raise ValueError("U_vals and T_vals must be 1D arrays")
    if len(U_vals) == 0 or len(T_vals) == 0:
        raise ValueError("U_vals and T_vals must be non-empty")
    if np.any(T_vals <= 0):
        raise ValueError("All temperatures must satisfy T > 0")
    rows = []

    for T in T_vals:
        beta = 1.0 / T
        metal_seed = _default_seed(M, "metal")
        ins_seed = _default_seed(M, "insulator")
        metal_branch: dict[float, dict[str, Any]] = {}
        ins_branch: dict[float, dict[str, Any]] = {}

        # metallic continuation: low U -> high U
        for U in U_vals:
            model = GhostDMFT_M(U=float(U), t=t, M=M, nquad=nquad, n_matsubara=n_matsubara)
            try:
                res = model.ghost_dmft(
                    beta, *metal_seed, mix=mix, tol=tol, maxiter=maxiter, verbose=False
                )
                metal_seed = (res["eta"], res["W"], res["eps"], res["V"])
                metal_branch[float(U)] = {
                    "res": res,
                    "F": total_free_energy(model, beta, res),
                    "Z_proxy": z_proxy(model, res["eta"], res["W"], sigma_inf=res["sigma_inf"]),
                }
            except Exception as exc:  # pragma: no cover - runtime robustness
                metal_branch[float(U)] = {"error": str(exc)}

        # insulating continuation: high U -> low U
        for U in U_vals[::-1]:
            model = GhostDMFT_M(U=float(U), t=t, M=M, nquad=nquad, n_matsubara=n_matsubara)
            try:
                res = model.ghost_dmft(
                    beta, *ins_seed, mix=mix, tol=tol, maxiter=maxiter, verbose=False
                )
                next_v = res["V"] if ins_seed_v_clip is None else np.minimum(res["V"], ins_seed_v_clip)
                ins_seed = (res["eta"], res["W"], res["eps"], next_v)
                ins_branch[float(U)] = {
                    "res": res,
                    "F": total_free_energy(model, beta, res),
                    "Z_proxy": z_proxy(model, res["eta"], res["W"], sigma_inf=res["sigma_inf"]),
                }
            except Exception as exc:  # pragma: no cover - runtime robustness
                ins_branch[float(U)] = {"error": str(exc)}

        for U in U_vals:
            u = float(U)
            row: dict[str, Any] = {"U": u, "T": float(T)}
            m = metal_branch.get(u, {})
            i = ins_branch.get(u, {})

            if "res" in m:
                mr = m["res"]
                row.update(
                    {
                        "docc_metal": float(mr["docc"]),
                        "nd_metal": float(mr["nd_imp"]),
                        "F_metal": float(m["F"]),
                        "Z_metal": float(mr["Z"]),
                        "Zp_metal": float(m["Z_proxy"]),
                        "eta_metal": float(mr["eta"][0]),
                        "W_metal": float(mr["W"][0]),
                        "eps_metal": float(mr["eps"][0]),
                        "V_metal": float(mr["V"][0]),
                        "conv_metal": bool(mr["converged"]),
                    }
                )
                row["metal_valid"] = bool(
                    np.isfinite(row["F_metal"])
                    and ((not require_converged_for_valid) or bool(mr["converged"]))
                )
            else:
                row.update(
                    {
                        "docc_metal": np.nan,
                        "nd_metal": np.nan,
                        "F_metal": np.nan,
                        "Z_metal": np.nan,
                        "Zp_metal": np.nan,
                        "eta_metal": np.nan,
                        "W_metal": np.nan,
                        "eps_metal": np.nan,
                        "V_metal": np.nan,
                        "conv_metal": False,
                        "metal_valid": False,
                    }
                )

            if "res" in i:
                ir = i["res"]
                row.update(
                    {
                        "docc_insulator": float(ir["docc"]),
                        "nd_insulator": float(ir["nd_imp"]),
                        "F_insulator": float(i["F"]),
                        "Z_insulator": float(ir["Z"]),
                        "Zp_insulator": float(i["Z_proxy"]),
                        "eta_insulator": float(ir["eta"][0]),
                        "W_insulator": float(ir["W"][0]),
                        "eps_insulator": float(ir["eps"][0]),
                        "V_insulator": float(ir["V"][0]),
                        "conv_insulator": bool(ir["converged"]),
                    }
                )
                row["ins_valid"] = bool(
                    np.isfinite(row["F_insulator"])
                    and ((not require_converged_for_valid) or bool(ir["converged"]))
                )
            else:
                row.update(
                    {
                        "docc_insulator": np.nan,
                        "nd_insulator": np.nan,
                        "F_insulator": np.nan,
                        "Z_insulator": np.nan,
                        "Zp_insulator": np.nan,
                        "eta_insulator": np.nan,
                        "W_insulator": np.nan,
                        "eps_insulator": np.nan,
                        "V_insulator": np.nan,
                        "conv_insulator": False,
                        "ins_valid": False,
                    }
                )

            if compat_mode:
                both_finite = bool(row["metal_valid"] and row["ins_valid"])
                if both_finite:
                    row["deltaF"] = row["F_metal"] - row["F_insulator"]
                    row["stable_phase"] = "metal" if row["deltaF"] <= 0 else "insulator"
                    row["coexistence"] = (
                        abs(row["docc_metal"] - row["docc_insulator"]) > 1e-6
                        or abs(row["deltaF"]) > 1e-10
                    )
                else:
                    row["deltaF"] = np.nan
                    row["stable_phase"] = "unknown"
                    row["coexistence"] = False
            else:
                if "res" in m:
                    row["metal_valid"] = _branch_valid(
                        m["res"], "metal",
                        z_metal_min, z_ins_max, v_ins_max, use_branch_filters
                    )
                if "res" in i:
                    row["ins_valid"] = _branch_valid(
                        i["res"], "insulator",
                        z_metal_min, z_ins_max, v_ins_max, use_branch_filters
                    )

                both_valid = bool(row["metal_valid"] and row["ins_valid"])
                branch_split = False
                if both_valid:
                    dz = abs(float(row["Z_metal"]) - float(row["Z_insulator"]))
                    ddocc = abs(float(row["docc_metal"]) - float(row["docc_insulator"]))
                    branch_split = (dz > coexist_z_tol) or (ddocc > coexist_docc_tol)
                row["coexistence"] = both_valid and branch_split
                if both_valid:
                    row["deltaF"] = row["F_metal"] - row["F_insulator"]
                    row["stable_phase"] = "metal" if row["deltaF"] <= 0 else "insulator"
                elif row["metal_valid"]:
                    row["deltaF"] = np.nan
                    row["stable_phase"] = "metal"
                elif row["ins_valid"]:
                    row["deltaF"] = np.nan
                    row["stable_phase"] = "insulator"
                else:
                    row["deltaF"] = np.nan
                    row["stable_phase"] = "unknown"
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["T", "U"]).reset_index(drop=True)
    boundaries = extract_phase_boundaries(df)
    return df, boundaries


def make_plots(df, boundaries, outprefix: str = "ghost_dmft"):
    """Generate standard scan plots and a phase-boundary panel."""
    plt = _plt()

    # stable D(U)
    plt.figure(figsize=(7, 5))
    for T in sorted(df["T"].unique()):
        sub = df[np.isclose(df["T"], T)].sort_values("U")
        y = np.where(
            sub["stable_phase"] == "metal",
            sub["docc_metal"],
            np.where(sub["stable_phase"] == "insulator", sub["docc_insulator"], np.nan),
        )
        plt.plot(sub["U"], y, marker="o", label=f"T={T:.3f}")
    plt.xlabel("Interaction U")
    plt.ylabel("Double occupancy D")
    plt.title("Stable-branch double occupancy")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outprefix}_D_vs_U.png", dpi=180)
    plt.close()

    # stable Z(U)
    plt.figure(figsize=(7, 5))
    for T in sorted(df["T"].unique()):
        sub = df[np.isclose(df["T"], T)].sort_values("U")
        y = np.where(
            sub["stable_phase"] == "metal",
            sub["Z_metal"],
            np.where(sub["stable_phase"] == "insulator", sub["Z_insulator"], np.nan),
        )
        plt.plot(sub["U"], y, marker="o", label=f"T={T:.3f}")
    plt.xlabel("Interaction U")
    plt.ylabel("Quasiparticle-weight proxy Z")
    plt.title("Stable-branch quasiparticle weight")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outprefix}_Z_vs_U.png", dpi=180)
    plt.close()

    # delta F
    plt.figure(figsize=(7, 5))
    for T in sorted(df["T"].unique()):
        sub = df[np.isclose(df["T"], T)].sort_values("U")
        plt.plot(sub["U"], sub["deltaF"], marker="o", label=f"T={T:.3f}")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Interaction U")
    plt.ylabel(r"$F_{\mathrm{metal}} - F_{\mathrm{insulator}}$")
    plt.title("Free-energy difference")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outprefix}_deltaF.png", dpi=180)
    plt.close()

    # coexistence map
    plt.figure(figsize=(7, 5))
    co = df[df["coexistence"] == True]
    no = df[df["coexistence"] == False]
    if len(no):
        plt.scatter(no["U"], no["T"], s=50, label="single branch")
    if len(co):
        plt.scatter(co["U"], co["T"], s=80, label="coexistence / two branch")
    plt.xlabel("Interaction U")
    plt.ylabel("Temperature T")
    plt.title("Coexistence indicator")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{outprefix}_coexistence.png", dpi=180)
    plt.close()

    # phase boundaries Uc1/Uc2/Uc_eq
    plt.figure(figsize=(7, 5))
    if len(boundaries):
        plt.plot(boundaries["Uc1"], boundaries["T"], marker="o", label=r"$U_{c1}(T)$")
        plt.plot(boundaries["Uc2"], boundaries["T"], marker="o", label=r"$U_{c2}(T)$")
        plt.plot(boundaries["Uc_eq"], boundaries["T"], marker="s", label=r"$U_c(T)$ from $\Delta F=0$")
    plt.xlabel("Interaction U")
    plt.ylabel("Temperature T")
    plt.title("Phase boundaries")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{outprefix}_phase_boundaries.png", dpi=180)
    plt.close()


def save_scan_outputs(df, boundaries, outprefix: str = "ghost_dmft"):
    df.to_csv(f"{outprefix}_phase_scan.csv", index=False)
    boundaries.to_csv(f"{outprefix}_phase_boundaries.csv", index=False)
    make_plots(df, boundaries, outprefix=outprefix)


def main():
    U_vals = np.linspace(2.0, 3.4, 30)
    T_vals = np.linspace(0.02, 0.20, 20)

    df, boundaries = run_phase_scan(
        U_vals=U_vals,
        T_vals=T_vals,
        M=1,
        t=0.5,
        nquad=120,
        n_matsubara=512,
        mix=0.05,
        tol=1e-4,
        maxiter=160,
    )
    save_scan_outputs(df, boundaries, outprefix="ghost_dmft")
    print(df)
    print(boundaries)
    print("Saved:")
    print("  ghost_dmft_phase_scan.csv")
    print("  ghost_dmft_phase_boundaries.csv")
    print("  ghost_dmft_D_vs_U.png")
    print("  ghost_dmft_Z_vs_U.png")
    print("  ghost_dmft_deltaF.png")
    print("  ghost_dmft_coexistence.png")
    print("  ghost_dmft_phase_boundaries.png")


if __name__ == "__main__":
    main()
