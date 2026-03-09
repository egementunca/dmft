"""Parameter containers and sign conventions for the two-ghost DMFT code.

Conventions (locked):
- Energy unit: half-bandwidth D = 2t = 1, so t = 0.5
- G^{-1}(iw) = iw + mu - h  (mu NOT in h)
- Delta(iw) = sum_l |V_l|^2 / (iw - eps_l)
- Sigma^(h)(iw) = sum_l |W_l|^2 / (iw - eta_l)
- Half-filling: mu = U/2, eps_d = 0
- Bethe self-consistency: Delta(iw) = t^2 * G_loc(iw)
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DMFTParams:
    """Physical and numerical parameters for the DMFT calculation."""

    t: float = 0.5          # hopping (half-bandwidth D = 2t = 1)
    U: float = 2.0          # Hubbard interaction
    beta: float = 50.0      # inverse temperature
    mu: float = 1.0         # chemical potential (U/2 at half-filling)
    eps_d: float = 0.0      # impurity level energy
    n_matsubara: int = 1024  # number of positive Matsubara frequencies
    M_g: int = 2            # number of bath (g) poles
    M_h: int = 2            # number of ghost (h) poles
    mix: float = 0.3        # linear mixing parameter for iteration
    max_iter: int = 100     # maximum DMFT iterations
    tol: float = 1e-6       # convergence tolerance on G_loc

    def __post_init__(self):
        if self.mu is None:
            self.mu = self.U / 2.0

    @classmethod
    def half_filling(cls, U: float, beta: float = 50.0, **kwargs):
        """Create params at half-filling where mu = U/2."""
        return cls(U=U, beta=beta, mu=U / 2.0, eps_d=0.0, **kwargs)


@dataclass
class PoleParams:
    """Pole parameters for bath hybridization and self-energy ghosts.

    Bath (g-levels):  Delta(iw) = sum_l |V_l|^2 / (iw - eps_l)
    Ghosts (h-levels): Sigma(iw) = sigma_inf + sum_l |W_l|^2 / (iw - eta_l)
    """

    eps: np.ndarray       # bath level energies, shape (M_g,)
    V: np.ndarray         # bath hybridization amplitudes, shape (M_g,)
    eta: np.ndarray       # ghost level energies, shape (M_h,)
    W: np.ndarray         # ghost hybridization amplitudes, shape (M_h,)
    sigma_inf: float      # static self-energy tail

    @classmethod
    def initial_symmetric(cls, M_g: int, M_h: int, U: float, t: float = 0.5):
        """Initialize with particle-hole symmetric pole positions.

        Bath poles span the bandwidth; ghost poles start near +/- U/2.
        """
        D = 2 * t  # half-bandwidth
        # Bath: symmetric around 0, spanning [-D, D]
        if M_g == 1:
            eps = np.array([0.0])
            V = np.array([t])
        else:
            eps = np.linspace(-D * 0.8, D * 0.8, M_g)
            V = np.full(M_g, t / np.sqrt(M_g))

        # Ghosts: symmetric around 0, near +/- U/2
        if M_h == 1:
            eta = np.array([0.0])
            W = np.array([U / 4.0])
        else:
            eta = np.linspace(-U / 2 * 0.8, U / 2 * 0.8, M_h)
            W = np.full(M_h, U / (4.0 * np.sqrt(M_h)))

        sigma_inf = U / 2.0  # half-filling initial guess

        return cls(eps=eps, V=V, eta=eta, W=W, sigma_inf=sigma_inf)

    def copy(self):
        return PoleParams(
            eps=self.eps.copy(),
            V=self.V.copy(),
            eta=self.eta.copy(),
            W=self.W.copy(),
            sigma_inf=self.sigma_inf,
        )
