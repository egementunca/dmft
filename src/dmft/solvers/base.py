"""Abstract impurity solver interface."""

from abc import ABC, abstractmethod
import numpy as np


class ImpuritySolver(ABC):
    """Base class for impurity solvers.

    All solvers take bath parameters and return the impurity Green's function,
    self-energy, and observables.
    """

    @abstractmethod
    def solve(self, iw: np.ndarray, mu: float, eps_d: float, U: float,
              V: np.ndarray, eps: np.ndarray, beta: float,
              sigma_inf: float) -> dict:
        """Solve the impurity problem.

        Parameters
        ----------
        iw : array, shape (N,)
            Imaginary frequencies (1j * w_n).
        mu : float
            Chemical potential.
        eps_d : float
            Impurity level energy.
        U : float
            Hubbard interaction.
        V : array, shape (M_g,)
            Bath hybridization amplitudes.
        eps : array, shape (M_g,)
            Bath level energies.
        beta : float
            Inverse temperature.
        sigma_inf : float
            Static self-energy tail passed by the outer loop.
            In the current project convention (Option A), impurity
            Hamiltonians are unshifted and this argument is informational.

        Returns
        -------
        dict with:
            'G_imp': array (N,) — impurity Green's function
            'Sigma_imp': array (N,) — impurity self-energy
            'n_imp': float — impurity occupancy per spin <n_sigma>
            'n_double': float — double occupancy <n_up n_down>
        """
        pass
