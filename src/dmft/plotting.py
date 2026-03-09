"""Plotting utilities for DMFT results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_green_function(wn, G_loc, title="Local Green's function"):
    """Plot Re and Im parts of G_loc(iw_n) vs w_n."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(wn, G_loc.real, 'b.-', markersize=2)
    ax1.set_xlabel(r'$\omega_n$')
    ax1.set_ylabel(r'Re $G_{\rm loc}(i\omega_n)$')
    ax1.set_xlim(0, min(wn[-1], 20))
    ax1.axhline(0, color='gray', lw=0.5)

    ax2.plot(wn, G_loc.imag, 'r.-', markersize=2)
    ax2.set_xlabel(r'$\omega_n$')
    ax2.set_ylabel(r'Im $G_{\rm loc}(i\omega_n)$')
    ax2.set_xlim(0, min(wn[-1], 20))
    ax2.axhline(0, color='gray', lw=0.5)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_self_energy(wn, Sigma, U=None, title="Self-energy"):
    """Plot Re and Im parts of Sigma(iw_n)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(wn, Sigma.real, 'b.-', markersize=2)
    ax1.set_xlabel(r'$\omega_n$')
    ax1.set_ylabel(r'Re $\Sigma(i\omega_n)$')
    ax1.set_xlim(0, min(wn[-1], 20))
    if U is not None:
        ax1.axhline(U / 2, color='gray', ls='--', label=f'U/2={U/2:.1f}')
        ax1.legend()

    ax2.plot(wn, Sigma.imag, 'r.-', markersize=2)
    ax2.set_xlabel(r'$\omega_n$')
    ax2.set_ylabel(r'Im $\Sigma(i\omega_n)$')
    ax2.set_xlim(0, min(wn[-1], 20))
    ax2.axhline(0, color='gray', lw=0.5)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_spectral_function(omega, A_omega, title="Spectral function"):
    """Plot A(omega) vs omega."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(omega, A_omega, 'b-', lw=1.5)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$A(\omega)$')
    ax.set_title(title)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    fig.tight_layout()
    return fig


def plot_convergence(history, title="DMFT convergence"):
    """Plot convergence metrics vs iteration."""
    iters = [h['iteration'] for h in history]
    diffs = [h['diff'] for h in history]
    Zs = [h['Z'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.semilogy(iters, diffs, 'ko-', markersize=3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Convergence (max |dSigma|/|Sigma|)')
    ax1.set_title('Convergence')

    ax2.plot(iters, Zs, 'bs-', markersize=3)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Z')
    ax2.set_ylim(0, 1)
    ax2.set_title('Quasiparticle weight')

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_z_vs_u(U_values, Z_values, title="Quasiparticle weight vs U"):
    """Plot Z as a function of U."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(U_values, Z_values, 'bo-', markersize=5)
    ax.set_xlabel(r'$U/D$')
    ax.set_ylabel(r'$Z$')
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.axhline(0, color='gray', lw=0.5)
    fig.tight_layout()
    return fig
