#!/usr/bin/env python3
"""
Plotting script for bond-scheme DMFT results (full T range).
Run from results/bond_scheme_verified/:
  python3 plot_results.py
"""
import numpy as np
import matplotlib.pyplot as plt

# Load full-range data
m1 = np.loadtxt('bond_M1_U1.3_t0.5_full.dat')
m2 = np.loadtxt('bond_M2_U1.3_t0.5_full.dat')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Ghost-DMFT Bond Scheme: U=1.3, t=0.5, Square Lattice', fontsize=14, y=0.98)

# Plot 1: Double occupancy vs T
ax = axes[0, 0]
ax.plot(m1[:, 0], m1[:, 1], 'o-', ms=3, label='single-site (M=1)', alpha=0.7)
ax.plot(m1[:, 0], m1[:, 2], 's-', ms=3, label='bond M=1', alpha=0.7)
ax.plot(m2[:, 0], m2[:, 2], '^-', ms=3, label='bond M=2', alpha=0.7)
ax.set_xlabel('Temperature T')
ax.set_ylabel('Double Occupancy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('Double Occupancy vs Temperature')

# Plot 2: Hopping vs T
ax = axes[0, 1]
ax.plot(m1[:, 0], m1[:, 5], 'o-', ms=3, label='M=1', alpha=0.7)
ax.plot(m2[:, 0], m2[:, 5], 's-', ms=3, label='M=2', alpha=0.7)
ax.set_xlabel('Temperature T')
ax.set_ylabel(r'Hopping $\langle d_1^\dagger d_2 \rangle$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('Hopping Parameter vs Temperature')

# Plot 3: docc1 and docc2 vs T
ax = axes[1, 0]
ax.plot(m1[:, 0], m1[:, 3], 'o-', ms=3, label=r'M=1 $d_1$', alpha=0.7)
ax.plot(m1[:, 0], m1[:, 4], 's-', ms=3, label=r'M=1 $d_2$', alpha=0.7)
ax.plot(m2[:, 0], m2[:, 3], '^-', ms=3, label=r'M=2 $d_1$', alpha=0.7, color='C2')
ax.plot(m2[:, 0], m2[:, 4], 'v-', ms=3, label=r'M=2 $d_2$', alpha=0.7, color='C3')
ax.set_xlabel('Temperature T')
ax.set_ylabel('Double Occupancy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('Site-Resolved Double Occupancy')

# Plot 4: Bond enhancement ratio
ax = axes[1, 1]
# docc_bpk / docc_ss shows how much the bond scheme enhances correlations
ratio_m1 = m1[:, 2] / m1[:, 1]
ratio_m2 = m2[:, 2] / m2[:, 1]
ax.plot(m1[:, 0], ratio_m1, 'o-', ms=3, label='M=1', alpha=0.7)
ax.plot(m2[:, 0], ratio_m2, 's-', ms=3, label='M=2', alpha=0.7)
ax.set_xlabel('Temperature T')
ax.set_ylabel(r'$d_{\mathrm{BPK}} / d_{\mathrm{ss}}$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(1, color='k', linestyle='--', alpha=0.3, label='no enhancement')
ax.set_title('Bond Enhancement Ratio')

plt.tight_layout()
plt.savefig('bond_scheme_results.png', dpi=150, bbox_inches='tight')
plt.savefig('bond_scheme_results.pdf', bbox_inches='tight')
print("Plots saved: bond_scheme_results.png / .pdf")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
diff_bpk = np.max(np.abs(m2[:, 2] - m1[:, 2]))
diff_hop = np.max(np.abs(m2[:, 5] - m1[:, 5]))
print(f"T range: [{m1[:,0].min():.2f}, {m1[:,0].max():.2f}]  ({len(m1)} points)")
print(f"Max |M2-M1| docc_bpk: {diff_bpk:.2e}")
print(f"Max |M2-M1| hop:      {diff_hop:.2e}")
print(f"Low T:  docc_bpk={m1[-1,2]:.4f}, hop={m1[-1,5]:.4f}, docc1={m1[-1,3]:.6f}")
print(f"High T: docc_bpk={m1[0,2]:.4f}, hop={m1[0,5]:.4f}, docc1={m1[0,3]:.6f}")
print(f"Bond enhancement at T=0.1: {ratio_m1[np.argmin(np.abs(m1[:,0]-0.1))]:.2f}x")
