#!/usr/bin/env python3
"""Clean figure for professor — M=2 bond scheme results only."""
import numpy as np
import matplotlib.pyplot as plt

m2 = np.loadtxt('bond_M1g2M2g2Mbg1_U1.3_free_eps.dat')
# Columns: T, docc_ss, docc_bpk, docc_1, docc_2, hop

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle(r'Bond-scheme DMFT: $M_{1g}=2,\; M_{2g}=2,\; M_{bg}=1$,  $U=1.3$, $t=0.5$',
             fontsize=13)

# Panel 1: Double occupancy
ax = axes[0]
ax.plot(m2[:, 0], m2[:, 1], 'o-', ms=3, label=r'$d_{\mathrm{ss}}$ (single-site)', color='C0', alpha=0.7)
ax.plot(m2[:, 0], m2[:, 2], 's-', ms=3, label=r'$d_{\mathrm{BPK}}$ (bond)', color='C1', alpha=0.7)
ax.set_xlabel('Temperature $T$')
ax.set_ylabel('Double Occupancy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Site-resolved docc
ax = axes[1]
ax.plot(m2[:, 0], m2[:, 3], 'o-', ms=3, label=r'$\langle n_{d_1\uparrow} n_{d_1\downarrow} \rangle$', color='C2', alpha=0.7)
ax.plot(m2[:, 0], m2[:, 4], 's-', ms=3, label=r'$\langle n_{d_2\uparrow} n_{d_2\downarrow} \rangle$', color='C3', alpha=0.7)
ax.set_xlabel('Temperature $T$')
ax.set_ylabel('Double Occupancy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Hopping
ax = axes[2]
ax.plot(m2[:, 0], m2[:, 5], 'o-', ms=3, color='C4', alpha=0.7)
ax.set_xlabel('Temperature $T$')
ax.set_ylabel(r'$\langle d_1^\dagger d_2 \rangle$')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bond_results_M2.png', dpi=150, bbox_inches='tight')
plt.savefig('bond_results_M2.pdf', bbox_inches='tight')
print("Saved bond_results_M2.png / .pdf")
