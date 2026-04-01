#!/usr/bin/env python3
"""
Quick plotting script for bond-scheme DMFT results.
Run locally with matplotlib installed.
"""
import numpy as np
import matplotlib.pyplot as plt

# Load data
m1 = np.loadtxt('bond_M1_U1.3_t0.5.dat')
m2 = np.loadtxt('bond_M2_U1.3_t0.5.dat')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Double occupancy vs T
ax = axes[0, 0]
ax.plot(m1[:, 0], m1[:, 1], 'o-', label='M=1 single-site', alpha=0.7)
ax.plot(m1[:, 0], m1[:, 2], 's-', label='M=1 bond', alpha=0.7)
ax.plot(m2[:, 0], m2[:, 2], '^-', label='M=2 bond', alpha=0.7)
ax.set_xlabel('Temperature T')
ax.set_ylabel('Double Occupancy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Double Occupancy vs Temperature')

# Plot 2: Hopping vs T
ax = axes[0, 1]
ax.plot(m1[:, 0], m1[:, 5], 'o-', label='M=1', alpha=0.7)
ax.plot(m2[:, 0], m2[:, 5], 's-', label='M=2', alpha=0.7)
ax.set_xlabel('Temperature T')
ax.set_ylabel('Hopping ⟨d₁†d₂⟩')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Hopping Parameter vs Temperature')

# Plot 3: docc1 and docc2 vs T
ax = axes[1, 0]
ax.plot(m1[:, 0], m1[:, 3], 'o-', label='M=1 docc₁', alpha=0.7)
ax.plot(m1[:, 0], m1[:, 4], 's-', label='M=1 docc₂', alpha=0.7)
ax.plot(m2[:, 0], m2[:, 3], '^-', label='M=2 docc₁', alpha=0.7)
ax.plot(m2[:, 0], m2[:, 4], 'v-', label='M=2 docc₂', alpha=0.7)
ax.set_xlabel('Temperature T')
ax.set_ylabel('Double Occupancy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Site-Resolved Double Occupancy')

# Plot 4: M1 vs M2 difference
ax = axes[1, 1]
diff_docc = m2[:, 2] - m1[:, 2]
diff_hop = m2[:, 5] - m1[:, 5]
ax.plot(m1[:, 0], diff_docc * 100, 'o-', label='Δ docc_bpk (%)', alpha=0.7)
ax.plot(m1[:, 0], diff_hop * 100, 's-', label='Δ hop (%)', alpha=0.7)
ax.set_xlabel('Temperature T')
ax.set_ylabel('Difference (%)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_title('M=2 vs M=1 Convergence')

plt.tight_layout()
plt.savefig('bond_scheme_results.png', dpi=150, bbox_inches='tight')
print("Plot saved as bond_scheme_results.png")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nHigh T (T={m1[0,0]:.2f}):")
print(f"  M1 docc_bpk: {m1[0,2]:.6f}")
print(f"  M2 docc_bpk: {m2[0,2]:.6f}")
print(f"  Difference:  {diff_docc[0]:.6f} ({100*diff_docc[0]/m1[0,2]:.3f}%)")

print(f"\nLow T (T={m1[-1,0]:.2f}):")
print(f"  M1 docc_bpk: {m1[-1,2]:.6f}")
print(f"  M2 docc_bpk: {m2[-1,2]:.6f}")
print(f"  Difference:  {diff_docc[-1]:.6f} ({100*diff_docc[-1]/m1[-1,2]:.3e}%)")

print(f"\nMax difference across all T:")
print(f"  docc_bpk: {np.max(np.abs(diff_docc)):.6f}")
print(f"  hop:      {np.max(np.abs(diff_hop)):.6f}")
