#!/usr/bin/env python3
"""Compare bond-scheme results from internal code vs professor's code.

Usage:
  python3 scripts/compare_results.py internal.dat professor.dat
  python3 scripts/compare_results.py bond_M1g2M2g2Mbg1_U1.3_free_eps.dat prof_results.dat
"""
import sys
import numpy as np

def load_dat(fname):
    """Load a bond-scheme .dat file (6 columns: T docc_ss docc_bpk docc1 docc2 hop)."""
    data = np.loadtxt(fname)
    cols = ['T', 'docc_ss', 'docc_bpk', 'docc1', 'docc2', 'hop']
    return {c: data[:, i] for i, c in enumerate(cols)}

def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <file1.dat> <file2.dat>')
        sys.exit(1)

    f1, f2 = sys.argv[1], sys.argv[2]
    d1 = load_dat(f1)
    d2 = load_dat(f2)

    # Match T values
    print(f'File 1: {f1}  ({len(d1["T"])} T points)')
    print(f'File 2: {f2}  ({len(d2["T"])} T points)')
    print()

    # Find common T values
    common = []
    for i, T1 in enumerate(d1['T']):
        for j, T2 in enumerate(d2['T']):
            if abs(T1 - T2) < 1e-6:
                common.append((i, j))
                break

    if not common:
        print('No matching T values found!')
        print(f'  File 1 T range: {d1["T"].min():.4f} - {d1["T"].max():.4f}')
        print(f'  File 2 T range: {d2["T"].min():.4f} - {d2["T"].max():.4f}')
        sys.exit(1)

    print(f'{len(common)} matching T values found.')
    print()

    obs = ['docc_bpk', 'docc1', 'docc2', 'hop', 'docc_ss']
    print(f'{"T":>8}', end='')
    for o in obs:
        print(f'  {"f1_"+o:>12}  {"f2_"+o:>12}  {"diff":>10}', end='')
    print()
    print('-' * (8 + len(obs) * 38))

    max_diffs = {o: 0.0 for o in obs}
    for i1, i2 in common:
        T = d1['T'][i1]
        print(f'{T:8.4f}', end='')
        for o in obs:
            v1, v2 = d1[o][i1], d2[o][i2]
            diff = abs(v1 - v2)
            max_diffs[o] = max(max_diffs[o], diff)
            print(f'  {v1:12.6f}  {v2:12.6f}  {diff:10.2e}', end='')
        print()

    print()
    print('Max absolute differences:')
    for o in obs:
        status = 'OK' if max_diffs[o] < 1e-4 else 'CHECK'
        print(f'  {o:12s}: {max_diffs[o]:.2e}  [{status}]')

if __name__ == '__main__':
    main()
