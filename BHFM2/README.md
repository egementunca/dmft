# M=2 Ghost-DMFT: Half-Filling T-Sweep AND Ferrero-Style Doping

Full handoff package for both:
1. **Half-filling T-sweep** at M=2, U=1.3
2. **Finite-doping sweep** at Ferrero et al params (U/t = 2.5, multiple fillings, low T)

## Files

### Source code
- `solve_min.py` — main solver (minimal formulation, no anti-bond)
- `ed_fast.py` — numba-accelerated dense ED
- `ed_sparse.py` — sparse ED (helpful at very low T only)
- `ghost_dmft_bond.py` — basis / sector building

### Runners
- `run_M2_sweep.py` — M=2 T-sweep at U=1.3, half filling
- `run_M2_doping.py` — M=2 sweep at U=1.25 (U/t=2.5 with t=0.5), fillings n=0.85,0.90,0.95, T=0.05-0.5

### Analysis
- `sigma_k.py` — extract momentum-resolved Sigma(k, iwn) from ghost params
- `plot_fermi_arcs.py` — generate Ferrero-style pseudogap/Fermi arc plots

### Reference data
- `M2_min_T1_best.pkl` — M=2 minimal at T=1, U=1.3 (||r||≈4e-4, partial)
- `M2_T2.5_ckpt.pkl` — M=2 minimal at T=2.5, U=1.3 (||r||≈2e-4, partial)
- `Tsweep_U1.3.pkl` — M=1 full anti-bond U=1.3 sweep (13 T points)
- `Tsweep_min_U1.3.pkl` — M=1 minimal U=1.3 sweep (13 T points)

## How to run

Cluster submission runbook (latest):
- `../jobs/BHFM2_SCC_INSTRUCTIONS.md`

### Half-filling T-sweep
```bash
python run_M2_sweep.py
```
Outputs: `Tsweep_M2_U1.3.pkl`, per-T checkpoints

### Ferrero-style doping sweep
```bash
python run_M2_doping.py
```
Outputs: `doping_M2_U1.25.pkl`, per-(T,n) checkpoints

Plans this run:
- U = 1.25 (Ferrero's U/t = 2.5 at t=0.5)
- Fillings: n = 0.95, 0.90, 0.85 (hole doping 5%, 10%, 15%)
- Temperatures: T = 0.5, 0.3, 0.2, 0.1, 0.05
- Strategy: start at mild doping + warm T, cool down and dope further
- Warm starts from nearest converged (T, n) — strongly recommends doing 0.95 first
- Safe to interrupt and resume

### Analysis / plots
```bash
python plot_fermi_arcs.py doping_M2_U1.25.pkl
```
Produces:
- `ferrero_ImSigma_vs_T.png` — Im Sigma at antinode (π,0) vs node (π/2,π/2) as function of T, for each filling
- `ferrero_Z_vs_k.png` — quasiparticle weight Z(k) along the FS
- `ferrero_Sigma_vs_wn.png` — Sigma(k, iwn) Matsubara curves at AN, N, and mid-arc

## Self-energy form

On the pair manifold our ghost code produces:

    Sigma(k, iwn) = Sigma_inf + sum_a W_a^2 / (iwn - eta_a)
                  + B_h^2 * (2 + cos(kx) + cos(ky)) / (2 * (iwn - eta_b))

Or equivalently,

    Sigma(k, iwn) = Sigma_loc(iwn) + (cos(kx) + cos(ky)) * Sigma_nn(iwn)

with

    Sigma_loc(iwn) = Sigma_inf + sum_a W_a^2/(iwn - eta_a) + B_h^2/(iwn - eta_b)
    Sigma_nn(iwn)  = B_h^2 / (2 * (iwn - eta_b))

## Expected physics (Ferrero et al 2008/2009)

At n ≈ 0.9, T ≈ 0.05, U/t ≈ 2.5:
- **Antinode** (π, 0): Im Σ divergent — Mott-insulating at antinode
- **Node** (π/2, π/2): Im Σ finite — metallic at node
- → **Fermi arcs**: quasiparticles survive only on nodal portion of FS

With our current pair-manifold form, gamma_k = 0 at BOTH node and antinode, so the bond-h channel gives the SAME contribution there. Differentiation comes entirely via Sigma_loc, and the Fermi arc signature may be weaker than in Ferrero's richer DCA treatment. If we don't see it, possible cures:
- Go to more complex bond structures (separate x/y channels)
- Higher M / more ghosts for richer self-energy

## Cost estimates (dense ED, M=2, Mb=1, N_orb=7)
- One imp2 call: ~7s
- One SCF residual: ~7-8s
- One LSQ Jacobian (27 params): ~3 min
- Convergence to ||r||~1e-3: ~15-30 min per (T, n) point
- Full doping grid (3 n × 5 T = 15 points): ~6-8 hours

## Tips for running
- `target_res = 1e-3` is loose — physics accurate to ~0.1%, fine for trends
- For sharper convergence (n_avg error < 1e-4), tighten to `target_res = 1e-4`
- Each per-T or per-(T,n) checkpoint lets you resume after interruption
- The doping sweep warm-starts intelligently; don't interrupt in the middle of a point unless necessary
