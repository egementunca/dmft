# Notes-Faithful Two-Ghost Variant B

Locked conventions used in code:
- Bethe lattice with `D=1`, `t=0.5`, and `G^{-1}(iw)=iw+\mu-h` (`\mu` not inside `h`).
- `\Delta(iw)=\sum_l |V_l|^2/(iw-\epsilon_l)`.
- `\Sigma(iw)=\sigma_\infty+\sum_l |W_l|^2/(iw-\eta_l)`.
- Half-filling runs use `\mu=U/2`, `\epsilon_d=0`, `\Delta_{\text{cav}}=t^2 G_{\text{loc}}`.
- Option A is fixed: `\sigma_\infty` is a tail constraint only (no impurity on-site shift).

Functional used for stationarity:
\[
F = \Omega[H_{\text{lat}}] + \Omega[H_{\text{imp}}] - \Omega[H_{\text{imp}}^{(0)}].
\]

By thermodynamic Hellmann-Feynman,
\[
\frac{\partial \Omega}{\partial \lambda}=\left\langle \frac{\partial H}{\partial \lambda}\right\rangle,
\]
so `\partial F/\partial \lambda=0` gives correlator matching constraints:
- h-sector matching (`\lambda=\eta_l,W_l`) between lattice and gateway:
  `\langle h_l^\dag h_l\rangle_{\text{lat}}=\langle h_l^\dag h_l\rangle_0`,
  `\langle d^\dag h_l\rangle_{\text{lat}}=\langle d^\dag h_l\rangle_0`.
- g-sector matching (`\lambda=\epsilon_l,V_l`) between impurity and gateway:
  `\langle g_l^\dag g_l\rangle_{\text{imp}}=\langle g_l^\dag g_l\rangle_0`,
  `\langle d^\dag g_l\rangle_{\text{imp}}=\langle d^\dag g_l\rangle_0`.

Implemented loop mapping:
1. Build pole `\Sigma` from current ghosts and compute Bethe `G_{\text{loc}}`.
2. Compute lattice h-correlators and update `{ \epsilon,V }` by h-matching.
3. Solve interacting impurity (unshifted, Option A), set `\sigma_\infty=U n_{\text{imp}}`.
4. Update `{ \eta,W }` by g-matching (default) or `ghost_update_mode="fit"` (debug only).
5. Rebuild `\Sigma`, enforce causality checks/backtracking, iterate.

Why this approaches full DMFT as `M\to\infty`:
- Pole sets become dense representations of causal `\Delta(iw)` and local `\Sigma(iw)`.
- Correlator constraints become projections of full functional stationarity conditions.
- In that limit, the restricted stationary manifold tends to the unconstrained DMFT saddle (within local-self-energy DMFT assumptions).
