# DMFT “Two‑Ghost” Notes — Compact Master Map + Mini‑Project Plan

This document is a **single, agent-friendly map** of the professor’s notes you’ve been studying (the “two‑ghost” / “gateway” DMFT functional construction) plus a **mini‑project plan** that turns the notes into a working numerical experiment.

---

## 0) What you should already be comfortable with (minimum prereqs)

### Imaginary time + Matsubara
- Thermal averages:  $\langle A\rangle = Z^{-1}\mathrm{Tr}(e^{-\beta H}A)$, $Z=\mathrm{Tr}\,e^{-\beta H}$.
- Fermionic imaginary time $\tau\in[0,\beta]$, antiperiodicity:
  $$
  G(\tau+\beta)=-G(\tau),\quad \omega_n=(2n+1)\pi/\beta.
  $$
- Fourier transform conventions and the crucial identity:
  $$
  \int_0^\beta d\tau\,e^{i\omega_n\tau}[-\partial_\tau G(\tau)] = i\omega_n G(i\omega_n)
  $$
  (boundary term cancels using antiperiodicity and $e^{i\omega_n\beta}=-1$).

### Quadratic Hamiltonians → matrix inverses
For a quadratic (one-body) Hamiltonian $H=\sum_{ab}\psi_a^\dagger h_{ab}\psi_b$,
$$
G(i\omega_n) = (i\omega_n+\mu-h)^{-1}.
$$
This is just the Fourier-transformed equation of motion.

### Schur complement (integrating out levels)
For block matrix $M=\begin{pmatrix}A&B\\C&D\end{pmatrix}$ (with $D$ invertible),
$$
(M^{-1})_{11}=(A-BD^{-1}C)^{-1}.
$$
**Physics meaning:** “integrate out / eliminate” the $D$-sector degrees of freedom and obtain an effective inverse propagator for the $A$-sector.

### EOM / spectral moments (high-frequency expansion)
For fermion $c$,
$$
G(i\omega) = \frac{\langle\{c,c^\dagger\}\rangle}{i\omega} + \frac{\langle\{[H,c],c^\dagger\}\rangle}{(i\omega)^2}+\cdots.
$$
For the Anderson/Hubbard impurity this gives the key tail constraint:
$$
\Sigma_\infty = U\langle n_{-\sigma}\rangle.
$$

---

## 1) Standard DMFT objects (to keep names straight)

### Impurity Dyson equation
Define Weiss/bare impurity propagator:
$$
\mathcal G_0^{-1}(i\omega) = i\omega+\mu-\epsilon_d-\Delta(i\omega).
$$
Then
$$
G_{\rm imp}^{-1}(i\omega) = \mathcal G_0^{-1}(i\omega)-\Sigma(i\omega)
= i\omega+\mu-\epsilon_d-\Delta(i\omega)-\Sigma(i\omega).
$$

### Lattice local Green’s function (with local self-energy)
$$
G_{\rm loc}(i\omega) = \sum_{\mathbf k}\frac{1}{i\omega+\mu-\epsilon_{\mathbf k}-\Sigma(i\omega)}
\quad\text{or}\quad
G_{\rm loc}(i\omega)=\int d\epsilon\,\rho(\epsilon)\frac{1}{i\omega+\mu-\epsilon-\Sigma(i\omega)}.
$$

### DMFT self-consistency
$$
G_{\rm imp}(i\omega)=G_{\rm loc}(i\omega).
$$
Rearranged Weiss field identity (often called “cavity/Weiss” relation):
$$
\Delta(i\omega) = i\omega+\mu-\epsilon_d-\Sigma(i\omega) - \frac{1}{G_{\rm loc}(i\omega)}.
$$

---

## 2) The core idea of the professor’s notes

### 2.1 Replace functions by **finite pole expansions**
Instead of treating $\Delta(i\omega)$ and $\Sigma(i\omega)$ as arbitrary functions, the notes parameterize them as **rational functions**:

**Bath / hybridization (g‑levels):**
$$
\Delta(i\omega)=\sum_{\ell=1}^{M_g}\frac{|V_\ell|^2}{i\omega+\mu-\epsilon_\ell}.
$$

**Self-energy (h‑ghosts):**
$$
\Sigma(i\omega)=\Sigma_\infty+\Sigma^{(h)}(i\omega),
\qquad
\Sigma^{(h)}(i\omega)=\sum_{\ell=1}^{M_h}\frac{|W_\ell|^2}{i\omega-\eta_\ell}.
$$

As $M\to\infty$, these pole bases can approximate a broad class of causal functions.

### 2.2 Why poles are natural: “integrating out a level”
If $d$ couples to a noninteracting level $g$ of energy $\epsilon$ with amplitude $V$, then Schur complement gives:
$$
G_{dd}^{-1}(i\omega)=i\omega+\mu-\epsilon_d-\frac{|V|^2}{i\omega+\mu-\epsilon}.
$$
So coupling to many $g_\ell$ levels gives the pole-sum $\Delta(i\omega)$.

Exactly the same mechanism with $h_\ell$ levels produces the pole-sum self-energy $\Sigma^{(h)}$.

---

## 3) The three Hamiltonians (what each one is for)

The notes introduce **three** Hamiltonians so that:
- lattice quantities with a chosen $\Sigma$ are computable by **linear algebra**;
- impurity interacting physics is captured by **one** interacting model;
- stationarity can be expressed as **matching equal-time correlators** between solvable (quadratic) models.

### 3.1 $H_{\rm lat}$: lattice + h‑ghosts (quadratic)
- Degrees of freedom: lattice $d_{i\sigma}$ and local ghosts $h_{i\sigma\ell}$.
- Role: compute lattice Green’s functions and correlators for a **chosen** pole self-energy $\Sigma(i\omega)$.
- Outcome via Schur complement: $h$-sector generates $\Sigma^{(h)}(i\omega)$.

### 3.2 $H_{\rm imp}$: interacting impurity + g‑bath (interacting)
- Degrees of freedom: one impurity $d_\sigma$, bath levels $g_{\sigma\ell}$, and interaction $U n_\uparrow n_\downarrow$.
- Role: this is the **only** place where strong correlation physics enters.
- Outcome: interacting impurity correlators (occupancy, etc.) and the “true” impurity response.

### 3.3 $H_{\rm imp}^{(0)}$: gateway (quadratic)
- Degrees of freedom: impurity $d_\sigma$ + **both** $g$-bath and $h$-ghost sectors.
- No $U$ term (hence superscript $^{(0)}$).
- Role: a solvable bridge model where both parameter sets $(\epsilon_\ell,V_\ell)$ and $(\eta_\ell,W_\ell)$ coexist, enabling **correlator matching**.

---

## 4) Why the free-energy functional is built as “+ lat + imp − imp(0)”

The notes define a functional (grand potential functional):
$$
F=\Omega[H_{\rm lat}] + \Omega[H_{\rm imp}] - \Omega[H_{\rm imp}^{(0)}],
\qquad
\Omega[H]=-\frac{1}{\beta}\ln\mathrm{Tr}\,e^{-\beta H}.
$$

**Interpretation (double-counting subtraction):**
- $\Omega[H_{\rm lat}]$: “embed” the chosen $\Sigma$ into the lattice (quadratic).
- $\Omega[H_{\rm imp}]$: add the true local interaction physics from $U$ (interacting).
- $-\Omega[H_{\rm imp}^{(0)}]$: subtract the overlapping **quadratic reference** contribution that would otherwise be counted twice. The gateway is the shared reference.

This structure is engineered so that derivatives w.r.t. parameters become simple expectation values.

---

## 5) Hellmann–Feynman for free energies → matching conditions

### 5.1 Thermodynamic Hellmann–Feynman identity
For any Hamiltonian $H(\lambda)$,
$$
\frac{\partial\Omega}{\partial\lambda}
=
\left\langle\frac{\partial H}{\partial\lambda}\right\rangle_H.
$$

### 5.2 Stationarity gives correlator equalities
Stationarity $\partial F/\partial\lambda=0$ yields correlator matching between the models that contain $\lambda$.

Examples:
- $\lambda=\eta_\ell$: since $\eta_\ell$ appears in $H_{\rm lat}$ and $H_{\rm imp}^{(0)}$,
  $$
  \langle h_\ell^\dagger h_\ell\rangle_{\rm lat}=\langle h_\ell^\dagger h_\ell\rangle_{0}.
  $$
- $\lambda=W_\ell$: gives a condition involving $\langle d^\dagger h_\ell\rangle$ between lattice and gateway.
- $\lambda=\epsilon_\ell,V_\ell$: gives impurity ↔ gateway matching involving $\langle g_\ell^\dagger g_\ell\rangle$, $\langle d^\dagger g_\ell\rangle$.

### 5.3 How correlators are computed from Green’s functions
Equal-time correlators from Matsubara Green’s functions:
$$
\langle b^\dagger a\rangle
=
\frac{1}{\beta}\sum_n e^{i\omega_n 0^+} G_{ab}(i\omega_n).
$$
The notes use this heavily for matching conditions.

---

## 6) The iterative algorithm (the “two-ghost” DMFT loop)

A practical (conceptual) iteration:

1) **Initialize** pole parameters for the self-energy ghosts:
   $\{\eta_\ell,W_\ell\}$ and $\Sigma_\infty$.
2) **Lattice step:** solve the quadratic $H_{\rm lat}$ (by Schur complement / resolvents) → compute lattice correlators in the $d/h$ sector.
3) **Lattice ↔ gateway matching:** choose/update $\{\epsilon_\ell,V_\ell\}$ so gateway correlators match lattice correlators.
4) **Impurity step:** solve the interacting $H_{\rm imp}$ with bath $\{\epsilon_\ell,V_\ell\}$ → compute impurity correlators and occupancy.
5) **Impurity ↔ gateway matching:** update $\{\eta_\ell,W_\ell\}$ so gateway correlators match impurity correlators.
6) **Update static tail:** $\Sigma_\infty \leftarrow U\langle n_{-\sigma}\rangle_{\rm imp}$.
7) Iterate until convergence.

**Key conceptual point:** Matching conditions enforce equality of a finite set of projections (onto pole kernels); as $M\to\infty$, full DMFT is recovered.

---

## 7) Mini‑project: Pole‑truncated DMFT on the Bethe lattice

### 7.1 Goal
Build a minimal working DMFT experiment aligned with the notes:
- See how pole truncation (small $M$) captures qualitative DMFT physics (quasiparticle peak, Hubbard bands, Mott trend).
- Observe convergence behavior as $M$ increases.

### 7.2 Minimal scope (recommended)
- Work at **half-filling** first (simplifies symmetry).
- Use **Bethe lattice** self-consistency:
  $$
  \Delta(i\omega)=t^2 G_{\rm loc}(i\omega),
  $$
  (with half-bandwidth conventions chosen consistently).
- Start with a simple impurity solver (fast) and upgrade later.

### 7.3 Solver choices (ladder)
You can choose a sequence:

1) **Hubbard-I / atomic + hybridization** (fastest intuition, rough).
2) **IPT** (often decent near half-filling).
3) **ED impurity solver** (most consistent with discrete bath poles).
4) **CT-QMC** (most accurate, heavier).

For a first mini-project consistent with pole baths: **ED** is the most natural; IPT is the fastest.

### 7.4 Concrete deliverables (plots / numbers)
- $G_{\rm loc}(i\omega_n)$ vs $\omega_n$ (real and imaginary parts).
- $\Sigma(i\omega_n)$ and its tail check $\Sigma_\infty = U\langle n_{-\sigma}\rangle$.
- Proxy for quasiparticle weight:
  $$
  Z \approx \left[1-\frac{\mathrm{Im}\,\Sigma(i\omega_0)}{\omega_0}\right]^{-1}
  \quad(\text{rough, Matsubara proxy}).
  $$
- Real-frequency spectral function $A(\omega)$ if using a pole representation (analytic continuation is easy because it’s rational).

### 7.5 Implementation outline (agent-ready)

#### Step A — choose conventions
- Set $t$, bandwidth $D$, temperature $T$, chemical potential $\mu$.
- Decide Matsubara cutoff $N_\omega$.

#### Step B — represent $\Delta$ and $\Sigma$ with poles
- Choose $M_g, M_h$ (start with 1–3).
- Store parameters $\{\epsilon_\ell,V_\ell\}$, $\{\eta_\ell,W_\ell\}$, $\Sigma_\infty$.

#### Step C — compute quadratic model Green’s functions by Schur complement
- For each $i\omega_n$, compute $G_{dd}(i\omega_n)$ and needed blocks $G_{h_\ell d}$, $G_{h_\ell h_\ell}$, etc.
- Convert blocks to correlators via Matsubara sums.

#### Step D — matching updates
- From lattice ↔ gateway matching: update $\{\epsilon_\ell,V_\ell\}$ (bath poles) to match $d/h$ correlators.
- From impurity ↔ gateway matching: update $\{\eta_\ell,W_\ell\}$ (self-energy poles) to match $d/g$ correlators.

*(Exact parameter-update method can be implemented as solving small nonlinear systems or least-squares on correlator constraints; start simple.)*

#### Step E — impurity solve
- Given bath $\{\epsilon_\ell,V_\ell\}$, solve impurity $H_{\rm imp}$.
- Output: $\langle n\rangle$, and correlators with $g$-levels (or impurity $G_{\rm imp}$, depending on solver).

#### Step F — update $\Sigma_\infty$
$$
\Sigma_\infty \leftarrow U\langle n_{-\sigma}\rangle.
$$

#### Step G — iterate + convergence
- Monitor change in parameters and/or $G_{\rm loc}(i\omega_n)$.
- Use mixing if needed.

### 7.6 Suggested milestone sequence
1) **Milestone 0:** implement pure quadratic Schur complement pipeline and verify:
   - bath-only gives $\Delta(i\omega)$ pole sum
   - ghost-only gives $\Sigma^{(h)}(i\omega)$ pole sum
2) **Milestone 1:** add Bethe self-consistency with a simple solver (Hubbard-I/IPT) and reproduce qualitative three-peak spectral shape.
3) **Milestone 2:** switch impurity solver to ED and compare results as bath size $M_g$ increases.
4) **Milestone 3:** scan $U$ and observe trend toward Mott transition.

---

## 8) Quick “board exam” checklist (what you should be able to reproduce)

1) $G(i\omega)=(i\omega+\mu-h)^{-1}$ for quadratic Hamiltonian.
2) Schur complement formula and “integrate out a level” giving $|V|^2/(i\omega-\epsilon)$.
3) EOM moment expansion and $\Sigma_\infty=U\langle n_{-\sigma}\rangle$.
4) Hellmann–Feynman $\partial\Omega/\partial\lambda=\langle \partial H/\partial\lambda\rangle$.
5) Why $F=\Omega_{\rm lat}+\Omega_{\rm imp}-\Omega_{\rm imp}^{(0)}$ produces matching conditions.

---

## 9) Notes for working with an agent
When delegating to an agent, ask them to:
- keep sign conventions consistent ($\mu$, $\epsilon_d$, Fourier sign),
- implement Schur complement robustly and test against direct inversion,
- add unit tests for the 2×2 and (1+M)×(1+M) cases,
- implement correlator-from-Matsubara-sum carefully with $e^{i\omega_n 0^+}$ factor,
- start with small $M$ and verify outputs against known DMFT qualitative behavior.
