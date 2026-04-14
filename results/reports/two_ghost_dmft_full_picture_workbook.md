# Two‑Ghost DMFT Notes — Full Picture Study & Re‑Derivation Workbook (Board‑Ready)

This is a **start-to-finish**, **board-derivation** guide for your professor’s “two-ghost / gateway” DMFT notes, with the minimum background filled in *only where needed*.  
Goal: you can **start from the right definitions**, and **derive the key equations** in a coherent order.

---

## How to use this workbook

- Treat each section as a **board module**.  
- Each module has:
  - **Start**: what you write first on the board
  - **Goal**: what you end with
  - **Derivation skeleton**: the minimal steps
  - **Sanity checks**: how to catch sign/definition drift

If you can do Modules 1–10, you can reproduce essentially all of the professor’s notes.

---

## Module 0 — Conventions (pick once; don’t drift)

### Imaginary time and Matsubara

- $\beta=1/T$, fermionic $\omega_n=(2n+1)\pi/\beta$.
- Imaginary-time Heisenberg (grand-canonical generator):
  $K=H-\mu N$, $O(\tau)=e^{\tau K} O e^{-\tau K}$.
- Fermion Green’s function:

$$
G_{ab}(\tau)\equiv -\langle T_\tau\, a(\tau)b^\dagger(0)\rangle,
\quad
G_{ab}(i\omega_n)=\int_0^\beta d\tau\,e^{i\omega_n\tau}G_{ab}(\tau).
$$

- Antiperiodicity: $G(\tau+\beta)=-G(\tau)$.

### Equal-time correlator from Matsubara

$$
\langle b^\dagger a\rangle
=
\frac{1}{\beta}\sum_n e^{i\omega_n0^+} G_{ab}(i\omega_n).
$$
This is your “bridge” from frequency-space Green’s functions to matching conditions.

---

## Module 1 — Quadratic Hamiltonian ⇒ $G=(i\omega+\mu-h)^{-1}$

## Start (board)

Write a quadratic Hamiltonian in one-body matrix form:
$$
H=\sum_{ab}\psi_a^\dagger h_{ab}\psi_b,
\qquad
K=H-\mu N=\sum_{ab}\psi_a^\dagger (h_{ab}-\mu\delta_{ab})\psi_b.
$$

## Goal

$$
G(i\omega_n)=(i\omega_n+\mu-h)^{-1}.
$$

## Derivation skeleton

1) EOM: $\partial_\tau\psi(\tau)=[K,\psi(\tau)]$.
2) Quadratic commutator identity:

$$
\Big[\sum_{bc}\psi_b^\dagger A_{bc}\psi_c,\psi_a\Big]= -\sum_c A_{ac}\psi_c.
$$
So $\partial_\tau\psi(\tau)=-(h-\mu)\psi(\tau)$.
3) Differentiate $G(\tau)$ (time-ordering produces delta):
$$
(\partial_\tau + h-\mu)G(\tau)=\delta(\tau)\,\mathbb 1.
$$
1) Fourier transform: $\partial_\tau\to i\omega_n$ (boundary term cancels using antiperiodicity & fermionic $\omega_n$).
2) Solve: $(i\omega_n+h-\mu)G=\mathbb 1\Rightarrow G=(i\omega_n+\mu-h)^{-1}$.

## Sanity checks

- One level $h=\epsilon$: $G=1/(i\omega_n+\mu-\epsilon)$.
- If you ever get $i\omega_n-\mu$ instead, you flipped the sign in $K=H-\mu N$.

---

## Module 2 — “Integrating out a level” = Schur complement

## Start

Block matrix inverse identity:
$$
M=\begin{pmatrix}A&B\\C&D\end{pmatrix},\ D^{-1}\ \text{exists}
\Rightarrow
(M^{-1})_{11}=(A-BD^{-1}C)^{-1}.
$$

## Goal

Show that coupling to auxiliary levels produces rational functions in $i\omega$.

## Derivation skeleton (2×2 example)

Take basis $(d,g)$ with one-body matrix
$h=\begin{pmatrix}\epsilon_d & V\\ V^* & \epsilon\end{pmatrix}$.
Then
$$
i\omega+\mu-h=
\begin{pmatrix}
i\omega+\mu-\epsilon_d & -V\\
-V^* & i\omega+\mu-\epsilon
\end{pmatrix}.
$$
Schur complement gives:
$$
G_{dd}(i\omega)
=
\frac{1}{i\omega+\mu-\epsilon_d-\frac{|V|^2}{i\omega+\mu-\epsilon}}.
$$

## Interpretation

The bath generates
$$
\Delta(i\omega)=\frac{|V|^2}{i\omega+\mu-\epsilon}.
$$
With $M$ bath levels $g_\ell$:
$$
\Delta(i\omega)=\sum_{\ell=1}^M \frac{|V_\ell|^2}{i\omega+\mu-\epsilon_\ell}.
$$

---

## Module 3 — Impurity action $S_{\rm imp}[\Delta]$ and what it means

## Start

Write the impurity partition function as a functional integral:
$$
Z_{\rm imp}[\Delta]=\int\mathcal D[d^\dagger,d]\,e^{-S_{\rm imp}[\Delta]},
\qquad
\Omega_{\rm imp}[\Delta]=-(1/\beta)\ln Z_{\rm imp}[\Delta].
$$

## Goal

Understand why the action contains a **double integral** and why $\Delta$ is “retarded”.

## The action (notes’ form)

$$
S_{\rm imp}[\Delta]=
-\int_0^\beta d\tau d\tau’\,
d^\dagger(\tau)
\Big[(\partial_\tau-\mu+\epsilon_d)\delta(\tau-\tau’)-\Delta(\tau-\tau’)\Big]
d(\tau’)
\; +\; U\int_0^\beta d\tau\, n_\uparrow(\tau)n_\downarrow(\tau).
$$

## Meaning

- $(\partial_\tau-\mu+\epsilon_d)\delta(\tau-\tau’)$: local-in-time bare level.
- $\Delta(\tau-\tau’)$: **memory kernel** (leave impurity, wander in bath, return later).
- The double integral is exactly what “nonlocal in time” means.

## Bridge to Hamiltonian bath

Start from Anderson impurity Hamiltonian with explicit bath $g_\ell$ and couplings $V_\ell$.
Integrate out (Gaussian) bath fields or Schur-complement them: the result is the above action with
$$
\Delta(i\omega_n)=\sum_{\ell}\frac{|V_\ell|^2}{i\omega_n+\mu-\epsilon_\ell}.
$$

---

# Module 4 — Functional derivative identity: $\delta\Omega_{\rm imp}/\delta\Delta = G_{\rm imp}$

## Start

$$
\Omega_{\rm imp}[\Delta]=-(1/\beta)\ln\int\mathcal D[d^\dagger,d]\,e^{-S_{\rm imp}[\Delta]}.
$$

## Goal

$$
\frac{\delta \Omega_{\rm imp}}{\delta \Delta(i\omega_n)}= G_{\rm imp}(i\omega_n).
$$

## Skeleton

1) $\delta\ln Z = \delta Z / Z$.
2) $\delta Z = -\int\mathcal D\, (\delta S) e^{-S}$.
3) So $\delta \Omega = (1/\beta)\langle \delta S\rangle$.
4) In $S_{\rm imp}$, $\Delta$ multiplies $d^\dagger d$ quadratically, so varying it “pulls down” $\langle d d^\dagger\rangle$, i.e. the impurity Green’s function.

---

# Module 5 — DMFT functional $F[\Delta,\Sigma]$ and stationarity ⇒ DMFT equations

## Start (notes’ key functional idea)

Write a functional of two functions $\Delta(i\omega)$ and $\Sigma(i\omega)$ such that stationarity gives:

- impurity Dyson equation
- $G_{\rm imp}=G_{\rm loc}$

## Goal (DMFT equations)

$$
G_{\rm imp}^{-1}(i\omega)= i\omega+\mu-\epsilon_d-\Delta(i\omega)-\Sigma(i\omega),
$$
$$
G_{\rm imp}(i\omega)=G_{\rm loc}(i\omega)=\sum_k\frac{1}{i\omega+\mu-\epsilon_k-\Sigma(i\omega)}.
$$

## Skeleton

- Vary $F$ w.r.t. $\Delta$: use Module 4 to get $G_{\rm imp}$, equate to the explicit propagator term → impurity Dyson.
- Vary $F$ w.r.t. $\Sigma$: lattice $\sum_k$ term appears; equate to the same impurity propagator term → $G_{\rm imp}=G_{\rm loc}$.

---

# Module 6 — High-frequency tail: $\Sigma_\infty = U\langle n_{-\sigma}\rangle$

## Start

EOM/spectral moment identity:
$$
G(i\omega)=\frac{\langle\{c,c^\dagger\}\rangle}{i\omega}+\frac{\langle\{[H,c],c^\dagger\}\rangle}{(i\omega)^2}+\cdots.
$$

## Goal

$$
\Sigma(i\omega)\to \Sigma_\infty = U\langle n_{-\sigma}\rangle.
$$

## Skeleton

1) Use the identity above for impurity $c=d_\sigma$.
2) Compute $[H_{\rm loc},d_\sigma]=-(\epsilon_d-\mu)d_\sigma - U n_{-\sigma}d_\sigma$.
3) Then $\{[H,d_\sigma],d_\sigma^\dagger\}=-(\epsilon_d-\mu)-U n_{-\sigma}$.
4) So

$$
G(i\omega)=\frac{1}{i\omega}+\frac{\epsilon_d-\mu+U\langle n_{-\sigma}\rangle}{(i\omega)^2}+\cdots.
$$
1) Compare with Dyson expansion:

$$
G(i\omega)=\frac{1}{i\omega}+\frac{\epsilon_d-\mu+\Sigma_\infty}{(i\omega)^2}+\cdots.
$$
Conclude $\Sigma_\infty=U\langle n_{-\sigma}\rangle$.

---

# Module 7 — Two-ghost parameterization: realizing $\Delta$ and $\Sigma$ as poles

## Start

Make explicit the approximations:
$$
\Delta(i\omega)=\sum_{\ell=1}^{M_g}\frac{|V_\ell|^2}{i\omega+\mu-\epsilon_\ell},
\qquad
\Sigma(i\omega)=\Sigma_\infty+\sum_{\ell=1}^{M_h}\frac{|W_\ell|^2}{i\omega-\eta_\ell}.
$$

## Goal

Understand that both are generated by *eliminating auxiliary fermions*.

---

# Module 8 — The three Hamiltonians and why $+\Omega_{\rm lat}+\Omega_{\rm imp}-\Omega_{\rm imp}^{(0)}$

## Start (define the three models conceptually)

1) $H_{\rm lat}$: lattice $d$ + local $h$-ghosts (quadratic).
2) $H_{\rm imp}$: impurity $d$ + bath $g$ + interaction $U$ (interacting).
3) $H_{\rm imp}^{(0)}$: gateway $d$ + $g$ + $h$ (quadratic).

## Goal

Explain the functional
$$
F=\Omega[H_{\rm lat}] + \Omega[H_{\rm imp}] - \Omega[H_{\rm imp}^{(0)}]
$$
and why stationarity gives **matching conditions**.

## Deep meaning (what you say)

- $\Omega[H_{\rm lat}]$: lattice “embedding” contribution for a chosen $\Sigma$.
- $\Omega[H_{\rm imp}]$: true local interaction contribution (the only place with $U$).
- $-\Omega[H_{\rm imp}^{(0)}]$: subtract the shared quadratic reference so you don’t double-count it; also makes derivatives become simple correlator equalities.

---

# Module 9 — Hellmann–Feynman and matching conditions

## Start

$$
\Omega[H]=-(1/\beta)\ln\mathrm{Tr}\,e^{-\beta H}.
$$

## Goal

$$
\frac{\partial\Omega}{\partial\lambda}=\left\langle\frac{\partial H}{\partial\lambda}\right\rangle,
$$
and thus $\partial F/\partial\lambda=0$ becomes correlator matching.

---

# Module 10 — Iteration algorithm (what is updated where)

You have three parameter sets:

- bath poles $\{\epsilon_\ell,V_\ell\}$ (control $\Delta$)
- self-energy poles $\{\eta_\ell,W_\ell\}$ (control $\Sigma^{(h)}$)
- tail $\Sigma_\infty$

**Half-step A (lattice → fix bath):**

1) guess $\{\eta,W\}$, $\Sigma_\infty$
2) solve $H_{\rm lat}$ (quadratic) → compute $\langle h^\dagger h\rangle_{\rm lat}, \langle d^\dagger h\rangle_{\rm lat}$
3) adjust $\{\epsilon,V\}$ so that in gateway $H_{\rm imp}^{(0)}$,

$$
\langle h^\dagger h\rangle_0=\langle h^\dagger h\rangle_{\rm lat},\quad
\langle d^\dagger h\rangle_0=\langle d^\dagger h\rangle_{\rm lat}.
$$

**Half-step B (impurity → fix self-energy poles):**
4) solve interacting $H_{\rm imp}$ with that $\{\epsilon,V\}$ → compute $\langle g^\dagger g\rangle_{\rm imp}, \langle d^\dagger g\rangle_{\rm imp}, \langle n\rangle$
5) adjust $\{\eta,W\}$ so that in gateway:
$$
\langle g^\dagger g\rangle_0=\langle g^\dagger g\rangle_{\rm imp},\quad
\langle d^\dagger g\rangle_0=\langle d^\dagger g\rangle_{\rm imp}.
$$
1) update $\Sigma_\infty \leftarrow U\langle n_{-\sigma}\rangle_{\rm imp}$
2) iterate to convergence.

---

# Module 11 — Two “limits”: $d\to\infty$ vs $M\to\infty$

## A) $d\to\infty$ (DMFT exactness)

- With hopping scaled as $t\sim 1/\sqrt{z}$, nonlocal corrections to $\Sigma$ vanish.
- Self-energy becomes purely local: $\Sigma(k,\omega)\to\Sigma(\omega)$.
- Cavity construction becomes exact: lattice ↔ impurity mapping is exact.

## B) $M\to\infty$ (pole completeness)

- Your prof’s scheme approximates $\Delta$ and $\Sigma$ by finite pole sums.
- As $M\to\infty$, the representable function class expands and the projections become complete → recover full DMFT within the local-self-energy assumption.

---

## Appendix — Board roadmap (how to start and connect)

If you’re asked to “derive the full picture” on a board, do it in this order:

1) Quadratic EOM → $G=(i\omega+\mu-h)^{-1}$
2) Schur complement → poles from eliminated auxiliaries
3) Impurity action $S_{\rm imp}[\Delta]$ and meaning of $\Delta$
4) $\delta\Omega_{\rm imp}/\delta\Delta = G_{\rm imp}$
5) Stationarity → DMFT equations
6) High-frequency tail $\Sigma_\infty$
7) Two-ghost pole parameterization
8) Three Hamiltonians + $+\Omega_{\rm lat}+\Omega_{\rm imp}-\Omega_{\rm imp}^{(0)}$
9) Hellmann–Feynman → matching
10) Iteration algorithm
11) Limits $d\to\infty$, $M\to\infty$

---

## What to do next

- Re-derive Modules 1–4 once cleanly.
- Then do Modules 8–10 carefully (this is “the notes”).
- Then start coding the mini-project (Bethe lattice + pole truncation).
