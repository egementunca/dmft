#!/usr/bin/env python3
"""
Ghost-DMFT bond (BPK) scheme — independent M per g-ghost family.

Fixed: M1h = M2h = Mbh = 1 (one h-ghost per family)
Free:  M1g, M2g, Mbg  (independent g-ghost counts)

Matching conditions:
  Step 2a: vary (eps1[M1g], V1[M1g])       -> match gw1 h^(1) sector  [2 eqs]
  Step 2b: vary (eps2[M2g], V2[M2g],
                 epsb[Mbg], Bg[Mbg])        -> match gw2 h^(2),h^(b)  [4 eqs]
  Step 4a: vary (eta1[1],   W1[1])          -> match gw1 g^(1) sector  [2*M1g eqs]
  Step 4b: vary (eta2[1],   W2[1],
                 etab[1],   Bh[1])          -> match gw2 g^(2),g^(b)  [2*(M2g+Mbg) eqs]

Square lattice z=4.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import eigh
from itertools import combinations
import sys, time, argparse

# ============================================================
# Utilities
# ============================================================

def fermi(e, beta):
    x = beta * np.asarray(e, dtype=float)
    out = np.empty_like(x)
    out[x >  50] = 0.0
    out[x < -50] = 1.0
    m = (x <= 50) & (x >= -50)
    out[m] = 1.0 / (np.exp(x[m]) + 1.0)
    return out

def c_op(dim, mode):
    C = np.zeros((dim, dim))
    for state in range(dim):
        if (state >> mode) & 1:
            lo = state & ((1 << mode) - 1)
            sgn = (-1) ** bin(lo).count('1')
            C[state ^ (1 << mode), state] = sgn
    return C

# ============================================================
# Square lattice
# ============================================================

def make_square_lattice(t, n_k=20):
    kx = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    KX, KY = np.meshgrid(kx, ky)
    eps_k   = -2*t*(np.cos(KX) + np.cos(KY))
    gamma_k = eps_k / 4
    weights = np.ones(n_k*n_k) / (n_k*n_k)
    return eps_k.ravel(), gamma_k.ravel(), weights, 4

# ============================================================
# Lattice statics  (M1h=M2h=Mbh=1 fixed)
# orbs per spin: d, h1, h2, hb
# ============================================================

def lattice_statics(beta, eta1, W1, eta2, W2, etab, Bh,
                    M1h, M2h, Mbh, eps_k, gamma_k, weights, shift):
    """Lattice one-body problem with independent M per h-ghost family."""
    eta1=np.atleast_1d(eta1); W1=np.atleast_1d(W1)
    eta2=np.atleast_1d(eta2); W2=np.atleast_1d(W2)
    etab=np.atleast_1d(etab); Bh=np.atleast_1d(Bh)
    Norb = 1 + M1h + M2h + Mbh
    nh1=np.zeros(M1h); dh1=np.zeros(M1h)
    nh2=np.zeros(M2h); dh2=np.zeros(M2h)
    nhb=np.zeros(Mbh); dhb=np.zeros(Mbh)
    nd_tot = 0.0
    for ek, gk, wk in zip(eps_k, gamma_k, weights):
        H = np.zeros((Norb, Norb))
        H[0,0] = ek + shift
        for l in range(M1h):
            i = 1+l
            H[i,i]=eta1[l]; H[0,i]=H[i,0]=W1[l]
        for l in range(M2h):
            i = 1+M1h+l
            H[i,i]=eta2[l]; H[0,i]=H[i,0]=W2[l]
        for l in range(Mbh):
            i = 1+M1h+M2h+l
            H[i,i]=etab[l]; H[0,i]=H[i,0]=Bh[l]*gk
        ev,U = eigh(H); f=fermi(ev,beta); rho=(U*f)@U.T
        nd_tot += wk*rho[0,0]
        for l in range(M1h):
            nh1[l]+=wk*rho[1+l,1+l]; dh1[l]+=wk*rho[0,1+l]
        for l in range(M2h):
            i=1+M1h+l; nh2[l]+=wk*rho[i,i]; dh2[l]+=wk*rho[0,i]
        for l in range(Mbh):
            i=1+M1h+M2h+l; nhb[l]+=wk*rho[i,i]; dhb[l]+=wk*gk*rho[0,i]
    return nh1, dh1, nh2, dh2, nhb, dhb, nd_tot

# ============================================================
# Gateway 1  (single-site, quadratic)
# orbs per spin: d, h1, g1_0..g1_{M1g-1}
# ============================================================

def gateway1_statics(beta, eta1, W1, eps1, V1, M1g, shift):
    eta1=float(np.atleast_1d(eta1)[0]); W1=float(np.atleast_1d(W1)[0])
    Norb = 2 + M1g
    H = np.zeros((Norb, Norb))
    H[0,0] = shift
    H[1,1] = eta1;  H[0,1] = H[1,0] = W1
    for l in range(M1g):
        H[2+l, 2+l] = eps1[l]
        H[0, 2+l] = H[2+l, 0] = V1[l]
    ev, U = eigh(H)
    f = fermi(ev, beta)
    rho = (U * f) @ U.T
    nh1 = rho[1,1];  dh1 = rho[0,1]
    ng1 = np.array([rho[2+l,2+l] for l in range(M1g)])
    dg1 = np.array([rho[0,2+l]   for l in range(M1g)])
    nd  = rho[0,0]
    return nh1, dh1, ng1, dg1, nd

# ============================================================
# Gateway 2  (two-site, quadratic)
# orbs per spin: d1, d2,
#   h2_0..h2_0 (M2h=1), hb_0 (Mbh=1),
#   g2_site1_0..g2_site1_{M2g-1}, g2_site2_0..g2_site2_{M2g-1},
#   gb_site1_0..gb_site1_{Mbg-1}, gb_site2_0..gb_site2_{Mbg-1}
# ============================================================

def gateway2_statics(beta, eta2, W2, etab, Bh,
                     eps2, V2, epsb, Bg, M2g, Mbg, t, shift):
    eta2=float(np.atleast_1d(eta2)[0]); W2=float(np.atleast_1d(W2)[0])
    etab=float(np.atleast_1d(etab)[0]); Bh=float(np.atleast_1d(Bh)[0])
    # orbital layout (per spin):
    # 0=d1, 1=d2
    # 2..2+M2h-1     = h2_site1  (local, couples to d1 only)
    # 2+M2h..2+2M2h-1 = h2_site2 (local, couples to d2 only)
    # 2+2*M2h..2+2*M2h+Mbh-1 = hb (shared bond, couples to both d1 and d2)
    # 2+2*M2h+Mbh..2+2*M2h+Mbh+M2g-1     = g2_site1 (local, d1)
    # 2+2*M2h+Mbh+M2g..2+2*M2h+Mbh+2M2g-1 = g2_site2 (local, d2)
    # 2+2*M2h+Mbh+2*M2g..+Mbg-1 = gb (shared bond, both d1 and d2)
    M2h = 1; Mbh = 1  # fixed
    Norb = 2 + 2*M2h + Mbh + 2*M2g + Mbg
    H = np.zeros((Norb, Norb))

    # d1, d2 onsite + hopping
    H[0,0] = shift;  H[1,1] = shift
    H[0,1] = H[1,0] = -t

    # h2: LOCAL per site (d1<->h2_site1, d2<->h2_site2)
    for l in range(M2h):
        i1 = 2+l;  i2 = 2+M2h+l
        H[i1,i1] = eta2;  H[i2,i2] = eta2
        H[0,i1] = H[i1,0] = W2   # d1 - h2_site1
        H[1,i2] = H[i2,1] = W2   # d2 - h2_site2

    # hb: SHARED bond orbital (both d1 and d2 couple to same hb)
    for l in range(Mbh):
        i = 2+2*M2h+l
        H[i,i] = etab
        H[0,i] = H[i,0] = Bh   # d1 - hb
        H[1,i] = H[i,1] = Bh   # d2 - hb

    # g2: LOCAL per site (d1<->g2_site1, d2<->g2_site2)
    off = 2+2*M2h+Mbh
    for l in range(M2g):
        i1 = off+l;  i2 = off+M2g+l
        H[i1,i1] = eps2[l];  H[i2,i2] = eps2[l]
        H[0,i1] = H[i1,0] = V2[l]   # d1 - g2_site1
        H[1,i2] = H[i2,1] = V2[l]   # d2 - g2_site2

    # gb: SHARED bond orbital (both d1 and d2 couple to same gb)
    off2 = off + 2*M2g
    for l in range(Mbg):
        i = off2+l
        H[i,i] = epsb[l]
        H[0,i] = H[i,0] = Bg[l]   # d1 - gb
        H[1,i] = H[i,1] = Bg[l]   # d2 - gb

    ev, U = eigh(H)
    f = fermi(ev, beta)
    rho = (U * f) @ U.T

    # h2: average over site1 and site2
    nh2 = np.mean([rho[2+l,2+l] for l in range(M2h)])
    dh2 = np.mean([rho[0,2+l] for l in range(M2h)])  # <d1† h2_site1>
    # hb: shared orbital — dhb = <d1†hb> + <d2†hb>
    nhb = np.mean([rho[2+2*M2h+l,2+2*M2h+l] for l in range(Mbh)])
    dhb = np.mean([rho[0,2+2*M2h+l]+rho[1,2+2*M2h+l] for l in range(Mbh)])

    # g2: average over site1 and site2
    ng2 = np.array([0.5*(rho[off+l,off+l]+rho[off+M2g+l,off+M2g+l]) for l in range(M2g)])
    dg2 = np.array([0.5*(rho[0,off+l]+rho[1,off+M2g+l]) for l in range(M2g)])

    # gb: shared orbital — dgb = <d1†gb> + <d2†gb>
    ngb = np.array([rho[off2+l,off2+l] for l in range(Mbg)])
    dgb = np.array([rho[0,off2+l]+rho[1,off2+l] for l in range(Mbg)])

    nd = 0.5*(rho[0,0]+rho[1,1])
    return nh2, dh2, nhb, dhb, ng2, dg2, ngb, dgb, nd

# ============================================================
# Impurity 1  (single-site, interacting)
# orbs: d, g1_0..g1_{M1g-1}
# ============================================================

def impurity1_statics(beta, eps1, V1, M1g, U, dmu):
    ed = -U/2.0 - dmu
    Norb = 1 + M1g
    Nmode = 2*Norb
    dim = 1 << Nmode
    C  = [c_op(dim, m) for m in range(Nmode)]
    Cd = [c.T for c in C]
    def n(o,s): return Cd[2*o+s] @ C[2*o+s]
    H = np.zeros((dim,dim))
    for s in range(2):
        H += ed * n(0,s)
        for l in range(M1g):
            H += eps1[l] * n(1+l, s)
            H += V1[l] * (Cd[2*0+s]@C[2*(1+l)+s] + Cd[2*(1+l)+s]@C[2*0+s])
    H += U * (n(0,0) @ n(0,1))
    ev, U_mat = eigh(H)
    E0 = ev.min()
    w = np.exp(-beta*(ev-E0)); prob = w/w.sum()
    def avg(O): return float(np.sum(prob * np.diag(U_mat.T @ O @ U_mat)))
    ng1 = np.array([avg(n(1+l,0)) for l in range(M1g)])
    dg1 = np.array([avg(Cd[0]@C[2*(1+l)]) for l in range(M1g)])
    nd   = avg(n(0,0))
    docc = avg(n(0,0)@n(0,1))
    return ng1, dg1, nd, docc

# ============================================================
# Impurity 2  (two-site, interacting, grand canonical via c_op)
# Orbs: d1(0), d2(1),
#   g2_site1(2..2+M2g-1), g2_site2(2+M2g..2+2*M2g-1)  [local]
#   gb(2+2*M2g..2+2*M2g+Mbg-1)                          [shared bond]
# ============================================================

def impurity2_statics(beta, eps2, V2, epsb, Bg, M2g, Mbg, U, t, dmu):
    ed = -U/2.0 - dmu
    Norb  = 2 + 2*M2g + Mbg
    Nmode = 2*Norb
    dim   = 1 << Nmode

    C  = [c_op(dim, m) for m in range(Nmode)]
    Cd = [c.T for c in C]
    def n(o, s): return Cd[2*o+s] @ C[2*o+s]

    H = np.zeros((dim, dim))

    # d onsite + U
    for s in range(2):
        H += ed * (n(0,s) + n(1,s))
    H += U * (n(0,0) @ n(0,1))
    H += U * (n(1,0) @ n(1,1))

    # d1-d2 hopping
    for s in range(2):
        H += (-t) * (Cd[2*0+s]@C[2*1+s] + Cd[2*1+s]@C[2*0+s])

    # g2: local (d1<->g2_site1, d2<->g2_site2)
    for l in range(M2g):
        g1 = 2+l; g2 = 2+M2g+l
        for s in range(2):
            H += eps2[l] * (n(g1,s) + n(g2,s))
            H += V2[l] * (Cd[2*0+s]@C[2*g1+s] + Cd[2*g1+s]@C[2*0+s])
            H += V2[l] * (Cd[2*1+s]@C[2*g2+s] + Cd[2*g2+s]@C[2*1+s])

    # gb: shared bond (both d1 and d2 couple to same gb)
    for l in range(Mbg):
        gb = 2+2*M2g+l
        for s in range(2):
            H += epsb[l] * n(gb,s)
            H += Bg[l] * (Cd[2*0+s]@C[2*gb+s] + Cd[2*gb+s]@C[2*0+s])
            H += Bg[l] * (Cd[2*1+s]@C[2*gb+s] + Cd[2*gb+s]@C[2*1+s])

    ev, Umat = eigh(H)
    E0 = ev.min()
    w = np.exp(-beta*(ev-E0)); prob = w/w.sum()

    def avg(O): return float(np.sum(prob * np.diag(Umat.T @ O @ Umat)))

    ng2 = np.array([0.5*(avg(n(2+l,0))+avg(n(2+M2g+l,0))) for l in range(M2g)])
    dg2 = np.array([0.5*(avg(Cd[0]@C[2*(2+l)])+avg(Cd[2]@C[2*(2+M2g+l)])) for l in range(M2g)])
    ngb = np.array([avg(n(2+2*M2g+l,0)) for l in range(Mbg)])
    dgb = np.array([avg(Cd[0]@C[2*(2+2*M2g+l)])+avg(Cd[2]@C[2*(2+2*M2g+l)]) for l in range(Mbg)])

    nd   = 0.5*(avg(n(0,0)) + avg(n(1,0)))
    docc = 0.5*(avg(n(0,0)@n(0,1)) + avg(n(1,0)@n(1,1)))
    hop  = avg(Cd[0]@C[2])

    return ng2, dg2, ngb, dgb, nd, docc, hop



# ============================================================
# Single-site reference
# ============================================================

def solve_singlesite(beta, M1g, U, t, eps_k, weights,
                     eta0=None, W0=None, eps0=None, V0=None,
                     mix=0.5, tol=1e-8, maxiter=200):
    _i1 = lambda x: 0. if x is None else float(np.atleast_1d(x)[0])
    _iM = lambda x,M: np.zeros(M) if x is None else np.asarray(x,float).copy()
    eta = _i1(eta0); W = _i1(W0)
    eps = _iM(eps0,M1g); V = _iM(V0,M1g)
    shift = 0.0; docc = 0.0
    gamma_k = np.zeros_like(eps_k)  # dummy — not used for single-site

    for it in range(maxiter):
        # Lattice: only d + h1 (M1h=1)
        nh1_l,dh1_l,_,_,_,_,_ = lattice_statics(
            beta, [eta], [W], [0.], [0.], [0.], [0.],
            1, 0, 0, eps_k, gamma_k, weights, shift)
        nh1_l=float(nh1_l[0]); dh1_l=float(dh1_l[0])

        # Step 2: find eps,V from gw1 h-sector
        def res2(p):
            e=p[:M1g]; v=p[M1g:]
            nh1_g,dh1_g,_,_,_ = gateway1_statics(beta,eta,W,e,v,M1g,shift)
            return np.array([nh1_g-nh1_l, dh1_g-dh1_l])
        sol2=least_squares(res2, np.concatenate([eps,V]), method='trf',
                           ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=5000,
                           bounds=([-8]*(2*M1g),[8]*(2*M1g)))
        eps_new=sol2.x[:M1g]; V_new=sol2.x[M1g:]

        # Step 3: impurity
        ng1,dg1,nd,docc = impurity1_statics(beta,eps_new,V_new,M1g,U,0.0)

        # Step 4: find eta,W from gw1 g-sector
        def res4(p):
            _,_,ng1_g,dg1_g,_ = gateway1_statics(beta,p[0],p[1],eps_new,V_new,M1g,shift)
            return np.concatenate([ng1_g-ng1, dg1_g-dg1])
        sol4=least_squares(res4,[eta,W],method='trf',
                           ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=5000,
                           bounds=([-8,-8],[8,8]))
        eta_new=sol4.x[0]; W_new=sol4.x[1]

        dp = abs(eta_new-eta)+abs(W_new-W)+float(np.sum(np.abs(eps_new-eps)+np.abs(V_new-V)))
        eta=mix*eta_new+(1-mix)*eta; W=mix*W_new+(1-mix)*W
        eps=mix*eps_new+(1-mix)*eps; V=mix*V_new+(1-mix)*V
        if dp<tol: break
    return dict(eta=eta,W=W,eps=eps,V=V,docc=docc,iters=it)

# ============================================================
# Bond self-consistency loop
# ============================================================

# ============================================================
# Bond self-consistency — simultaneous solve of all conditions
# Variables: eta1[M1h],W1[M1h], eta2[M2h],W2[M2h], etab[Mbh],Bh[Mbh],
#            eps1[M1g],V1[M1g], eps2[M2g],V2[M2g], epsb[Mbg],Bg[Mbg]
# Equations: 2*M1h (h1-match) + 2*(M2h+Mbh) (h2,hb-match)
#          + 2*M1g (g1-match) + 2*(M2g+Mbg) (g2,gb-match)
# Square system requires: M1h=M1g, M2h+Mbh=M2g+Mbg
# ============================================================

def solve_bond(beta, M1g, M2g, Mbg, M1h, M2h, Mbh, U, t,
               eps_k, gamma_k, weights, z=4,
               p0=None, mix=0.5, tol=1e-7, maxiter=50, verbose=True):

    # All parameters free except mu=U/2 and Sigma_infinity (shift)
    n1h=M1h; n2h=M2h; nbh=Mbh; n1g=M1g; n2g=M2g; nbg=Mbg
    Np = 2*(n1h+n2h+nbh+n1g+n2g+nbg)

    def unpack(p):
        i = 0
        eta1=p[i:i+n1h]; i+=n1h; W1=p[i:i+n1h]; i+=n1h
        eta2=p[i:i+n2h]; i+=n2h; W2=p[i:i+n2h]; i+=n2h
        etab=p[i:i+nbh]; i+=nbh; Bh=p[i:i+nbh]; i+=nbh
        eps1=p[i:i+n1g]; i+=n1g; V1=p[i:i+n1g]; i+=n1g
        eps2=p[i:i+n2g]; i+=n2g; V2=p[i:i+n2g]; i+=n2g
        epsb=p[i:i+nbg]; i+=nbg; Bg=p[i:i+nbg]; i+=nbg
        return eta1,W1,eta2,W2,etab,Bh,eps1,V1,eps2,V2,epsb,Bg

    if p0 is None:
        p = np.zeros(Np)
        i0 = 2*(n1h+n2h+nbh)
        p[n1h:2*n1h] = 0.3              # W1
        p[2*n1h+n2h:2*n1h+2*n2h] = 0.3 # W2
        p[i0+n1g:i0+2*n1g] = 0.4        # V1
        p[i0+2*n1g+n2g:i0+2*n1g+2*n2g] = 0.4  # V2
    else:
        p = np.asarray(p0, dtype=float).copy()
        assert len(p) == Np

    # Initialize impurity correlators
    shift = 0.0
    ng1_i = np.zeros(n1g); dg1_i = np.zeros(n1g)
    ng2_i = np.zeros(n2g); dg2_i = np.zeros(n2g)
    ngb_i = np.zeros(nbg); dgb_i = np.zeros(nbg)
    docc1 = docc2 = hop = nd_total = docc_bpk = 0.0
    PENALTY = 1e4

    for it in range(1, maxiter+1):

        eta1,W1,eta2,W2,etab,Bh,eps1,V1,eps2,V2,epsb,Bg = unpack(p)

        # Step 1+2+4: simultaneously match all gateway conditions
        # Impurity correlators (ng1_i etc) are fixed for this inner solve
        def residuals(p_):
            et1,W1_,et2,W2_,etb,Bh_,e1,v1,e2,v2,eb,bg = unpack(p_)

            nh1_l,dh1_l,nh2_l,dh2_l,nhb_l,dhb_l,_ = lattice_statics(
                beta, et1, W1_, et2, W2_, etb, Bh_,
                M1h, M2h, Mbh, eps_k, gamma_k, weights, shift)

            nh1_g,dh1_g,ng1_g,dg1_g,_ = gateway1_statics(
                beta, et1, W1_, e1, v1, n1g, shift)
            nh2_g,dh2_g,nhb_g,dhb_g,ng2_g,dg2_g,ngb_g,dgb_g,nd2_g = gateway2_statics(
                beta, et2, W2_, etb, Bh_, e2, v2, eb, bg, n2g, nbg, t, shift)

            r = np.concatenate([
                nh1_g - nh1_l,   dh1_g - dh1_l,
                nh2_g - nh2_l,   dh2_g - dh2_l,
                nhb_g - nhb_l,   dhb_g - dhb_l,
                ng1_g - ng1_i,   dg1_g - dg1_i,
                ng2_g - ng2_i,   dg2_g - dg2_i,
                ngb_g - ngb_i,   dgb_g - dgb_i,
                [nd2_g - 0.5],   # enforce half-filling in gateway2
            ])
            if docc1<=0 or docc2<=0 or docc_bpk<=0:
                r *= PENALTY
            return r

        sol = least_squares(residuals, p, method='trf',
                            ftol=1e-10, xtol=1e-10, gtol=1e-10,
                            max_nfev=50000, bounds=([-8]*Np, [8]*Np))

        p_new = sol.x
        dp = float(np.linalg.norm(p_new - p))
        p = mix*p_new + (1-mix)*p
        eta1,W1,eta2,W2,etab,Bh,eps1,V1,eps2,V2,epsb,Bg = unpack(p)

        # Re-solve impurities; use dmu bisection to enforce nd2=0.5
        ng1_i,dg1_i,nd1_i,docc1 = impurity1_statics(beta,eps1,V1,n1g,U,0.)

        # Bisect dmu to enforce half-filling in impurity2
        dmu2 = 0.0
        for _ in range(30):
            ng2_i,dg2_i,ngb_i,dgb_i,nd2_i,docc2,hop = impurity2_statics(
                beta,eps2,V2,epsb,Bg,n2g,nbg,U,t,dmu2)
            err = nd2_i - 0.5
            if abs(err) < 1e-6: break
            dmu2 -= err / (beta * nd2_i * (1-nd2_i) * 2 + 1e-6)
            dmu2 = float(np.clip(dmu2, -2., 2.))

        nd_total = (1-z)*nd1_i + z*nd2_i
        docc_bpk = (1-z)*docc1 + z*docc2

        rnorm = float(np.linalg.norm(sol.fun))

        if verbose:
            physical = docc1>0 and docc2>0 and docc_bpk>0 and abs(nd_total-0.5)<0.05
            if physical:
                print(f'  it={it:3d}  dp={dp:.2e}  |r|={rnorm:.2e}  '
                      f'nd={nd_total:.4f}  docc1={docc1:.5f}  '
                      f'docc2={docc2:.5f}  docc_bpk={docc_bpk:.5f}  hop={hop:.5f}')
            else:
                print(f'  it={it:3d}  dp={dp:.2e}  |r|={rnorm:.2e}  '
                      f'[unphysical: nd={nd_total:.4f}  docc1={docc1:.4f}  '
                      f'docc2={docc2:.4f}  docc_bpk={docc_bpk:.4f}]')
            sys.stdout.flush()

        if dp < tol and rnorm < tol:
            if verbose: print(f'  Converged at it={it}')
            break

    return dict(p=p, eta1=eta1, W1=W1, eta2=eta2, W2=W2, etab=etab, Bh=Bh,
                eps1=eps1, V1=V1, eps2=eps2, V2=V2, epsb=epsb, Bg=Bg,
                docc1=docc1, docc2=docc2, docc_bpk=docc_bpk,
                hop=hop, nd_total=nd_total, iters=it, dp=dp, rnorm=rnorm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--M1g',    type=int,   default=1)
    parser.add_argument('--M2g',    type=int,   default=1)
    parser.add_argument('--Mbg',    type=int,   default=1)
    parser.add_argument('--M1h',    type=int,   default=1)
    parser.add_argument('--M2h',    type=int,   default=1)
    parser.add_argument('--Mbh',    type=int,   default=1)
    parser.add_argument('--U',      type=float, default=1.3)
    parser.add_argument('--t',      type=float, default=0.5)
    parser.add_argument('--nk',     type=int,   default=20)
    parser.add_argument('--mix',    type=float, default=0.5)
    parser.add_argument('--tol',    type=float, default=1e-7)
    parser.add_argument('--maxiter',type=int,   default=100)
    parser.add_argument('--Tmin',   type=float, default=0.1)
    parser.add_argument('--Tmax',   type=float, default=1.0)
    parser.add_argument('--nT',     type=int,   default=10)
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--tag',    type=str,   default='')
    args = parser.parse_args()

    M1g=args.M1g; M2g=args.M2g; Mbg=args.Mbg
    M1h=args.M1h; M2h=args.M2h; Mbh=args.Mbh
    U=args.U; t=args.t

    # Check system is square
    assert M1h == M1g, f'Need M1h={M1g} (=M1g) for square system, got M1h={M1h}'
    assert M2h+Mbh == M2g+Mbg, \
        f'Need M2h+Mbh={M2g+Mbg} (=M2g+Mbg) for square system, got {M2h+Mbh}'

    eps_k, gamma_k, weights, z = make_square_lattice(t, n_k=args.nk)
    T_vals = np.linspace(args.Tmax, args.Tmin, args.nT)

    print(f'\nGhost-DMFT bond  M1g={M1g} M2g={M2g} Mbg={Mbg}  '
          f'M1h={M1h} M2h={M2h} Mbh={Mbh}  U={U}  t={t}  z={z}')
    print(f'{"T":>6}  {"docc_ss":>9}  {"docc_bpk":>10}  '
          f'{"docc1":>8}  {"docc2":>8}  {"hop":>8}  {"nd":>7}  {"its":>4}')
    print('-'*85)

    p0 = None  # warm-start parameter vector
    results = []

    for T in T_vals:
        beta = 1./T
        t0 = time.time()

        # Single-site reference
        ss = solve_singlesite(beta, M1g, U, t, eps_k, weights,
                              eta0=0., W0=0.3,
                              eps0=np.zeros(M1g), V0=0.4*np.ones(M1g),
                              mix=0.5, tol=1e-8, maxiter=200)

        if args.verbose:
            print(f'\nT={T:.4f}  beta={beta:.2f}  ss docc={ss["docc"]:.6f}')

        # Build p0: [eta1(M1h), W1(M1h), eta2(M2h), W2(M2h), etab(Mbh), Bh(Mbh),
        #            eps1(M1g), V1(M1g), eps2(M2g), V2(M2g), epsb(Mbg), Bg(Mbg)]
        if p0 is None:
            Np = 2*(M1h+M2h+Mbh+M1g+M2g+Mbg)
            p0 = np.zeros(Np)
            i = 0
            p0[i:i+M1h]=ss['eta']; i+=M1h   # eta1
            p0[i:i+M1h]=ss['W'];   i+=M1h   # W1
            p0[i:i+M2h]=ss['eta']; i+=M2h   # eta2
            p0[i:i+M2h]=ss['W'];   i+=M2h   # W2
            i += 2*Mbh                        # etab=0, Bh=0
            i += M1g                          # eps1=0 initially
            p0[i:i+M1g]=ss['V'];   i+=M1g   # V1
            i += M2g                          # eps2=0 initially
            p0[i:i+M2g]=ss['V'][0]*np.ones(M2g); i+=M2g  # V2
            # epsb=0, Bg=0 initially

        rb = solve_bond(beta, M1g, M2g, Mbg, M1h, M2h, Mbh, U, t,
                        eps_k, gamma_k, weights, z=z,
                        p0=p0, mix=args.mix, tol=args.tol,
                        maxiter=args.maxiter, verbose=args.verbose)

        dt = time.time()-t0
        d1=rb['docc1']; d2=rb['docc2']; dbpk=rb['docc_bpk']
        nd=rb['nd_total']; hp=rb['hop']
        physical = d1>0 and d2>0 and dbpk>0 and abs(nd-0.5)<0.05

        if physical:
            print(f'  {T:6.4f}  {ss["docc"]:9.6f}  {dbpk:10.6f}  '
                  f'{d1:8.6f}  {d2:8.6f}  {hp:8.5f}  {nd:7.4f}  '
                  f'{rb["iters"]:4d}  ({dt:.1f}s)')
            print(f'    eta1={np.array2string(rb["eta1"],precision=4)}  W1={np.array2string(rb["W1"],precision=4)}')
            print(f'    eta2={np.array2string(rb["eta2"],precision=4)}  W2={np.array2string(rb["W2"],precision=4)}')
            print(f'    etab={np.array2string(rb["etab"],precision=4)}  Bh={np.array2string(rb["Bh"],precision=4)}')
            print(f'    eps1={np.array2string(rb["eps1"],precision=4)}  V1={np.array2string(rb["V1"],precision=4)}')
            print(f'    eps2={np.array2string(rb["eps2"],precision=4)}  V2={np.array2string(rb["V2"],precision=4)}')
            print(f'    epsb={np.array2string(rb["epsb"],precision=4)}  Bg={np.array2string(rb["Bg"],precision=4)}')
            p0 = rb['p']  # warm start next T
        else:
            print(f'  {T:6.4f}  {ss["docc"]:9.6f}  [unphysical]'
                  f'  iters={rb["iters"]}  |r|={rb["rnorm"]:.2e}  ({dt:.1f}s)')
        sys.stdout.flush()

        results.append([T, ss['docc'], dbpk, d1, d2, hp])

        # Save observables after every T
        fname = f'bond_M1g{M1g}M2g{M2g}Mbg{Mbg}_U{U}{("_"+args.tag) if args.tag else ""}.dat'
        np.savetxt(fname, np.array(results),
                   header='T docc_ss docc_bpk docc1 docc2 hop', fmt='%.8f')

        # Save parameters after every T
        pfname = f'bond_M1g{M1g}M2g{M2g}Mbg{Mbg}_U{U}_params.dat'
        with open(pfname, 'a') as pf:
            pf.write(f'T={T:.4f}  eta1={rb["eta1"]}  W1={rb["W1"]}  '
                     f'eta2={rb["eta2"]}  W2={rb["W2"]}  '
                     f'etab={rb["etab"]}  Bh={rb["Bh"]}  '
                     f'V1={rb["V1"]}  V2={rb["V2"]}  Bg={rb["Bg"]}\n')
        sys.stdout.flush()

    print("Done. Saved %d temperature points to %s" % (len(results), fname))
