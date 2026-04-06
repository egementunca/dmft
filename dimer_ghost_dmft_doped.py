#!/usr/bin/env python3
"""
dimer_ghost_dmft_doped.py
==========================
Dimer ghost-DMFT at arbitrary filling.

FILLING CONVENTION (used everywhere, no exceptions):
  n_dimer = total electrons on the two d-sites (A and B), both spins
  Range: 0 to 4.   Half-filling: n_dimer = 2.

  From impurity ED:
    occ_op(norb, dA) -> n_dA = electrons on site A, both spins, in [0,2]
    n_dimer_imp = n_dA + n_dB = 2 * n_dA  (by A<->B symmetry)

  Sigma_inf = U * n_{-sigma}
    n_{-sigma} = electrons of one spin on one site = n_dimer_imp / 4
    So: Sigma_inf = U * n_dimer_imp / 4

  From lattice (one-body rho, per-spin):
    rho[dA,dA] = per-spin occupation of site A, in [0,1]
    rho[dB,dB] = per-spin occupation of site B, in [0,1]
    n_dimer_lat = 2 * (rho[dA,dA] + rho[dB,dB])
                  ^--- factor 2 for both spins
    At half-filling: rho[dA,dA]=rho[dB,dB]=0.5 -> n_dimer_lat=2

  Bisection: find mu such that n_dimer_lat(mu) = n_target
    Half-filling: n_target = 2.0

Sequential structure each iteration:
  (1) Lattice(mu, Sigma_inf) -> h-targets
  (2) Gateway fit -> eps_g, V_g [, t_h]    h-sector: gw = lat
  (3) Impurity(mu) -> g-targets + n_dimer_imp
      Sigma_inf = U * n_dimer_imp / 4
      mu via bisection: n_dimer_lat(mu) = n_target
  (4) Gateway fit -> eta_h, W_h [, t_g]    g-sector: gw = imp

Usage:
  python dimer_ghost_dmft_doped.py --n_target 1.6 --U 2.0   # 20% hole doped
  python dimer_ghost_dmft_doped.py --n_target 2.0 --U 2.0   # half-filling
"""

import numpy as np
from functools import lru_cache
from scipy.optimize import least_squares, brentq
from scipy.linalg import eigh as scipy_eigh
import argparse


# =============================================================================
# 1.  Fermi function
# =============================================================================
def fermi(x, beta):
    x = np.asarray(x, dtype=float)
    y = beta * x
    out = np.empty_like(y)
    out[y >  60] = 0.0
    out[y < -60] = 1.0
    m = (y >= -60) & (y <= 60)
    out[m] = 1.0 / (np.exp(y[m]) + 1.0)
    return out


# =============================================================================
# 2.  Fock-space engine
# =============================================================================
def make_index(norb):
    return np.arange(4**norb, dtype=np.int32)

def c_action(state, site, spin, dag):
    bit = 2*site + spin
    occupied = (state >> bit) & 1
    if dag:
        if occupied: return -1, 0
        sign = (-1)**bin(state & ((1<<bit)-1)).count('1')
        return state | (1<<bit), sign
    else:
        if not occupied: return -1, 0
        sign = (-1)**bin(state & ((1<<bit)-1)).count('1')
        return state ^ (1<<bit), sign

def build_H(norb, onsite, hoppings, U_terms):
    dim=4**norb; basis=np.arange(dim,dtype=np.int32)
    idx=make_index(norb); H=np.zeros((dim,dim)); diag=np.zeros(dim)
    for site,eps in onsite:
        nu=(basis>>(2*site))&1; nd=(basis>>(2*site+1))&1
        diag+=eps*(nu+nd)
    for site,U in U_terms:
        nu=(basis>>(2*site))&1; nd=(basis>>(2*site+1))&1
        diag+=U*nu*nd
    np.fill_diagonal(H,diag)
    for (si,sj,amp) in hoppings:
        if abs(amp)<1e-14: continue
        for spin in range(2):
            bj=2*sj+spin; bi=2*si+spin
            occ_j=(basis>>bj)&1; emp_i=1-((basis>>bi)&1)
            active=np.where(occ_j&emp_i)[0]
            if not len(active): continue
            states=basis[active]
            sg1=np.where(np.array([bin(int(s)&((1<<bj)-1)).count('1')%2
                                   for s in states]),-1.,1.)
            s1=states^(1<<bj)
            sg2=np.where(np.array([bin(int(s)&((1<<bi)-1)).count('1')%2
                                   for s in s1]),-1.,1.)
            s2=s1|(1<<bi)
            H[idx[s2],active]+=amp*sg1*sg2; H[active,idx[s2]]+=amp*sg1*sg2
    return H

@lru_cache(maxsize=None)
def _get_NSz_blocks(norb):
    dim=4**norb; basis=np.arange(dim,dtype=np.int32)
    N_up=np.zeros(dim,dtype=np.int32); N_dn=np.zeros(dim,dtype=np.int32)
    for s in range(norb):
        N_up+=(basis>>(2*s))&1; N_dn+=(basis>>(2*s+1))&1
    sectors={}; N_arr=N_up+N_dn; Sz2=N_up-N_dn
    for N in range(2*norb+1):
        for S in range(-N,N+1,2):
            mask=np.where((N_arr==N)&(Sz2==S))[0]
            if len(mask): sectors[(N,S)]=mask
    return sectors

def thermal_obs(H, beta, diag_ops, off_ops):
    dim=H.shape[0]; norb=int(round(np.log(dim)/np.log(4)))
    sectors=_get_NSz_blocks(norb)
    all_E=[]; all_V=[]
    for mask in sectors.values():
        Hb=H[np.ix_(mask,mask)]
        try: e,v=np.linalg.eigh(Hb)
        except: e,v=scipy_eigh(Hb)
        all_E.append(e); all_V.append((mask,v))
    Ef=np.concatenate(all_E)-min(e.min() for e in all_E)
    bw=np.exp(-beta*Ef); p=bw/bw.sum()
    res={}
    for name,diag in diag_ops.items():
        val=0.; ptr=0
        for (mask,v) in all_V:
            n=len(mask); Oe=np.einsum('in,i,in->n',v,diag[mask],v)
            val+=np.dot(p[ptr:ptr+n],Oe); ptr+=n
        res[name]=float(val)
    for name,op in off_ops.items():
        val=0.; ptr=0
        for (mask,v) in all_V:
            n=len(mask); Oe=np.diag(v.T@op[np.ix_(mask,mask)]@v)
            val+=np.dot(p[ptr:ptr+n],Oe.real); ptr+=n
        res[name]=float(val)
    return res

@lru_cache(maxsize=None)
def _occ_op_cached(norb,site):
    dim=4**norb; basis=np.arange(dim)
    return (((basis>>(2*site))&1)+((basis>>(2*site+1))&1)).astype(float)

@lru_cache(maxsize=None)
def _docc_op_cached(norb,site):
    dim=4**norb; basis=np.arange(dim)
    return (((basis>>(2*site))&1)*((basis>>(2*site+1))&1)).astype(float)

@lru_cache(maxsize=None)
def _cdag_c_op_cached(norb,si,sj):
    dim=4**norb; basis=np.arange(dim,dtype=np.int32)
    idx=make_index(norb); op=np.zeros((dim,dim))
    for m,state in enumerate(basis):
        for spin in range(2):
            s1,sg1=c_action(state,sj,spin,dag=False)
            if s1<0: continue
            s2,sg2=c_action(s1,si,spin,dag=True)
            if s2<0: continue
            op[idx[s2],m]+=sg1*sg2
    return op

def occ_op(norb,site):      return _occ_op_cached(norb,site)
def docc_op(norb,site):     return _docc_op_cached(norb,site)
def cdag_c_op(norb,si,sj): return _cdag_c_op_cached(norb,si,sj)


# =============================================================================
# 3.  Impurity
# =============================================================================
def impurity_obs(beta, mu, U, t_b, M, eps_g, V_g, t_g, hop):
    dA=0; dB=1
    gA=[2+m for m in range(M)]; gB=[2+M+m for m in range(M)]
    norb=2+2*M
    onsite=[(dA,-mu),(dB,-mu)]; hoppings=[(dA,dB,t_b)]
    for m in range(M):
        onsite+=[(gA[m],eps_g[m]),(gB[m],eps_g[m])]
        hoppings+=[(dA,gA[m],V_g[m]),(dB,gB[m],V_g[m])]
        if hop: hoppings+=[(gA[m],gB[m],t_g[m])]
    H=build_H(norb,onsite,hoppings,[(dA,U),(dB,U)])
    diag_ops={'docc':docc_op(norb,dA), 'n_dA':occ_op(norb,dA)}
    off_ops={}
    for m in range(M):
        diag_ops[f'n_gA_{m}']=occ_op(norb,gA[m])
        off_ops[f'd_gA_{m}']=cdag_c_op(norb,dA,gA[m])
        if hop: off_ops[f'ghop_{m}']=cdag_c_op(norb,gA[m],gB[m])
    res=thermal_obs(H,beta,diag_ops,off_ops)

    n_dA = res['n_dA']
    n_dimer_imp = 2.0 * n_dA

    out = {'docc': res['docc'],
           'n_dimer_imp': n_dimer_imp}
    out['n_g']  = np.array([res[f'n_gA_{m}']/2. for m in range(M)])
    out['d_g']  = np.array([res[f'd_gA_{m}']/2. for m in range(M)])
    out['ghop'] = (np.array([res[f'ghop_{m}']/2. for m in range(M)])
                   if hop else np.zeros(M))
    return out


# =============================================================================
# 4.  Gateway
# =============================================================================
def gateway_obs(beta, mu, Sigma_inf, t_b, M,
                eps_g, V_g, t_g, eta_h, W_h, t_h, hop):
    dA=0; dB=1
    gA=[2+m for m in range(M)]; gB=[2+M+m for m in range(M)]
    hA=[2+2*M+m for m in range(M)]; hB=[2+3*M+m for m in range(M)]
    sz=2+4*M
    H1b=np.zeros((sz,sz))
    dlev = Sigma_inf - mu
    H1b[dA,dA]=dlev; H1b[dB,dB]=dlev
    H1b[dA,dB]=H1b[dB,dA]=-t_b
    for m in range(M):
        H1b[gA[m],gA[m]]=eps_g[m]; H1b[gB[m],gB[m]]=eps_g[m]
        H1b[dA,gA[m]]=H1b[gA[m],dA]=V_g[m]
        H1b[dB,gB[m]]=H1b[gB[m],dB]=V_g[m]
        H1b[hA[m],hA[m]]=eta_h[m]; H1b[hB[m],hB[m]]=eta_h[m]
        H1b[dA,hA[m]]=H1b[hA[m],dA]=W_h[m]
        H1b[dB,hB[m]]=H1b[hB[m],dB]=W_h[m]
        if hop:
            H1b[gA[m],gB[m]]=H1b[gB[m],gA[m]]=-t_g[m]
            H1b[hA[m],hB[m]]=H1b[hB[m],hA[m]]=-t_h[m]
    e,U=np.linalg.eigh(H1b)
    rho=(U*fermi(e,beta)[None,:])@U.T
    out={}
    out['n_gA']=np.array([rho[gA[m],gA[m]] for m in range(M)])
    out['d_gA']=np.array([rho[dA,   gA[m]] for m in range(M)])
    out['ghop']=np.array([rho[gA[m],gB[m]] for m in range(M)])
    out['n_hA']=np.array([rho[hA[m],hA[m]] for m in range(M)])
    out['d_hA']=np.array([rho[dA,   hA[m]] for m in range(M)])
    out['hhop']=np.array([rho[hA[m],hB[m]] for m in range(M)])
    return out


# =============================================================================
# 5.  Lattice
# =============================================================================
def square_lattice_kgrid(t_d, nk=20):
    k=np.linspace(-np.pi,np.pi,nk,endpoint=False)
    kx,ky=np.meshgrid(k,k,indexing='ij')
    eps_k=(-2.*t_d*(np.cos(kx)+np.cos(ky))).ravel()
    return eps_k, np.ones(eps_k.size)/eps_k.size

def lattice_obs(beta, mu, Sigma_inf, t_b, M,
                eta_h, W_h, t_h, hop, eps_k, wk):
    dA=0; dB=1
    hA=[2+m for m in range(M)]; hB=[2+M+m for m in range(M)]
    sz=2+2*M; Nk=len(eps_k)
    dlev = Sigma_inf - mu
    H=np.zeros((Nk,sz,sz))
    H[:,dA,dA] = eps_k + dlev
    H[:,dB,dB] = dlev
    H[:,dA,dB] = H[:,dB,dA] = -t_b
    for m in range(M):
        H[:,hA[m],hA[m]]=eta_h[m]; H[:,hB[m],hB[m]]=eta_h[m]
        H[:,dA,hA[m]]=H[:,hA[m],dA]=W_h[m]
        H[:,dB,hB[m]]=H[:,hB[m],dB]=W_h[m]
        if hop: H[:,hA[m],hB[m]]=H[:,hB[m],hA[m]]=-t_h[m]
    e,U=np.linalg.eigh(H)
    y=beta*e
    f=np.where(y>60,0.,np.where(y<-60,1.,1./(np.exp(np.clip(y,-60,60))+1.)))
    rho=np.einsum('kin,kn,kjn->kij',U,f,U)

    n_dimer_lat = 2.0 * float(np.dot(wk, rho[:,dA,dA] + rho[:,dB,dB]))

    n_hA=np.zeros(M); d_hA=np.zeros(M); hhop=np.zeros(M)
    for m in range(M):
        n_hA[m]=np.dot(wk,rho[:,hA[m],hA[m]])
        d_hA[m]=np.dot(wk,rho[:,dA,   hA[m]])
        hhop[m]=np.dot(wk,rho[:,hA[m],hB[m]])
    return dict(n_dimer_lat=n_dimer_lat, n_hA=n_hA, d_hA=d_hA, hhop=hhop)


# =============================================================================
# 6.  Self-consistency
# =============================================================================
def solve_T_doped(T, x0, n_target=2.0, Uval=1.3, t_d=0.5, t_b=0.3,
                  M=1, hop=True, nk=32,
                  mix=0.5, tol=1e-8, maxiter=500, verbose=False):
    beta = 1.0/T
    eps_k, wk = square_lattice_kgrid(t_d, nk=nk)

    x0    = np.array(x0, dtype=float)
    eta_h = x0[0*M:1*M].copy()
    W_h   = x0[1*M:2*M].copy()
    t_h   = x0[2*M:3*M].copy()
    t_g   = x0[3*M:4*M].copy()
    eps_g = x0[4*M:5*M].copy()
    V_g   = x0[5*M:6*M].copy()
    mu        = float(x0[6*M])   if len(x0)>6*M   else Uval/2.
    Sigma_inf = float(x0[6*M+1]) if len(x0)>6*M+1 else Uval/2.

    lsq = dict(method='trf', ftol=1e-11, xtol=1e-11, gtol=1e-11,
                max_nfev=5000, bounds=(-15.,15.))
    docc = 0.25

    for it in range(1, maxiter+1):

        lat = lattice_obs(beta, mu, Sigma_inf, t_b, M,
                          eta_h, W_h, t_h, hop, eps_k, wk)
        n_hA_lat = lat['n_hA']
        d_hA_lat = lat['d_hA']
        hhop_lat = lat['hhop']

        p2 = np.concatenate([eps_g, V_g] + ([t_h] if hop else []))
        def r2(p):
            eg_=p[0:M]; Vg_=p[M:2*M]; th_=p[2*M:3*M] if hop else t_h
            gw=gateway_obs(beta,mu,Sigma_inf,t_b,M,eg_,Vg_,t_g,
                           eta_h,W_h,th_,hop)
            res=list(gw['n_hA']-n_hA_lat)+list(gw['d_hA']-d_hA_lat)
            if hop: res+=list(gw['hhop']-hhop_lat)
            return res
        sol2=least_squares(r2,p2,**lsq)
        eps_g_n=sol2.x[0:M].copy(); V_g_n=sol2.x[M:2*M].copy()
        t_h_n=sol2.x[2*M:3*M].copy() if hop else t_h.copy()

        imp = impurity_obs(beta, mu, Uval, t_b, M,
                           eps_g_n, V_g_n, t_g, hop)
        docc          = imp['docc']
        n_dimer_imp   = imp['n_dimer_imp']
        n_g_imp       = imp['n_g']
        d_g_imp       = imp['d_g']
        ghop_imp      = imp['ghop']

        Sigma_inf_n = Uval * n_dimer_imp / 4.0

        def filling(mu_val):
            l=lattice_obs(beta,mu_val,Sigma_inf_n,t_b,M,
                          eta_h,W_h,t_h_n,hop,eps_k,wk)
            return l['n_dimer_lat'] - n_target
        try:
            mu_n = brentq(filling, mu-4., mu+4., xtol=1e-8, maxiter=100)
        except ValueError:
            mu_n = brentq(filling, -10., 10., xtol=1e-8, maxiter=200)

        p4 = np.concatenate([eta_h, W_h] + ([t_g] if hop else []))
        def r4(p):
            eh_=p[0:M]; Wh_=p[M:2*M]; tg_=p[2*M:3*M] if hop else t_g
            gw=gateway_obs(beta,mu_n,Sigma_inf_n,t_b,M,
                           eps_g_n,V_g_n,tg_,eh_,Wh_,t_h_n,hop)
            res=list(gw['n_gA']-n_g_imp)+list(gw['d_gA']-d_g_imp)
            if hop: res+=list(gw['ghop']-ghop_imp)
            return res
        sol4=least_squares(r4,p4,**lsq)
        eta_h_n=sol4.x[0:M].copy(); W_h_n=sol4.x[M:2*M].copy()
        t_g_n=sol4.x[2*M:3*M].copy() if hop else t_g.copy()

        for arr in [eta_h_n,W_h_n,eps_g_n,V_g_n]:
            np.clip(arr,-10.,10.,out=arr)

        x_new=np.concatenate([eta_h_n,W_h_n,t_h_n,t_g_n,eps_g_n,V_g_n])
        x_old=np.concatenate([eta_h,  W_h,  t_h,  t_g,  eps_g,  V_g  ])
        dp=float(np.linalg.norm(np.concatenate(
            [x_new-x_old,[mu_n-mu,Sigma_inf_n-Sigma_inf]])))
        xm=mix*x_new+(1.-mix)*x_old

        eta_h=xm[0*M:1*M]; W_h=xm[1*M:2*M]
        t_h=xm[2*M:3*M];   t_g=xm[3*M:4*M]
        eps_g=xm[4*M:5*M]; V_g=xm[5*M:6*M]
        Sigma_inf = Sigma_inf_n
        mu        = mu_n

        if verbose:
            print(f'  it={it:3d}  dp={dp:.2e}  D={docc:.5f}'
                  f'  n_dimer={n_dimer_imp:.4f}'
                  f'  mu={mu:.4f}  Sinf={Sigma_inf:.4f}'
                  f'  n_lat={lat["n_dimer_lat"]:.4f}')
        if dp < tol:
            break

    x_out=np.concatenate([eta_h,W_h,t_h,t_g,eps_g,V_g,[mu,Sigma_inf]])
    return dict(T=T, iters=it, dp=dp, docc=docc,
                mu=mu, Sigma_inf=Sigma_inf,
                n_dimer_lat=lat['n_dimer_lat'],
                n_dimer_imp=n_dimer_imp,
                n_target=n_target,
                eps_g=eps_g.copy(), V_g=V_g.copy(),
                eta_h=eta_h.copy(), W_h=W_h.copy(),
                t_g=t_g.copy(), t_h=t_h.copy(),
                x=x_out)


# =============================================================================
# 7.  Temperature sweep
# =============================================================================
def run_sweep_doped(n_target=2.0, Uval=1.3, t_d=0.5, t_b=0.3,
                    M=1, hop=True, nk=32,
                    nT=20, T_max=5., T_min=0.1,
                    mix=0.5, tol=1e-8, maxiter=500, verbose=False,
                    x0=None):
    Ts = np.logspace(np.log10(T_max), np.log10(T_min), nT)
    if x0 is None:
        if M == 1:
            x0 = np.array([0.01, 0.20, 0.01, 0.01, -0.01, 0.20,
                            Uval/2., Uval/2.])
        elif M == 2:
            x0 = np.array([-0.30,  0.30,
                             0.20,  0.20,
                             0.01,  0.01,
                             0.01,  0.01,
                            -0.30,  0.30,
                             0.20,  0.20,
                            Uval/2., Uval/2.])
        else:
            sp = np.linspace(-0.4, 0.4, M)
            x0 = np.concatenate([sp, np.full(M,0.2), np.full(M,0.01),
                                  np.full(M,0.01), sp, np.full(M,0.2),
                                  [Uval/2., Uval/2.]])
    x0 = np.array(x0, dtype=float)

    mode = 'hop' if hop else 'no-hop'
    print(f'\nDimer ghost-DMFT doped  M={M}  [{mode}]'
          f'  n_target={n_target:.2f} (half-fill=2)'
          f'  U={Uval}  t_d={t_d}  t_b={t_b}  nk={nk}')
    print(f'Impurity dim = {4**(2+2*M)}')

    cols=['T','D','mu','Sigma_inf','n_dimer_lat','n_dimer_imp',
          'eps_g[0]','V_g[0]','eta_h[0]','W_h[0]','iters','dp']
    hdr='  '.join(f'{c:>12}' for c in cols)
    print(hdr); print('-'*len(hdr))

    results=[]; xp=None; x2=None
    for T in Ts:
        if   x2 is not None: xi=np.clip(2*xp-x2,-5.,5.)
        elif xp is not None: xi=xp.copy()
        else:                xi=x0.copy()

        r=solve_T_doped(T,xi,n_target=n_target,Uval=Uval,
                        t_d=t_d,t_b=t_b,M=M,hop=hop,nk=nk,
                        mix=mix,tol=tol,maxiter=maxiter,verbose=verbose)

        vals=[T,r['docc'],r['mu'],r['Sigma_inf'],
              r['n_dimer_lat'],r['n_dimer_imp'],
              r['eps_g'][0],r['V_g'][0],r['eta_h'][0],r['W_h'][0]]
        row='  '.join(f'{v:12.5f}' for v in vals)
        print(row+f'  {r["iters"]:8d}  {r["dp"]:9.2e}')
        results.append(r); x2=xp; xp=r['x'].copy()
    return results


# =============================================================================
# 8.  Sanity check
# =============================================================================
def check_halffill(Uval=1.3, T=1.0):
    print(f'Sanity check: half-filling n_target=2, U={Uval}, T={T}')
    x0=np.array([0.01,0.20,0.01,0.01,-0.01,0.20,Uval/2.,Uval/2.])
    r=solve_T_doped(T,x0,n_target=2.0,Uval=Uval,nk=20,
                    mix=0.5,tol=1e-8,maxiter=300,verbose=False)
    print(f'  mu={r["mu"]:.6f}         (should be {Uval/2.:.6f})')
    print(f'  Sigma_inf={r["Sigma_inf"]:.6f}  (should be {Uval/2.:.6f})')
    print(f'  n_dimer_lat={r["n_dimer_lat"]:.6f}  (should be 2.000000)')
    print(f'  n_dimer_imp={r["n_dimer_imp"]:.6f}  (should be 2.000000)')
    print(f'  D={r["docc"]:.6f}')
    ok = (abs(r['mu']-Uval/2.)<1e-4 and
          abs(r['Sigma_inf']-Uval/2.)<1e-4 and
          abs(r['n_dimer_lat']-2.0)<1e-4 and
          abs(r['n_dimer_imp']-2.0)<1e-4)
    print(f'  {"PASSED" if ok else "FAILED"}')
    return ok


# =============================================================================
# 9.  CLI
# =============================================================================
if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n_target', type=float, default=2.0,
                   help='Total dimer electrons. Half-filling=2.0')
    p.add_argument('--U',        type=float, default=1.3)
    p.add_argument('--t_d',      type=float, default=0.5)
    p.add_argument('--t_b',      type=float, default=0.3)
    p.add_argument('--M',        type=int,   default=1)
    p.add_argument('--nk',       type=int,   default=32)
    p.add_argument('--nT',       type=int,   default=20)
    p.add_argument('--T_max',    type=float, default=5.0)
    p.add_argument('--T_min',    type=float, default=0.1)
    p.add_argument('--mix',      type=float, default=0.5)
    p.add_argument('--tol',      type=float, default=1e-8)
    p.add_argument('--maxiter',  type=int,   default=500)
    p.add_argument('--hop',      action='store_true', default=True)
    p.add_argument('--verbose',  action='store_true')
    p.add_argument('--check',    action='store_true')
    args=p.parse_args()

    if args.check:
        check_halffill(Uval=args.U)
        print()

    run_sweep_doped(n_target=args.n_target, Uval=args.U,
                    t_d=args.t_d, t_b=args.t_b,
                    M=args.M, hop=args.hop, nk=args.nk,
                    nT=args.nT, T_max=args.T_max, T_min=args.T_min,
                    mix=args.mix, tol=args.tol, maxiter=args.maxiter,
                    verbose=args.verbose)
