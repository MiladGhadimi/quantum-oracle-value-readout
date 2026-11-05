"""Reproduce the manuscript's numerical examples from explicit new instances.

Run: OPENBLAS_NUM_THREADS=1 python numerics/reproduce.py
Outputs: results.json, certificates.npz, mesh_data.npz, and Fig. 4.
The new seeds/configurations are not reconstructions of unavailable old data.
"""
from pathlib import Path
import json, platform, time, warnings
import numpy as np
from mesh_core import (herm, mineig, fourier, equal_tail, coherence, shifted, beam, beam_derivatives, decompose, mesh_unitaries, generators, correlation_data, quadratic_samples, coefficients, statistics, simulate)
import scipy
import cvxpy as cp
import clarabel
import scs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 20261102
N_PAIRS = 3000
N_SCAN_PAIRS = 25000
SIGMA = 1e-3
SOLVER_OPTIONS = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                      tol_feas=1e-10, max_iter=300)
CERT = {}
MESH_DATA = {}


def solve(problem):
    problem.solve(solver='CLARABEL',**SOLVER_OPTIONS)
    if problem.status not in ['optimal','optimal_inaccurate']:
        raise RuntimeError(problem.status)
    return problem.status

def discrimination(states, key, factor=1.):
    """Independent primal/dual solves plus numerical feasible repairs."""
    states=np.asarray(states)*factor;n,d,_=states.shape
    ms=[cp.Variable((d,d),hermitian=True) for _ in range(n)]
    primal=cp.Problem(cp.Maximize(sum(cp.real(cp.trace(r@m)) for r,m in zip(states,ms))),
                      [m>>0 for m in ms]+[sum(ms)==np.eye(d)])
    ps=solve(primal);mv=np.asarray([herm(m.value) for m in ms])
    y=cp.Variable((d,d),hermitian=True)
    dual=cp.Problem(cp.Minimize(cp.real(cp.trace(y))),[y-r>>0 for r in states])
    ds=solve(dual);yv=herm(y.value)
    pobj=float(np.real(np.einsum('aij,aji->',states,mv)));dobj=float(np.trace(yv).real)
    pmin=min(mineig(m) for m in mv);dmin=min(mineig(yv-r) for r in states)
    # Clip to PSD, then congruence-normalize the POVM; inflate the dual.
    fixed=[]
    for m in mv:
        ev,u=np.linalg.eigh(m);fixed.append((u*np.maximum(ev,0))@u.conj().T)
    total=sum(fixed);ev,u=np.linalg.eigh(total);inv=(u*(1/np.sqrt(ev)))@u.conj().T
    mf=np.asarray([herm(inv@m@inv) for m in fixed])
    yf=yv+(max(0.,-dmin)+1e-11)*np.eye(d)
    lower=float(np.real(np.einsum('aij,aji->',states,mf)));upper=float(np.trace(yf).real)
    CERT[key+'_states']=states;CERT[key+'_primal']=mv;CERT[key+'_dual']=yv
    CERT[key+'_primal_repaired']=mf;CERT[key+'_dual_repaired']=yf
    return dict(primal=pobj,dual=dobj,gap=dobj-pobj,completeness=float(np.linalg.norm(mv.sum(axis=0)-np.eye(d))),
                primal_min_eig=pmin,dual_slack_min_eig=dmin,lower=lower,upper=upper,
                repaired_completeness=float(np.linalg.norm(mf.sum(axis=0)-np.eye(d))),
                repaired_primal_min_eig=min(mineig(m) for m in mf),
                repaired_dual_slack_min_eig=min(mineig(yf-r) for r in states),
                primal_status=ps,dual_status=ds)

def robustness(rho,key):
    d=len(rho);t=cp.Variable(d)
    pr=cp.Problem(cp.Minimize(cp.sum(t)),[cp.diag(t)-rho>>0]);ps=solve(pr)
    x=cp.Variable((d,d),hermitian=True)
    du=cp.Problem(cp.Maximize(cp.real(cp.trace(rho@x))),[x>>0,cp.diag(x)==1]);ds=solve(du)
    tv=np.asarray(t.value);xv=herm(x.value)
    CERT[key+'_robust_diagonal']=tv;CERT[key+'_robust_witness']=xv
    return dict(C_R=float(tv.sum()-1),dual_C_R=float(np.trace(rho@xv).real-1),
                primal_slack_min_eig=mineig(np.diag(tv)-rho),dual_min_eig=mineig(xv),
                dual_diag_residual=float(np.linalg.norm(np.diag(xv)-1)),
                gap=float(tv.sum()-np.trace(rho@xv).real),primal_status=ps,dual_status=ds)

def scs_crosscheck(orbit,key):
    n=len(orbit);d=len(orbit[0]);ms=[cp.Variable((d,d),hermitian=True) for _ in orbit]
    problem=cp.Problem(cp.Maximize(sum(cp.real(cp.trace(r@m)) for r,m in zip(orbit,ms))/n),
                       [m>>0 for m in ms]+[sum(ms)==np.eye(d)])
    problem.solve(solver='SCS',eps=1e-9,max_iters=100000)
    mv=np.asarray([herm(m.value) for m in ms]);CERT[key+'_scs_primal']=mv
    return dict(value=float(problem.value),status=problem.status,
                completeness=float(np.linalg.norm(mv.sum(axis=0)-np.eye(d))),
                primal_min_eig=min(mineig(m) for m in mv))

def sdp_examples():
    d=4;p=np.array([.7,.1,.1,.1]);theta=np.array([0,.2,-.4,.7])
    eta=np.sqrt(p)*np.exp(1j*theta);pure=np.outer(eta,eta.conj());states=[]
    for gamma in [0.,.3,1.]:states.append((f'parent_gamma_{gamma:g}',np.exp(-gamma)*pure+(1-np.exp(-gamma))*np.diag(p)))
    rng=np.random.default_rng(SEED+400)
    for k in range(3):
        a=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d));rho=a@a.conj().T;rho/=np.trace(rho)
        states.append((f'generic_{k+1}',rho))
    rows=[]
    for name,rho in states:
        orbit=[]
        for a in range(d):
            t=np.diag(np.exp(2j*np.pi*np.arange(d)*a/d));orbit.append(t@rho@t.conj().T)
        disc=discrimination(orbit,'mixed_'+name,1/d);rob=robustness(rho,'mixed_'+name)
        CERT['mixed_'+name+'_rho_character']=rho
        c=float(abs(rho).sum()-np.trace(rho).real)
        assert abs(disc['primal']-(1+rob['C_R'])/d)<2e-8
        cross=scs_crosscheck(orbit,'mixed_'+name) if name.startswith('generic') else None
        if cross is not None:assert abs(cross['value']-disc['primal'])<2e-8
        rows.append(dict(name=name,discrimination=disc,robustness=rob,C=c,l1_bound=(1+c)/d,scs_crosscheck=cross))
        print('SDP',name,disc['primal'],disc['gap'],flush=True)
    return rows

def loss_examples():
    rng=np.random.default_rng(SEED+500);rows=[];k=0
    for d in range(2,7):
        f=fourier(d);weights=[equal_tail(d,.9),equal_tail(d,.5),rng.dirichlet(np.ones(d))]
        transmissions=[np.full(d,.7),np.linspace(.02,1,d),rng.uniform(.02,1,d)]
        for wi,p in enumerate(weights):
            for ti,tau in enumerate(transmissions):
                for pi in range(4):
                    theta=np.zeros(d) if pi==0 else rng.uniform(0,2*np.pi,d)
                    eta=f@(np.sqrt(p)*np.exp(1j*theta));v=np.sqrt(tau)[:,None]*shifted(eta)
                    states=np.einsum('ia,ja->aij',v,v.conj());key=f'loss_{k:03d}'
                    diag=discrimination(states,key);bound=float(tau.mean()*(1+coherence(p)))
                    w=(abs(v)**2).sum(axis=0)
                    CERT[key+'_weights']=p;CERT[key+'_phases']=theta;CERT[key+'_transmission']=tau
                    rows.append(dict(d=d,weight=wi,profile=ti,phase=pi,key=key,bound=bound,
                                     min_survival=float(w.min()),diagnostics=diag,
                                     relative_gap=(bound-diag['primal'])/bound,
                                     certified_gap_lower=(bound-diag['upper'])/bound,
                                     certified_gap_upper=(bound-diag['lower'])/bound))
                    k+=1
        print('Loss SDPs complete for d =',d,flush=True)
    return rows

def meshes():
    result={};table=[]
    for d in [8,16]:
        rng=np.random.default_rng(SEED+d);theta=rng.uniform(-np.pi,np.pi,d)
        f=fourier(d);target=(f*np.exp(-1j*theta))@f.conj().T
        mesh=decompose(target);gs=generators(mesh);q,r,v=correlation_data(gs)
        MESH_DATA[f'd{d}_phases']=theta;MESH_DATA[f'd{d}_target']=target
        MESH_DATA[f'd{d}_mesh_ops']=np.asarray(mesh['ops']);MESH_DATA[f'd{d}_output']=mesh['output']
        MESH_DATA[f'd{d}_generators']=gs;MESH_DATA[f'd{d}_Q']=q;MESH_DATA[f'd{d}_R']=r
        # Independent check of physical tangent and the stationary Hessian.
        direction=np.random.default_rng(SEED+2*d).normal(size=d*d);h=1e-6
        numerical=(mesh_unitaries(mesh,(h*direction)[None,:])[0]-mesh_unitaries(mesh,(-h*direction)[None,:])[0])/(2*h)
        predicted=1j*np.einsum('j,jab->ab',direction,gs)@target
        tangent_error=float(np.linalg.norm(numerical-predicted));assert tangent_error<1e-7
        skew=np.arange(1,d+1,dtype=float);skew/=skew.sum()
        labels=['uniform','point mass','Phi=0.95','Phi=0.50','skewed']
        states=[np.ones(d)/d,equal_tail(d,1),equal_tail(d,.95),equal_tail(d,.5),skew]
        z=np.random.default_rng(SEED+100+d).normal(size=(N_PAIRS,d*d))
        direct=simulate(mesh,target,gs,states,z,SIGMA)
        for name,p,mc in zip(labels,states,direct):
            coeff=coefficients(gs,p);table.append(dict(d=d,state=name,p=p.tolist(),**coeff,**mc))
            print('Mesh',d,name,coeff['full'],mc['direct']/SIGMA**2,flush=True)
        result[str(d)]=dict(theta=theta.tolist(),reconstruction_residual=mesh['reconstruction_residual'],
                            tangent_residual=tangent_error,V=v.tolist())
        if d==8:
            identity=decompose(np.eye(d));gi=generators(identity)
            phases=gs[np.r_[np.arange(1,2*len(mesh['ops']),2),np.arange(2*len(mesh['ops']),d*d)]]
            splitting=gs[np.arange(0,2*len(mesh['ops']),2)]
            profiles={name:correlation_data(x)[2] for name,x in [('identity',gi),('all',gs),('phase',phases),('splitting',splitting)]}
            result['profiles']={k:(v/v.mean()).tolist() for k,v in profiles.items()}
            result['boundary_scan']=[dict(Phi=phi,**coefficients(gs,equal_tail(d,phi))) for phi in [1,.95,.8,.6,.5,.4,.2,1/d]]
            zscan=np.random.default_rng(SEED+300).normal(size=(N_SCAN_PAIRS,d*d))
            scales=[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,.1,.2]
            scan=[]
            for sigma in scales:
                mc=simulate(mesh,target,gs,[equal_tail(d,.5)],zscan,sigma)[0]
                scan.append(dict(sigma=sigma,**mc));print('Scale',sigma,mc['direct']/sigma**2,flush=True)
            result['scale_scan']=scan;result['scale_c']=coefficients(gs,equal_tail(d,.5))['full']
    result['table']=table
    return result

def plot_scan(mesh):
    s=np.array([r['sigma'] for r in mesh['scale_scan']]);y=np.array([r['direct'] for r in mesh['scale_scan']]);e=np.array([r['direct_se'] for r in mesh['scale_scan']]);c=mesh['scale_c']
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42})
    fig,(a,b)=plt.subplots(2,1,figsize=(4.3,4.7),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.35,1]})
    a.errorbar(s,y,yerr=e,fmt='o',ms=4,color='#205781',label='Direct mesh simulation')
    a.plot(s,c*s*s,'--',color='#bf5b2f',label=rf'$c\sigma^2$, $c={c:.6f}$')
    a.set(xscale='log',yscale='log',ylabel=r'Mean loss $-\mathbb{E}[\Delta P]$');a.legend(frameon=False,fontsize=8)
    b.errorbar(s,y/s**2,yerr=e/s**2,fmt='o-',ms=4,color='#205781')
    b.axhline(c,ls='--',color='#bf5b2f');b.set(xscale='log',xlabel=r'Parameter-error standard deviation $\sigma$',ylabel=r'Mean loss / $\sigma^2$')
    for ax in [a,b]:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'sigma_scan_expectation.pdf');plt.close(fig)

def main():
    start=time.time()
    result={'seed':SEED,'n_pairs':N_PAIRS,'n_scan_pairs':N_SCAN_PAIRS,'sigma':SIGMA,
            'solver':'CLARABEL','solver_options':SOLVER_OPTIONS,
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                        'cvxpy':cp.__version__,'scs':scs.__version__,'clarabel':clarabel.__version__,'matplotlib':matplotlib.__version__}}
    result['mixed']=sdp_examples()
    result['mesh']=meshes()
    result['loss']=loss_examples()
    np.savez_compressed(HERE/'certificates.npz',**CERT)
    np.savez_compressed(HERE/'mesh_data.npz',**MESH_DATA)
    result['elapsed_seconds']=time.time()-start
    (HERE/'results.json').write_text(json.dumps(result,indent=2))
    plot_scan(result['mesh'])
    print('Complete:',result['elapsed_seconds'],'seconds',flush=True)

if __name__=='__main__':main()
