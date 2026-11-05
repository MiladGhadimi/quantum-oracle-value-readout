"""Independent checks of stored certificates and query-channel identities.

Run after reproduce.py. No solver is called here. All certificate objectives
and residuals are recomputed from the saved matrices rather than trusted JSON.
"""
from pathlib import Path
import json
import numpy as np
from mesh_core import fourier, mesh_unitaries, coefficients, equal_tail

HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'results.json').read_text())
c=np.load(HERE/'certificates.npz')
m=np.load(HERE/'mesh_data.npz')
checked=0
max_complete=0.
for key in c.files:
    if not key.endswith('_states'): continue
    base=key[:-7]; states=c[key]; M=c[base+'_primal_repaired'];Y=c[base+'_dual_repaired']
    d=Y.shape[0]
    comp=np.linalg.norm(M.sum(0)-np.eye(d));max_complete=max(max_complete,comp)
    assert comp<1e-11, (base,comp)
    assert np.linalg.eigvalsh(M).min()>-1e-12
    assert min(np.linalg.eigvalsh(Y-a)[0] for a in states)>0
    lower=np.einsum('aij,aji->',states,M).real
    upper=np.trace(Y).real
    assert lower<=upper+1e-12
    if base.startswith('mixed_'):
        row=next(x for x in r['mixed'] if 'mixed_'+x['name']==base)
        diagnostic=row['discrimination']
        rho=c[base+'_rho_character'];t=c[base+'_robust_diagonal'];X=c[base+'_robust_witness']
        assert abs(t.sum()-1-row['robustness']['C_R'])<1e-12
        assert abs(np.trace(rho@X).real-1-row['robustness']['dual_C_R'])<1e-12
        assert row['l1_bound']>=upper-1e-7
        if row['name'].startswith('generic'):
            assert row['l1_bound']-upper>1e-3
            assert abs(np.einsum('aij,aji->',states,c[base+'_scs_primal']).real-row['scs_crosscheck']['value'])<1e-12
    else:
        row=next(x for x in r['loss'] if x['key']==base)
        diagnostic=row['diagnostics']
        if row['phase']==0 or row['profile']==0:
            assert lower-1e-10<=row['bound']<=upper+1e-10
        else:
            assert row['bound']>upper
    assert abs(lower-diagnostic['lower'])<1e-12
    assert abs(upper-diagnostic['upper'])<1e-12
    checked+=1

# A single explicit random direction checks the mesh's finite-error curvature
# against the analytic expectation formula for a deterministic generator.
curvature_checks=[]
for d in [8,16]:
    ops=[tuple(x) for x in m[f'd{d}_mesh_ops']]
    ops=[(int(i),int(j),t,p) for i,j,t,p in ops]
    mesh={'d':d,'ops':ops,'output':m[f'd{d}_output']}
    target=m[f'd{d}_target'];gs=m[f'd{d}_generators']
    assert np.linalg.norm(mesh_unitaries(mesh,np.zeros((1,d*d)))[0]-target)<1e-12
    z=np.random.default_rng(773+d).normal(size=d*d)
    G=np.einsum('j,jab->ab',z,gs)
    p=equal_tail(d,.5);phi=fourier(d)@np.sqrt(p)
    S=np.stack([np.roll(phi,a) for a in range(d)],axis=1)
    p0=(np.sqrt(p).sum()**2)/d
    analytic=coefficients(G[None,:,:],p)['full']
    h=1e-4
    u=mesh_unitaries(mesh,h*np.array([z,-z]))@target.conj().T
    numeric=(p0-(abs(np.einsum('naz,za->na',u,S))**2).mean())/h**2
    rel=abs(numeric-analytic)/analytic
    assert rel<1e-5,(d,rel)
    curvature_checks.append({'d':d,'relative_error':float(rel)})

# Check the channel kernel, its sign, its Fourier inversion, and the weighted
# coherence formula against an explicit controlled unitary and partial trace.
rng=np.random.default_rng(337)
kernel_residuals=[]
for d in [3,5,8]:
    n=d+2;f=np.r_[np.arange(d),[0,1]]
    Z=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d));rho=Z@Z.conj().T;rho/=np.trace(rho)
    F=fourier(d);p=np.diag(F.conj().T@rho@F).real
    alpha=rng.normal(size=n)+1j*rng.normal(size=n);alpha/=np.linalg.norm(alpha)
    sigma=np.outer(alpha,alpha.conj())
    T=[np.roll(np.eye(d),a,axis=0) for a in range(d)]
    O=np.zeros((n*d,n*d),complex)
    for x in range(n):O[x*d:(x+1)*d,x*d:(x+1)*d]=T[f[x]]
    joint=O@np.kron(sigma,rho)@O.conj().T
    channel=np.trace(joint.reshape(n,d,n,d),axis1=1,axis2=3)
    K=np.array([np.trace(rho@T[g]) for g in range(d)])
    kernel=K[(f[:,None]-f[None,:])%d]*sigma
    res=np.linalg.norm(channel-kernel);assert res<1e-12
    recovered=np.array([sum(K[g]*np.exp(-2j*np.pi*k*g/d) for g in range(d))/d for k in range(d)])
    assert np.linalg.norm(recovered-p)<1e-12
    weighted=abs(np.outer(alpha,alpha.conj()))*abs(K[(f[:,None]-f[None,:])%d]);np.fill_diagonal(weighted,0)
    cx=abs(channel).sum()-abs(np.diag(channel)).sum()
    assert abs(cx-weighted.sum())<1e-12
    kernel_residuals.append(float(res))

result={'stored_discrimination_certificates_checked':checked,
        'max_repaired_completeness_residual':float(max_complete),
        'mesh_curvature_checks':curvature_checks,'kernel_residuals':kernel_residuals,
        'status':'passed'}
(HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
