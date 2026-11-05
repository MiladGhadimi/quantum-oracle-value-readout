"""NumPy-only mesh model shared by simulation and saved-data validation."""
import numpy as np

def herm(a):
    return (a + a.conj().T) / 2


def mineig(a):
    return float(np.linalg.eigvalsh(herm(a))[0])


def fourier(d):
    z = np.arange(d)
    return np.exp(-2j*np.pi*np.outer(z,z)/d)/np.sqrt(d)


def equal_tail(d, phi):
    p = np.full(d,(1-phi)/(d-1)); p[0] = phi
    return p


def coherence(p):
    return float(np.sqrt(p).sum()**2-1)


def shifted(phi):
    return np.stack([np.roll(phi,a) for a in range(len(phi))],axis=1)


def beam(theta, phase):
    c,s,e = np.cos(theta),np.sin(theta),np.exp(1j*phase)
    return np.array([[e*c,-s],[e*s,c]],complex)


def beam_derivatives(theta, phase):
    c,s,e = np.cos(theta),np.sin(theta),np.exp(1j*phase)
    return (np.array([[-e*s,-c],[e*c,-s]],complex),
            np.array([[1j*e*c,0],[1j*e*s,0]],complex))


def decompose(target):
    """Adjacent-row elimination of target^dagger; U=D B_m ... B_1."""
    d = len(target); a = np.asarray(target.conj().T,dtype=complex).copy(); ops=[]
    for col in range(d-1):
        for row in range(d-1,col,-1):
            i,j=row-1,row; x,y=a[i,col],a[j,col]
            if abs(y)<1e-14: theta,phase=0.,0.
            elif abs(x)<1e-14: theta,phase=np.pi/2,0.
            else:
                theta=np.arctan2(abs(y),abs(x))
                phase=np.angle(y)-np.angle(x)+np.pi
            b=beam(theta,phase);a[[i,j],:]=b@a[[i,j],:]
            ops.append((i,j,float(theta),float(phase)))
    output=-np.angle(np.diag(a))
    assert np.linalg.norm(a-np.diag(np.diag(a)))<2e-13
    mesh={'d':d,'ops':ops,'output':output}
    built=mesh_unitaries(mesh,np.zeros((1,d*d)))[0]
    mesh['reconstruction_residual']=float(np.linalg.norm(built-target))
    assert mesh['reconstruction_residual']<3e-13
    return mesh


def mesh_unitaries(mesh, errors):
    n=errors.shape[0];d=mesh['d'];m=len(mesh['ops'])
    u=np.broadcast_to(np.eye(d,dtype=complex),(n,d,d)).copy()
    for k,(i,j,t,p) in enumerate(mesh['ops']):
        th=t+errors[:,2*k];ph=p+errors[:,2*k+1]
        c=np.cos(th)[:,None];s=np.sin(th)[:,None];e=np.exp(1j*ph)[:,None]
        x,y=u[:,i,:].copy(),u[:,j,:].copy()
        u[:,i,:]=e*c*x-s*y;u[:,j,:]=e*s*x+c*y
    return np.exp(1j*(mesh['output'][None,:]+errors[:,2*m:]))[:,:,None]*u


def generators(mesh):
    d=mesh['d'];ops=mesh['ops'];m=len(ops)
    bs=[];prefix=[np.eye(d,dtype=complex)]
    for i,j,t,p in ops:
        b=np.eye(d,dtype=complex);b[np.ix_([i,j],[i,j])]=beam(t,p)
        bs.append(b);prefix.append(b@prefix[-1])
    suffix=[None]*(m+1);suffix[m]=np.diag(np.exp(1j*mesh['output']))
    for k in range(m-1,-1,-1):suffix[k]=suffix[k+1]@bs[k]
    u=suffix[0]; gs=[]
    for k,(i,j,t,p) in enumerate(ops):
        for db in beam_derivatives(t,p):
            mat=np.zeros((d,d),complex);mat[np.ix_([i,j],[i,j])]=db
            deriv=suffix[k+1]@mat@prefix[k]
            g=-1j*deriv@u.conj().T
            assert np.linalg.norm(g-g.conj().T)<2e-13
            gs.append(herm(g))
    for k in range(d):
        g=np.zeros((d,d),complex);g[k,k]=1.;gs.append(g)
    return np.asarray(gs)


def correlation_data(gs):
    d=gs.shape[1];a=np.arange(d)
    shifts=np.stack([gs[:,a,(a+w)%d] for w in range(d)],axis=2)
    q=np.einsum('jaw,jav->wv',shifts,shifts.conj())
    gs2=gs@gs
    r=np.array([gs2[:,a,(a+w)%d].sum() for w in range(d)])
    v=np.diag(q).real
    return q,r,v


def quadratic_samples(gs, p):
    d=len(p);phi=fourier(d)@np.sqrt(p);s=shifted(phi)
    g=np.einsum('jaz,za->ja',gs,s)
    h=np.einsum('jaz,za->ja',gs@gs,s)
    return (phi[0].real*h.real.sum(axis=1)-(abs(g)**2).sum(axis=1))/d


def coefficients(gs, p):
    d=len(p);phi=fourier(d)@np.sqrt(p);c=coherence(p);q,r,v=correlation_data(gs)
    full=float(quadratic_samples(gs,p).sum())
    via_qr=float((phi[0].real*np.real(r@phi)-np.real(np.einsum('wv,w,v->',q,phi,phi.conj())))/d)
    assert abs(full-via_qr)<2e-12
    profile=float(v@(phi[0].real**2-abs(phi)**2)/d)
    scalar=float(v[1:].mean()*c/d);iso=float(v.mean()*c/d)
    if c<1e-13: full=profile=scalar=iso=0.
    return dict(C=c,full=full,profile=profile,scalar=scalar,isotropic=iso)


def statistics(a):
    return float(a.mean()),float(a.std(ddof=1)/np.sqrt(len(a)))


def simulate(mesh, target, gs, states, z, sigma):
    """Antithetic pairs; each z row is one independent sampling unit."""
    n=len(z);d=mesh['d'];sp=[shifted(fourier(d)@np.sqrt(p)) for p in states]
    p0=np.array([(np.sqrt(p).sum()**2)/d for p in states])
    loss=np.empty((n,len(states)));quad=np.empty_like(loss)
    for start in range(0,n,512):
        zz=z[start:start+512]; size=len(zz)
        g=np.einsum('nj,jab->nab',zz,gs,optimize=True)
        for l,p in enumerate(states):quad[start:start+size,l]=quadratic_samples(g,p)*sigma**2
        losses=[]
        for sign in [1,-1]:
            u=mesh_unitaries(mesh,sign*sigma*zz)
            w=u@target.conj().T
            probs=np.array([(abs(np.einsum('naz,za->na',w,s))**2).mean(axis=1) for s in sp]).T
            losses.append(p0[None,:]-probs)
        loss[start:start+size]=.5*(losses[0]+losses[1])
    out=[]
    for l,p in enumerate(states):
        avg,se=statistics(loss[:,l]);qm,qse=statistics(quad[:,l]);diff,diffse=statistics(loss[:,l]-quad[:,l])
        out.append(dict(direct=avg,direct_se=se,paired_quadratic=qm,
                        paired_quadratic_se=qse,paired_difference=diff,
                        paired_difference_se=diffse))
    return out

