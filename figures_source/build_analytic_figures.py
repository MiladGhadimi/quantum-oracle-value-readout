"""Regenerate both analytic figures; no existing PDF or simulation is required."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent.parent

plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,
    'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'savefig.bbox':'tight'})
phi=np.linspace(0,1,1401)
def limits(d):
    lo=(np.sqrt(phi)+np.sqrt(1-phi))**2/d
    hi=(np.sqrt(phi)+np.sqrt((d-1)*(1-phi)))**2/d
    return lo,hi

d=16
lo,hi=limits(d)
ys=np.linspace(0,1,1000)
P,Y=np.meshgrid(phi,ys)
data=np.ma.masked_where((Y<lo[None,:]) | (Y>hi[None,:]),d*Y)
fig,ax=plt.subplots(figsize=(4.4,3.4),layout='constrained')
mesh=ax.pcolormesh(P,Y,data,cmap='viridis',vmin=1,vmax=d,rasterized=True,shading='auto')
ax.plot(phi,hi,color='#172b45',lw=1.7,label=r'$P_+(\Phi)$')
ax.plot(phi,lo,color='#172b45',lw=1.2,ls='--',label=r'$P_-(\Phi)$')
ax.axhline(1/d,color='gray',ls=':',lw=1)
ax.scatter([1/d,1],[1,1/d],s=25,c='#b62b32',zorder=5,clip_on=False)
# Label each effective-support level only once, avoiding the original duplicate.
for eff in [2,4,8,12]:
    valid=np.where((eff/d>=lo)&(eff/d<=hi))[0]
    if len(valid):
        a,b=phi[valid[0]],phi[valid[-1]]
        ax.plot([a,b],[eff/d,eff/d],color='white',lw=.6,alpha=.65)
        ax.text(a+.03*(b-a),eff/d+.012,rf'$S_{{\rm eff}}={eff}$',
                fontsize=7,color='white',ha='left',va='bottom')
ax.set(xlim=(0,1),ylim=(0,1.035),xlabel=r'Target-character fidelity $\Phi$',
       ylabel=r'Optimal readout probability $P_{\rm val}$')
fig.colorbar(mesh,ax=ax,pad=.025,label=r'$S_{\rm eff}=dP_{\rm val}$',ticks=[1,4,8,12,16])
ax.legend(loc='upper right',frameon=False,fontsize=8)
fig.savefig(HERE/'oracle_capability_map.pdf')
plt.close(fig)

fig,ax=plt.subplots(figsize=(6.2,3.5),layout='constrained')
for d,color in zip([2,4,16,64],['#314c80','#1b8a7a','#c67d26','#a34b78']):
    _,hi=limits(d)
    ax.plot(phi,hi,lw=1.7,color=color,label=rf'$d={d}$')
    ax.scatter([1/d],[1],s=22,color=color,clip_on=False)
ax.set(xlim=(0,1),ylim=(0,1.04),xlabel=r'Target-character fidelity $\Phi$',
       ylabel=r'Upper readout boundary $P_+(\Phi)$')
ax.legend(frameon=False,loc='lower left',ncol=2)
ax.grid(alpha=.15)
fig.savefig(HERE/'tradeoff.pdf')
plt.close(fig)
print('Regenerated 2 analytic plots.')
