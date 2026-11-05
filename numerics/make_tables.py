"""Render manuscript tables and numerical summary macros from results.json."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'results.json').read_text())

def sci(x,digits=2):
    if abs(x)<1e-30:return '0'
    a,b=f'{x:.{digits}e}'.split('e')
    return a+r'\times10^{'+str(int(b))+'}'

def table(filename,columns,header,rows,caption,label):
    text='\\begin{table}[!htbp]\n\\centering\n\\begin{ruledtabular}\n'
    text+='\\begin{tabular}{'+columns+'}\n'+header+'\\\\\n\\colrule\n'
    text+='\n'.join(' & '.join(row)+r'\\' for row in rows)
    text+='\n\\end{tabular}\n\\end{ruledtabular}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{table}\n'
    (HERE/filename).write_text(text)

rows=[]
for x in r['mixed']:
    name=x['name'];label=(r'parent, $\gamma='+name.split('_')[-1]+'$') if name.startswith('parent') else 'generic mixed '+name[-1]
    cr=x['robustness']['C_R'];p=x['discrimination']['primal'];bound=x['l1_bound']
    rows.append([label,f'${cr:.6f}$',f'${p:.6f}$',f'${(1+cr)/4:.6f}$',f'${bound:.6f}$',f'${bound-p:.6f}$'])
table('table_mixed.tex','lccccc',r'state & $C_R$ & $P_{\rm val}$ & $[1+C_R]/4$ & $[1+C]/4$ & $\ell_1$ gap',rows,
      r'Mixed-state readout at $d=4$. The first three rows use the common parent in Eq.~\eqref{eq:table-common-parent}; the remaining rows use the three stored Ginibre states. The discrimination and robustness columns come from independent SDPs. The last column is $[1+C]/4-P_{\rm val}$. Display rounding is six decimals; feasibility diagnostics and matrices are retained separately.', 'tab:sdp')

profiles=r['mesh']['profiles'];rows=[]
for w in range(8):rows.append([f'${w}$']+[f'${profiles[k][w]:.3f}$' for k in ['identity','all','phase','splitting']])
table('table_profiles.tex','ccccc',r'$w$ & identity & all parameters & phase settings only & mixing angles only',rows,
      r'Normalized profiles $V(w)/\overline V$ for the $d=8$ triangular mesh. The identity column uses zero gate and output phases; all other columns use the same analyzer as Tables~\ref{tab:reck} and \ref{tab:d8scan} and Fig.~\ref{fig:quadraticscaling}. Profiles are evaluated exactly from tangent generators, without Monte Carlo estimation.','tab:profiles')

rows=[]
for x in r['mesh']['table']:
    state={'uniform':r'$p$ uniform','point mass':'point mass','Phi=0.95':r'$\Phi=0.95$','Phi=0.50':r'$\Phi=0.50$','skewed':'skewed'}[x['state']]
    direct='$0$' if x['C']==0 else rf"${x['direct']*1e6:.3f}\pm{x['direct_se']*1e6:.3f}$"
    rows.append([f"${x['d']}$",state,direct]+[f'${x[k]:.3f}$' for k in ['full','profile','scalar','isotropic']])
table('table_mesh.tex','llccccc',r'$d$ & state & direct mean $\pm$ SE & full & profile & off-diagonal scalar & all-shift iso.',rows,
      r'Coherent-error losses in units of $10^{-6}$ at $\sigma=10^{-3}$. Direct values use $3000$ independent antithetic Gaussian pairs; SE is the standard error over pair averages. All state rows at a given dimension share the same Gaussian vectors and their sign reversals. Full and reduced columns are ensemble quadratic predictions, not estimates on the same finite sample. The profile and off-diagonal scalar agree on upper-boundary states; both skewed rows are evaluated using the explicit weights in the text. The point-mass row is zero analytically, with direct numerical residual below $10^{-15}$.','tab:reck')

rows=[]
for x in r['mesh']['boundary_scan']:
    dev='---' if x['full']==0 else rf"${100*abs(x['profile']/x['full']-1):.3f}\%$"
    rows.append([f"${x['Phi']:.3f}$",f"${x['C']:.6f}$",f"${x['full']:.6f}$",f"${x['profile']:.6f}$",dev])
table('table_boundary.tex','ccccc',r'$\Phi$ & $C$ & full coefficient & profile/scalar coefficient & relative deviation',rows,
      r'Boundary scan for the same $d=8$ mesh as Table~\ref{tab:reck}. Entries are coefficients of $\sigma^2$. The $\Phi=0.50$ row is also the quadratic prediction in Fig.~\ref{fig:quadraticscaling}. The undefined relative deviation at zero loss is left blank; no numerical quantity is missing.','tab:d8scan')

groups=[('Table I',[x['discrimination'] for x in r['mixed']]),('Loss study (180 instances)',[x['diagnostics'] for x in r['loss']])]
rows=[]
for name,items in groups:
    vals=[max(x['completeness'] for x in items),max(max(0,-x['primal_min_eig']) for x in items),max(max(0,-x['dual_slack_min_eig']) for x in items),max(x['upper']-x['lower'] for x in items)]
    rows.append([name]+['$'+sci(x)+'$' for x in vals])
table('table_diagnostics.tex','lcccc',r'instances & $\max r_{\rm comp}$ & $\max e_{\rm pr}$ & $\max e_{\rm du}$ & $\max(P_U-P_L)$',rows,
      r'Numerical discrimination diagnostics in the original complex matrices. The first three columns precede feasibility repair; the last column uses repaired primal and dual matrices. All values are computed from stored arrays. These are floating-point certificates, not interval-arithmetic proofs.','tab:diagnostics')

nonzero=[x for x in r['mesh']['table'] if x['full']>0]
loss=[x for x in r['loss'] if x['profile']!=0 and x['phase']!=0]
macros={
 'MeshQuadraticC':f"{r['mesh']['scale_c']:.6f}",
 'MeshMaxProfileDeviation':f"{max(abs(x['profile']/x['full']-1) for x in nonzero)*100:.2f}",
 'MeshIsoMinBias':f"{min(x['isotropic']/x['full']-1 for x in nonzero)*100:.2f}",
 'MeshIsoMaxBias':f"{max(x['isotropic']/x['full']-1 for x in nonzero)*100:.2f}",
 'MeshMaxPairedRelative':sci(max(abs(x['paired_difference']/x['paired_quadratic']) for x in nonzero)),
 'MeshMaxSE':f"{max(abs(x['direct']-x['full']*1e-6)/x['direct_se'] for x in nonzero):.2f}",
 'LossGapMinPercent':f"{min(x['relative_gap'] for x in loss)*100:.4f}",
 'LossGapMaxPercent':f"{max(x['relative_gap'] for x in loss)*100:.4f}",
 'LossCertifiedMinPercent':f"{min(x['certified_gap_lower'] for x in loss)*100:.4f}",
 'MixedIdentityResidual':sci(max(abs(x['discrimination']['primal']-(1+x['robustness']['C_R'])/4) for x in r['mixed'])),
 'RobustnessMaxGap':sci(max(abs(x['robustness']['gap']) for x in r['mixed'])),
 'SCSMaxDifference':sci(max(abs(x['scs_crosscheck']['value']-x['discrimination']['primal']) for x in r['mixed'] if x['scs_crosscheck'])),
 'MeshMaxReconstruction':sci(max(r['mesh'][str(d)]['reconstruction_residual'] for d in [8,16])),
}
(HERE/'numeric_values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
print(json.dumps(macros,indent=2))
