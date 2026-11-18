"""Quantitative report: existing100-time-unit GPU TOV trajectories, one resolution."""
from pathlib import Path
import csv,hashlib,json,math,re,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PLOT_ONLY='--plot-only' in sys.argv
R=Path(__file__).resolve().parent;F=R/'figures';F.mkdir(exist_ok=True)
fields=json.loads((R/'FIELDS.json').read_text());method=json.loads((R/'METHOD.json').read_text())
assert len(fields)==21 and fields[-1]['nominal_time']==100
plt.rcParams.update({'font.family':['DejaVu Sans','sans-serif'],'font.size':14,'axes.linewidth':2,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
colors={'PANGU':'#0F4D92','AthenaK':'#B64342'};styles={'PANGU':'-','AthenaK':'--'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(name,rows):
 if PLOT_ONLY:return
 with (R/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def hist(path):return np.atleast_2d(np.loadtxt(path))
if PLOT_ONLY:
 data=json.loads((R/'HISTORIES.json').read_text());mass={c:(np.array(v['time']),np.array(v['baryon_mass'])) for c,v in data.items()}
else:
 pah=hist(R/'source/PANGU/sync_grmhd_tov.out2.hst');aah=hist(R/'source/AthenaK/athenak_tov_collapse_n64.mhd.hst')
 assert np.isfinite(pah).all() and np.isfinite(aah).all() and pah[-1,0]==100 and aah[-1,0]==100
 mass={'PANGU':(pah[:,0],pah[:,4]),'AthenaK':(aah[:,0],aah[:,2])}
 (R/'HISTORIES.json').write_text(json.dumps({c:dict(time=t.tolist(),baryon_mass=m.tolist()) for c,(t,m) in mass.items()},indent=2)+'\n')
t=np.asarray([r['nominal_time'] for r in fields]);pre=np.asarray([r['pre_first_reported_AH'] for r in fields])
table=[]
for row in fields:
 x=dict(nominal_time=row['nominal_time'],pre_first_reported_AH=row['pre_first_reported_AH'])
 for code in colors:
  for k,v in row[code].items():x[code+'_'+k]=v
 table.append(x)
write_csv('FIELD_COMPARISON.csv',table)
common=np.linspace(0,100,1001)
write_csv('GLOBAL_BARYON_ACCOUNTING.csv',[dict(time=float(v),PANGU_relative_change=float(np.interp(v,*mass['PANGU'])/mass['PANGU'][1][0]-1),AthenaK_relative_change=float(np.interp(v,*mass['AthenaK'])/mass['AthenaK'][1][0]-1)) for v in common])
if PLOT_ONLY:
 def csv_rows(name):
  return [{k:float(v) for k,v in x.items()} for x in csv.DictReader((R/name).open())]
 pa=csv_rows('PANGU_AH_QUALIFIED.csv');trace=csv_rows('ATHENAK_AH_PLOTTING_TRACE.csv');matched=csv_rows('AH_MATCHED_DIAGNOSTICS.csv')
else:
 pa=list(csv.DictReader(l for l in (R/'source/PANGU/nr_horizon.csv').read_text().splitlines() if not l.startswith('#')))
 pa=[dict(time=float(x['time']),Mirr=float(x['irreducible_mass']),Mch=float(x['christodoulou_mass']),scaled_RMS=float(x['scaled_rms_expansion']),tolerance=float(x['expansion_tolerance'])) for x in pa if int(x['found'])==1 and float(x['scaled_rms_expansion'])<=float(x['expansion_tolerance'])*(1+1e-10)]
 found=set(float(x) for x in re.findall(r'^# iter = \d+, Time = (\S+)',(R/'source/AthenaK/athenak_tov_collapse_n64.horizon_shape_0.txt').read_text(),re.M))
 aa=[]
 for line in (R/'source/AthenaK/athenak_tov_collapse_n64.horizon_summary_0.txt').read_text().splitlines():
  if not line.strip() or line.startswith('#'):continue
  v=list(map(float,line.split()))
  if v[1] in found and v[7]>0 and v[8]>=0 and all(math.isfinite(z) for z in [v[1],v[2],v[7],v[8]]):
   aa.append(dict(time=v[1],Mch=v[2],Mirr=math.sqrt(v[7]/(16*math.pi)),native_RMS=math.sqrt(v[8])))
 at=np.array([x['time'] for x in aa]);matched=[]
 for x in pa:
  j=int(np.argmin(abs(at-x['time'])));a=aa[j]
  if abs(a['time']-x['time'])>.01:continue
  matched.append(dict(time_PANGU=x['time'],time_AthenaK=a['time'],PANGU_Mirr=x['Mirr'],AthenaK_Mirr=a['Mirr'],Mirr_relative_difference=a['Mirr']/x['Mirr']-1,PANGU_scaled_RMS_over_tolerance=x['scaled_RMS']/x['tolerance'],AthenaK_native_RMS=a['native_RMS']))
 assert matched
 write_csv('AH_MATCHED_DIAGNOSTICS.csv',matched)
 write_csv('PANGU_AH_QUALIFIED.csv',pa)
 # Preserve full numeric comparisons in matched rows; export a compact native plotting trace.
 stride=max(1,len(aa)//2000);trace=aa[::stride]
 if trace[-1] is not aa[-1]:trace.append(aa[-1])
 write_csv('ATHENAK_AH_PLOTTING_TRACE.csv',trace)
fig,ax=plt.subplots(2,2,figsize=(12.8,8.4))
for code in colors:
 rho=np.array([r[code]['rho_max'] if r['pre_first_reported_AH'] else np.nan for r in fields])
 H=np.array([r[code]['H_shell_proper_RMS'] for r in fields])
 shell=np.array([r[code]['shell_baryon_inventory'] for r in fields])
 ax[0,0].plot(t[pre],rho[pre]/rho[0],styles[code],color=colors[code],lw=2.3,marker='o',ms=4,label=code)
 ax[0,1].semilogy(t,H,styles[code],color=colors[code],lw=2.3,marker='o',ms=3,label=code)
 ax[1,0].plot(t,shell/mass[code][1][0],styles[code],color=colors[code],lw=2.3,marker='o',ms=3,label=code)
 mt,mm=mass[code];ax[1,1].plot(mt,mm/mm[0]-1,styles[code],color=colors[code],lw=2.3,label=code)
ax[0,0].set(title='(a) Before either reported horizon',ylabel=r'$\rho_{\max}(t)/\rho_{\max}(0)$',xlim=(0,40))
ax[0,1].set(title='(b) Hamiltonian constraint outside BH',ylabel=r'Proper-volume RMS of $H$',xlim=(0,100))
ax[1,0].set(title='(c) Baryon inventory in fixed shell',ylabel=r'$M_b(3\leq r\leq30)/M_b(0)$',xlim=(0,100))
ax[1,1].set(title='(d) Global baryon accounting',ylabel=r'$[M_b(t)-M_b(0)]/M_b(0)$',xlim=(0,100))
ax[1,1].axvspan(method['AthenaK_first_reported_AH_time'],100,color='#CFCECE',alpha=.25,zorder=0)
ax[1,1].text(.04,.93,'Endpoint mass difference:\n>99.99999% within r < 1.8\n(inside reported AHs)',transform=ax[1,1].transAxes,va='top',fontsize=11)
for a in ax.flat:a.set_xlabel('t (code units)');a.legend(fontsize=11);a.tick_params(direction='out')
ax[1,1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig.suptitle('Magnetized TOV collapse: PANGU and AthenaK, N64',fontsize=16)
fig.text(.5,.005,'Same stellar model/cells; native FD4 / FD6 and different damping/floor settings. Shell inventory has boundary fluxes.',ha='center',fontsize=10)
fig.tight_layout(pad=2,rect=(0,.025,1,.96))
for fmt in ['png','pdf','svg']:fig.savefig(F/('tov_n64_long_field_comparison.'+fmt),dpi=300)
plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(16,4.9))
for code,rows in [('PANGU',pa),('AthenaK',trace)]:
 ax[0].plot([x['time'] for x in rows],[x['Mirr'] for x in rows],styles[code],color=colors[code],lw=2.1,label=code+(' (qualified)' if code=='PANGU' else ' (native found)'))
ax[0].set(title='(a) Raw horizon mass diagnostics',xlabel='t (code units)',ylabel=r'$M_{\rm irr}$',xlim=(40,100));ax[0].legend(fontsize=10)
ax[1].plot([x['time'] for x in pa],[x['scaled_RMS']/x['tolerance'] for x in pa],color=colors['PANGU'],lw=2,marker='o',ms=3);ax[1].axhline(1,color='#767676',ls=':',lw=1.5)
ax[1].set(title='(b) PANGU finder quality',xlabel='t (code units)',ylabel='Scaled RMS / native tolerance',ylim=(0,1.08),xlim=(40,100))
ax[2].plot([x['time'] for x in trace],[x['native_RMS'] for x in trace],color=colors['AthenaK'],ls='--',lw=2)
ax[2].set(title='(c) AthenaK finder diagnostic',xlabel='t (code units)',ylabel='Native unscaled RMS expansion',xlim=(40,100))
fig.suptitle('AH diagnostics: finder residuals use separate native normalizations',fontsize=16)
fig.text(.5,.015,'A native found flag is not a common residual certificate. Raw mass differences cannot be assigned to evolution alone.',ha='center',fontsize=11)
fig.tight_layout(pad=2,rect=(0,.045,1,.95))
for fmt in ['png','pdf','svg']:fig.savefig(F/('tov_n64_long_horizon_diagnostics.'+fmt),dpi=300)
plt.close(fig)
for vector in F.glob('*.svg'):
 vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')
if PLOT_ONLY:
 print('Rebuilt figures from published reductions; no evolution or raw field extraction.');sys.exit(0)
rho_rel=[r['AthenaK']['rho_max']/r['PANGU']['rho_max']-1 for r in fields if r['pre_first_reported_AH']]
initial_mass=mass['PANGU'][1][0]
endpoint=json.loads((R/'ENDPOINT_BARYON_REGIONS.json').read_text())
summary=dict(endpoint_baryon_localization=endpoint,status='N64_100_time_unit_quantitative_comparison_completed',physical_acceptance='Not a convergence or common-finder certificate',
 pre_AH_density_samples=int(pre.sum()),pre_AH_density_end_time=method['pre_AH_density_end_time'],maximum_sampled_relative_rho_max_difference=max(abs(x) for x in rho_rel),
 maximum_sampled_absolute_shell_inventory_difference_over_initial_baryon_mass=max(abs(r['AthenaK']['shell_baryon_inventory']-r['PANGU']['shell_baryon_inventory']) for r in fields)/initial_mass,
 first_AH_sampling= {k:v for k,v in method.items() if 'AH_time' in k or 'last_failed' in k or 'sampling_bracket' in k},
 global_baryon_relative_changes={code:dict(final=float(mm[-1]/mm[0]-1),maximum_absolute=float(np.max(abs(mm/mm[0]-1)))) for code,(mt,mm) in mass.items()},
 final_H_shell_proper_RMS={code:fields[-1][code]['H_shell_proper_RMS'] for code in colors},
 reported_AH_matched_samples=len(matched),last_raw_AH_diagnostic_comparison=matched[-1],
 raw_AH_mass_difference_is_not_solver_error=True,method=method,
 source_text_sha256={str(p.relative_to(R)):sha(p) for p in (R/'source').rglob('*') if p.is_file()},
 analysis_input_sha256={n:sha(R/n) for n in ['FIELDS.json','METHOD.json','FIELD_ARCHIVE_HASHES.json']},
 numeric_and_figure_sha256={str(p.relative_to(R)):sha(p) for p in list(R.glob('*.csv'))+list(F.iterdir()) if p.is_file()})
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['source_text_sha256','numeric_and_figure_sha256','method']},indent=2))
