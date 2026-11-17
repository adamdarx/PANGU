"""Use preserved native AthenaK AH series, with original residuals and caveats."""
from pathlib import Path
import bisect,csv,hashlib,json,math,re
ROOT=Path('/lustre/home/2601110206/CODEX_WORKPLACE');OLD=ROOT/'SinglePuncture/20261030-spin-accuracy-rerun';R=ROOT/'Operations/20261117-single-continuation';OUT=R/'existing-athenak-comparison';OUT.mkdir(exist_ok=True)
jobs={'stationary':[(96,'16457629','17313453'),(128,'16457630','17260149'),(192,'16457631','17260150')],'spinning':[(96,'16696486','17313451'),(128,'16696487','17313452'),(192,'16696488','17260147')]};report=[]
for kind,cases in jobs.items():
 for n,aa_job,pa_job in cases:
  case='d200_%s_n%d'%(kind,n);a=OLD/'runs'/case/aa_job/'attempt-0/athenak';p=OLD/'runs'/case/pa_job/'attempt-0/pangu';summary=next((a/'output').glob('*.horizon_summary_0.txt'));shape=next((a/'output').glob('*.horizon_shape_0.txt'))
  found=set(float(x) for x in re.findall(r'^# iter = \d+, Time = (\S+)',shape.read_text(),re.M));aa=[]
  for line in summary.read_text().splitlines():
   if not line.strip() or line.startswith('#'):continue
   v=list(map(float,line.split()));t,m,j,area,h2=v[1],v[2],v[6],v[7],v[8]
   if t in found and area>0 and h2>=0 and all(math.isfinite(x) for x in (t,m,j,area,h2)):
    aa.append(dict(time=t,Mch=m,J=j,Mirr=math.sqrt(area/(16*math.pi)),native_H_rms=math.sqrt(h2)))
  assert aa;aa.sort(key=lambda x:x['time']);times=[x['time'] for x in aa]
  pa_file=p/'output/diagnostics/nr_horizon.csv';assert pa_file.exists(),str(pa_file)
  pa_raw=pa_file.read_bytes();pa=list(csv.DictReader(line for line in pa_raw.decode().splitlines() if not line.startswith('#')))
  raw_dir=OUT/'source-records'/case;raw_dir.mkdir(parents=True,exist_ok=True)
  (raw_dir/'PANGU_AH_snapshot.csv').write_bytes(pa_raw)
  (raw_dir/'AthenaK_horizon_summary.txt').write_bytes(summary.read_bytes())
  (raw_dir/'AthenaK_found_times.json').write_text(json.dumps(sorted(found))+'\n')
  for src,name in [(a/'input.in','AthenaK_input.in'),(p/'input.in','PANGU_input.in'),(a/'phase-validated.json','AthenaK_phase-validated.json')]:
   (raw_dir/name).write_bytes(src.read_bytes())
  rows=[]
  for row in pa:
   t=float(row['time'])
   if not (0<=t<=min(25,times[-1])):continue
   if int(row['found'])!=1 or float(row['scaled_rms_expansion'])>float(row['expansion_tolerance'])*(1+1e-10):continue
   i=bisect.bisect_left(times,t);choices=[j for j in (i-1,i) if 0<=j<len(aa)];other=min((aa[j] for j in choices),key=lambda v:abs(v['time']-t))
   if abs(other['time']-t)>.01:continue
   mirr,mch,spin=map(float,(row['irreducible_mass'],row['christodoulou_mass'],row['spin']))
   if not all(math.isfinite(x) for x in (mirr,mch,spin)) or min(mirr,mch)<=0:continue
   rows.append(dict(time_PANGU=t,time_AthenaK=other['time'],PANGU_Mirr=mirr,AthenaK_Mirr=other['Mirr'],Mirr_relative_difference=other['Mirr']/mirr-1,PANGU_Mch=mch,AthenaK_Mch=other['Mch'],Mch_relative_difference=other['Mch']/mch-1,PANGU_J=spin,AthenaK_J=other['J'],J_absolute_difference=other['J']-spin,PANGU_scaled_H_rms=float(row['scaled_rms_expansion']),PANGU_H_tolerance=float(row['expansion_tolerance']),AthenaK_native_H_rms=other['native_H_rms']))
  assert rows
  name=case+'.csv'
  with (OUT/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  archive_receipts=[]
  for receipt in sorted((a/'output/rst').rglob('*.00001.rst.archive.json')):archive_receipts.append(dict(source=str(receipt),receipt=json.loads(receipt.read_text())))
  receipt=json.loads((a/'phase-validated.json').read_text())
  report.append(dict(case=case,AthenaK_job=aa_job,PANGU_job=pa_job,matched_samples=len(rows),time_range=[rows[0]['time_PANGU'],rows[-1]['time_PANGU']],last=rows[-1],max_abs_Mirr_relative_difference=max(abs(x['Mirr_relative_difference']) for x in rows),max_abs_J_difference=max(abs(x['J_absolute_difference']) for x in rows),AthenaK_raw_series=str(summary),AthenaK_raw_series_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),AthenaK_binary_sha256=receipt['binary_sha256'],PANGU_AH_source=str(pa_file),PANGU_AH_source_snapshot_sha256=hashlib.sha256(pa_raw).hexdigest(),checkpoint_archive_receipts=archive_receipts,matched_csv=name,matched_csv_sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest()))
result=dict(status='six_existing_AthenaK_series_used_in_time_matched_diagnostic_comparison',not_a_long_run_or_convergence_acceptance=True,caveats=['Native AH residuals have different normalizations. Preserve both columns; do not interpret raw finder mass discrepancies as solver error.','PANGU rows satisfy their native scaled expansion tolerance; AthenaK rows require finite values and a recorded found shape, but are not certified against the PANGU residual tolerance.','Diagnostic timestamps matched to<=0.01M, no time interpolation. These are existing0-25M windows, not500M results.','Native fixed-AMR N96/N128/N192 may share finest spacing. These comparisons are not the N32/N48/N64 convergence series.','Native rank-separated checkpoint restoration/repartition requires a separate GPU gate. Archive receipts alone do not prove currently accessible bytes.'],cases=report)
(OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=result['status'],cases=[dict(case=x['case'],samples=x['matched_samples'],end=x['time_range'][-1]) for x in report]),indent=2))
