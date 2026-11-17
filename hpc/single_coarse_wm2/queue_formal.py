from pathlib import Path
import json,subprocess,os,signal
R=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-single-coarse-convergence')
w=json.loads((R/'FORMAL_WATCHER.json').read_text());pid=w['pid']
try:
 if b'submit_formal.py' in Path('/proc/%d/cmdline'%pid).read_bytes():os.kill(pid,signal.SIGTERM)
except FileNotFoundError:pass
v=json.loads((R/'SUBMISSIONS.json').read_text());assert not v['formal'],'Already queued; do not duplicate'
deps=[]
for key,j in v['admission'].items():
 d=json.loads((R/('ADMISSION_'+key+'.json')).read_text())
 if d['job_id']!=j or d['status']!='admission_passed':deps.append(j)
for key in v['admission']:
 kind,n=key.rsplit('_n',1);partition='GPU80G' if key in v.get('GPU80_retry',{}) else 'GPU40G'
 cmd=['sbatch','--parsable','--partition='+partition,'--qos=low','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=4','--time=5-00:00:00','--no-requeue','--kill-on-invalid-dep=yes','--job-name=SP500_COARSE_'+key,'--output='+str(R/'logs/%j.out'),'--error='+str(R/'logs/%j.err')]
 proof=json.loads((R/('ADMISSION_'+key+'.json')).read_text());case_deps=[] if proof['status']=='admission_passed' and proof['job_id']==v['admission'][key] else [v['admission'][key]]
 if case_deps:cmd+=['--dependency=afterok:'+':'.join(case_deps)]
 cmd += [str(R/'case.slurm'),'formal',kind,n]
 v['formal'][key]=subprocess.check_output(cmd).decode().strip().split(';')[0]
 p=R/'SUBMISSIONS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
v.update(status='all_six_formal500M_queued_with_success_dependencies',formal_dependencies=deps);(R/'SUBMISSIONS.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
