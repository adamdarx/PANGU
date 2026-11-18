from pathlib import Path
import datetime,json,shutil,socket,subprocess
assert socket.gethostname().startswith('wm2-login')
base=Path('/lustre/home/2601110206/CODEX_WORKPLACE/Operations');r=base/'20261117-single-continuation';out=r/'RETRY_20261118.json';assert not out.exists()
v=dict(utc=datetime.datetime.utcnow().isoformat()+'Z',single={},reason='Both four-GPU restart trajectories succeeded exactly, but measured reserve7575MiB failed8192MiB policy. Preserve threshold; test eight GPUs across two nodes. Original jobs unchanged.')
def save():out.write_text(json.dumps(v,indent=2)+'\n')
def submit(root,name,mode,args,nodes=1,ranks=4,time=None,dep=None):
 cmd=['sbatch','--parsable','--partition=GPU80G','--qos='+('high' if mode in ['admission','io'] else 'low'),'--nodes='+str(nodes),'--ntasks='+str(ranks),'--ntasks-per-node='+str(ranks//nodes),'--cpus-per-task=16','--gres=gpu:'+str(ranks//nodes),'--time='+(time or ('01:00:00' if mode in ['admission','io'] else '5-00:00:00')),'--no-requeue','--kill-on-invalid-dep=yes','--job-name='+name,'--output='+str(root/'logs/%x-%j.log')]
 if dep:cmd+=['--dependency='+dep]
 cmd+=[str(root/('formal.slurm' if 'bbh' in root.name else 'case.slurm')),mode]+args
 return subprocess.check_output(cmd).decode().strip().split(';')[0]
for kind,old in [('stationary','17260150'),('spinning','17260147')]:
 p=r/('ADMISSION_'+kind+'.json');d=json.loads(p.read_text());assert d['status']=='admission_failed' and d['restart_state_scaled_max_difference']==0 and d['restart_step8']['final']['cycle']==d['checkpoint']['cycle']+8
 shutil.copy2(str(p),str(r/('ADMISSION_'+kind+'.fourGPU.json')))
 gate=submit(r,'SP192_RESTART8_'+kind,'admission',[kind],nodes=2,ranks=8)
 v['single'][kind]=dict(original=old,admission=gate);save()
 formal=submit(r,'SP192_CONT500_8_'+kind,'formal',[kind],nodes=2,ranks=8,dep='afterany:'+old+',afterok:'+gate);v['single'][kind]['formal']=formal;save()
b=base/'20261117-bbh-convergence';shutil.copy2(str(b/'FULL_IO_ADMISSION.json'),str(b/'FULL_IO_ADMISSION.compression_timeout.json'))
gate=submit(b,'BBH_QC_FULL_IO0','io',[],time='01:45:00');v['bbh']=dict(full_io_gate=gate,formal={});save()
for n in (32,48,64):
 v['bbh']['formal'][str(n)]=submit(b,'BBH_QC_IO0_N%d_T300'%n,'formal',[str(n)],dep='afterok:'+gate);save()
b=base/'20261117-einstein-toolkit-gpu/bns-reference';(b/'logs').mkdir(exist_ok=True);shutil.copy2(str(b/'N96_ADMISSION.json'),str(b/'N96_ADMISSION.twoGPU_speed_failed.json'))
gate=submit(b,'ET_BNS96_4GPU_SPEED','admission',['96']);v['ET_BNS96']=dict(admission=gate);save();v['ET_BNS96']['formal']=submit(b,'ET_BNS96_4GPU_T200','formal',['96'],dep='afterok:'+gate);save();print(json.dumps(v,indent=2))
