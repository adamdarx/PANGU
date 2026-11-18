"""Measure each MPI rank's assigned GPU, including ranks on the second node."""
from pathlib import Path
import json,os,socket,subprocess,sys,time
folder=Path(os.environ['CONT_GPU_DIR']);rank=int(os.environ['OMPI_COMM_WORLD_RANK']);gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert len(gpu.split(','))==1
record=dict(rank=rank,host=socket.gethostname(),gpu=gpu,peak_MiB=0,minimum_reserve_MiB=None,actual_GPU_observed=False)
path=folder/('rank%d.json'%rank)
def write():
 t=path.with_suffix('.tmp');t.write_text(json.dumps(record));t.replace(path)
p=subprocess.Popen(sys.argv[1:])
try:
 while p.poll() is None:
  used,total=map(float,subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().split(','));record['peak_MiB']=max(record['peak_MiB'],used);record['minimum_reserve_MiB']=min(record['minimum_reserve_MiB'] if record['minimum_reserve_MiB'] is not None else total,total-used)
  apps=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid,process_name','--format=csv,noheader']).decode();record['actual_GPU_observed']|=any(line.split(',')[0].strip()==str(p.pid) and 'pangu' in line for line in apps.splitlines());write();time.sleep(1)
finally:
 if p.poll() is None:p.terminate();p.wait(timeout=20)
 record['returncode']=p.returncode;write()
sys.exit(p.returncode)
