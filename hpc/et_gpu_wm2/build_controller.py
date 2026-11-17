"""LOGIN-only compile and freeze; never run the solver on LOGIN."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import socket
import subprocess

ROOT=Path(__file__).resolve().parent
assert socket.gethostname().startswith('wm2-login')
record=dict(status='login_cuda_build_running',started_utc=datetime.datetime.utcnow().isoformat()+'Z')
expected={Path(item['path']).name:item['commit'] for item in json.loads((ROOT/'FETCH_RESULT.json').read_text())['repositories']}
for item in json.loads((ROOT/'COMPATIBLE_SOURCE_SELECTION.json').read_text())['repos']:
    expected[item['name']]=item['head']
source_commits={name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT/'Cactus/repos'/name)).decode().strip()
                for name in ['CarpetX','SpacetimeX','AsterX']}
assert all(source_commits[name]==expected[name] for name in source_commits),source_commits
record['source_commits']=source_commits
record['source_overrides_manifest']=str(ROOT/'COMPATIBLE_SOURCE_SELECTION.json')
receipt=ROOT/'BUILD_STATUS.json'
def save():
    temporary=receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(record,indent=2)+"\n")
    temporary.replace(receipt)
save()
with (ROOT/'logs/build.log').open('w') as log:
    result=subprocess.run(['bash','-lc','bash '+str(ROOT/'build.sh')],stdout=log,stderr=subprocess.STDOUT)
record.update(returncode=result.returncode,finished_utc=datetime.datetime.utcnow().isoformat()+'Z')
if result.returncode:
    record['status']='login_cuda_build_failed';save();raise SystemExit(result.returncode)
assert all(subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT/'Cactus/repos'/name)).decode().strip()==revision
           for name,revision in source_commits.items()),'Source revision changed while building'
exe=ROOT/'Cactus/exe/cactus_wm2cuda'
assert exe.exists()
with exe.open('rb') as stream:digest=hashlib.sha256(stream.read()).hexdigest()
target=ROOT/'bin'/('cactus-wm2cuda-'+digest[:12]);target.parent.mkdir(exist_ok=True)
shutil.copyfile(exe,target);target.chmod(0o555)
with (ROOT/'logs/cuda-images.txt').open('w') as log:
    check=subprocess.run(['/lustre/software/cuda/12.6.0/bin/cuobjdump','--list-elf',str(target)],stdout=log,stderr=subprocess.STDOUT)
images=(ROOT/'logs/cuda-images.txt').read_text()
assert check.returncode==0 and 'ELF file' in images,'CUDA device images were not verified'
record.update(status='gpu_binary_compiled_frozen_compute_smoke_pending',binary=str(target),sha256=digest,
              CUDA_device_images_verified=True,source_manifest=str(ROOT/'FETCH_RESULT.json'),
              selection=str(ROOT/'BUILD_SELECTION.json'),runtime_validation='Pending Slurm compute allocation; no solver run on LOGIN/DATA.')
save();print(json.dumps(record,indent=2))
