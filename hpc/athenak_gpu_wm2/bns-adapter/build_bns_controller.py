"""Compile and freeze the pinned native BNS GPU reference on LOGIN only."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parent
assert socket.gethostname().startswith('wm2-login')
assert (ROOT / 'STAGED_SOURCE.json').exists()
record = dict(status='BNS_sm80_CUDA_build_running',
              started_utc=datetime.datetime.utcnow().isoformat()+'Z',
              formal_reference_resolutions=[64,96],
              validation='Initial-data adapter only; matched GPU initial-state validation pending; no formal submission')
receipt = ROOT / 'BNS_BUILD_STATUS.json'
def save():
    temporary = receipt.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(receipt)
save()
with (ROOT/'logs/build-bns.log').open('w') as log:
    result = subprocess.run(['bash',str(ROOT/'build-bns.sh')],stdout=log,stderr=subprocess.STDOUT)
record.update(returncode=result.returncode,finished_utc=datetime.datetime.utcnow().isoformat()+'Z')
if result.returncode:
    record['status']='BNS_sm80_CUDA_build_failed';save();raise SystemExit(result.returncode)
source=ROOT/'build-tov/src/athena'
with source.open('rb') as stream:digest=hashlib.sha256(stream.read()).hexdigest()
target=ROOT/'bin'/('athenak-bns-sm80-'+digest[:12]);target.parent.mkdir(exist_ok=True)
shutil.copyfile(source,target);target.chmod(0o555)
with (ROOT/'logs/bns-cuda-images.txt').open('w') as log:
    check=subprocess.run(['/lustre/software/cuda/12.6.0/bin/cuobjdump','--list-elf',str(target)],stdout=log,stderr=subprocess.STDOUT)
assert check.returncode==0 and 'sm_80' in (ROOT/'logs/bns-cuda-images.txt').read_text()
record.update(status='BNS_sm80_GPU_binary_frozen_input_validation_pending',binary=str(target),sha256=digest,
              CUDA_device_images_verified=True,source_snapshot=json.loads((ROOT/'STAGED_SOURCE.json').read_text()))
save();print(json.dumps(record,indent=2))
