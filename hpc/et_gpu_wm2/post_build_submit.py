"""LOGIN-only bounded watcher; submit one GPU regression allocation after freezing."""
from pathlib import Path
import datetime
import json
import socket
import subprocess
import time

root = Path(__file__).resolve().parent
assert socket.gethostname().startswith('wm2-login')
receipt = root / 'GPU_SMOKE_SUBMISSION.json'
assert not receipt.exists(), 'A smoke allocation has already been submitted'
deadline = time.monotonic() + 86400
while time.monotonic() < deadline:
    status = json.loads((root / 'BUILD_STATUS.json').read_text())
    if status['status'] == 'login_cuda_build_failed':
        raise RuntimeError('Build failed; no GPU job submitted')
    if status['status'] == 'gpu_binary_compiled_frozen_compute_smoke_pending':
        result = subprocess.run(['sbatch', '--parsable', str(root / 'gpu-smoke.slurm')],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        assert result.returncode == 0, result.stderr
        job_id = result.stdout.strip().split(';')[0]
        assert job_id.isdigit(), result.stdout
        receipt.write_text(json.dumps(dict(job_id=job_id,
            submitted_utc=datetime.datetime.utcnow().isoformat()+'Z',
            binary=status['binary'], sha256=status['sha256'],
            status='submitted_not_yet_validated'), indent=2)+'\n')
        print(receipt.read_text());break
    time.sleep(30)
else:
    raise TimeoutError('Build watcher reached its one-day limit; no GPU job submitted')
