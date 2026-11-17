"""Slurm-only official WaveToyX and coupled AsterX/Z4c regression tests."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import subprocess
import threading
import time

root = Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID'), 'Evolution is allowed only in a compute allocation'
assert int(os.environ['SLURM_NTASKS']) == 1
build = json.loads((root / 'BUILD_STATUS.json').read_text())
assert build['status'] == 'gpu_binary_compiled_frozen_compute_smoke_pending'
binary = Path(build['binary'])
with binary.open('rb') as stream:
    digest = hashlib.sha256(stream.read()).hexdigest()
assert digest == build['sha256'] and build['CUDA_device_images_verified']
base = root / 'runs' / os.environ['SLURM_JOB_ID']
base.mkdir(parents=True, exist_ok=False)
record = dict(status='gpu_official_tests_running', job_id=os.environ['SLURM_JOB_ID'],
              binary=str(binary), sha256=digest,
              started_utc=datetime.datetime.utcnow().isoformat() + 'Z',
              tests=['WaveToyX/standing', 'AsterX/magTOV_Z4c_AMR_order4'],
              scientific_acceptance='Technical GPU regression only; quantitative research comparison remains pending')
receipt = base / 'GPU_SMOKE_STATUS.json'
def save():
    receipt.write_text(json.dumps(record, indent=2) + '\n')
save()
subprocess.check_call(['nvidia-smi', '-L'])
stop = threading.Event()
def monitor():
    with (base / 'gpu-processes.log').open('w') as log:
        while not stop.is_set():
            log.write(datetime.datetime.utcnow().isoformat() + 'Z\n')
            subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                            '--format=csv,noheader'], stdout=log, stderr=subprocess.STDOUT)
            log.flush()
            stop.wait(.5)
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
env = dict(os.environ)
env.update(TESTS_DIR=str(base / 'TEST'), CCTK_TESTSUITE_RUN_PROCESSORS='1',
           CCTK_TESTSUITE_PARALLEL_TESTS='1',
           CCTK_TESTSUITE_RUN_TESTS=' '.join(record['tests']),
           CCTK_TESTSUITE_RUN_COMMAND='mpirun -np 1 --bind-to none ' + str(binary) + ' $parfile')
try:
    with (base / 'official-tests.log').open('w') as log:
        result = subprocess.run(['perl', 'lib/sbin/RunTest.pl', 'no', str(root / 'Cactus'), 'wm2cuda'],
                                cwd=str(root / 'Cactus'), env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=1000)
    summaries = list((base / 'TEST').rglob('summary.log'))
    assert len(summaries) == 1, summaries
    summary = summaries[0].read_text()
    (base / 'OFFICIAL_SUMMARY.txt').write_text(summary)
    failed = re.search(r'Number failed\s*->\s*(\d+)', summary)
    assert result.returncode == 0 and failed and int(failed.group(1)) == 0, summary
    for label, expected in [('Runnable tests', 2), ('Unrunnable tests', 0),
                            ('Number of tests passed', 2)]:
        count = re.search(re.escape(label) + r'\s*->\s*(\d+)', summary)
        assert count and int(count.group(1)) == expected, summary
    for name in ['standing', 'magTOV_Z4c_AMR_order4']:
        assert name in summary, 'Expected test missing from summary: ' + name
    stop.set(); thread.join(timeout=5)
    assert binary.name in (base / 'gpu-processes.log').read_text(), 'Actual GPU process was not observed'
    record.update(status='official_gpu_regressions_passed', summary=str(summaries[0]),
                  actual_gpu_process_observed=True)
except Exception as error:
    record.update(status='official_gpu_regressions_failed', error=repr(error))
    raise
finally:
    stop.set(); thread.join(timeout=5)
    record['finished_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'
    save()
