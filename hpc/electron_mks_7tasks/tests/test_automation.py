#!/usr/bin/env python3
"""Offline tests of the seven-case orchestration; standard library only."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

BUNDLE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('checkpoint', BUNDLE / 'scripts/checkpoint.py')
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)
SIGNATURE = b'\x89HDF\r\n\x1a\n'


def restart(path, t, body=b'good'):
    path.write_bytes(SIGNATURE + body)
    Path(str(path) + '.xdmf').write_text(
        f'<Xdmf><Domain><Grid><Information Name="Cycle" Value="{int(t*10)}"/>'
        f'<Time Value="{t}"/></Grid></Domain></Xdmf>')


class Automation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='pangu-seven-')
        root = self.root = Path(self.tmp.name)
        self.bundle = root / 'bundle with spaces'
        shutil.copytree(BUNDLE, self.bundle, ignore=shutil.ignore_patterns('run', '__pycache__'))
        config = self.bundle / 'server.conf'
        config.write_text(config.read_text() + '\nPOLL_SECONDS=0.05\n')
        self.run = self.bundle / 'run/mad_a0'
        (self.run / 'logs').mkdir(parents=True)
        self.bin = root / 'bin'
        self.bin.mkdir()
        self.repo = root / 'repo'
        exe = self.repo / 'build-electron-mks/src/pangu'
        exe.parent.mkdir(parents=True)
        exe.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
if '--pangu-version' in sys.argv:
    print('Metric mks\\nGeometry mode static'); sys.exit(0)
if '--check-input' in sys.argv:
    sys.exit(0)
if '-m' in sys.argv:
    p = pathlib.Path(sys.argv[sys.argv.index('-r')+1])
    sys.exit(1 if b'bad_payload' in p.read_bytes() else 0)
if os.environ.get('MOCK_FAIL'):
    print('Driver failed.'); sys.exit(2)
if '-r' in sys.argv:
    assert pathlib.Path(sys.argv[sys.argv.index('-r')+1]).exists()
assert '-t' not in sys.argv
t = float(os.environ.get('MOCK_TIME', '100'))
p = pathlib.Path('gr_torus_sane.restart.final.rhdf')
p.write_bytes(b'\\x89HDF\\r\\n\\x1a\\n' + b'good')
pathlib.Path(str(p)+'.xdmf').write_text(f'<Xdmf><Information Name="Cycle" Value="{int(t*10)}"/><Time Value="{t}"/></Xdmf>')
print('Driver completed.' if t >= 30000 else 'Driver timed out.  Restart to continue.')
print(f'time={t} cycle=10')
sys.exit(0 if t >= 30000 else 1)
''')
        exe.chmod(0o755)
        self.script('module', '#!/bin/bash\nexit 0\n')
        self.script('nohup', '#!/bin/bash\nexit 0\n')
        self.script('mpirun', '''#!/bin/bash
[[ "$1" == -np && "$2" == 2 ]] || exit 2
shift 2
while [[ "$1" == --mca ]]; do shift 3; done
exec "$@"
''')
        self.script('squeue', '#!/bin/bash\nprintf "%s" "${MOCK_QUEUE:-}"\n')
        self.script('sacct', '#!/bin/bash\nprintf "%s\\n" "${MOCK_STATE:-TIMEOUT}"\n')
        self.script('sbatch', '''#!/usr/bin/env python3
import json, os, sys, pathlib
with open(os.environ['SUBMISSIONS'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
if os.environ.get('STOP_AFTER_SUBMIT'):
    pathlib.Path(os.environ['TEST_RUN'], 'STOP').touch()
print('200')
''')
        self.env = dict(os.environ, REPO=str(self.repo),
                        PATH=f'{self.bin}:{os.environ["PATH"]}',
                        SLURM_SUBMIT_DIR=str(self.bundle), SLURM_JOB_ID='100',
                        SUBMISSIONS=str(root / 'submissions'), TEST_RUN=str(self.run))

    def tearDown(self):
        self.tmp.cleanup()

    def script(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)

    def run_script(self, name, *args, **env):
        return subprocess.run(['bash', str(self.bundle / 'scripts' / name), 'mad_a0', *args],
                              env=self.env | env, text=True, capture_output=True)

    def submissions(self):
        path = Path(self.env['SUBMISSIONS'])
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def test_all_seven_inputs(self):
        files = list((self.bundle / 'cases').glob('*.in'))
        self.assertEqual(len(files), 7)
        for path in files:
            text = path.read_text()
            self.assertIn('tlim = 30000.0', text)
            self.assertEqual(text.count('dt = 100.0'), 3)
            self.assertIn('nx1 = 384', text)
            for key in ('constant', 'howes', 'kawazura', 'werner', 'rowan', 'sharma'):
                self.assertIn(f'{key} = true', text)
            if path.name.startswith('mad'):
                for entry in ('magnetic_topology = mad', 'potential_r_pow = 3.0',
                              'potential_falloff = 400.0', 'potential_cutoff = 0.2'):
                    self.assertIn(entry, text)

    def test_submission_resources_and_case(self):
        p = self.run_script('submit.sh')
        self.assertEqual(p.returncode, 0, p.stderr)
        args = self.submissions()[0]
        for arg in ('--partition=GPU80G', '--qos=low', '--ntasks=2', '--cpus-per-task=1',
                    '--gres=gpu:2', '--time=120:00:00', 'mad_a0'):
            self.assertIn(arg, args)
        self.assertFalse(any('dependency' in x for x in args))

    def test_newer_numbered_beats_stale_final_and_archive(self):
        restart(self.run / 'gr_torus_sane.restart.final.rhdf', 100)
        restart(self.run / 'gr_torus_sane.restart.saved-1.rhdf', 200)
        latest = self.run / 'gr_torus_sane.restart.00003.rhdf'
        restart(latest, 300)
        self.assertEqual(checkpoint.candidates(self.run)[0][2], str(latest))

    def test_incomplete_xml_and_empty_files_skipped(self):
        path = self.run / 'gr_torus_sane.restart.00003.rhdf'
        restart(path, 300)
        Path(str(path) + '.xdmf').write_text('<Xdmf>')
        self.assertEqual(checkpoint.candidates(self.run), [])

    def test_initial_run_no_prequeue_no_wall_timer(self):
        p = self.run_script('job.sh')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse(self.submissions())
        self.assertFalse((self.run / 'DONE').exists())

    def test_restart_fallback_validated_by_pangu(self):
        good = self.run / 'gr_torus_sane.restart.00001.rhdf'
        restart(good, 100)
        restart(self.run / 'gr_torus_sane.restart.00002.rhdf', 200, b'bad_payload')
        p = self.run_script('job.sh', '99', MOCK_TIME='300')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('Rejected restart', p.stdout)
        self.assertIn('00001.rhdf', (self.run / 'logs/command-100.txt').read_text())

    def test_ignores_old_completed_logs(self):
        (self.run / 'logs/run-0.log').write_text('Driver completed.\ntime=30000 cycle=100\n')
        restart(self.run / 'gr_torus_sane.restart.final.rhdf', 100)
        p = self.run_script('job.sh', '99', MOCK_TIME='200')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse((self.run / 'DONE').exists())

    def test_reaches_target_stops(self):
        p = self.run_script('job.sh', MOCK_TIME='30000')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue((self.run / 'DONE').exists())

    def test_rounded_xml_time_cannot_mark_complete(self):
        restart(self.run / 'gr_torus_sane.restart.final.rhdf', 30000)
        p = self.run_script('job.sh', '99', MOCK_TIME='29999.99')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse((self.run / 'DONE').exists())

    def test_all_restarts_bad_refuses_cold_start(self):
        restart(self.run / 'gr_torus_sane.restart.00001.rhdf', 100, b'bad_payload')
        p = self.run_script('job.sh', '99')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('refusing', p.stderr)

    def test_monitor_submits_after_timeout_only(self):
        (self.run / 'job.id').write_text('99\n')
        restart(self.run / 'gr_torus_sane.restart.00001.rhdf', 100)
        p = self.run_script('watch.sh', STOP_AFTER_SUBMIT='1')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.submissions()[0][-2:], ['mad_a0', '99'])

    def test_active_job_not_duplicated(self):
        (self.run / 'job.id').write_text('99\n')
        p = self.run_script('submit.sh', MOCK_QUEUE='RUNNING')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse(self.submissions())

    def test_monitor_failure_and_stop(self):
        (self.run / 'job.id').write_text('99\n')
        p = self.run_script('watch.sh', MOCK_STATE='FAILED')
        self.assertNotEqual(p.returncode, 0)
        self.assertFalse(self.submissions())
        (self.run / 'STOP').touch()
        self.assertEqual(self.run_script('watch.sh').returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
